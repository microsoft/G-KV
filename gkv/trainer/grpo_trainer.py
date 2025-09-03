from gkv.trainer.grpo_utils.experience_maker import ExperienceMaker, SamplesGenerator
from gkv.trainer.grpo_utils.actor import Actor
from gkv.trainer.grpo_utils.experience_maker import Experience
from typing import List
import torch
from gkv.trainer.grpo_utils.utils import masked_mean
import os
from datetime import datetime
from tqdm import tqdm
import torch.distributed as dist


class Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        optimizer,
        scheduler,
        train_dataloader,
        eval_dataloader,
        accelerator,
        reward_fn,
        args,
    ):
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.accelerator = accelerator
        self.args = args
        self.actor = Actor(model, tokenizer, accelerator, args)
        self.samples_generator = SamplesGenerator(
            self.actor, tokenizer, accelerator, args
        )
        self.experience_maker = ExperienceMaker(reward_fn, tokenizer, accelerator, args)
        self.max_train_steps = args.max_train_steps

        if self.accelerator.is_main_process:
            log_dir = os.path.join(args.output_dir, args.exp_name, "runs")
            # tensorboard init
            try:
                from tensorboardX import SummaryWriter

                self.tb_writer = SummaryWriter(log_dir=log_dir)
            except ImportError:
                print("tensorboardX is not installed, skipping tensorboard logging.")
                self.tb_writer = None
            # wandb init
            try:
                import wandb

                wandb.init(
                    project=getattr(args, "wandb_project", "default_project"),
                    name=args.exp_name + datetime.now().strftime("%Y%m%d_%H%M%S"),
                    dir=args.output_dir,
                    config=vars(args) if args is not None else {},
                    reinit=True,
                )
                self.wandb = wandb
            except ImportError:
                print("wandb is not installed, skipping wandb logging.")
                self.wandb = None

        else:
            self.tb_writer = None
            self.wandb = None

    def train(self):
        steps = 0
        step_per_spisode = len(self.train_dataloader)
        if self.accelerator.is_main_process:
            tqdm_bar = tqdm(
                total=self.max_train_steps,
                desc="Training",
            )
        else:
            tqdm_bar = None
        while steps < self.max_train_steps:
            for batch in self.train_dataloader:
                experiences = self.samples_generator.generate_samples(batch)
                experiences = self.experience_maker.make_experience_batch(experiences)
                torch.cuda.empty_cache()
                status_dict = self.train_step(experiences)
                # evaluation
                if (steps + 1) % self.args.eval_steps == 0:
                    eval_acc, eval_length = self.evaluate()
                    status_dict["eval/length"] = eval_length
                    status_dict["eval/acc"] = eval_acc

                # log info
                status_dict["train/rewards"] = []
                for exp in experiences:
                    for key, value in exp.info.items():
                        if key == "output_texts":
                            continue
                        if "gen/" + key not in status_dict:
                            status_dict["gen/" + key] = []
                        status_dict["gen/" + key].extend(value)
                    rewards = exp.rewards.mean().item()
                    status_dict["train/rewards"].append(rewards)
                # reduce local
                for key, value in status_dict.items():
                    if isinstance(value, float):
                        continue
                    status_dict[key] = sum(value) / len(value)
                # reduce across all processes
                world_size = torch.distributed.get_world_size()
                obj_list = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(obj_list, status_dict)
                for key in status_dict.keys():
                    status_dict[key] = sum([obj[key] for obj in obj_list]) / len(
                        obj_list
                    )

                status_dict["train/lr"] = self.scheduler.get_last_lr()[0]
                self.scheduler.step()
                steps += 1
                status_dict["train/epoch"] = steps / step_per_spisode
                if tqdm_bar is not None:
                    tqdm_bar.update(1)
                    simple_status_dict = {
                        "seq_l": status_dict["gen/sequence_length"],
                        "in_l": status_dict["gen/input_len"],
                        "out_l": status_dict["gen/output_len"],
                        "rw": status_dict["train/rewards"],
                        "loss": status_dict["train/loss"],
                        "etp": status_dict["train/entropy"],
                        "lr": status_dict["train/lr"],
                        "epc": status_dict["train/epoch"],
                    }
                    tqdm_bar.set_postfix(**simple_status_dict)

                self.log_and_save(status_dict, steps)

    def train_step(self, batched_experiences: List[Experience]):
        self.actor.model.train()
        status_dict = {"train/loss": [], "train/entropy": []}
        for micro_batch_exp in batched_experiences:
            micro_batch_exp.to_device(self.accelerator.device)
            log_probs, mean_entropy = self.actor.forward(
                micro_batch_exp.sequences,
                micro_batch_exp.attention_mask,
                micro_batch_exp.sparse_mask,
                action_mask=micro_batch_exp.action_mask,
            )
            loss = self.compute_policy_loss(
                log_probs, micro_batch_exp.advantages, micro_batch_exp.action_mask
            )
            loss = loss / self.args.gradient_accumulation_steps
            self.accelerator.backward(loss)
            status_dict["train/loss"].append(loss.item())
            status_dict["train/entropy"].append(mean_entropy.item())
        # deepspeed will automatically step optimizer when backward is called
        # and the gradient will be accumulated by deepspeed
        self.optimizer.step()
        self.optimizer.zero_grad()
        return status_dict

    def compute_policy_loss(
        self,
        log_prob,
        advantages,
        response_mask,
    ):
        """
        Compute the clipped policy objective and related metrics for PPO.

        Adapted from
        https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

        Args:
            old_log_prob (torch.Tensor):
                Log-probabilities of actions under the old policy, shape (batch_size, response_length).
            log_prob (torch.Tensor):
                Log-probabilities of actions under the current policy, shape (batch_size, response_length).
            advantages (torch.Tensor):
                Advantage estimates for each action, shape (batch_size, response_length).
            response_mask (torch.Tensor):
                Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        """
        # only support on policy
        old_log_prob = log_prob.detach()
        negative_approx_kl = log_prob - old_log_prob
        # all values is 1
        ratio = torch.exp(negative_approx_kl)
        pg_loss = -advantages[:, :-1] * ratio
        pg_loss = masked_mean(pg_loss, response_mask[:, :-1])
        return pg_loss

    def log_and_save(self, status_dict, step):
        if self.accelerator.is_main_process:
            # tensorboard
            if self.tb_writer is not None:
                for key, value in status_dict.items():
                    self.tb_writer.add_scalar(key, value, step)
                self.tb_writer.flush()
            # wandb
            if self.wandb is not None:
                self.wandb.log(status_dict)

            # save checkpoint
            if step % self.args.save_steps == 0:
                self.save_checkpoint(step)
        dist.barrier()

    def evaluate(self):
        acc = []
        length = []
        for batch in self.eval_dataloader:
            prompts = []
            answers = []
            for item in batch:
                for _ in range(self.args.eval_sample_n):
                    prompts.append(item["prompt"])
                    answers.append(item["answer"])
            for i in range(0, len(prompts), self.args.eval_batch_size_per_gpu):
                batch_prompts = prompts[i : i + self.args.eval_batch_size_per_gpu]
                batch_answers = answers[i : i + self.args.eval_batch_size_per_gpu]
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    add_special_tokens=False,
                    padding=True,
                ).to(self.accelerator.device)
                sequences, _, _ = self.actor.generate(
                    inputs,
                    do_sample=self.args.eval_do_sample,
                    temperature=self.args.eval_temperature,
                )
                output_ids = sequences[:, inputs.input_ids.shape[1] :]
                output_texts = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                num_pad = (output_ids == self.tokenizer.pad_token_id).sum(dim=1)
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    num_pad = torch.clamp(num_pad - 1, min=0)
                seq_length = (output_ids.shape[1] - num_pad).tolist()
                length.extend(seq_length)
                for j in range(len(batch_prompts)):
                    acc.append(
                        self.experience_maker.reward_fn(
                            batch_answers[j], output_texts[j]
                        )
                    )

            acc = sum(acc) / len(acc)
            avg_length = sum(length) / len(length)
            torch.cuda.empty_cache()
            torch.distributed.barrier()
            return acc, avg_length

    def save_checkpoint(self, step):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        save_path = os.path.join(
            self.args.output_dir, f"{self.args.exp_name}/checkpoint-{step}"
        )
        unwrapped_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
