import torch
import torch.distributed as dist
from tqdm import tqdm
from accelerate import Accelerator
import os
import json
from transformers import PreTrainedTokenizer
from datetime import datetime
import torch.nn.functional as F
from gkv.reward.math_reward_fn import compute_score


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        dataloader: torch.utils.data.DataLoader,
        eval_dataloader: torch.utils.data.DataLoader,
        accelerator: Accelerator,
        ref_model: torch.nn.Module,
        args=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.eval_dataloader = eval_dataloader
        self.accelerator: Accelerator = accelerator
        self.ref_model = ref_model
        self.args = args
        self.max_train_steps = args.max_train_steps
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.evaluate_fn = compute_score

        if self.args.method == "sepllm":
            self.sep_ids = torch.LongTensor(
                self.tokenizer(args.kept_sep, add_special_tokens=False).input_ids
            ).reshape(-1)
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
        self.model.train()
        if self.accelerator.is_main_process:
            tqdm_bar = tqdm(
                total=self.max_train_steps,
                desc="Training",
            )
        else:
            tqdm_bar = None
        step = 0
        update_step = 0
        acc_loss = 0
        step_per_epoch = len(self.dataloader)

        while update_step < self.max_train_steps:
            for _, batch in enumerate(self.dataloader):
                if update_step >= self.max_train_steps:
                    break
                if self.args.method == "sepllm":
                    batch["sep_ids"] = self.sep_ids
                labels = batch.pop("labels")
                self.model.train()
                output = self.model(**batch, use_cache=False)
                output.logits = output.logits.to(torch.float32)
                if self.args.use_kl_loss:
                    if self.args.ref_model_divice is None:
                        # each process will load a ref model
                        with torch.no_grad():
                            self.ref_model.to(self.accelerator.device)
                            ref_logits = self.ref_model(
                                **batch, use_cache=False
                            ).logits.to(torch.float32)
                            if self.args.ref_model_offload:
                                self.ref_model.to("cpu")
                        loss, token_mean_loss = self.kl_loss(
                            output.logits, ref_logits, labels
                        )
                        token_mean_loss /= self.gradient_accumulation_steps
                    else:
                        # allreduce input to main process
                        # scatter output to each process
                        raise NotImplementedError("Not implemented")
                else:
                    loss = self.cross_entropy_loss(
                        output.logits, labels, self.model.module.config.vocab_size
                    )
                torch.cuda.empty_cache()
                loss = loss / self.gradient_accumulation_steps
                self.accelerator.backward(loss)
                if self.args.use_kl_loss:
                    acc_loss += token_mean_loss
                else:
                    acc_loss += loss.item()
                step += 1
                if (step) % self.gradient_accumulation_steps == 0:
                    update_step += 1
                    # deepspeed will automatically step optimizer when backward is called
                    # the following code actually does nothing
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    lr = self.scheduler.get_last_lr()[0]
                    status_dict = {
                        "train/lr": lr,
                        "train/loss": acc_loss,
                    }
                    if update_step % self.args.eval_steps == 0 and self.eval_dataloader is not None:
                        acc = self.evaluate()
                        status_dict["eval/acc"] = acc
                    # tensor /= torch.distributed.get_world_size()
                    world_size = torch.distributed.get_world_size()
                    obj_list = [None for _ in range(world_size)]
                    torch.distributed.all_gather_object(obj_list, status_dict)
                    for key in status_dict.keys():
                        status_dict[key] = sum([obj[key] for obj in obj_list]) / len(obj_list)
                    status_dict["train/epoch"] = step / step_per_epoch
                    if tqdm_bar is not None:
                        tqdm_bar.update(1)
                        tqdm_bar.set_postfix(**status_dict)
                    self.scheduler.step()
                    self.log_and_save(status_dict, update_step)
                    torch.distributed.barrier()
                    acc_loss = 0
                

                torch.cuda.empty_cache()
            if self.accelerator.is_main_process:
                pass

    def cross_entropy_loss(self, logits, labels, vocab_size):
        logits = logits[:, :-1, :]
        labels = labels[:, 1:]
        loss = F.cross_entropy(
            logits.view(-1, vocab_size), labels.view(-1), ignore_index=-100
        )
        return loss

    def kl_loss(self, logits, ref_logits, label, **kwargs):
        """
        soft target loss
        """
        p = F.log_softmax(logits.div(self.args.temperature), dim=-1)
        q = F.softmax(ref_logits.div(self.args.temperature), dim=-1)
        loss = F.kl_div(p, q, reduction="none").sum(dim=-1)
        mask = label != -100
        loss = loss * mask
        with torch.no_grad():
            token_mean_loss = (loss.sum() / mask.sum()).item()
        loss = loss.sum(dim=-1) / logits.shape[0]
        return loss * (self.args.temperature**2), token_mean_loss

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

    def evaluate(self):
        acc = []
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
                sequences = self.generate(
                    inputs,
                    do_sample=self.args.eval_do_sample,
                )
                output_ids = sequences[:, inputs.input_ids.shape[1] :]
                output_texts = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                for j in range(len(batch_prompts)):
                    acc.append(self.evaluate_fn(batch_answers[j], output_texts[j]))

            acc = sum(acc) / len(acc)
            torch.cuda.empty_cache()
            torch.distributed.barrier()
            return acc

    def generate(self, inputs, do_sample=True, temperature=None):
        unwrap_model = self.accelerator.unwrap_model(self.model)
        unwrap_model.eval()
        if do_sample:
            sample_args = {
                "do_sample": True,
                "max_new_tokens": self.args.max_new_tokens,
                "temperature": (
                    self.args.temperature if temperature is None else temperature
                ),
                "top_p": self.args.top_p,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "return_dict_in_generate": True,
            }
        else:
            sample_args = {
                "do_sample": False,
                "max_new_tokens": self.args.max_new_tokens,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "return_dict_in_generate": True,
                "top_p": None,
            }
        outputs = unwrap_model.generate(
            **inputs,
            **sample_args,
        )
        return outputs.sequences

    def save_checkpoint(self, step):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        save_path = os.path.join(
            self.args.output_dir, f"{self.args.exp_name}/checkpoint-{step}"
        )
        unwrapped_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
