import torch
import torch.distributed as dist
from tqdm import tqdm
from accelerate import Accelerator
import os
import json
from transformers import PreTrainedTokenizer
from datetime import datetime
import torch.nn.functional as F


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        dataloader: torch.utils.data.DataLoader,
        accelerator: Accelerator,
        ref_model: torch.nn.Module,
        args=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.accelerator: Accelerator = accelerator
        self.ref_model = ref_model
        self.args = args
        self.max_train_steps = args.max_train_steps
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        if self.args.sparse_mode == "sepllm":
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
                if self.args.sparse_mode == "sepllm":
                    batch["sep_ids"] = self.sep_ids
                logits = self.model(**batch)
                if self.args.use_kl_loss:
                    if self.args.ref_model_divice is None:
                        # each process will load a ref model
                        labels = batch.pop("labels")
                        with torch.no_grad():
                            ref_logits = self.ref_model(**batch).logits
                        loss, token_mean_loss = self.kl_loss(logits, ref_logits, labels)
                        token_mean_loss /= self.gradient_accumulation_steps
                    else:
                        # allreduce input to main process
                        # scatter output to each process
                        raise NotImplementedError("Not implemented")
                else:
                    loss = self.cross_entropy_loss(
                        logits, batch["labels"], self.model.module.config.vocab_size
                    )
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
                    if tqdm_bar is not None:
                        tqdm_bar.update(1)
                    lr = self.scheduler.get_last_lr()[0]
                    status_dict = {
                        "train/lr": lr,
                    }
                    reduced_loss = torch.tensor(
                        acc_loss, device=self.accelerator.device
                    )
                    torch.distributed.all_reduce(
                        reduced_loss, op=torch.distributed.ReduceOp.AVG
                    )
                    # tensor /= torch.distributed.get_world_size()
                    status_dict["train/loss"] = reduced_loss.item()
                    status_dict["train/epoch"] = step / step_per_epoch
                    if tqdm_bar is not None:
                        tqdm_bar.set_postfix(**status_dict)

                    self.optimizer.zero_grad()
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

    def save_checkpoint(self, step):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        save_path = os.path.join(
            self.args.output_dir, f"{self.args.exp_name}/checkpoint-{step}"
        )
        unwrapped_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
