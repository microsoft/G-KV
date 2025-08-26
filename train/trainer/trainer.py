import torch
import torch.distributed as dist
from tqdm import tqdm
from accelerate import Accelerator
import os
import json
from tensorboardX import SummaryWriter
from transformers import PreTrainedTokenizer


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        dataloader: torch.utils.data.DataLoader,
        accelerator: Accelerator,
        args=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataloader = dataloader
        self.accelerator: Accelerator = accelerator
        self.args = args
        self.max_train_steps = args.max_train_steps
        self.gradient_accumulation_steps = args.gradient_accumulation_steps

        log_dir = os.path.join(args.output_dir, args.exp_name, "runs")
        self.tb_writer = SummaryWriter(log_dir=log_dir)

    def train(self):
        self.model.train()

        tqdm_bar = tqdm(
            total=self.max_train_steps,
            desc="Training",
        )
        step = 0
        update_step = 0
        acc_loss = 0

        while update_step < self.max_train_steps:
            for _, batch in enumerate(self.dataloader):
                if update_step >= self.max_train_steps:
                    break
                loss = self.model(**batch) / self.gradient_accumulation_steps
                self.accelerator.backward(loss)
                acc_loss += loss.item()
                step += 1
                if (step) % self.gradient_accumulation_steps == 0:
                    update_step += 1
                    # deepspeed will automatically step optimizer when backward is called
                    # the following code actually does nothing
                    self.optimizer.step()
                    tqdm_bar.update(1)
                    lr = self.scheduler.get_last_lr()[0]
                    status_dict = {
                        "lr": lr,
                    }
                    reduced_loss = torch.tensor(
                        acc_loss, device=self.accelerator.device
                    )
                    torch.distributed.all_reduce(
                        reduced_loss, op=torch.distributed.ReduceOp.AVG
                    )
                    # tensor /= torch.distributed.get_world_size()
                    status_dict['loss'] = reduced_loss.item()
                    tqdm_bar.set_postfix(**status_dict)

                    self.optimizer.zero_grad()
                    self.scheduler.step()
                    self.log_and_save(status_dict, update_step)
                    torch.distributed.barrier()
                    acc_loss = 0

                torch.cuda.empty_cache()
            if self.accelerator.is_main_process:
                pass

    def log_and_save(self, status_dict, step):
        if self.accelerator.is_main_process:
            for key, value in status_dict.items():
                self.tb_writer.add_scalar(key, value, step)
            self.tb_writer.flush()
            if step % self.args.save_steps == 0:
                self.save_checkpoint(step)

    def save_checkpoint(self, step):
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        save_path = os.path.join(
            self.args.output_dir, f"{self.args.exp_name}/checkpoint-{step}"
        )
        unwrapped_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
