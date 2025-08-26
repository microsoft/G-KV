import random
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from transformers import AutoTokenizer, AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from train.model.modeling_sparse_qwen2 import Qwen2SparseModelForCausalLM
from train.dataloader.dataloader import get_dataloader
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from accelerate import Accelerator
from train.trainer.trainer import Trainer
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from transformers import get_scheduler
from torch.optim import AdamW





def main(args):
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)
    config.compress_step = args.compress_step
    config.window_size = args.window_size
    config.kv_budget = args.kv_budget
    config.alpha = args.alpha
    config.mix_lambda = args.mix_lambda

    accelerator = Accelerator()

    model = Qwen2SparseModelForCausalLM.from_pretrained(
        args.model_name,
        config=config
    )
    model.model.gradient_checkpointing_enable()
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
    # dataset
    dataloader = get_dataloader(args.dataset_path, tokenizer,args.max_output_len)

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler)
    
    if accelerator.deepspeed_plugin is not None and hasattr(accelerator.deepspeed_plugin, 'gradient_accumulation_steps'):
        args.gradient_accumulation_steps = accelerator.deepspeed_plugin.gradient_accumulation_steps
    else:
        args.gradient_accumulation_steps = 1


    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        dataloader=dataloader,
        accelerator=accelerator,
        args=args,
    )
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--model_name", type=str, required=True)

    # dataset
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--max_output_len", type=int, default=None)
    # sparse
    parser.add_argument("--compress_step", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--kv_budget", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--mix_lambda", type=float, default=0.5)
    parser.add_argument(
        "--sparse_mode",
        type=str,
        choices=["sepllm", "stream", "dynamic"],
        default="sparse",
    )
    # train
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--deepspeed_config", type=str, default='./configs/deepspeed_stage2.json')

    # other
    parser.add_argument("--output_dir", type=str, default='./checkpoints')
    parser.add_argument("--exp_name", type=str, default='default')
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
