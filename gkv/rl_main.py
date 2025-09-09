from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import get_scheduler
from transformers import AutoTokenizer, AutoConfig
import torch.distributed as dist
from .dataloader.rl_dataloader import get_dataloader
from .trainer.grpo_trainer import Trainer
from .model.gen_patch import patch_sample
from .reward.math_reward_fn import compute_score
from .model import AutoModelForCausalLM


def check_bsz(args, accelerator):
    args.gradient_accumulation_steps = (
        accelerator.deepspeed_plugin.gradient_accumulation_steps
    )
    update_bsz_per_gpu = (
        args.gradient_accumulation_steps * args.train_micro_batch_size_per_gpu
    )
    assert (
        args.train_batch_size_per_gpu * args.sample_n
    ) == update_bsz_per_gpu, "only suport online RL, make sure train_batch_size_per_gpu*sample_n is equal to update_bsz_per_gpu"


def main(args):
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left")
    # set KV compression config
    config = AutoConfig.from_pretrained(args.model_name)
    compression_config = {
        "method": args.method,
        "method_config": {
            "budget": args.budget,
            "window_size": args.window_size,
            "compress_mode": args.compress_mode,
            "compress_ratio": args.compress_ratio,
            "sink_len": args.sink_len,
            "enable_pooling": args.enable_pooling,
            "suppressing_redundancy": args.suppressing_redundancy,
            "mix_lambda": args.mix_lambda,
            "retain_ratio": args.retain_ratio,
            "retain_direction": args.retain_direction,
            "enable_score_cache": args.enable_score_cache,
            "smooth_method": args.smooth_method,
            "alpha": args.alpha,
            "disable_norm": args.disable_norm,
        },
        "record_pos_ids": True,
        "return_sparse_mask": True,
        "divide_length": args.divide_length,
    }
    config.update(compression_config)
    patch_sample()
    #
    accelerator = Accelerator()
    # dataset
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    train_dataloader, eval_dataloader = get_dataloader(
        args.dataset_path,
        tokenizer,
        args.train_batch_size_per_gpu,
        args.eval_split_len,
        world_size,
    )
    # model

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        attn_implementation="flash_attention_2",
    )
    model.model.gradient_checkpointing_enable()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_warmup_steps = int(args.max_train_steps * args.warmup_ratio * world_size)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps * world_size,
    )
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
    )
    check_bsz(args, accelerator)
    reward_fn = compute_score
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        accelerator=accelerator,
        reward_fn=reward_fn,
        args=args,
    )
    trainer.train()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    # sample
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--sample_n", type=int, default=8)
    # kv compression
    parser.add_argument(
        "--method",
        type=str,
        default="score",
        choices=["score", "sepllm", "streamingllm", "fullkv"],
    )
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--divide_length", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--enable_pooling", action="store_true", default=False)

    parser.add_argument("--suppressing_redundancy", action="store_true", default=False)
    parser.add_argument("--retain_ratio", type=float, default=0.2)
    parser.add_argument(
        "--retain_direction", type=str, default="last", choices=["last", "first"]
    )
    parser.add_argument("--mix_lambda", type=float, default=0.5)
    parser.add_argument(
        "--compress_mode", type=str, default="budget", choices=["budget", "ratio"]
    )
    parser.add_argument("--compress_ratio", type=float, default=0.2)

    parser.add_argument("--enable_score_cache", action="store_true", default=False)
    parser.add_argument(
        "--smooth_method", type=str, default="max", choices=["mean", "max"]
    )
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--disable_norm", action="store_true", default=False)

    parser.add_argument("--record_pos_ids", action="store_true", default=False)
    parser.add_argument("--sink_len", type=int, default=4)
    # train
    parser.add_argument("--trunk_length", type=int, default=6144)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="constant_with_warmup")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload_sparse_mask", action="store_true", default=False)
    parser.add_argument("--clip_overlength_advantage", action="store_true", default=False)    
    # eval
    parser.add_argument("--eval_do_sample", action="store_true", default=False)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--eval_batch_size_per_gpu", type=int, default=128)
    parser.add_argument("--eval_sample_n", type=int, default=4)
    parser.add_argument("--eval_temperature", type=float, default=0.6)
    parser.add_argument("--eval_split_len", type=int, default=32)
    # set in deepspeed config
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    # parser.add_argument("--micro_batch_size_per_gpu", type=int, default=1)
    parser.add_argument(
        "--train_batch_size_per_gpu",
        type=int,
        default=2,
        help="data batch size",
    )
    parser.add_argument("--train_micro_batch_size_per_gpu", type=int, default=1)
    parser.add_argument("--generate_batch_size_per_gpu", type=int, default=128)

    # log and save
    parser.add_argument(
        "--log_method",
        type=str,
        choices=["wandb", "tensorboard", "none"],
        default=["wandb", "tensorboard"],
    )
    parser.add_argument("--wandb_project", type=str, default="sparse_kv_rl")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--exp_name", type=str, default="exp")
    parser.add_argument("--save_steps", type=int, default=100)

    args = parser.parse_args()
    if not args.eval_do_sample:
        args.eval_sample_n = 1
    main(args)
