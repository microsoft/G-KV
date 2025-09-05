import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoConfig
from .dataloader.sft_dataloader import get_dataloader
from accelerate import Accelerator
from .trainer.sft_trainer import Trainer
from accelerate.utils import set_seed
from transformers import get_scheduler
from torch.optim import AdamW
from gkv.model.gen_patch import patch_sample


def main(args):
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)

    compression_config = {
        "method": args.method,
        "method_config": {
            # for evaluation and rebuild sparse mask
            "enable_score_cache": True,
            "suppressing_redundancy": True,
            "budget": args.budget,
            "window_size": args.window_size,
            "sink_len": args.sink_len,
            "mix_lambda": args.mix_lambda,
            "alpha": args.alpha,
        },
        "record_pos_ids": False,
        "return_sparse_mask": False,
        "divide_length": args.divide_length,
    }
    config.update(compression_config)

    accelerator = Accelerator()

    from gkv.model import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        config=config,
        attn_implementation="flash_attention_2",
    )
    
    patch_sample()
    model.model.gradient_checkpointing_enable()
    model.train()
    ref_model = None

    if args.use_kl_loss:
        from transformers import AutoModelForCausalLM

        ref_model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    else:
        world_size = 1
    num_warmup_steps = int(args.max_train_steps * args.warmup_ratio * world_size)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=args.max_train_steps * world_size,
    )
    # dataset
    dataloader = get_dataloader(args.dataset_path, tokenizer, args.max_output_len)
    eval_dataloader = None
    if args.eval_dataset_path is not None:
        from gkv.dataloader.sft_dataloader import get_eval_dataloader

        eval_dataloader = get_eval_dataloader(
            args.eval_dataset_path,
            tokenizer,
            args.eval_split_len,
            world_size,
        )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )
    if eval_dataloader is not None:
        eval_dataloader = accelerator.prepare(eval_dataloader)

    if accelerator.deepspeed_plugin is not None and hasattr(
        accelerator.deepspeed_plugin, "gradient_accumulation_steps"
    ):
        args.gradient_accumulation_steps = (
            accelerator.deepspeed_plugin.gradient_accumulation_steps
        )
    else:
        args.gradient_accumulation_steps = 1

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=lr_scheduler,
        dataloader=dataloader,
        eval_dataloader=eval_dataloader,
        accelerator=accelerator,
        ref_model=ref_model,
        args=args,
    )
    trainer.train()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # model
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--ref_model_divice", type=int, default=None)
    parser.add_argument("--ref_model_offload", action="store_true")

    # dataset
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--max_output_len", type=int, default=None)
    # sparse
    parser.add_argument("--divide_length", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=16)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--mix_lambda", type=float, default=0.5)
    parser.add_argument(
        "--method",
        type=str,
        choices=["sepllm", "streamingllm", "score"],
        default="dynamic",
    )
    parser.add_argument("--sink_len", type=int, default=4)
    parser.add_argument(
        "--kept_sep",
        type=str,
        nargs="+",
        default=[".", ",", "?", "!", ";", ":", " ", "\t", "\n"],
    )
    # train
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--use_kl_loss", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.6)
    # eval
    parser.add_argument("--eval_dataset_path", type=str, default=None)
    parser.add_argument("--eval_steps", type=int, default=10)
    parser.add_argument("--eval_batch_size_per_gpu", type=int, default=128)
    parser.add_argument("--eval_sample_n", type=int, default=4)
    parser.add_argument("--eval_split_len", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=6144)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--eval_do_sample", action="store_true")

    # log
    parser.add_argument(
        "--log_method",
        type=str,
        choices=["wandb", "tensorboard", "none"],
        default=["wandb", "tensorboard"],
    )
    parser.add_argument("--wandb_project", type=str, default="sparse_kv_training")

    # other
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--exp_name", type=str, default="default")
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.eval_do_sample:
        args.eval_sample_n = 1
    main(args)
