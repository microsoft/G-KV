import json
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model import replace_llama, replace_qwen2
from time import time

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def set_seed(seed):
    """
    set seed
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_path):
    prompts = []
    data = []
    with open(dataset_path) as f:
        for index, line in enumerate(f):
            item = json.loads(line)
            question = item["question"]
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            prompt = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            item["prompt"] = prompt
            item["index"] = index
            for _ in range(args.n_sample):
                prompts.append(prompt)
                data.append(item)
    return prompts, data


def loop(args):
    fout = open(args.save_path, "w")

    times = []
    all_scores = []
    pos_ids_cache = []

    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        if i + args.eval_batch_size > len(prompts):
            batch_prompts = prompts[i:]
        else:
            batch_prompts = prompts[i : i + args.eval_batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
        ).to("cuda")

        prefill_lengths = tokenized_prompts["attention_mask"].sum(dim=1).tolist()
        start_time = time()
        with torch.no_grad():
            if args.attn_implementation == "flash_attention_2":
                output = model.generate(
                    **tokenized_prompts,
                    **sample_args,
                )
            else:
                # mixed precision acceleration
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    output = model.generate(
                        **tokenized_prompts,
                        **sample_args,
                    )
        end_time = time()
        times.append(end_time - start_time)

        if args.method == "score":
            # clear the score cache
            scores = model.clear_score_cache()
            if args.record_scores:
                all_scores.append(scores)

        if hasattr(model, "pos_ids_cache"):
            pos_ids_cache.append(model.pos_ids_cache)
            model.pos_ids_cache = None

        batch_token_stats = []
        for j in range(output.size(0)):
            total_tokens = int((output[j] != tokenizer.pad_token_id).sum().item())

            prefill = prefill_lengths[j]
            output_tokens = total_tokens - prefill

            batch_token_stats.append(
                {
                    "sample_idx": i + j,
                    "prefill_tokens": prefill,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                }
            )

        batch_outputs = tokenizer.batch_decode(
            [output[j][prefill_lengths[j] :] for j in range(output.size(0))],
            skip_special_tokens=True,
        )

        torch.cuda.empty_cache()

        for j in range(len(batch_outputs)):
            sample_idx = batch_token_stats[j]["sample_idx"]
            data[sample_idx]["prompt"] = batch_prompts[j]
            data[sample_idx]["output"] = batch_outputs[j]
            data[sample_idx]["prefill_tokens"] = batch_token_stats[j]["prefill_tokens"]
            data[sample_idx]["output_tokens"] = batch_token_stats[j]["output_tokens"]
            data[sample_idx]["total_tokens"] = batch_token_stats[j]["total_tokens"]
            data[sample_idx]["sample_idx"] = batch_token_stats[j]["sample_idx"]

            fout.write(json.dumps(data[sample_idx], ensure_ascii=False) + "\n")

    fout.close()
    np.save(
        args.save_path.replace(".jsonl", "_info.npy"),
        np.array(
            {"times": times, "scores": all_scores, "pos_ids": pos_ids_cache},
            dtype=object,
        ),
    )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    # sampling config
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)

    # method config
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["score", "sepllm", "streamingllm", "fullkv"],
    )
    # basic
    parser.add_argument("--kv_budget", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument(
        "--compress_mode", type=str, default="budget", choices=["budget", "ratio"]
    )
    parser.add_argument("--compress_ratio", type=float, default=0.2)
    parser.add_argument("--divide_length", type=int, default=128)
    # StreamingLLM
    parser.add_argument("--sink_len", type=int, default=4)
    # SepLLM
    parser.add_argument(
        "--kept_sep",
        type=str,
        nargs="+",
        default=[".", ",", "?", "!", ";", ":", " ", "\t", "\n"],
    )
    # SnapKV
    parser.add_argument("--enable_pooling", action="store_true", default=False)
    # R-KV
    parser.add_argument("--suppressing_redundancy", action="store_true", default=False)
    parser.add_argument("--mix_lambda", type=float, default=0.1)
    parser.add_argument("--retain_ratio", type=float, default=0.2)
    parser.add_argument(
        "--retain_direction", type=str, default="last", choices=["last", "first"]
    )
    # G-KV
    parser.add_argument("--enable_score_cache", action="store_true", default=False)
    parser.add_argument(
        "--smooth_method", type=str, default="max", choices=["mean", "max"]
    )
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--disable_norm", action="store_true", default=False)

    # Info
    parser.add_argument("--record_pos_ids", action="store_true", default=False)
    parser.add_argument("--record_scores", action="store_true", default=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)

    # ====== build compression config ======
    compression_config = {
        "method": args.method,
        "method_config": {
            "budget": args.kv_budget,
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
        "record_pos_ids": args.record_pos_ids,
    }
    model_config = {
        "divide_length": args.divide_length,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # apply monkey patch
    if args.method.lower() != "fullkv":
        if "llama" in args.model_path.lower():
            replace_llama(compression_config)
        elif "qwen" in args.model_path.lower():
            replace_qwen2(compression_config)
        else:
            raise ValueError(f"Unsupported model: {args.model_path}")

    if args.attn_implementation == "flash_attention_2":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            use_cache=True,
            attn_implementation=args.attn_implementation,
        ).cuda()
    else:
        # bf16 is numerically unstable, bf16 need to use with flash attention 2
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            use_cache=True,
            attn_implementation=args.attn_implementation,
        ).cuda()

    model.eval()

    model.config.update(model_config)

    # sampling config
    if args.do_sample:
        sample_args = {
            "do_sample": True,
            "max_length": args.max_length,
            "top_p": args.top_p,
            "temperature": args.temperature,
        }
    else:
        sample_args = {
            "do_sample": False,
            "max_length": args.max_length,
        }

    prompts, data = load_dataset(args.dataset_path)
    loop(args)
