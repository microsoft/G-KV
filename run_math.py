import json
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ikv.monkeypatch import replace_llama, replace_qwen2, replace_qwen3


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


def main(args):
    fout = open(args.save_path, "w")
    system_prompt = (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )

    prompts = []
    test_data = []
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

    with open(args.dataset_path) as f:
        for index, line in enumerate(f):
            item = json.loads(line)
            question = item["question"]
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]
            prompt = tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            item["prompt"] = prompt
            item["index"] = index
            for _ in range(args.n_sample):
                prompts.append(prompt)
                test_data.append(item)

    for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
        if i+args.eval_batch_size>len(prompts):
            batch_prompts=prompts[i:]
        else:
            batch_prompts = prompts[i : i + args.eval_batch_size]
        tokenized_prompts = tokenizer(
            batch_prompts,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False,
        ).to("cuda")

        prefill_lengths = tokenized_prompts["attention_mask"].sum(dim=1).tolist()

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
            test_data[sample_idx]["prompt"] = batch_prompts[j]
            test_data[sample_idx]["output"] = batch_outputs[j]
            test_data[sample_idx]["prefill_tokens"] = batch_token_stats[j][
                "prefill_tokens"
            ]
            test_data[sample_idx]["output_tokens"] = batch_token_stats[j][
                "output_tokens"
            ]
            test_data[sample_idx]["total_tokens"] = batch_token_stats[j]["total_tokens"]
            test_data[sample_idx]["sample_idx"] = batch_token_stats[j]["sample_idx"]

            fout.write(json.dumps(test_data[sample_idx], ensure_ascii=False) + "\n")

    fout.close()


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
    parser.add_argument("--do_sample", action='store_true', default=False)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)

    # method config
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["rkv", "fullkv", "snapkv", "streamingllm", "h2o", "ikv"],
    )
    parser.add_argument("--kv_budget", type=int, default=128)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--first_tokens", type=int, default=4)
    parser.add_argument("--mix_lambda", type=float, default=0.1)
    parser.add_argument("--retain_ratio", type=float, default=0.2)
    parser.add_argument("--update_kv", type=bool, default=True)
    parser.add_argument(
        "--retain_direction", type=str, default="last", choices=["last", "first"]
    )
    parser.add_argument("--cross_salience_score", action='store_true', default=False)
    parser.add_argument("--enable_pooling", action='store_true', default=False)
    parser.add_argument("--suppressing_redundancy", action='store_true', default=False)
    parser.add_argument("--num_group", type=int, default=2)
    parser.add_argument("--enable_score_cache", action='store_true', default=False)
    parser.add_argument("--alpha", type=float, default=0.8)

    # model config
    parser.add_argument(
        "--divide_method",
        type=str,
        default="step_length",
        choices=["newline", "step_length"],
    )
    parser.add_argument("--divide_length", type=int, default=128)
    parser.add_argument(
        "--compression_content",
        type=str,
        default="all",
        choices=["think", "all"],
        help="whether to compress the whole model output or only the think part",
    )

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
            "mix_lambda": args.mix_lambda,
            "retain_ratio": args.retain_ratio,
            "retain_direction": args.retain_direction,
            "first_tokens": args.first_tokens,
            "suppressing_redundancy": args.suppressing_redundancy,
            "enable_pooling": args.enable_pooling,
            # ikv config
            "cross_salience_score": args.cross_salience_score,
            "num_group": args.num_group,
            "enable_score_cache": args.enable_score_cache,
            "alpha": args.alpha,
        },
        "compression": None,
        "update_kv": args.update_kv,
    }
    model_config = {
        "divide_method": args.divide_method,
        "divide_length": args.divide_length,
        "compression_content": args.compression_content,
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
        elif "qwen3" in args.model_path.lower():
            replace_qwen3(compression_config)
        elif "qwen" in args.model_path.lower():
            replace_qwen2(compression_config)
        else:
            raise ValueError(f"Unsupported model: {args.model_path}")
    
    if args.attn_implementation == "flash_attention_2":
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=True,
            attn_implementation=args.attn_implementation,
        )
    else:
        # bf16 is numerically unstable, bf16 need to use with flash attention 2
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            use_cache=True,
            attn_implementation=args.attn_implementation,
        )

    model.eval()

    model.config.update(model_config)

    if args.method.lower() != "fullkv":
        model.newline_token_ids = [
            tokenizer.encode("\n")[-1],
            tokenizer.encode(".\n")[-1],
            tokenizer.encode(")\n")[-1],
            tokenizer.encode("\n\n")[-1],
            tokenizer.encode(".\n\n")[-1],
            tokenizer.encode(")\n\n")[-1],
        ]

        model.after_think_token_ids = [
            tokenizer.encode("</think>")[-1],
        ]

    main(args)
