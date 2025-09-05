import json
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoConfig
from time import time
from datasets import load_dataset

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

INPUT_KEY = {
    "math-ai/aime24": "problem",
    "zwhe99/amc23": "question",
    "agentica-org/DeepScaleR-Preview-Dataset": "problem",
}
TARGET_KEY = {
    "math-ai/aime24": "solution",
    "zwhe99/amc23": "answer",
    "agentica-org/DeepScaleR-Preview-Dataset": "answer",
}
SPLIT = {
    "math-ai/aime24": "test",
    "zwhe99/amc23": "test",
    "agentica-org/DeepScaleR-Preview-Dataset": "train",
}


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


def load_eval_dataset(
    dataset_name_or_path, tokenizer, input_key=None, target_key=None, split_len=None
):
    split = "test"
    if dataset_name_or_path in INPUT_KEY:
        input_key = INPUT_KEY[dataset_name_or_path]
        target_key = TARGET_KEY[dataset_name_or_path]
        split = SPLIT[dataset_name_or_path]
    else:
        assert (
            input_key is not None
        ), "input_key is not provided for dataset: {dataset_name_or_path}"
        assert (
            target_key is not None
        ), "target_key is not provided for dataset: {dataset_name_or_path}"

    prompts = []
    data = []
    dataset = load_dataset(dataset_name_or_path, split=split)

    if split_len is not None:
        dataset = dataset.select(range(0, split_len))

    for index, item in enumerate(dataset):
        question = item[input_key]
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        for _ in range(args.n_sample):
            prompts.append(prompt)
            if not isinstance(item[target_key], str):
                item[target_key] = str(item[target_key])
            data.append(
                {"question": question, "answer": item[target_key], "index": index}
            )
    return prompts, data


def process_output(output, input_len, tokenizer):
    output_dis = output.sequences[:, input_len:]
    num_pad = (output_dis == tokenizer.pad_token_id).sum(dim=1)
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        num_pad = torch.clamp(num_pad - 1, min=0)

    output_tokens = (output_dis.shape[1] - num_pad).tolist()

    batch_outputs = tokenizer.batch_decode(
        output_dis,
        skip_special_tokens=True,
    )

    pos_ids = None
    if hasattr(output.past_key_values, "pos_ids_cache"):
        pos_ids_list = []
        for layer_idx in sorted(output.past_key_values.pos_ids_cache.keys()):
            pos_ids_list.append(
                output.past_key_values.pos_ids_cache[layer_idx].unsqueeze(1)
            )
        # (bsz, layer, num_kv_heads, seq_len)
        pos_ids = torch.cat(pos_ids_list, dim=1).cpu().numpy()
    return batch_outputs, output_tokens, pos_ids


@torch.no_grad()
def generate(model, tokenizer, batch_prompts, sample_args):

    inputs = tokenizer(
        batch_prompts,
        padding="longest",
        return_tensors="pt",
        add_special_tokens=False,
    ).to("cuda")

    prefill_tokens = inputs["attention_mask"].sum(dim=1).tolist()

    start_time = time()
    output = model.generate(
        **inputs,
        **sample_args,
        return_dict_in_generate=True,
    )
    end_time = time()

    batch_outputs_text, output_tokens, pos_ids = process_output(
        output, inputs["input_ids"].shape[1], tokenizer
    )
    torch.cuda.empty_cache()
    return (
        prefill_tokens,
        output_tokens,
        batch_outputs_text,
        end_time - start_time,
        pos_ids,
    )


def main(args):
    set_seed(args.seed)

    # ====== build compression config ======
    config = AutoConfig.from_pretrained(args.model_path)
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
        "record_pos_ids": args.record_pos_ids,
        "return_sparse_mask": False,
        "divide_length": args.divide_length,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, padding_side="left"
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.method.lower() == "fullkv":
        from transformers import AutoModelForCausalLM
    else:
        from gkv.model import AutoModelForCausalLM
        from gkv.model.gen_patch import patch_sample

        config.update(compression_config)
        patch_sample()

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).cuda()

    model.eval()

    prompts, data = load_eval_dataset(
        args.dataset_path,
        tokenizer,
        input_key=args.input_key,
        target_key=args.target_key,
        split_len=args.split_len,
    )
    # sampling config
    if args.do_sample:
        sample_args = {
            "do_sample": True,
            "max_new_tokens": args.max_new_tokens,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
    else:
        sample_args = {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "top_p": None,
            "use_cache": True,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }

    info = {
        "time_per_batch": [],
        "tokens_per_batch": [],
        "pos_ids": [],
    }
    with open(args.save_path, "w") as f:
        for i in tqdm(range(0, len(prompts), args.eval_batch_size)):
            batch_prompts = prompts[i : i + args.eval_batch_size]
            prefill_tokens, output_tokens, output_text, time_per_batch, pos_ids = (
                generate(model, tokenizer, batch_prompts, sample_args)
            )
            all_tokens = sum(output_tokens)

            info["time_per_batch"].append(time_per_batch)
            info["pos_ids"].append(pos_ids)
            info["tokens_per_batch"].append(all_tokens)

            tqdm.write(f"throughput: {all_tokens / time_per_batch:.2f} tokens/s")
            for j in range(len(output_text)):
                data[i + j]["output_text"] = output_text[j]
                data[i + j]["prefill_tokens"] = prefill_tokens[j]
                data[i + j]["output_tokens"] = output_tokens[j]
                f.write(json.dumps(data[i + j], ensure_ascii=False) + "\n")

    print(f"total tokens: {sum(info['tokens_per_batch'])}")
    print(f"total time: {sum(info['time_per_batch'])}")
    print(
        f"average throughput: {sum(info['tokens_per_batch']) / sum(info['time_per_batch']):.2f} tokens/s"
    )

    np.save(args.save_path.replace(".jsonl", "_info.npy"), np.array(info, dtype=object))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    # dataset
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--input_key", type=str, default=None)
    parser.add_argument("--target_key", type=str, default=None)
    parser.add_argument("--split_len", type=int, default=None)

    parser.add_argument("--save_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--eval_batch_size", type=int, default=1)

    # sampling config
    parser.add_argument("--n_sample", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true", default=False)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)

    # method config
    parser.add_argument(
        "--method",
        type=str,
        default="score",
        choices=["score", "sepllm", "streamingllm", "fullkv"],
    )
    # basic
    parser.add_argument("--budget", type=int, default=128)
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
    args = parser.parse_args()

    main(args)
