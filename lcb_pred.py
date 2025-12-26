import argparse

from lcb_runner.utils.scenarios import Scenario
from lcb_runner.lm_styles import LanguageModelStore
from utils.lcb_utils import load_lcb_codegeneration_dataset, GKVRunner

from lcb_runner.runner.vllm_runner import VLLMRunner
import json


def main(args):
    model = LanguageModelStore[args.model]
    benchmark, format_prompt = load_lcb_codegeneration_dataset(args)
    if not args.method == "fullkv":
        runner = GKVRunner(args, model)
    else:
        runner = VLLMRunner(args, model)
    results: list[list[str]] = runner.run_main(benchmark, format_prompt)
    save_info = {
        "outputs": results,
    }
    if not args.method == "fullkv":
        save_info["input_len"] = runner.loginfo["input_len"]
        save_info["output_len"] = runner.loginfo["output_len"]
        save_info["method"] = args.method
        save_info["budget"] = args.budget
        save_info["window_size"] = args.window_size

    with open(args.save_path, "w") as f:
        json.dump(save_info, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
    )
    parser.add_argument("--save_path", type=str, default="outputs/lcb_test.json")
    parser.add_argument(
        "--scenario",
        type=Scenario,
        default=Scenario.codegeneration,
        help="Type of scenario to run",
    )
    parser.add_argument(
        "--n", type=int, default=10, help="Number of samples to generate"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--codegen_n",
        type=int,
        default=10,
        help="Number of samples for which code generation was run (used to map the code generation file during self-repair)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for sampling"
    )
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for sampling")
    parser.add_argument(
        "--max_new_tokens", type=int, default=2000, help="Max tokens for sampling"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default=None,
        help="Start date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default=None,
        help="End date for the contest to filter the evaluation file (format - YYYY-MM-DD)",
    )

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
        "--smooth_method", type=str, default="max", choices=["mean", "max", "sum"]
    )
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--disable_norm", action="store_true", default=False)

    # Info
    parser.add_argument("--record_pos_ids", action="store_true", default=False)

    args = parser.parse_args()
    # vllm args
    args.dtype = "bfloat16"
    args.enable_prefix_caching = True
    args.trust_remote_code = True
    args.tensor_parallel_size = 1
    args.max_tokens = args.max_new_tokens
    args.model_name = args.model
    args.stop = None
    args.num_process_evaluate = 32
    args.timeout = 6
    args.local_model_path = None
    if not args.method == "fullkv":
        args.use_cache = True
        args.cache_batch_size = args.batch_size
    else:
        args.use_cache = False

    main(args)
