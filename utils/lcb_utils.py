import datasets
from lcb_runner.runner.base_runner import BaseRunner
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.prompts.code_generation import format_prompt_generation
from transformers import AutoConfig, AutoTokenizer
from gkv.inference_main import generate
import torch


def load_code_generation_dataset_not_fast() -> list[CodeGenerationProblem]:
    dataset = datasets.load_dataset(
        "json", data_files="datasets/livecodebench/test.jsonl", split="train"
    )
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset


def load_lcb_codegeneration_dataset(args) -> list[CodeGenerationProblem]:

    benchmark = load_code_generation_dataset_not_fast()
    benchmark = sorted(benchmark, key=lambda x: x.question_id)
    format_prompt = format_prompt_generation
    return benchmark, format_prompt


class GKVRunner(BaseRunner):
    def __init__(self, args, model):
        super().__init__(args, model)
        from gkv.model import AutoModelForCausalLM
        from gkv.model.gen_patch import patch_sample

        config = AutoConfig.from_pretrained(args.model)

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

        config.update(compression_config)
        patch_sample()

        self.llm = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)

        self.sampling_params = {
            "do_sample": True,
            "max_new_tokens": args.max_new_tokens,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "use_cache": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        self.sample_n = args.n

    def _run_single(self, prompt: str) -> list[str]:
        pass

    def run_batch(self, prompts: list[str]) -> list[list[str]]:
        outputs = [[] for _ in prompts]
        indices = []
        repeated_prompts = []
        for i in range(len(prompts)):
            indices.extend([i] * self.sample_n)
            repeated_prompts.extend([prompts[i]] * self.sample_n)

        prefill_tokens, output_tokens, output_text, time_per_batch, pos_ids = generate(
            self.llm, self.tokenizer, repeated_prompts, self.sampling_params
        )
        for i, output in zip(indices, output_text):
            outputs[i].append(output)
        return outputs
