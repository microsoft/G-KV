from datasets import load_dataset
from torch.utils.data import DataLoader
from functools import partial
import torch
from typing import Optional

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def collate_fn(batch, tokenizer):
    prompts = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["problem"]},
        ]
        for item in batch
    ]
    prompt_texts = tokenizer.apply_chat_template(
        prompts, tokenize=False, add_generation_prompt=True
    )
    for i in range(len(prompt_texts)):
        batch[i]["prompt"] = prompt_texts[i]
    return batch


def get_dataloader(
    dataset_path, tokenizer, bsz_per_gpu, eval_split_len=32, world_size=1
):
    dataset = load_dataset(dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)
    eval_dataset = dataset.select(range(0, eval_split_len))
    train_dataset = dataset.select(range(eval_split_len, len(dataset)))

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        batch_size=bsz_per_gpu,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=partial(collate_fn, tokenizer=tokenizer),
        batch_size=eval_split_len // world_size,
    )
    return train_dataloader, eval_dataloader
