from datasets import load_dataset
from torch.utils.data import DataLoader
from functools import partial
import torch
from typing import Optional

SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def collate_fn(batch, tokenizer, max_output_len: Optional[int] = None):
    prompts = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["question"]},
        ]
        for item in batch
    ]
    prompt_texts = tokenizer.apply_chat_template(
        prompts, tokenize=False, add_generation_prompt=True
    )
    output_text = [item["output"] for item in batch]
    prompt_ids = tokenizer(prompt_texts, add_special_tokens=False)["input_ids"]
    output_ids = tokenizer(output_text, add_special_tokens=False)["input_ids"]
    # add eos token
    output_ids = [
        output_ids[i] + [tokenizer.eos_token_id] for i in range(len(output_ids))
    ]
    input_len = max([len(prompt_ids[i]) for i in range(len(prompt_ids))])
    output_len = max(len(output_ids[i]) for i in range(len(output_ids)))
    res_dict = {
        "input_ids": [],
        "attention_mask": [],
        "position_ids": [],
        "labels": [],
        "input_length": input_len,
        "logits_to_keep": min(output_len,max_output_len),
    }

    for i in range(len(prompt_ids)):
        num_left_padding = input_len - len(prompt_ids[i])
        num_right_padding = output_len - len(output_ids[i])
        trunked_len=input_len+min(output_len,max_output_len)
        input_ids = (
            [tokenizer.pad_token_id] * num_left_padding
            + prompt_ids[i]
            + output_ids[i]
            + [tokenizer.pad_token_id] * num_right_padding
        )[:trunked_len]

        res_dict["input_ids"].append(input_ids)
        res_dict["attention_mask"].append(
            ([0] * num_left_padding + [1] * (len(input_ids) - num_left_padding))[:trunked_len]
        )
        position_ids = (
            [0] * num_left_padding
            + list(range(len(prompt_ids[i]) + len(output_ids[i])))
            + [0] * num_right_padding
        )[:trunked_len]
        res_dict["position_ids"].append(position_ids)
        res_dict["labels"].append((output_ids[i] + [-100] * num_right_padding)[:max_output_len])

    res_dict["input_ids"] = torch.tensor(res_dict["input_ids"])
    res_dict["attention_mask"] = torch.tensor(res_dict["attention_mask"])
    res_dict["position_ids"] = torch.tensor(res_dict["position_ids"])
    res_dict["labels"] = torch.tensor(res_dict["labels"])

    return res_dict


def get_dataloader(dataset_path,  tokenizer,max_output_len:Optional[int]=None):
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=partial(collate_fn, tokenizer=tokenizer,max_output_len=max_output_len),
    )
    return dataloader
