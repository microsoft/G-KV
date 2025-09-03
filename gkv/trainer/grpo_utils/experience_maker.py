import torch
from typing import Union
from dataclasses import dataclass
from typing import List, Any
from dataclasses import fields
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
# from gkv.model.utils import zero_pad_sequences
from gkv.trainer.grpo_utils.actor import Actor


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


@dataclass
class Experience:
    """Experience is a batch of data for RLHF training.

    Shapes of each tensor:
    sequences: (B, S)
    attention_mask: (B, S)
    action_mask: (B, S)
    sparse_mask: (B, L, H, S, S)
    advantages: (B, S)
    info: dict[str, list]
    """

    sequences: torch.Tensor = None
    attention_mask: torch.LongTensor = None
    action_mask: torch.BoolTensor = None
    advantages: torch.Tensor = None
    sparse_mask: torch.BoolTensor = None
    rewards: torch.Tensor = None  # used for advantage calculation

    prompts: list[str] = None
    labels: list[str] = None
    info: dict[str, torch.Tensor] = None

    def __init__(
        self,
        sequences=None,
        advantages=None,
        attention_mask=None,
        action_mask=None,
        sparse_mask=None,
        prompts=None,
        labels=None,
        rewards=None,
        scores=None,
        info=None,
    ):
        self.sequences = sequences
        self.advantages = advantages
        self.attention_mask = attention_mask
        self.action_mask = action_mask
        self.sparse_mask = sparse_mask
        self.rewards = rewards
        self.prompts = prompts or []
        self.labels = labels or []
        self.info = info or []

    @torch.no_grad()
    def to_device(self, device: torch.device):
        """Move all tensor fields to the specified device."""
        for field, value in self.__dict__.items():
            if field == "sparse_mask":
                # keep the sparse mask on cpu
                continue
            if isinstance(value, dict):
                setattr(
                    self, field, {key: to(val, device) for key, val in value.items()}
                )
            else:
                setattr(self, field, to(value, device))

        return self

    @staticmethod
    def _merge_item(
        items: List, pad_value: int = 0
    ) -> Union[torch.Tensor, list, dict, Any]:
        """Merge a list of items into a single item.
        Recursively merge tensors, lists and dicts.
        For tensors, use zero_pad_sequences to merge sequences of different lengths.

        Args:
            items: List of items to merge
            pad_value: Value used for padding tensors
        """
        if isinstance(items[0], torch.Tensor):
            # add left pad
            if len(items[0].shape) == 2:
                max_len = max([item.shape[1] for item in items])
                pad_tensor = []
                for item in items:
                    pad_tensor.append(
                        torch.cat(
                            [
                                torch.full(
                                    (item.shape[0], max_len - item.shape[1]),
                                    pad_value,
                                    dtype=item.dtype,
                                    device=item.device,
                                ),
                                item,
                            ],
                            dim=1,
                        )
                    )
                return torch.cat(pad_tensor, dim=0)
            elif len(items[0].shape) == 5:
                # sparse mask is a 5D tensor, pad the last two dimension
                max_len = max([item.shape[-1] for item in items])
                pad_tensor = []
                for item in items:
                    new_tensor = torch.full(
                        (item.shape[0], item.shape[1], item.shape[2], max_len, max_len),
                        pad_value,
                        dtype=item.dtype,
                        device=item.device,
                    )
                    new_tensor[:, :, :, -item.shape[-2] :, -item.shape[-1] :] = item
                    pad_tensor.append(new_tensor)
                return torch.cat(pad_tensor, dim=0)
            else:
                raise ValueError(f"Unsupported tensor shape: {items[0].shape}")
        elif isinstance(items[0], list):
            return sum(items, [])
        elif isinstance(items[0], dict):
            result = {}
            # Collect all values for each key
            for d in items:
                for key, value in d.items():
                    if key not in result:
                        result[key] = []
                    result[key].append(value)
            # Merge all values for each key at once
            return {
                key: Experience._merge_item(values, pad_value)
                for key, values in result.items()
            }
        elif items[0] is None:
            return None
        else:
            raise ValueError(f"Unsupported type: {type(items[0])}")

    @staticmethod
    def concat_experiences(
        experiences_list: List["Experience"], pad_token_id
    ) -> "Experience":
        """Concatenate multiple experiences into one large experience.

        Args:
            experiences_list: List of Experience to concatenate
            pad_token_id: Token id used for padding sequences

        Returns:
            A new Experience instance containing all the concatenated data
        """
        if not experiences_list:
            return Experience()

        if len(experiences_list) == 1:
            return experiences_list[0]

        # Get all field names from the dataclass
        field_names = [f.name for f in fields(Experience)]

        # Create result dictionary
        result = {}

        # Merge all fields
        for field in field_names:
            values = [getattr(e, field) for e in experiences_list]
            # Use pad_token_id for sequences field, 0 for others
            pad_value = pad_token_id if field == "sequences" else 0
            result[field] = Experience._merge_item(values, pad_value)

        return Experience(**result)


def process_sample(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    sparse_mask: torch.Tensor,
    input_len: int,
    pad_token_id: int,
    eos_token_id: int,
    prompts: str,
    output_texts: str,
    answers: str,
    trunk_length: Optional[int],
):
    # remove padding
    num_left_pad = (attention_mask == 0).sum().item()
    output_ids = sequences[input_len:]
    if pad_token_id == eos_token_id:
        num_right_pad = max(0, (output_ids == pad_token_id).sum().item() - 1)
    else:
        num_right_pad = (output_ids == pad_token_id).sum().item()
    if num_right_pad > 0:
        sequences = sequences[num_left_pad:-num_right_pad].unsqueeze(0).cpu()
    else:
        sequences = sequences[num_left_pad:].unsqueeze(0).cpu()
    
    seq_len=sequences.shape[1]
    if trunk_length is not None:
        trunked_seq_len = min(seq_len, trunk_length)
        sequences = sequences[:, :trunked_seq_len]
    else:
        trunked_seq_len = seq_len
    output_len = seq_len - (input_len - num_left_pad)
    trunked_output_len = trunked_seq_len - (input_len - num_left_pad)

    info = {
        "output_len": [output_len], # log the original output length
        "input_len": [input_len - num_left_pad],
        "output_texts": [output_texts],
        "sequence_length": [seq_len],
    }
    # build mask
    attention_mask = torch.tensor([1] * (trunked_seq_len)).unsqueeze(0)
    sparse_mask = (
        sparse_mask[
            :,
            :,
            num_left_pad : input_len + trunked_output_len,
            num_left_pad : input_len + trunked_output_len,
        ]
        .unsqueeze(0)
        .cpu()
    )
    action_mask = torch.BoolTensor(
        [False] * info["input_len"][0] + [True] * trunked_output_len
    ).unsqueeze(0)

    exp = Experience(
        sequences=sequences,
        attention_mask=attention_mask,
        sparse_mask=sparse_mask,
        action_mask=action_mask,
        prompts=[prompts],
        labels=[answers],
        info=info,
    )
    return exp


class SamplesGenerator:
    def __init__(self, actor: Actor, tokenizer, accelerator, args):
        self.actor = actor
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.args = args
        self.generate_batch_size_per_gpu = args.generate_batch_size_per_gpu
        self.sample_n = args.sample_n

    def generate_samples(self, batch):
        prompts = []
        answers = []
        for item in batch:
            for _ in range(self.sample_n):
                prompts.append(item["prompt"])
                answers.append(item["answer"])

        i = 0
        all_experiences = []
        for i in range(0, len(prompts), self.generate_batch_size_per_gpu):
            batch_prompts = prompts[i : i + self.generate_batch_size_per_gpu]
            experiences = self.generate_batch(batch_prompts, answers)
            all_experiences.extend(experiences)
        return all_experiences

    def generate_batch(
        self, batch_prompts: List[str], answers: List[str]
    ) -> List[Experience]:
        inputs = self.tokenizer(
            batch_prompts, return_tensors="pt", add_special_tokens=False, padding=True
        ).to(self.accelerator.device)
        input_len = inputs.input_ids.shape[1]
        batch_ids, sparse_mask, attention_mask = self.actor.generate(inputs,output_sparse_mask=True)
        output_dis = batch_ids[:, input_len:]
        output_texts = self.tokenizer.batch_decode(output_dis, skip_special_tokens=True)
        exp_list = []
        for i in range(len(batch_prompts)):
            item = {
                "sequences": batch_ids[i],
                "input_len": input_len,
                "sparse_mask": sparse_mask[i],
                "attention_mask": attention_mask[i][:input_len],
                "prompts": batch_prompts[i],
                "output_texts": output_texts[i],
                "answers": answers[i],
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "trunk_length": self.args.trunk_length,
            }
            exp_list.append(process_sample(**item))
        return exp_list


class ExperienceMaker:
    def __init__(self, reward_fn, tokenizer, accelerator, args):
        self.reward_fn = reward_fn
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.args = args
        self.train_micro_batch_size_per_gpu = args.train_micro_batch_size_per_gpu
        self.sample_n = args.sample_n

    def compute_advantages(self, experiences):
        """
        compute the group advantages (GRPO) for the experiences
        """
        for i in range(0, len(experiences), self.sample_n):
            group_rewards = []
            for j in range(i, i + self.sample_n):
                group_rewards.append(experiences[j].rewards)
            group_rewards = torch.cat(group_rewards, dim=0)
            group_advantages = group_rewards - group_rewards.mean()
            group_advantages = group_advantages / (group_advantages.std() + 1e-8)
            for j in range(i, i + self.sample_n):
                sequence_len = experiences[j].action_mask.shape[1]
                experiences[j].advantages = (
                    group_advantages[j - i].unsqueeze(0).expand(1, sequence_len)
                )
        return experiences

    def split_experience_micro_batch(self, experiences):
        """
        split the experiences to the micro batch size
        """
        batched_experiences = []
        for i in range(0, len(experiences), self.train_micro_batch_size_per_gpu):
            batched_experiences.append(
                Experience.concat_experiences(
                    experiences[i : i + self.train_micro_batch_size_per_gpu],
                    self.tokenizer.pad_token_id,
                )
            )
        return batched_experiences

    def make_experience_batch(self, experiences: List[Experience]):
        """
        compute the advantages for the experiences and concatenate experiences to the micro batch size
        """
        for i in range(0, len(experiences)):
            outputs = experiences[i].info["output_texts"][0]
            answers = experiences[i].labels[0]
            rewards = self.reward_fn(outputs, answers)
            experiences[i].rewards = torch.tensor(
                [rewards], dtype=torch.float32
            ).unsqueeze(0)

        experiences = self.compute_advantages(experiences)
        experiences = self.split_experience_micro_batch(experiences)
        return experiences
