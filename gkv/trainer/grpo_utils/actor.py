import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import log_probs_from_logits, compute_entropy


class Actor(nn.Module):
    def __init__(self, model, tokenizer, accelerator, args):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.args = args
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def forward(self, sequences, attention_mask, sparse_mask, action_mask):

        rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        outputs = self.model(
            sequences,
            attention_mask,
            sparse_mask=sparse_mask,
            position_ids=position_ids,
            use_cache=False,
        )

        outputs["logits"] = outputs["logits"].div_(self.args.temperature)

        with torch.no_grad():
            entropy = compute_entropy(outputs["logits"])
            entropy = entropy[:, -action_mask.shape[1] :] * action_mask.float()
            mean_entropy = entropy.sum() / action_mask.sum().float()

        outputs["logits"] = outputs["logits"].to(torch.float32)

        log_probs = log_probs_from_logits(outputs["logits"], rolled_sequences)
        log_probs = log_probs[:, :-1]
        return log_probs, mean_entropy

    @torch.no_grad()
    def generate(
        self, batch_input, do_sample=True, temperature=None, output_sparse_mask=False
    ):
        unwrap_model = self.accelerator.unwrap_model(self.model)
        unwrap_model.eval()
        unwrap_model.config.return_sparse_mask = output_sparse_mask
        unwrap_model.config.record_pos_ids = output_sparse_mask
        if do_sample:
            sample_args = {
                "do_sample": True,
                "max_new_tokens": self.args.max_new_tokens,
                "temperature": (
                    self.args.temperature if temperature is None else temperature
                ),
                "top_p": self.args.top_p,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
                "return_dict_in_generate": True,
            }
        else:
            sample_args = {
                "do_sample": False,
                "max_new_tokens": self.args.max_new_tokens,
                "eos_token_id": self.eos_token_id,
                "pad_token_id": self.pad_token_id,
                "return_dict_in_generate": True,
                "top_p": None,
            }
        outputs = unwrap_model.generate(**batch_input, **sample_args)
        sequences = outputs.sequences
        sparse_mask = outputs.past_key_values.sparse_mask_cache
        attention_mask = outputs.past_key_values.attention_mask
        del outputs
        return sequences, sparse_mask, attention_mask
