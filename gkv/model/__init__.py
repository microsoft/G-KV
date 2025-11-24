class AutoModelForCausalLM:
    def from_pretrained(model_name, config, *args, **kwargs):
        if config.model_type == "qwen2":
            from .modeling_qwen2 import Qwen2ForCausalLM

            return Qwen2ForCausalLM.from_pretrained(
                model_name, config=config, *args, **kwargs
            )
        elif config.model_type == "llama":
            from .modeling_llama import LlamaForCausalLM

            return LlamaForCausalLM.from_pretrained(
                model_name, config=config, *args, **kwargs
            )
        elif config.model_type == "qwen3":
            from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention,Qwen3ForCausalLM
            from .modeling_qwen3 import qwen3_attn_init, forward
            print("qwen3 only support inference")

            Qwen3Attention.__init__ = qwen3_attn_init
            Qwen3Attention.forward = forward
            return Qwen3ForCausalLM.from_pretrained(
                model_name, config=config, *args, **kwargs
            )

        else:
            raise ValueError(f"Unsupported method: {config.method}")
