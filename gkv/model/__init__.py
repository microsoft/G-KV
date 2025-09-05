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
        else:
            raise ValueError(f"Unsupported method: {config.method}")
