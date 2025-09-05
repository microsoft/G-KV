class AutoModelForCausalLM:
    def from_pretrained(model_name, config, *args, **kwargs):
        if config.model_type == "qwen2":
            from .rl_modeling_qwen2 import Qwen2ForCausalLM
            return Qwen2ForCausalLM.from_pretrained(model_name, config=config, *args, **kwargs)
        elif config.model_type == "llama":
            pass
        else:
            raise ValueError(f"Unsupported method: {config.method}")