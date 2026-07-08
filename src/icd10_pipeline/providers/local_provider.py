"""Local open-source LLM provider (Hugging Face transformers).

Default model is small enough to run on CPU in a demo Space. For better
quality swap LOCAL_MODEL for e.g. Qwen2.5-7B-Instruct on a GPU Space,
or replace this class with a vLLM / llama.cpp client.
"""

import os
from .base import LLMProvider

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class LocalProvider(LLMProvider):
    name = "local"

    def __init__(self, model: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model or os.getenv("LOCAL_MODEL", DEFAULT_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def complete(self, prompt: str, max_tokens: int = 2000) -> str:
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(
            inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
