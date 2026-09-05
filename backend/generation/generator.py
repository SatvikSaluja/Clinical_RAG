"""LLM generation provider abstraction (spec section 10 / 21).

Two interchangeable providers, selected via `settings.llm_provider`:
- LocalHFProvider: a local Hugging Face instruct model (no API key needed).
- OpenAIProvider: any OpenAI-compatible chat completions endpoint.

Both expose the same `generate(system_prompt, user_prompt) -> str` interface
so the rest of the pipeline (and the ablation/baseline harness) never has to
know which one is active.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from backend.config.settings import settings

logger = logging.getLogger(__name__)

_local_model_cache: dict[str, tuple] = {}


@dataclass
class GenerationResult:
    text: str
    latency_seconds: float
    provider: str
    model: str


class BaseGenerator:
    provider_name: str = "base"

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        raise NotImplementedError


class LocalHFProvider(BaseGenerator):
    provider_name = "local"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.local_llm_model
        self.tokenizer, self.model = self._load(self.model_name)

    @staticmethod
    def _load(model_name: str):
        if model_name in _local_model_cache:
            return _local_model_cache[model_name]
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading local LLM '%s' (CPU) - this may take a while on first run", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        model.eval()
        _local_model_cache[model_name] = (tokenizer, model)
        return tokenizer, model

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        import torch

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=4096)

        start = time.time()
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=settings.llm_max_new_tokens,
                temperature=max(settings.llm_temperature, 1e-4),
                do_sample=settings.llm_temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        latency = time.time() - start

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        return GenerationResult(text=text, latency_seconds=latency, provider=self.provider_name, model=self.model_name)


class OpenAIProvider(BaseGenerator):
    provider_name = "openai"

    def __init__(self, model_name: str | None = None):
        from openai import OpenAI

        self.model_name = model_name or settings.openai_model
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

    def generate(self, system_prompt: str, user_prompt: str) -> GenerationResult:
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_new_tokens,
        )
        latency = time.time() - start
        text = response.choices[0].message.content or ""
        return GenerationResult(text=text.strip(), latency_seconds=latency, provider=self.provider_name, model=self.model_name)


_generator_singleton: BaseGenerator | None = None


def get_generator(provider: str | None = None) -> BaseGenerator:
    """Return a cached generator instance for the configured (or explicitly
    requested) provider. Used both by the live pipeline and by the
    baseline/ablation harness so model-loading cost is paid once.
    """
    global _generator_singleton
    provider = provider or settings.llm_provider
    if _generator_singleton is not None and _generator_singleton.provider_name == provider:
        return _generator_singleton
    if provider == "openai":
        _generator_singleton = OpenAIProvider()
    else:
        _generator_singleton = LocalHFProvider()
    return _generator_singleton
