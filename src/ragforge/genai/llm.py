from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LLMConfig:
    """Configuration for a pretrained causal language model."""

    model_name: str = "distilgpt2"
    device: str = "cpu"


class PretrainedLLM:
    """
    Wrapper around a pretrained causal language model.

    This component provides a clean interface for loading
    an open-source LLM and generating text.
    """

    def __init__(
        self,
        config: LLMConfig,
    ):
        self.config = config

        self.device = torch.device(
            config.device
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name
        )

        self.model.to(self.device)
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """Generate text from a prompt."""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(
            self.device
        )

        attention_mask = inputs[
            "attention_mask"
        ].to(self.device)

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )