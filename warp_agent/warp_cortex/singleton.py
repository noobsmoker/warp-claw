"""
Singleton weight sharing for Warp-Cortex architecture.
Ensures only one instance of the transformer model exists in memory.
"""

import threading
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class WarpCortexSingleton:
    """Singleton class for sharing transformer model weights across agents."""

    _instance: Optional['WarpCortexSingleton'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'WarpCortexSingleton':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._model: Optional[AutoModelForCausalLM] = None
            self._tokenizer: Optional[AutoTokenizer] = None
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model_name = "gpt2"  # Default model

    def load_model(self, model_name: str = "gpt2") -> None:
        """Load the transformer model if not already loaded."""
        if self._model is None or model_name != self._model_name:
            self._model_name = model_name
            self._model = AutoModelForCausalLM.from_pretrained(model_name).to(self._device)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

    def generate(self, prompt: str, max_length: int = 100, **kwargs) -> str:
        """Generate text using the shared model."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model.generate(
                inputs.input_ids,
                max_length=max_length,
                pad_token_id=self._tokenizer.pad_token_id,
                **kwargs
            )

        return self._tokenizer.decode(outputs[0], skip_special_tokens=True)

    @property
    def model(self) -> AutoModelForCausalLM:
        """Get the shared model instance."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """Get the shared tokenizer instance."""
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        return self._tokenizer