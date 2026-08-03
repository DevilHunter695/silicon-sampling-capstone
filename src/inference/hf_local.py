"""HuggingFace Transformers inference with logprob extraction."""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .base import CachedInferenceRunner

logger = logging.getLogger(__name__)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed")


class HFLocalInferenceRunner(CachedInferenceRunner):
    """Inference using HuggingFace models locally with logprob extraction."""

    def __init__(self, model_id: str, device: str = "auto", load_in_4bit: bool = True):
        """
        Args:
            model_id: HuggingFace model ID (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            device: "cuda", "cpu", or "auto"
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers required. Install with: pip install transformers torch"
            )

        super().__init__(f"hf-{model_id.split('/')[-1].lower()}")

        self.model_id = model_id
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Loading {model_id} on device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        try:
            if load_in_4bit and self.device == "cuda":
                from transformers import BitsAndBytesConfig

                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_id)
                self.model = self.model.to(self.device)

        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

        logger.info(f"✓ Loaded {model_id}")

    def infer_single(
        self,
        prompt: str,
        answer_options: List[str],
    ) -> Tuple[Optional[str], Optional[np.ndarray], Dict]:
        """
        Run inference and extract logprobs over answer tokens.

        Returns:
            (predicted_answer, logprobs_over_options, metadata)
        """
        try:
            with torch.no_grad():
                # Tokenize prompt
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                # Generate continuation (just one token)
                outputs = self.model(
                    **inputs,
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=1,
                )

                # Extract logits for the generated token
                logits = outputs.scores[0][0]  # Shape: (vocab_size,)

                # Get tokens for each answer option
                option_tokens = []
                for opt in answer_options:
                    # Tokenize the option and get first token
                    tokens = self.tokenizer.encode(opt, add_special_tokens=False)
                    if tokens:
                        option_tokens.append(tokens[0])
                    else:
                        option_tokens.append(None)

                # Compute logprobs for each option
                logprobs = []
                for token_id in option_tokens:
                    if token_id is not None:
                        logprob = torch.log_softmax(logits, dim=0)[token_id].item()
                    else:
                        logprob = -np.inf
                    logprobs.append(logprob)

                logprobs = np.array(logprobs)

                # Find best option
                valid_logprobs = np.where(np.isfinite(logprobs), logprobs, -np.inf)
                best_idx = np.argmax(valid_logprobs)

                # Convert logprobs to probabilities
                logprobs = logprobs - np.max(logprobs)  # Numerical stability
                probs = np.exp(logprobs)
                probs = probs / probs.sum()

                pred_answer = answer_options[best_idx] if np.isfinite(valid_logprobs[best_idx]) else None

                metadata = {
                    "refusal": False,
                    "error": None,
                }

                return pred_answer, probs, metadata

        except Exception as e:
            logger.error(f"Inference error: {e}")

            probs = np.ones(len(answer_options)) / len(answer_options)
            metadata = {
                "refusal": False,
                "error": str(e),
            }

            return None, probs, metadata
