"""Gemini 2.5 Flash inference runner (using free tier via google-generativeai)."""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import CachedInferenceRunner

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google.generativeai not installed. Install via: pip install google-generativeai")


class GeminiInferenceRunner(CachedInferenceRunner):
    """Inference using Gemini 2.5 Flash via AI Studio free tier."""

    def __init__(self, api_key: Optional[str] = None):
        super().__init__("gemini-2.5-flash")

        if not GENAI_AVAILABLE:
            raise ImportError("google.generativeai required. Install with: pip install google-generativeai")

        # Get API key
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not found. "
                "Get free key from https://aistudio.google.com/app/apikey"
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        logger.info("✓ Initialized Gemini 2.5 Flash")

    def infer_single(
        self,
        prompt: str,
        answer_options: List[str],
    ) -> Tuple[Optional[str], Optional[np.ndarray], Dict]:
        """
        Run inference and extract logprobs if available.

        Gemini free tier may not have logprobs, so we fall back to sampling
        or use temperature-0 for deterministic outputs.
        """
        try:
            # Try with response_mime_type to constrain output
            generation_config = {
                "temperature": 0.0,  # Deterministic
                "top_p": 1.0,
                "max_output_tokens": 10,  # Short response
            }

            full_prompt = prompt + "\n\nAnswer with ONLY the number or letter (e.g., '1' or 'A')."

            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config,
            )

            answer_text = response.text.strip()

            # Parse answer
            pred_answer = self._parse_answer(answer_text, answer_options)

            # Logprobs unavailable in free tier
            # For now, use uniform distribution as placeholder
            logprobs = np.ones(len(answer_options)) / len(answer_options)

            metadata = {
                "refusal": False,
                "error": None,
                "raw_text": answer_text,
            }

            return pred_answer, logprobs, metadata

        except Exception as e:
            logger.error(f"Gemini inference error: {e}")

            logprobs = np.ones(len(answer_options)) / len(answer_options)
            metadata = {
                "refusal": "blocked" in str(e).lower(),
                "error": str(e),
            }

            return None, logprobs, metadata

    def _parse_answer(self, text: str, answer_options: List[str]) -> Optional[str]:
        """Parse model's text response into an answer option."""
        text = text.strip().lower()

        # Try exact match
        for opt in answer_options:
            if text == opt.lower():
                return opt

        # Try first word
        first_word = text.split()[0] if text else ""
        for opt in answer_options:
            if first_word == opt.lower():
                return opt

        # Try numeric index
        try:
            idx = int(first_word)
            if 0 <= idx < len(answer_options):
                return answer_options[idx]
            if 1 <= idx <= len(answer_options):
                return answer_options[idx - 1]
        except ValueError:
            pass

        # No valid answer parsed
        logger.warning(f"Could not parse answer: '{text}' against options {answer_options}")
        return None
