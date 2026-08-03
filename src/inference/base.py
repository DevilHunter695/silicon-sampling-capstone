"""Base inference runner with caching and resumability."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.config import CACHE_DIR

logger = logging.getLogger(__name__)


class CachedInferenceRunner:
    """
    Base class for cached, resumable inference.

    Results are cached as JSON to disk, keyed by:
    (respondent_id, item_id, model, condition)
    """

    def __init__(self, model_name: str, cache_dir: Path = CACHE_DIR):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(
        self, respondent_id: int, item_id: str, condition: str
    ) -> Path:
        """Get cache file path for a single prediction."""
        return self.cache_dir / f"{self.model_name}_{respondent_id}_{item_id}_{condition}.json"

    def load_cached_result(
        self, respondent_id: int, item_id: str, condition: str
    ) -> Optional[Dict]:
        """Load cached result if it exists."""
        cache_path = self.get_cache_path(respondent_id, item_id, condition)

        if cache_path.exists():
            with open(cache_path, 'r') as f:
                return json.load(f)

        return None

    def save_result(
        self,
        respondent_id: int,
        item_id: str,
        condition: str,
        result: Dict,
    ):
        """Save result to cache."""
        cache_path = self.get_cache_path(respondent_id, item_id, condition)

        with open(cache_path, 'w') as f:
            json.dump(result, f)

    def infer_single(
        self,
        prompt: str,
        answer_options: List[str],
    ) -> Tuple[Optional[str], np.ndarray, Dict]:
        """
        Run inference on a single prompt.

        Returns:
            (predicted_answer, logprobs_over_options, metadata)

        Subclasses should implement this.
        """
        raise NotImplementedError("Subclasses must implement infer_single()")

    def infer_batch(
        self,
        respondent_ids: List[int],
        item_ids: List[str],
        prompts: List[str],
        answer_options_list: List[List[str]],
        condition: str,
        resume: bool = True,
    ) -> Dict:
        """
        Run batch inference with caching and resumability.

        Args:
            respondent_ids: List of respondent IDs
            item_ids: List of item IDs
            prompts: List of prompts
            answer_options_list: List of answer option lists
            condition: Prompt condition (P0, P1, P2, P3)
            resume: If True, skip cached results

        Returns: {
            "results": [
                {
                    "respondent_id": int,
                    "item_id": str,
                    "condition": str,
                    "predicted_answer": str or None,
                    "logprobs": [...],  # log-probs over answer options
                    "refusal": bool,
                    "error": str or None,
                }
            ]
        }
        """
        assert (
            len(respondent_ids)
            == len(item_ids)
            == len(prompts)
            == len(answer_options_list)
        )

        results = []
        n_cached = 0
        n_new = 0

        for respondent_id, item_id, prompt, answer_options in zip(
            respondent_ids, item_ids, prompts, answer_options_list
        ):
            # Check cache first
            if resume:
                cached = self.load_cached_result(respondent_id, item_id, condition)
                if cached is not None:
                    results.append(cached)
                    n_cached += 1
                    continue

            try:
                # Run inference
                pred_answer, logprobs, metadata = self.infer_single(prompt, answer_options)

                result = {
                    "respondent_id": respondent_id,
                    "item_id": item_id,
                    "condition": condition,
                    "predicted_answer": pred_answer,
                    "logprobs": logprobs.tolist() if logprobs is not None else None,
                    "refusal": metadata.get("refusal", False),
                    "error": metadata.get("error"),
                }

                # Save to cache
                self.save_result(respondent_id, item_id, condition, result)

                results.append(result)
                n_new += 1

            except Exception as e:
                logger.error(f"Error for respondent {respondent_id}, item {item_id}: {e}")
                result = {
                    "respondent_id": respondent_id,
                    "item_id": item_id,
                    "condition": condition,
                    "predicted_answer": None,
                    "logprobs": None,
                    "refusal": False,
                    "error": str(e),
                }
                results.append(result)
                n_new += 1

        logger.info(
            f"Batch inference complete: {n_cached} cached, {n_new} new predictions"
        )

        return {"results": results, "n_cached": n_cached, "n_new": n_new}
