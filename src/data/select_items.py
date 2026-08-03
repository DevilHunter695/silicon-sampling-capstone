"""Select survey items for the study based on HANDOFF criteria."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import entropy
from src.config import (
    DATA_PROCESSED,
    MAX_MISSINGNESS_PCT,
    MIN_MODAL_ENTROPY,
    MIN_RESPONSE_SCALE_SIZE,
)

logger = logging.getLogger(__name__)

# WVS question metadata (extracted from worldvaluesbench repo pattern)
# Format: Q_ID -> (scale_type, thematic_block)
WVS_QUESTION_META = {
    # Values/morality
    "Q1": ("ordinal_4pt", "ethics"),
    "Q3": ("ordinal_4pt", "ethics"),
    "Q5": ("ordinal_4pt", "ethics"),
    "Q6": ("ordinal_4pt", "ethics"),
    "Q7": ("ordinal_4pt", "ethics"),
    "Q8": ("ordinal_4pt", "ethics"),
    # Trust
    "Q57": ("ordinal_4pt", "trust"),
    "Q58": ("ordinal_4pt", "trust"),
    "Q59": ("ordinal_4pt", "trust"),
    "Q60": ("ordinal_4pt", "trust"),
    # Economic values
    "Q106": ("ordinal_10pt", "economics"),
    "Q107": ("ordinal_10pt", "economics"),
    "Q108": ("ordinal_10pt", "economics"),
    # Politics/governance
    "Q142": ("ordinal_5pt", "politics"),
    "Q143": ("ordinal_5pt", "politics"),
    "Q144": ("ordinal_5pt", "politics"),
    # Religion
    "Q164": ("ordinal_3pt", "religion"),
    "Q165": ("ordinal_3pt", "religion"),
    "Q166": ("ordinal_3pt", "religion"),
}


def is_ordinal_scale(df: pd.Series, question_id: str) -> bool:
    """Check if a question has a clean ordinal scale."""
    # Use metadata if available
    if question_id in WVS_QUESTION_META:
        scale_type, _ = WVS_QUESTION_META[question_id]
        return "ordinal" in scale_type or "categorical" in scale_type

    # Fallback heuristic: small number of unique integer values
    unique_vals = df.dropna().unique()
    if len(unique_vals) < MIN_RESPONSE_SCALE_SIZE or len(unique_vals) > 20:
        return False

    # Check if mostly integers
    return all(isinstance(v, (int, np.integer)) for v in unique_vals)


def compute_missingness(df: pd.Series) -> float:
    """Compute missingness percentage."""
    return (df.isna().sum() / len(df)) * 100


def compute_modal_entropy(df: pd.Series) -> float:
    """Compute normalized entropy of response distribution."""
    counts = df.value_counts()
    if len(counts) == 0:
        return 0.0

    # Normalize
    probs = counts / counts.sum()

    # Entropy (normalized to 0-1)
    max_entropy = np.log(len(probs))
    if max_entropy == 0:
        return 0.0

    return entropy(probs) / max_entropy


def get_thematic_block(question_id: str) -> str:
    """Get thematic block for a question."""
    if question_id in WVS_QUESTION_META:
        _, block = WVS_QUESTION_META[question_id]
        return block
    return "other"


def select_items(
    df: pd.DataFrame,
    n_target: int = 45,
    missingness_threshold: float = MAX_MISSINGNESS_PCT,
    entropy_threshold: float = MIN_MODAL_ENTROPY,
) -> list[str]:
    """
    Select survey items based on HANDOFF criteria.

    Filters by:
    1. Clean ordinal/categorical scale
    2. Low missingness (<threshold %)
    3. Non-degenerate variance (entropy > threshold)
    4. Domain coverage (stratify across thematic blocks)

    Returns: List of selected question IDs
    """
    logger.info("Starting item selection...")

    candidates = []

    for col in df.columns:
        if not col.startswith("Q"):
            continue

        # Criterion 1: Ordinal scale
        if not is_ordinal_scale(df[col], col):
            continue

        # Criterion 2: Low missingness
        miss_pct = compute_missingness(df[col])
        if miss_pct > missingness_threshold:
            continue

        # Criterion 3: Non-degenerate variance
        ent = compute_modal_entropy(df[col])
        if ent < entropy_threshold:
            continue

        # Passed all filters
        block = get_thematic_block(col)
        candidates.append({
            "question_id": col,
            "missingness_pct": miss_pct,
            "entropy": ent,
            "block": block,
        })

    logger.info(f"Found {len(candidates)} candidate items")

    # Stratify by thematic block for coverage
    if len(candidates) == 0:
        logger.error("No candidate items found! Adjust thresholds.")
        return []

    df_candidates = pd.DataFrame(candidates)
    block_counts = df_candidates["block"].value_counts()
    logger.info(f"Thematic block distribution:\n{block_counts}")

    # Sort by entropy (descending) within each block, then sample evenly
    selected = []
    for block in df_candidates["block"].unique():
        block_items = df_candidates[df_candidates["block"] == block].sort_values(
            "entropy", ascending=False
        )
        # Allocate roughly n_target / n_blocks items per block
        n_per_block = max(1, n_target // len(df_candidates["block"].unique()))
        selected.extend(block_items.head(n_per_block)["question_id"].tolist())

    # If we have too few, add highest-entropy items overall
    if len(selected) < n_target:
        remaining = n_target - len(selected)
        extra = df_candidates[~df_candidates["question_id"].isin(selected)].sort_values(
            "entropy", ascending=False
        )
        selected.extend(extra.head(remaining)["question_id"].tolist())

    # Keep only n_target
    selected = selected[:n_target]

    logger.info(f"Selected {len(selected)} items for final study")

    # Log selected items
    selected_df = df_candidates[df_candidates["question_id"].isin(selected)].sort_values(
        "question_id"
    )
    logger.info(f"\nSelected items by block:\n{selected_df.groupby('block').size()}")

    return selected


def main(input_path: Path = None, output_path: Path = None):
    """Load data, select items, and save to JSON."""
    if input_path is None:
        input_path = DATA_PROCESSED / "ind_wvs7.parquet"
    if output_path is None:
        output_path = DATA_PROCESSED / "selected_items.json"

    if not input_path.exists():
        logger.error(f"Input data not found: {input_path}")
        logger.info("Run: python -m src.data.load_wvs --country IND")
        return False

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} respondents")

    selected = select_items(df)

    if len(selected) == 0:
        logger.error("Item selection failed")
        return False

    # Save
    output = {
        "selected_items": selected,
        "n_items": len(selected),
        "criteria": {
            "max_missingness_pct": MAX_MISSINGNESS_PCT,
            "min_entropy": MIN_MODAL_ENTROPY,
            "min_scale_size": MIN_RESPONSE_SCALE_SIZE,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"✓ Saved selected items to {output_path}")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = main()
    exit(0 if success else 1)
