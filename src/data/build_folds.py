"""Build stratified cross-validation folds for India WVS data."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from src.config import DATA_PROCESSED, N_FOLDS, RANDOM_SEED

logger = logging.getLogger(__name__)


def create_stratification_target(df: pd.DataFrame) -> np.ndarray:
    """
    Create stratification target that balances key subgroups.
    Combines urban/rural × income tercile for rough balance across cells.
    """
    strat_cols = []

    # Urban/rural (H_URBRURAL)
    if "H_URBRURAL" in df.columns:
        strat_cols.append(df["H_URBRURAL"].fillna("Unknown"))

    # Income tercile (Q288 or similar)
    income_col = None
    for col in ["Q288", "income", "Q274"]:
        if col in df.columns:
            income_col = col
            break

    if income_col:
        income_terciles = pd.qcut(
            df[income_col].fillna(df[income_col].median()),
            q=3,
            labels=["Low", "Mid", "High"],
            duplicates="drop",
        )
        strat_cols.append(income_terciles)

    if len(strat_cols) < 2:
        # Fallback: use sex if urban/rural not available
        if "Q260" in df.columns:
            strat_cols.append(df["Q260"].fillna("Unknown"))

    # Combine strat columns into single target
    strat_target = pd.concat(strat_cols, axis=1).fillna("Unknown")
    strat_str = strat_target.apply(lambda x: "_".join(x.astype(str)), axis=1)

    return strat_str.values


def build_folds(
    df: pd.DataFrame, n_splits: int = N_FOLDS, random_state: int = RANDOM_SEED
) -> dict:
    """
    Build stratified k-fold splits.

    Returns: {
        "respondent_ids": [...],
        "folds": [
            {"train": [...], "test": [...]},
            ...
        ]
    }
    """
    logger.info(f"Building {n_splits}-fold stratified CV...")

    # Create respondent ID column if not present
    if "respondent_id" not in df.columns:
        df["respondent_id"] = range(len(df))

    respondent_ids = df["respondent_id"].values

    # Stratification
    strat_target = create_stratification_target(df)

    # StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(df, strat_target)):
        fold = {
            "fold": fold_idx,
            "train": respondent_ids[train_idx].tolist(),
            "test": respondent_ids[test_idx].tolist(),
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        }
        folds.append(fold)
        logger.info(f"Fold {fold_idx}: train={fold['n_train']}, test={fold['n_test']}")

    # Verify no respondent appears in test set of multiple folds
    all_test_ids = set()
    for fold in folds:
        test_set = set(fold["test"])
        overlap = all_test_ids & test_set
        assert (
            len(overlap) == 0
        ), f"Respondent appears in multiple test sets!"
        all_test_ids.update(test_set)

    result = {
        "n_respondents": len(df),
        "n_folds": n_splits,
        "random_seed": random_state,
        "folds": folds,
    }

    logger.info(f"✓ Built {n_splits}-fold CV with {len(df)} respondents")
    return result


def main(input_path: Path = None, output_path: Path = None):
    """Load data and create folds."""
    if input_path is None:
        input_path = DATA_PROCESSED / "ind_wvs7.parquet"
    if output_path is None:
        output_path = DATA_PROCESSED / "folds.json"

    if not input_path.exists():
        logger.error(f"Input data not found: {input_path}")
        return False

    df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(df)} respondents")

    fold_config = build_folds(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(fold_config, f, indent=2)

    logger.info(f"✓ Saved fold configuration to {output_path}")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    success = main()
    exit(0 if success else 1)
