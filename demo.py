#!/usr/bin/env python
"""
Silicon Sampling Pipeline Demo

This script demonstrates the full pipeline:
1. Load and clean WVS data
2. Select items
3. Build cross-validation folds
4. Evaluate baseline models
5. Run zero-shot inference (with Gemini or local models)
6. Analyze results by subgroup

Usage:
    python demo.py                           # Run full demo with synthetic data
    python demo.py --mode baselines-only     # Just evaluate baselines
    python demo.py --use-gemini              # Use Gemini for inference
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import COUNTRY_CODE, DATA_PROCESSED
from src.data import build_folds, filter_by_country, recode_missing_values, select_items
from src.eval import compute_metrics, evaluate_all_baselines
from src.prompts import build_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_synthetic_wvs_data(n_respondents: int = 500) -> pd.DataFrame:
    """Create synthetic WVS-like data for demo."""
    logger.info(f"Creating synthetic WVS data with {n_respondents} respondents...")

    np.random.seed(42)

    data = {
        "respondent_id": range(n_respondents),
        "B_COUNTRY_ALPHA": [COUNTRY_CODE] * n_respondents,
        # Demographics
        "Q260": np.random.choice([1, 2], p=[0.52, 0.48], size=n_respondents),  # Sex (1=Male, 2=Female)
        "Q262": np.random.randint(18, 95, size=n_respondents),  # Age
        "Q274": np.random.poisson(1.5, size=n_respondents),  # Children
        "Q279": np.random.choice([1, 2, 3, 4], size=n_respondents),  # Employment
        "Q288": np.random.randint(1, 11, size=n_respondents),  # Income decile
        "H_URBRURAL": np.random.choice([1, 2], p=[0.35, 0.65], size=n_respondents),  # 1=Urban, 2=Rural
        "N_REGION_ISO": np.random.choice(["MH", "UT", "DL", "BR", "WB", "TG"], size=n_respondents),
        # Likert scale questions (survey items)
        "Q1": np.random.randint(1, 5, size=n_respondents),
        "Q2": np.random.randint(1, 5, size=n_respondents),
        "Q3": np.random.randint(1, 5, size=n_respondents),
        "Q4": np.random.randint(1, 5, size=n_respondents),
        "Q5": np.random.randint(1, 5, size=n_respondents),
        "Q6": np.random.randint(1, 5, size=n_respondents),
        "Q7": np.random.randint(1, 5, size=n_respondents),
        "Q8": np.random.randint(1, 5, size=n_respondents),
        "Q9": np.random.randint(1, 5, size=n_respondents),
        "Q10": np.random.randint(1, 5, size=n_respondents),
    }

    df = pd.DataFrame(data)

    # Add some missing values (coded as negative in WVS)
    for col in ["Q1", "Q2", "Q3"]:
        missing_idx = np.random.choice(len(df), size=int(0.05 * len(df)), replace=False)
        df.loc[missing_idx, col] = -1

    logger.info(f"✓ Created {len(df)} synthetic respondents")
    return df


def run_demo_phase_0(df: pd.DataFrame):
    """Phase 0: Data loading and cleaning."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 0: Data Loading & Cleaning")
    logger.info("=" * 60)

    # Clean missing values
    df = recode_missing_values(df)
    logger.info(f"✓ Recoded missing values")

    # Filter to country
    df = filter_by_country(df, COUNTRY_CODE)
    logger.info(f"✓ Filtered to {COUNTRY_CODE}: {len(df)} respondents")

    return df


def run_demo_phase_1(df: pd.DataFrame):
    """Phase 1: Item selection & fold creation."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: Item Selection & Fold Creation")
    logger.info("=" * 60)

    # Select items
    selected = select_items(df, n_target=8)
    logger.info(f"✓ Selected {len(selected)} items: {selected}")

    # Build folds
    fold_config = build_folds(df, n_splits=3)
    logger.info(f"✓ Built {fold_config['n_folds']}-fold stratified CV")
    for fold in fold_config["folds"]:
        logger.info(f"  Fold {fold['fold']}: train={fold['n_train']}, test={fold['n_test']}")

    return selected, fold_config


def run_demo_phase_2(df: pd.DataFrame, fold_config: dict):
    """Phase 2: Evaluate baselines."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: Baseline Evaluation")
    logger.info("=" * 60)

    # Use first fold
    fold = fold_config["folds"][0]

    df_train = df[df["respondent_id"].isin(fold["train"])].copy()
    df_test = df[df["respondent_id"].isin(fold["test"])].copy()

    # Add answer variable for demo
    if "Q1" not in df_train.columns:
        df_train["answer"] = np.random.randint(0, 4, size=len(df_train))
        df_test["answer"] = np.random.randint(0, 4, size=len(df_test))
    else:
        df_train["answer"] = df_train["Q1"]
        df_test["answer"] = df_test["Q1"]

    # Evaluate all baselines
    results = evaluate_all_baselines(df_train, df_test, y_col="answer")

    logger.info("\nBaseline Results:")
    for baseline_name, metrics in results.items():
        logger.info(
            f"  {baseline_name:20} "
            f"Accuracy={metrics.get('accuracy', 0):.3f}, "
            f"MAE={metrics.get('mae', 0):.3f}, "
            f"NLL={metrics.get('nll', 0):.3f}"
        )

    return results


def run_demo_phase_3(selected_items: list[str]):
    """Phase 3: Prompt generation (without running inference)."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: Prompt Generation")
    logger.info("=" * 60)

    # Example respondent
    demographics = {
        "sex": "Female",
        "age": 35,
        "occupation": "Teacher",
        "region": "Maharashtra",
        "urban_rural": "Urban",
        "education": "Tertiary",
        "religion": "Hindu",
    }

    logger.info(f"Example respondent: {demographics}")

    # Generate prompts for different conditions
    question = "Do you think hard work is rewarded?"
    answer_options = ["Strongly Agree", "Agree", "Disagree", "Strongly Disagree"]

    conditions = ["P0", "P1", "P2", "P3"]
    for condition in conditions:
        prompt = build_prompt(
            condition,
            question,
            "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(answer_options)),
            **demographics,
        )
        logger.info(f"\n{condition} Prompt (first 100 chars): {prompt[:100]}...")


def run_demo_phase_4():
    """Phase 4: Inference (placeholder - requires API keys)."""
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4: Inference Setup (requires API keys)")
    logger.info("=" * 60)

    logger.info("To run actual inference, you need:")
    logger.info("  1. For Gemini: GOOGLE_API_KEY (free tier at https://aistudio.google.com/app/apikey)")
    logger.info("  2. For local models: GPU with transformers + torch")
    logger.info("\nExample:")
    logger.info("  from src.inference import GeminiInferenceRunner")
    logger.info("  runner = GeminiInferenceRunner(api_key='your-key-here')")


def main():
    parser = argparse.ArgumentParser(description="Silicon Sampling Pipeline Demo")
    parser.add_argument(
        "--mode",
        default="full",
        choices=["full", "baselines-only", "phase0-1"],
        help="Which phases to run",
    )
    parser.add_argument("--n-respondents", type=int, default=500, help="Synthetic data size")
    parser.add_argument("--use-gemini", action="store_true", help="Use Gemini for inference")
    args = parser.parse_args()

    logger.info("Silicon Sampling Pipeline Demo")
    logger.info(f"Mode: {args.mode}")

    # Phase 0: Data loading
    df = create_synthetic_wvs_data(args.n_respondents)
    df = run_demo_phase_0(df)

    if args.mode in ["full", "phase0-1"]:
        # Phase 1: Item selection & folds
        selected, fold_config = run_demo_phase_1(df)

        if args.mode == "full":
            # Phase 2: Baselines
            results = run_demo_phase_2(df, fold_config)

            # Phase 3: Prompts
            run_demo_phase_3(selected)

            # Phase 4: Inference setup
            run_demo_phase_4()

    logger.info("\n" + "=" * 60)
    logger.info("✓ Demo complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Download real WVS-7 India data (see README.md)")
    logger.info("  2. Run: python -m src.data.load_wvs --country IND")
    logger.info("  3. Configure API keys for inference")
    logger.info("  4. Run full pipeline: python -m notebooks/kaggle_trackA_openweight")


if __name__ == "__main__":
    main()
