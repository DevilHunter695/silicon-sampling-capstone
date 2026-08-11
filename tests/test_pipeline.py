"""Integration tests for the silicon sampling pipeline."""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import COUNTRY_CODE, N_FOLDS, RANDOM_SEED
from src.data import build_folds, filter_by_country, load_wvs_raw, recode_missing_values, select_items
from src.eval import (
    compute_metrics,
    DemographicCellLookup,
    evaluate_all_baselines,
    NationalMarginalBaseline,
)
from src.prompts import build_prompt, format_p0_control, format_p1_minimal, format_p2_structured

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def sample_wvs_data():
    """Create synthetic WVS-like data for testing."""
    n_respondents = 100

    data = {
        "respondent_id": range(n_respondents),
        "B_COUNTRY_ALPHA": [COUNTRY_CODE] * n_respondents,
        "Q260": np.random.choice([1, 2], size=n_respondents),  # Sex
        "Q262": np.random.randint(18, 80, size=n_respondents),  # Age
        "Q274": np.random.randint(1, 5, size=n_respondents),  # Children
        "Q288": np.random.randint(1, 11, size=n_respondents),  # Income decile
        "H_URBRURAL": np.random.choice([1, 2], size=n_respondents),  # Urban/rural
        "N_REGION_ISO": np.random.choice(["MH", "UT", "DL", "BR"], size=n_respondents),  # Region
        # Survey items (Likert scales)
        "Q1": np.random.randint(1, 5, size=n_respondents),
        "Q2": np.random.randint(1, 5, size=n_respondents),
        "Q3": np.random.randint(1, 5, size=n_respondents),
        "Q4": np.random.randint(1, 5, size=n_respondents),
        "Q5": np.random.randint(1, 5, size=n_respondents),
    }

    df = pd.DataFrame(data)

    # Add some missing values (coded as negative in WVS)
    df.loc[df.sample(5).index, "Q1"] = -1
    df.loc[df.sample(3).index, "Q2"] = -2

    return df


class TestDataLoading:
    """Test data loading and cleaning."""

    def test_recode_missing_values(self, sample_wvs_data):
        """Test that missing value codes are recoded to NaN."""
        df = sample_wvs_data.copy()

        # Verify negatives exist before
        assert (df == -1).any().any() or (df == -2).any().any()

        df = recode_missing_values(df)

        # Verify no negative codes remain
        for col in df.select_dtypes(include=['number']).columns:
            assert not (df[col].isin([-1, -2, -3, -4, -5])).any(), f"Missing codes remain in {col}"

        logger.info("✓ test_recode_missing_values passed")

    def test_filter_by_country(self, sample_wvs_data):
        """Test filtering by country."""
        df = sample_wvs_data.copy()

        df_filtered = filter_by_country(df, COUNTRY_CODE)

        assert len(df_filtered) == len(df)
        logger.info("✓ test_filter_by_country passed")


class TestItemSelection:
    """Test survey item selection."""

    def test_select_items(self, sample_wvs_data):
        """Test that items are selected based on criteria."""
        df = sample_wvs_data.copy()
        df = recode_missing_values(df)

        # Select items
        selected = select_items(df, n_target=3)

        # Should have selected some items
        assert len(selected) > 0
        assert all(isinstance(item, str) for item in selected)

        logger.info(f"✓ test_select_items passed (selected {len(selected)} items)")


class TestFolds:
    """Test fold creation."""

    def test_build_folds(self, sample_wvs_data):
        """Test stratified fold creation."""
        df = sample_wvs_data.copy()

        fold_config = build_folds(df, n_splits=3)

        # Verify structure
        assert fold_config["n_respondents"] == len(df)
        assert fold_config["n_folds"] == 3
        assert len(fold_config["folds"]) == 3

        # Verify no overlap
        for fold in fold_config["folds"]:
            assert len(fold["train"]) + len(fold["test"]) <= len(df)

        logger.info("✓ test_build_folds passed")


class TestPrompts:
    """Test prompt formatting."""

    def test_format_prompts(self):
        """Test prompt template generation."""
        # P0
        p0 = format_p0_control()
        assert "survey question" in p0.lower()

        # P1
        p1 = format_p1_minimal(age=45, sex="Male", region="Maharashtra")
        assert "45" in p1
        assert "Male" in p1

        # P2
        p2 = format_p2_structured(sex="Female", age=30, region="Delhi", education="Tertiary")
        assert "Female" in p2
        assert "30" in p2

        # Full prompt
        prompt = build_prompt(
            "P0",
            "Do you trust people in general?",
            "1. Yes\n2. No",
        )
        assert "trust people" in prompt
        assert "1. Yes" in prompt

        logger.info("✓ test_format_prompts passed")


class TestMetrics:
    """Test evaluation metrics."""

    def test_compute_metrics(self):
        """Test metric computation."""
        y_true = np.array([0, 1, 2, 1, 0, 2, 1])
        y_pred = np.array([0, 1, 2, 2, 0, 2, 1])

        # Without probabilities
        metrics = compute_metrics(y_true, y_pred)

        assert "accuracy" in metrics
        assert "mae" in metrics
        assert metrics["accuracy"] == 6 / 7

        # With probabilities
        pred_probs = np.random.dirichlet([1, 1, 1], size=len(y_true))
        pred_probs[np.arange(len(y_true)), y_pred] += 0.3  # Boost correct predictions
        pred_probs /= pred_probs.sum(axis=1, keepdims=True)

        metrics = compute_metrics(y_true, y_pred, pred_probs)

        assert "nll" in metrics
        assert "jsd" in metrics
        assert metrics["nll"] > 0

        logger.info("✓ test_compute_metrics passed")


class TestBaselines:
    """Test baseline models."""

    def test_national_marginal_baseline(self):
        """Test national marginal baseline."""
        y_train = np.array([0, 0, 1, 1, 2, 2, 2])

        baseline = NationalMarginalBaseline()
        baseline.fit(y_train)

        # Check marginal
        probs = baseline.predict_proba(5)
        assert probs.shape == (5, 3)
        assert np.allclose(probs[0], probs[1])  # All same distribution

        logger.info("✓ test_national_marginal_baseline passed")

    def test_demographic_cell_lookup(self, sample_wvs_data):
        """Test demographic cell lookup baseline."""
        df = sample_wvs_data.copy()

        # Create a simple target variable
        df["answer"] = np.random.randint(0, 4, size=len(df))

        # Split into train/test
        train_idx = np.arange(len(df))[:80]
        test_idx = np.arange(len(df))[80:]

        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

        # Fit baseline
        baseline = DemographicCellLookup()
        baseline.fit(df_train, "answer", ["H_URBRURAL", "Q260"])

        # Predict
        probs = baseline.predict_proba(df_test)

        assert probs.shape[0] == len(df_test)
        assert np.allclose(probs.sum(axis=1), 1.0)

        logger.info("✓ test_demographic_cell_lookup passed")

    def test_evaluate_all_baselines(self, sample_wvs_data):
        """Test comprehensive baseline evaluation."""
        df = sample_wvs_data.copy()
        df["answer"] = np.random.randint(0, 4, size=len(df))

        train_idx = np.arange(len(df))[:80]
        test_idx = np.arange(len(df))[80:]

        df_train = df.iloc[train_idx]
        df_test = df.iloc[test_idx]

        results = evaluate_all_baselines(df_train, df_test, y_col="answer")

        # Check all baselines ran
        assert "uniform" in results
        assert "marginal" in results
        assert "cell_lookup" in results
        assert "logistic" in results
        assert "gbm" in results

        # Check metrics
        for baseline_name, metrics in results.items():
            assert "accuracy" in metrics
            assert 0 <= metrics["accuracy"] <= 1

        logger.info("✓ test_evaluate_all_baselines passed")


class TestEndToEnd:
    """End-to-end pipeline tests."""

    def test_full_pipeline(self, sample_wvs_data):
        """Test full pipeline end-to-end."""
        logger.info("Starting end-to-end pipeline test...")

        df = sample_wvs_data.copy()
        df = recode_missing_values(df)
        df["answer"] = np.random.randint(0, 4, size=len(df))

        # Step 1: Item selection
        selected = select_items(df, n_target=2)
        assert len(selected) > 0
        logger.info(f"Selected {len(selected)} items")

        # Step 2: Build folds
        fold_config = build_folds(df, n_splits=2)
        assert fold_config["n_folds"] == 2
        logger.info("Built folds")

        # Step 3: Evaluate baselines
        fold = fold_config["folds"][0]
        df_train = df[df["respondent_id"].isin(fold["train"])]
        df_test = df[df["respondent_id"].isin(fold["test"])]

        results = evaluate_all_baselines(df_train, df_test, y_col="answer")
        assert len(results) == 5
        logger.info("Baselines evaluated")

        logger.info("✓ test_full_pipeline passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
