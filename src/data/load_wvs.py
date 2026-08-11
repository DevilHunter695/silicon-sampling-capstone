"""Load and preprocess World Values Survey data."""

import argparse
import logging
from pathlib import Path

import pandas as pd
from src.config import DATA_RAW, DATA_PROCESSED, COUNTRY_CODE, WVS_INDIA_N

logger = logging.getLogger(__name__)

# WVS missing value codes (negative numbers)
MISSING_CODES = {-1, -2, -3, -4, -5}


def load_wvs_raw(csv_path: Path) -> pd.DataFrame:
    """Load raw WVS CSV file."""
    logger.info(f"Loading WVS CSV from {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info(f"Loaded {len(df)} total respondents, {len(df.columns)} columns")
    return df


def recode_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Recode WVS missing value codes (-1..-5) to NaN."""
    logger.info("Recoding missing values...")

    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        mask = df[col].isin(MISSING_CODES)
        if mask.any():
            df.loc[mask, col] = pd.NA

    logger.info(f"Recoded {mask.sum()} missing value codes")
    return df


def filter_by_country(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    """Filter respondents by country code."""
    # WVS uses 'B_COUNTRY_ALPHA' or similar field
    country_col = None
    for col in ['B_COUNTRY_ALPHA', 'Country', 'country']:
        if col in df.columns:
            country_col = col
            break

    if country_col is None:
        logger.warning("Country column not found, returning all data")
        return df

    df_country = df[df[country_col] == country_code].copy()
    logger.info(f"Filtered to {len(df_country)} respondents in {country_code}")

    assert len(df_country) > 0, f"No respondents found for {country_code}"

    return df_country


def verify_india_demographics(df: pd.DataFrame) -> dict:
    """Verify and report India demographic marginals."""
    results = {}

    # Sex ratio (Q260)
    if 'Q260' in df.columns:
        sex_dist = df['Q260'].value_counts(normalize=True)
        results['sex'] = sex_dist.to_dict()
        logger.info(f"Sex distribution: {sex_dist.to_dict()}")

    # Urban/rural (H_URBRURAL)
    if 'H_URBRURAL' in df.columns:
        urb_dist = df['H_URBRURAL'].value_counts(normalize=True)
        results['urban_rural'] = urb_dist.to_dict()
        logger.info(f"Urban/rural distribution: {urb_dist.to_dict()}")

    # Age (Q262)
    if 'Q262' in df.columns:
        age_stats = df['Q262'].describe()
        results['age_stats'] = age_stats.to_dict()
        logger.info(f"Age stats: mean={age_stats['mean']:.1f}, std={age_stats['std']:.1f}")

    return results


def main(country: str = COUNTRY_CODE):
    """Load, clean, and save WVS data for specified country."""
    csv_path = DATA_RAW / "WVS_Cross_Wave_1981_2022_CSV_v5_0.csv"

    if not csv_path.exists():
        logger.error(
            f"WVS CSV not found at {csv_path}\n"
            f"Download from: https://www.worldvaluessurvey.org/\n"
            f"Then place in: {DATA_RAW}/"
        )
        return False

    # Load and clean
    df = load_wvs_raw(csv_path)
    df = recode_missing_values(df)
    df = filter_by_country(df, country)

    # Verify
    verify_india_demographics(df)

    # Save processed
    output_path = DATA_PROCESSED / f"{country.lower()}_wvs7.parquet"
    df.to_parquet(output_path)
    logger.info(f"Saved {len(df)} respondents to {output_path}")

    # Verify no missing codes survived
    for col in df.select_dtypes(include=['number']).columns:
        if (df[col].isin(MISSING_CODES)).any():
            logger.error(f"Missing codes survived in {col}!")
            return False

    logger.info("✓ Data load and clean complete")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", default=COUNTRY_CODE, help="Country code (e.g., IND)")
    args = parser.parse_args()

    success = main(args.country)
    exit(0 if success else 1)
