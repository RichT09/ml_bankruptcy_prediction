#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 01: Data Loading & Feature Selection
Handles data I/O, target construction, and feature selection

Author: Richard Tschumi
Institution: HEC Lausanne

"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
import logging

logger = logging.getLogger(__name__)

# Add project root to path (bkty_ml folder)
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # bkty_ml/
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import config
except ImportError:
    from config import config


# ============================================================================
# DATA LOADING
# ============================================================================

def ensure_dirs():
    """Create necessary directories"""
    dirs = [
        config.data_dir,
        config.plots_dir,
        config.models_dir,
        config.metrics_dir,
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    logger.info("✓ Directories verified")


def load_raw_data(validate: bool = True) -> pd.DataFrame:
    """
    Load raw CSV data with robust validation
    
    Args:
        validate: If True, validate data after loading
    
    Returns:
        DataFrame with raw data
    """
    path = config.raw_data_path
    
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {path}\n"
            f"Expected location: {path.absolute()}"
        )
    
    logger.info(f"Loading data from: {path.name}")
    
    # Try multiple separators
    for sep in [';', ',', '\t']:
        try:
            df = pd.read_csv(path, sep=sep, nrows=5)
            if len(df.columns) > 5:
                df = pd.read_csv(path, sep=sep)
                logger.info(f"✓ Loaded with separator: '{sep}'")
                break
        except:
            continue
    else:
        df = pd.read_csv(path)
    
    logger.info(f"✓ Raw data loaded: {df.shape[0]:,} rows × {df.shape[1]} cols")
    
    # Validate required columns
    required_cols = [
        config.data.id_col,
        config.data.year_col,
        config.data.status_col
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Filter year range
    year_col = config.data.year_col
    df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
    
    before_filter = len(df)
    df = df[df[year_col].notna()]
    df = df[df[year_col].between(config.data.year_start, config.data.year_end)]
    after_filter = len(df)
    
    if after_filter == 0:
        raise ValueError(
            f"No data after year filtering "
            f"[{config.data.year_start}, {config.data.year_end}]"
        )
    
    logger.info(f"✓ Year filter: {before_filter:,} → {after_filter:,} rows")
    logger.info(f"  Range: {int(df[year_col].min())} → {int(df[year_col].max())}")
    
    # Sort by firm and year
    df = df.sort_values([config.data.id_col, year_col]).reset_index(drop=True)
    
    # Handle duplicates
    logger.info("\n--- Handling Duplicate (Firm, Year) Pairs ---")
    before_dedup = len(df)
    
    # Strategy 1: Keep annual reports (STD format)
    if 'datafmt' in df.columns:
        df = df[df['datafmt'] == 'STD']
        logger.info(f"  Filtered datafmt='STD': {len(df):,} rows")
    
    # Strategy 2: Keep consolidated reports
    if 'consol' in df.columns:
        df = df[df['consol'] == 'C']
        logger.info(f"  Filtered consol='C': {len(df):,} rows")
    
    # Strategy 3: Remove remaining duplicates
    n_before_final = len(df)
    df = df.drop_duplicates(subset=[config.data.id_col, year_col], keep='first')
    n_after_final = len(df)
    
    if n_before_final > n_after_final:
        logger.info(f"  Removed {n_before_final - n_after_final:,} remaining duplicates")
    
    after_dedup = len(df)
    removed = before_dedup - after_dedup
    
    if removed > 0:
        logger.info(f"✓ Deduplication: removed {removed:,} obs ({removed/before_dedup*100:.2f}%)")
        logger.info(f"  Final: {after_dedup:,} unique (firm, year) pairs")
    
    if validate:
        validate_raw_data(df)
    
    return df


def validate_raw_data(df: pd.DataFrame):
    """Validate data quality"""
    issues = []
    
    id_col = config.data.id_col
    year_col = config.data.year_col
    status_col = config.data.status_col
    
    # Check status values
    valid_statuses = ['active', 'failure']
    invalid_statuses = df[~df[status_col].isin(valid_statuses)][status_col].unique()
    if len(invalid_statuses) > 0:
        issues.append(f"Invalid status values: {invalid_statuses.tolist()}")
    
    # Check missing in key columns
    key_cols = [id_col, year_col, status_col]
    for col in key_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            issues.append(f"Missing values in {col}: {n_missing}")
    
    if issues:
        warnings.warn("Data validation warnings:\n" + "\n".join(f"  - {i}" for i in issues))
    else:
        logger.info("✓ Data validation passed")


# ============================================================================
# DATA IMPUTATION & MISSING DATA HANDLING
# ============================================================================

def impute_missing_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Impute missing financial data using forward-fill within company, then median
    
    Strategy:
    1. Forward-fill within each company (by fyear)
    2. For "active" companies: DROP rows with remaining NaN in critical features
    3. For "failure" companies: Fill remaining NaN with median (across all companies)
    4. Backward-fill for companies with NaN at the start
    
    Args:
        df: DataFrame with financial data and status column
        verbose: Print statistics
    
    Returns:
        DataFrame with imputed missing values
    """
    df = df.copy()
    id_col = config.data.id_col
    year_col = config.data.year_col
    status_col = config.data.status_col
    feature_cols = config.data.feature_cols
    
    if verbose:
        logger.info(f"\n--- Imputing Missing Data ---")
        total_missing_before = df[feature_cols].isna().sum().sum()
        logger.info(f"Total missing values: {total_missing_before:,}")
    
    # Step 1: Forward-fill within each company
    if verbose:
        logger.info(f"\nStep 1: Forward-fill within each company...")
    
    df = df.sort_values([id_col, year_col]).reset_index(drop=True)
    df[feature_cols] = df.groupby(id_col)[feature_cols].fillna(method='ffill')
    
    total_after_ffill = df[feature_cols].isna().sum().sum()
    if verbose:
        logger.info(f"  After forward-fill: {total_after_ffill:,} missing")
    
    # Step 2: Backward-fill (for companies with missing first observation)
    if verbose:
        logger.info(f"\nStep 2: Backward-fill within each company...")
    
    df[feature_cols] = df.groupby(id_col)[feature_cols].fillna(method='bfill')
    
    total_after_bfill = df[feature_cols].isna().sum().sum()
    if verbose:
        logger.info(f"  After backward-fill: {total_after_bfill:,} missing")
    
    # Step 3: Handle remaining NaN differently for "active" vs "failure" companies
    if verbose:
        logger.info(f"\nStep 3: Handle remaining NaN (status-dependent)...")
    
    # For "failure" companies: use median
    if verbose:
        logger.info(f"  - Failure companies: fill with median")
    failure_mask = df[status_col] == "failure"
    medians = df[feature_cols].median()
    df.loc[failure_mask, feature_cols] = df.loc[failure_mask, feature_cols].fillna(medians)
    
    # For "active" companies: drop rows with NaN
    if verbose:
        logger.info(f"  - Active companies: drop rows with NaN")
    
    before_drop = len(df)
    active_mask = df[status_col] == "active"
    df_active = df[active_mask].dropna(subset=feature_cols)
    df_failure = df[~active_mask]
    
    df = pd.concat([df_active, df_failure], ignore_index=True)
    df = df.sort_values([id_col, year_col]).reset_index(drop=True)
    
    after_drop = len(df)
    
    if verbose:
        total_after_impute = df[feature_cols].isna().sum().sum()
        logger.info(f"\n✓ Imputation complete:")
        logger.info(f"  Before: {before_drop:,} rows")
        logger.info(f"  After:  {after_drop:,} rows")
        logger.info(f"  Removed: {before_drop - after_drop:,} rows ({(before_drop-after_drop)/before_drop*100:.2f}%)")
        logger.info(f"  Remaining missing values: {total_after_impute:,}")
    
    return df


# ============================================================================
# TARGET CONSTRUCTION
# ============================================================================

def build_target_from_status(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Build failed_next_year target from dlrsn status
    
    Logic:
    - Firms with dlrsn='failure': mark LAST year as failed_next_year=1
    - Firms with dlrsn='active': all years = 0
    
    Args:
        df: DataFrame with status column
        verbose: Print statistics
    
    Returns:
        DataFrame with target column
    """
    df = df.copy()
    id_col = config.data.id_col
    year_col = config.data.year_col
    status_col = config.data.status_col
    target_col = config.data.target_col
    
    # Initialize target
    df[target_col] = 0
    
    # Find failure firms
    failure_firms = df.loc[df[status_col] == "failure", id_col].unique()
    
    if verbose:
        logger.info(f"\n--- Building Target Variable ---")
        logger.info(f"Total firms: {df[id_col].nunique():,}")
        logger.info(f"Failed firms (dlrsn='failure'): {len(failure_firms):,}")
        logger.info(f"Healthy firms (dlrsn='active'): {df[id_col].nunique() - len(failure_firms):,}")
    
    # Mark last year of failed firms
    for firm_id in failure_firms:
        firm_data = df[df[id_col] == firm_id]
        
        if firm_data.empty:
            continue
        
        last_year = firm_data[year_col].max()
        mask = (df[id_col] == firm_id) & (df[year_col] == last_year)
        df.loc[mask, target_col] = 1
    
    # Validate
    unique_values = df[target_col].unique()
    assert set(unique_values) <= {0, 1}, f"Invalid target values: {unique_values}"
    
    if verbose:
        n_failures = df[target_col].sum()
        failure_rate = n_failures / len(df) * 100
        logger.info(f"\n Target variable created:")
        logger.info(f"  Total observations: {len(df):,}")
        logger.info(f"  Failed next year (target=1): {int(n_failures):,} ({failure_rate:.2f}%)")
        logger.info(f"  Survived (target=0): {len(df) - int(n_failures):,} ({100-failure_rate:.2f}%)")
        
        firms_with_failure = df[df[target_col] == 1][id_col].nunique()
        logger.info(f"  Unique firms that will fail: {firms_with_failure:,}")
        
        if failure_rate < 5:
            logger.warning(
                f"⚠️  Severe class imbalance: {failure_rate:.2f}%. "
                "SMOTE recommended."
            )
    
    return df


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features(df: pd.DataFrame, validate: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Select relevant features for modeling
    
    Args:
        df: DataFrame with all columns
        validate: Validate feature availability
        verbose: Print statistics
    
    Returns:
        DataFrame with selected features only
    """
    keep_cols = (
        [config.data.id_col, config.data.ticker_col, 
         config.data.name_col, config.data.year_col]
        + [config.data.target_col]
        + config.data.feature_cols
    )
    
    # Check missing columns
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        logger.warning(f"⚠️  Missing columns: {missing}")
        keep_cols = [c for c in keep_cols if c in df.columns]
    
    df_features = df[keep_cols].copy()
    
    if verbose:
        logger.info(f"\n--- Feature Selection ---")
        logger.info(f"Selected {len(config.data.feature_cols)} features")
        logger.info(f"Dataset shape: {df_features.shape}")
        
        # Feature availability
        available_features = [f for f in config.data.feature_cols if f in df_features.columns]
        logger.info(f"Available features: {len(available_features)}/{len(config.data.feature_cols)}")
        
        if len(available_features) < len(config.data.feature_cols):
            missing_features = set(config.data.feature_cols) - set(available_features)
            logger.warning(f"⚠️  Missing features: {missing_features}")
    
    if validate:
        validate_features(df_features)
    
    return df_features


def validate_features(df: pd.DataFrame):
    """Validate feature quality"""
    feature_cols = [f for f in config.data.feature_cols if f in df.columns]
    
    logger.info("\n--- Feature Validation ---")
    
    # Missing values
    missing_pct = (df[feature_cols].isna().sum() / len(df) * 100).sort_values(ascending=False)
    high_missing = missing_pct[missing_pct > 30]
    
    if len(high_missing) > 0:
        logger.warning(f"⚠️  Features with >30% missing:")
        for feat, pct in high_missing.items():
            logger.warning(f"    {feat}: {pct:.2f}%")
    
    # Zero variance
    zero_var = df[feature_cols].var()
    zero_var_features = zero_var[zero_var == 0].index.tolist()
    if zero_var_features:
        logger.warning(f"⚠️  Zero variance features: {zero_var_features}")
    
    logger.info(f"✓ Feature validation complete")


def save_processed_data(df: pd.DataFrame, filename: str = "dataset_with_features.csv"):
    """Save processed dataset"""
    out_path = config.data_dir / filename
    df.to_csv(out_path, index=False)
    logger.info(f"\n✓ Saved to: {out_path}")
    logger.info(f"  Shape: {df.shape}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_data_features_pipeline(verbose: bool = True) -> pd.DataFrame:
    """
    Complete data loading + feature selection + imputation pipeline
    
    Steps:
    1. Load raw CSV data
    2. Impute missing data (forward-fill, then drop for active / median for failure)
    3. Build target variable (from status)
    4. Select features
    5. Save processed data
    
    Returns:
        DataFrame ready for preprocessing
    """
    print("\n" + "="*70)
    print("MODULE 01: DATA LOADING & FEATURE SELECTION")
    print("="*70)
    
    # Create directories
    ensure_dirs()
    
    # Step 1: Load raw data
    df_raw = load_raw_data(validate=True)
    
    # Step 2: Impute missing data
    df_imputed = impute_missing_data(df_raw, verbose=verbose)
    
    # Step 3: Build target
    df_with_target = build_target_from_status(df_imputed, verbose=verbose)
    
    # Step 4: Select features
    df_features = select_features(df_with_target, validate=True, verbose=verbose)
    
    # Step 5: Save
    save_processed_data(df_features)
    
    print("")
    print("=" * 70)
    print("MODULE 01 COMPLETED")
    print("=" * 70)
    
    return df_features


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    df = run_data_features_pipeline(verbose=True)
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Target distribution:\n{df[config.data.target_col].value_counts()}")
