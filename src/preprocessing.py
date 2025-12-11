#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 02: Preprocessing
Data cleaning, train/test split, scaling, and SMOTE resampling

Author: Richard Tschumi
Institution: HEC Lausanne
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import logging

logger = logging.getLogger(__name__)

try:
    from . import config
except ImportError:
    import config


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feature_cols = config.FEATURE_COLS

    medians = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(medians)

    lower = df[feature_cols].quantile(0.01)
    upper = df[feature_cols].quantile(0.99)
    df[feature_cols] = df[feature_cols].clip(lower=lower, upper=upper, axis=1)

    variances = df[feature_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        df = df.drop(columns=zero_var)

    return df


def time_based_split(df: pd.DataFrame):
    df_train = df[df[config.YEAR_COL] <= config.SPLIT_YEAR].copy()
    df_test = df[df[config.YEAR_COL] > config.SPLIT_YEAR].copy()

    X_train = df_train[config.FEATURE_COLS]
    y_train = df_train[config.TARGET_COL].astype(int)
    X_test = df_test[config.FEATURE_COLS]
    y_test = df_test[config.TARGET_COL].astype(int)

    return df_train, df_test, X_train, X_test, y_train, y_test


def scale_for_linear_models(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return scaler, X_train_scaled, X_test_scaled


def compute_class_weights(y_train, fp_weight_multiplier=1.0):
    """
    Compute class weights for imbalanced classification
    
    Formula:
        1. balanced_weight = n_samples / (n_classes * n_samples_per_class)
           → With 1% failures: weight[1] ≈ 86x weight[0]
        
        2. adjusted_weight[1] = balanced_weight[1] * fp_weight_multiplier
           → multiplier < 1 reduces positive class weight
           → This leads to fewer false positives (model less aggressive)
        
        3. scale_pos_weight = weight[1] / weight[0]
           → Used by XGBoost
    
    Example with 1% failure rate:
        - multiplier=1.0  → scale_pos_weight ≈ 86 (fully balanced)
        - multiplier=0.45 → scale_pos_weight ≈ 39 (fewer FP, lower recall)
    
    Args:
        y_train: Target labels
        fp_weight_multiplier: Factor to adjust positive class weight
                              < 1 = fewer false positives, lower recall
                              > 1 = more false positives, higher recall
    
    Returns:
        class_weight_dict: {0: w0, 1: w1} for sklearn models
        scale_pos_weight: w1/w0 for XGBoost
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    
    # Adjust positive class weight
    weights[1] *= fp_weight_multiplier
    
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    scale_pos_weight = float(weights[1] / weights[0]) if len(classes) == 2 else 1.0
    return class_weight_dict, scale_pos_weight


# ============ SMOTE WITH CONTROLLED RATIO ============
def apply_smote(X_train, y_train, sampling_strategy=0.15, verbose=True):
    """
    SMOTE with controlled ratio (not 50/50)
    
    For precision-focused models:
    - sampling_strategy: enerates synthetic minorities 
    - Keeps false positive rate lower than 50/50 balance
    - Better precision while maintaining some recall
    
    Args:
        X_train: Feature matrix
        y_train: Target labels
        sampling_strategy: Ratio of minority to majority class (default 0.15 = 15%)
        verbose: Print statistics
    
    Returns:
        X_resampled, y_resampled
    """
    if verbose:
        n_before = len(y_train)
        n_fail_before = y_train.sum()
        logger.info(f"Before: {n_before:,} samples | {n_fail_before} failures ({n_fail_before/n_before*100:.2f}%)")
    
    sm = SMOTE(
        sampling_strategy=sampling_strategy,  # Key parameter: not 'auto' (50/50)
        k_neighbors=5,
        random_state=config.RANDOM_STATE
    )
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    if verbose:
        n_after = len(y_res)
        n_fail_after = y_res.sum()
        fail_ratio = n_fail_after / n_after * 100
        logger.info(f"After:  {n_after:,} samples | {n_fail_after} failures ({fail_ratio:.2f}%)")
        logger.info(f"Added {n_after - n_before:,} synthetic samples")
        logger.info(f"Class ratio: {sampling_strategy*100:.2f}% minority")
    
    return X_res, y_res


def generate_preprocessing_summary(
    n_raw: int,
    n_fail_raw: int,
    n_train: int,
    n_fail_train: int,
    n_test: int,
    n_fail_test: int,
    n_smote: int = None,
    n_fail_smote: int = None,
    output_dir=None
) -> pd.DataFrame:
    """
    Generate a summary table tracking data transformation at each step.
    
    This provides full transparency for the thesis report.
    
    Args:
        n_raw: Total samples in raw dataset
        n_fail_raw: Failures in raw dataset
        n_train: Training samples
        n_fail_train: Training failures
        n_test: Test samples
        n_fail_test: Test failures
        n_smote: Samples after SMOTE (optional)
        n_fail_smote: Failures after SMOTE (optional)
        output_dir: Directory to save CSV (optional)
    
    Returns:
        DataFrame with transformation summary
    """
    rows = [
        {
            'Step': 'Data',
            'n_samples': n_raw,
            'n_failures': n_fail_raw,
            'failure_rate': n_fail_raw / n_raw * 100 if n_raw > 0 else 0
        },
        {
            'Step': 'Train Set',
            'n_samples': n_train,
            'n_failures': n_fail_train,
            'failure_rate': n_fail_train / n_train * 100 if n_train > 0 else 0
        }
    ]
    
    # SMOTE row comes after Train Set, before Test Set
    if n_smote is not None:
        rows.append({
            'Step': 'After SMOTE (Train)',
            'n_samples': n_smote,
            'n_failures': n_fail_smote,
            'failure_rate': n_fail_smote / n_smote * 100 if n_smote > 0 else 0
        })
    
    # Test Set is always last
    rows.append({
        'Step': 'Test Set',
        'n_samples': n_test,
        'n_failures': n_fail_test,
        'failure_rate': n_fail_test / n_test * 100 if n_test > 0 else 0
    })
    
    df_summary = pd.DataFrame(rows)
    df_summary['failure_rate'] = df_summary['failure_rate'].round(2)
    
    # Save to CSV if output_dir provided
    if output_dir is not None:
        csv_path = output_dir / "preprocessing_summary.csv"
        df_summary.to_csv(csv_path, index=False)
        # Note: save confirmation logged after table display
    
    return df_summary, csv_path if output_dir else None


# ============ pipeline ============

def run_preprocessing_pipeline(df, verbose=True):
    """
    Complete preprocessing pipeline -
    
    Strategy for improving precision :
    1. Clean features + handle outliers
    2. Split by time
    3. Scale features
    4. Compute class weights with FP penalty (base models)
    5. Apply SMOTE (tuned models)
    
    Returns:
        Dictionary with both base and tuned data for model comparison
    
    Args:
        df: DataFrame with features
        verbose: Print progress
    
    Returns:
        Dictionary with all preprocessed data
    """
    if verbose:
        print("")
        print("=" * 70)
        print("MODULE 02: PREPROCESSING")
        print("=" * 70)
    
    # Step 1: Clean features
    if verbose:
        logger.info("--- Cleaning Features ---")
    df_clean = clean_features(df)
    if verbose:
        logger.info(f"Cleaned dataset: {df_clean.shape}")
    
    # Save cleaned dataset
    out_path = config.DATA_DIR / "dataset_clean.csv"
    df_clean.to_csv(out_path, index=False)
    if verbose:
        logger.info(f"Saved: {out_path}")
    
    # Step 2: Train/Test Split
    if verbose:
        logger.info("--- Train/Test Split (Time-Based) ---")
    df_train, df_test, X_train, X_test, y_train, y_test = time_based_split(df_clean)
    
    if verbose:
        logger.info(f"Train: {df_train[config.YEAR_COL].min():.0f} -> {df_train[config.YEAR_COL].max():.0f}")
        logger.info(f"  Observations: {len(df_train):,}")
        logger.info(f"  Failures: {y_train.sum():,} ({y_train.sum()/len(df_train)*100:.2f}%)")
        logger.info(f"Test:  {df_test[config.YEAR_COL].min():.0f} -> {df_test[config.YEAR_COL].max():.0f}")
        logger.info(f"  Observations: {len(df_test):,}")
        logger.info(f"  Failures: {y_test.sum():,} ({y_test.sum()/len(df_test)*100:.2f}%)")
    
    # Step 3: Scaling (BEFORE SMOTE to avoid data leakage)
    if verbose:
        logger.info("--- Scaling Features ---")
    scaler, X_train_scaled, X_test_scaled = scale_for_linear_models(X_train, X_test)
    if verbose:
        logger.info(f"Scaled features: {X_train_scaled.shape}")
    
    # Step 4: Class Weights
    if verbose:
        logger.info(f"--- Computing Class Weights (FP Penalty = {config.config.models.fp_weight_multiplier}x) ---")
    class_weights, scale_pos_weight = compute_class_weights(
        y_train, 
        fp_weight_multiplier=config.config.models.fp_weight_multiplier
    )
    if verbose:
        logger.info(f"LR/RF: class_weight = {{0: {class_weights[0]:.2f}, 1: {class_weights[1]:.2f}}}")
        logger.info(f"XGB:   scale_pos_weight = {scale_pos_weight:.2f}")
        logger.info(f"→ All models: positive class weighted {scale_pos_weight:.2f}x higher")
    
    # Step 5: SMOTE with controlled ratio (from config)
    X_train_smote = None
    X_train_smote_scaled = None
    y_train_smote = None
    
    if config.USE_SMOTE:
        smote_ratio = config.config.models.smote_ratio_tuned
        if verbose:
            logger.info(f"--- Applying SMOTE ({smote_ratio*100:.2f}% ratio) ---")
        
        X_train_smote, y_train_smote = apply_smote(
            X_train, y_train, 
            sampling_strategy=smote_ratio,
            verbose=verbose
        )
        X_train_smote_scaled = scaler.transform(X_train_smote)
        
        if verbose:
            logger.info(f"After SMOTE: {len(y_train_smote):,} samples")
            logger.info(f"  Ratio: {y_train_smote.sum() / len(y_train_smote) * 100:.2f}% failures")
    
    if verbose:
        smote_pct = config.config.models.smote_ratio_tuned * 100
        print("")
        print("=" * 70)
        print("MODULE 02 COMPLETED")
        print("=" * 70)
        logger.info("Strategy Summary:")
        logger.info("   BASE MODELS:  Unsampled data + class weight (imbalanced data)")
        logger.info(f"   TUNED MODELS: Optimized hyperparameters + SMOTE {smote_pct:.2f}% + class weight")
    
    # Generate preprocessing summary table (for thesis transparency)
    n_raw = len(df)
    n_fail_raw = int(df[config.TARGET_COL].sum())
    
    summary_df, csv_path = generate_preprocessing_summary(
        n_raw=n_raw,
        n_fail_raw=n_fail_raw,
        n_train=len(y_train),
        n_fail_train=int(y_train.sum()),
        n_test=len(y_test),
        n_fail_test=int(y_test.sum()),
        n_smote=len(y_train_smote) if y_train_smote is not None else None,
        n_fail_smote=int(y_train_smote.sum()) if y_train_smote is not None else None,
        output_dir=config.DATA_DIR
    )
    
    if verbose:
        logger.info("Data Transformation Summary:")
        print(f"{summary_df.to_string(index=False)}")
        if csv_path:
            logger.info(f"✓ Preprocessing summary saved: {csv_path}")
    
    # Return dictionary compatible with main.py
    return {
        'df_clean': df_clean,
        'df_train': df_train,
        'df_test': df_test,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'scaler': scaler,
        'class_weights': class_weights,
        'scale_pos_weight': scale_pos_weight,
        'X_train_smote': X_train_smote,
        'X_train_smote_scaled': X_train_smote_scaled,
        'y_train_smote': y_train_smote,
    }


def main():
    """
    Run preprocessing pipeline standalone
    """
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Load dataset from Module 01
    path = config.DATA_DIR / "dataset_with_features.csv"
    if not path.exists():
        logger.error(f"File not found: {path}")
        logger.error("Run Module 01 (data_features.py) first.")
        return
    
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset_with_features: {df.shape}")
    
    # Run full preprocessing pipeline
    data = run_preprocessing_pipeline(df, verbose=True)


if __name__ == "__main__":
    main()