# 02_preprocessing.py

"""
Module 02: Preprocessing
Data cleaning, train/test split, scaling, SMOTE
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import config
except ImportError:
    import config


# ============================================================================
# DATA CLEANING
# ============================================================================

def clean_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Clean features: handle missing values, outliers, zero variance
    
    Args:
        df: DataFrame with features
        verbose: Print statistics
    
    Returns:
        Cleaned DataFrame
    """
    df = df.copy()
    feature_cols = [f for f in config.data.feature_cols if f in df.columns]
    
    if verbose:
        logger.info("\n--- Feature Cleaning ---")
        logger.info(f"Initial shape: {df.shape}")
    
    # 1. Handle missing values (median imputation)
    n_missing_before = df[feature_cols].isna().sum().sum()
    medians = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(medians)
    
    if verbose and n_missing_before > 0:
        logger.info(f"✓ Imputed {n_missing_before:,} missing values (median)")
    
    # 2. Handle outliers (winsorization)
    if config.preprocessing.outlier_method == "winsorize":
        lower = df[feature_cols].quantile(config.preprocessing.outlier_lower_pct / 100)
        upper = df[feature_cols].quantile(config.preprocessing.outlier_upper_pct / 100)
        df[feature_cols] = df[feature_cols].clip(lower=lower, upper=upper, axis=1)
        
        if verbose:
            logger.info(f"✓ Winsorized outliers ({config.preprocessing.outlier_lower_pct}%, {config.preprocessing.outlier_upper_pct}%)")
    
    # 3. Remove zero variance features
    variances = df[feature_cols].var()
    zero_var = variances[variances == 0].index.tolist()
    if zero_var:
        df = df.drop(columns=zero_var)
        if verbose:
            logger.info(f"✓ Removed {len(zero_var)} zero-variance features")
    
    # 4. Replace inf values
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(medians)
    
    if verbose:
        logger.info(f"Final shape: {df.shape}")
    
    return df


# ============================================================================
# TRAIN/TEST SPLIT
# ============================================================================

def time_based_split(df: pd.DataFrame, verbose: bool = True):
    """
    Time-based train/test split (no data leakage)
    
    Train: years <= split_year
    Test: years > split_year
    
    Args:
        df: DataFrame with cleaned features
        verbose: Print statistics
    
    Returns:
        df_train, df_test, X_train, X_test, y_train, y_test
    """
    year_col = config.data.year_col
    target_col = config.data.target_col
    feature_cols = [f for f in config.data.feature_cols if f in df.columns]
    
    # Split
    df_train = df[df[year_col] <= config.data.split_year].copy()
    df_test = df[df[year_col] > config.data.split_year].copy()
    
    # Extract X, y
    X_train = df_train[feature_cols]
    y_train = df_train[target_col].astype(int)
    
    X_test = df_test[feature_cols]
    y_test = df_test[target_col].astype(int)
    
    if verbose:
        logger.info("\n--- Train/Test Split (Time-Based) ---")
        logger.info(f"Train: {df_train[year_col].min():.0f} → {df_train[year_col].max():.0f}")
        logger.info(f"  Observations: {len(df_train):,}")
        logger.info(f"  Failures: {y_train.sum():,} ({y_train.sum()/len(y_train)*100:.2f}%)")
        
        logger.info(f"\nTest: {df_test[year_col].min():.0f} → {df_test[year_col].max():.0f}")
        logger.info(f"  Observations: {len(df_test):,}")
        logger.info(f"  Failures: {y_test.sum():,} ({y_test.sum()/len(y_test)*100:.2f}%)")
        
        logger.info(f"\nFeatures: {len(feature_cols)}")
    
    return df_train, df_test, X_train, X_test, y_train, y_test


# ============================================================================
# SCALING
# ============================================================================

def scale_for_linear_models(X_train, X_test, verbose: bool = True):
    """
    Scale features for linear models (LogReg, SVM)
    
    Args:
        X_train: Training features
        X_test: Test features
        verbose: Print info
    
    Returns:
        scaler, X_train_scaled, X_test_scaled
    """
    method = config.preprocessing.scaling_method
    
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if verbose:
        logger.info(f"\n--- Scaling ({method}) ---")
        logger.info(f"✓ Scaler fitted on train set")
        logger.info(f"  Train scaled: {X_train_scaled.shape}")
        logger.info(f"  Test scaled: {X_test_scaled.shape}")
    
    return scaler, X_train_scaled, X_test_scaled


# ============================================================================
# CLASS WEIGHTS
# ============================================================================

def compute_class_weights(y_train, verbose: bool = True):
    """
    Compute class weights for imbalanced data
    
    Args:
        y_train: Training target
        verbose: Print weights
    
    Returns:
        class_weight_dict, scale_pos_weight (for XGBoost)
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced", 
        classes=classes, 
        y=y_train
    )
    
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    
    # XGBoost scale_pos_weight
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0
    
    if verbose:
        logger.info("\n--- Class Weights ---")
        logger.info(f"Class distribution: {dict(pd.Series(y_train).value_counts())}")
        logger.info(f"Class weights: {class_weight_dict}")
        logger.info(f"XGBoost scale_pos_weight: {scale_pos_weight:.4f}")
    
    return class_weight_dict, scale_pos_weight


# ============================================================================
# SMOTE
# ============================================================================

def apply_smote(X_train, y_train, verbose: bool = True):
    """
    Apply SMOTE oversampling to balance classes
    
    Args:
        X_train: Training features
        y_train: Training target
        verbose: Print statistics
    
    Returns:
        X_resampled, y_resampled
    """
    sm = SMOTE(
        sampling_strategy= 0.2,  # Minority class to 20% of majority
        k_neighbors=3,
        random_state=config.models.random_state
    )
    
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    if verbose:
        logger.info("\n--- SMOTE Oversampling ---")
        logger.info(f"Before SMOTE: {X_train.shape[0]:,} samples")
        logger.info(f"  Class 0: {(y_train == 0).sum():,}")
        logger.info(f"  Class 1: {(y_train == 1).sum():,}")
        
        logger.info(f"\nAfter SMOTE: {X_res.shape[0]:,} samples")
        logger.info(f"  Class 0: {(y_res == 0).sum():,}")
        logger.info(f"  Class 1: {(y_res == 1).sum():,}")
    
    return X_res, y_res


# ============================================================================
# SAVE PREPROCESSED DATA
# ============================================================================

def save_preprocessed_data(df_clean, scaler, class_weights, scale_pos_weight):
    """Save cleaned data and preprocessing objects"""
    import pickle
    
    # Save cleaned dataset
    out_csv = config.data_dir / "dataset_clean.csv"
    df_clean.to_csv(out_csv, index=False)
    logger.info(f"\n✓ Saved cleaned dataset: {out_csv}")
    
    # Save scaler
    scaler_path = config.models_dir / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"✓ Saved scaler: {scaler_path}")
    
    # Save class weights
    weights_path = config.models_dir / "class_weights.pkl"
    with open(weights_path, 'wb') as f:
        pickle.dump({
            'class_weights': class_weights,
            'scale_pos_weight': scale_pos_weight
        }, f)
    logger.info(f"✓ Saved class weights: {weights_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_preprocessing_pipeline(df_features: pd.DataFrame, verbose: bool = True):
    """
    Complete preprocessing pipeline
    
    Args:
        df_features: DataFrame from module 01
        verbose: Print progress
    
    Returns:
        Dictionary with all preprocessed data
    """
    logger.info("\n" + "="*70)
    logger.info("MODULE 02: PREPROCESSING")
    logger.info("="*70)
    
    # 1. Clean features
    df_clean = clean_features(df_features, verbose=verbose)
    
    # 2. Train/test split
    df_train, df_test, X_train, X_test, y_train, y_test = time_based_split(
        df_clean, verbose=verbose
    )
    
    # 3. Scaling (for LogReg & SVM)
    scaler, X_train_scaled, X_test_scaled = scale_for_linear_models(
        X_train, X_test, verbose=verbose
    )
    
    # 4. Class weights
    class_weights, scale_pos_weight = compute_class_weights(y_train, verbose=verbose)
    
    # 5. SMOTE (optional)
    X_train_smote = None
    X_train_smote_scaled = None
    y_train_smote = None
    
    if config.models.use_smote:
        X_train_smote, y_train_smote = apply_smote(X_train, y_train, verbose=verbose)
        X_train_smote_scaled = scaler.transform(X_train_smote)
        logger.info("✓ SMOTE applied and scaled")
    
    # 6. Save
    save_preprocessed_data(df_clean, scaler, class_weights, scale_pos_weight)
    
    logger.info("\n" + "="*70)
    logger.info("✅ MODULE 02 COMPLETED")
    logger.info("="*70)
    
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
        'X_train_smote': X_train_smote,
        'X_train_smote_scaled': X_train_smote_scaled,
        'y_train_smote': y_train_smote,
        'scaler': scaler,
        'class_weights': class_weights,
        'scale_pos_weight': scale_pos_weight,
    }


if __name__ == "__main__":
    # Load data from module 01
    df_path = config.data_dir / "dataset_with_features.csv"
    if not df_path.exists():
        logger.error(f"❌ {df_path} not found. Run module 01 first.")
        sys.exit(1)
    
    df = pd.read_csv(df_path)
    logger.info(f"Loaded dataset: {df.shape}")
    
    # Run preprocessing
    data = run_preprocessing_pipeline(df, verbose=True)
    
    print(f"\n📊 Preprocessing complete!")
    print(f"  Train: {data['X_train'].shape}")
    print(f"  Test: {data['X_test'].shape}")
    if data['X_train_smote'] is not None:
        print(f"  Train (SMOTE): {data['X_train_smote'].shape}")
