#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 03: Model Training

IMPROVED: TimeSeriesSplit for CV + RFE feature selection

============================================================================
MODEL ARCHITECTURE - BASE vs TUNED
============================================================================

| Model  | Dataset      | Hyperparams          |  CV        | SMOTE | class_weight |
|--------|--------------|----------------------|------------|-------|--------------|
| BASE   | Train normal | Fixed (from config)  |  NO        |  NO   | YES-balanced |
| TUNED  | Train SMOTE  | RandomizedSearchCV   |  YES-TSCV  |  YES  | YES-balanced |

BASE models: Conservative, no tuning, original imbalanced data
TUNED models: Optimized via TimeSeriesSplit CV, SMOTE resampled data

Author: Richard Tschumi
Institution: HEC Lausanne
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.feature_selection import RFE
import pickle
import logging
from pathlib import Path
import sys
from typing import Dict, Tuple, List
import time

logger = logging.getLogger(__name__)

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import config
except ImportError:
    import config


# ============================================================================
# FEATURE SELECTION (RFE)
# ============================================================================

def select_features_rfe(
    X_train, 
    y_train, 
    n_features_to_select: int = 20,
    verbose: bool = True
) -> Tuple[List[str], np.ndarray]:
    """
    Recursive Feature Elimination using XGBoost
    
    - Removes redundant/correlated features
    - Reduces overfitting risk
    - Faster training
    - More interpretable models
    
    Args:
        X_train: Training features (DataFrame or array)
        y_train: Training target
        n_features_to_select: Number of features to keep (default 20)
        verbose: Print progress
    
    Returns:
        selected_features: List of selected feature names
        feature_mask: Boolean mask for selected features
    """
    if verbose:
        logger.info(f"\n--- RFE Feature Selection (targeting {n_features_to_select} features) ---")
        logger.info(f"Starting with {X_train.shape[1]} features")
    
    # Use XGBoost as base estimator (fast, handles imbalance well)
    base_estimator = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        verbosity=0
    )
    
    # RFE
    rfe = RFE(
        estimator=base_estimator,
        n_features_to_select=n_features_to_select,
        step=5,  # Remove 5 features at a time (faster)
        verbose=0
    )
    
    start_time = time.time()
    rfe.fit(X_train, y_train)
    elapsed = time.time() - start_time
    
    # Get selected features
    feature_mask = rfe.support_
    
    if hasattr(X_train, 'columns'):
        selected_features = X_train.columns[feature_mask].tolist()
    else:
        selected_features = [f"feature_{i}" for i in range(X_train.shape[1]) if feature_mask[i]]
    
    if verbose:
        logger.info(f"RFE completed in {elapsed:.2f}s")
        logger.info(f"Selected {len(selected_features)} features:")
        for i, feat in enumerate(selected_features, 1):
            logger.info(f"   {i:2d}. {feat}")
    
    return selected_features, feature_mask


# ============================================================================
# MODEL BUILDING - BASE VERSION
# ============================================================================

def build_base_models(class_weights, scale_pos_weight, verbose: bool = True) -> Dict:
    """
    Build base models with default/conservative hyperparameters
    
    Models:
    - LogisticRegression: Linear baseline
    - RandomForest: Tree-based baseline
    - XGBoost: Gradient boosting baseline
    
    Args:
        class_weights: Dictionary of class weights
        scale_pos_weight: XGBoost weight parameter
        verbose: Print model info
    
    Returns:
        Dictionary of base models
    """
    if verbose:
        logger.info("\n--- Building BASE Models (Conservative Hyperparameters) ---")
    
    models_base = {}
    
    # Logistic Regression
    logreg = LogisticRegression(**config.models.logreg_base)
    models_base['LogisticRegression_base'] = logreg
    if verbose:
        logger.info("✓ LogisticRegression (base)")
    
    # Random Forest - pass class_weights for consistent FP penalty
    rf_params = config.models.rf_base.copy()
    rf_params['class_weight'] = class_weights
    rf = RandomForestClassifier(**rf_params)
    models_base['RandomForest_base'] = rf
    if verbose:
        logger.info("✓ RandomForest (base)")
    
    # XGBoost
    xgb_params = config.models.xgb_base.copy()
    xgb_params['scale_pos_weight'] = scale_pos_weight
    xgb = XGBClassifier(**xgb_params)
    models_base['XGBoost_base'] = xgb
    if verbose:
        logger.info("✓ XGBoost (base)")
    
    return models_base


# ============================================================================
# MODEL BUILDING - TUNED VERSION
# ============================================================================

def build_tuned_models(class_weights, scale_pos_weight, verbose: bool = True) -> Dict:
    """
    Build tuned models with optimized hyperparameters
    
    Models:
    - LogisticRegression: Tuned hyperparameters
    - RandomForest: Deeper trees and more estimators
    - XGBoost: Higher learning capacity
    
    Args:
        class_weights: Dictionary of class weights
        scale_pos_weight: XGBoost weight parameter
        verbose: Print model info
    
    Returns:
        Dictionary of tuned models
    """
    if verbose:
        logger.info("\n--- Building TUNED Models (Optimized Hyperparameters) ---")
    
    models_tuned = {}
    
    # Logistic Regression
    logreg = LogisticRegression(**config.models.logreg_tuned)
    models_tuned['LogisticRegression_tuned'] = logreg
    if verbose:
        logger.info("✓ LogisticRegression (tuned)")
    
    # Random Forest - pass class_weights for consistent FP penalty
    rf_params = config.models.rf_tuned.copy()
    rf_params['class_weight'] = class_weights
    rf = RandomForestClassifier(**rf_params)
    models_tuned['RandomForest_tuned'] = rf
    if verbose:
        logger.info("✓ RandomForest (tuned)")
    
    # XGBoost
    xgb_params = config.models.xgb_tuned.copy()
    xgb_params['scale_pos_weight'] = scale_pos_weight
    xgb = XGBClassifier(**xgb_params)
    models_tuned['XGBoost_tuned'] = xgb
    if verbose:
        logger.info("✓ XGBoost (tuned)")
    
    return models_tuned


# ============================================================================
# MODEL TRAINING
# ============================================================================

def fit_models(
    models: Dict,
    X_train,
    X_train_scaled,
    y_train,
    X_train_smote=None,
    X_train_smote_scaled=None,
    y_train_smote=None,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    Train all models with Cross-Validation for hyperparameter tuning
    
    Strategy for data selection:
    - Linear models (LogReg) use scaled data
    - Tree models (RF, XGBoost) use raw data
    
    Data source depends on model type:
    - BASE models: Original unsampled data (no tuning)
    - TUNED models: SMOTE resampled data + GridSearchCV with TimeSeriesSplit
    
    Args:
        models: Dictionary of models to train
        X_train: Raw training features
        X_train_scaled: Scaled training features
        y_train: Training target
        X_train_smote: SMOTE resampled features (optional)
        X_train_smote_scaled: SMOTE scaled features (optional)
        y_train_smote: SMOTE target (optional)
        verbose: Print progress
    
    Returns:
        Tuple of (fitted_models, training_metrics)
    """
    use_smote = X_train_smote is not None and config.models.use_smote
    training_metrics = {}
    
    # TimeSeriesSplit for temporal CV (prevents future data leakage)
    tscv = TimeSeriesSplit(n_splits=3)
    
    if verbose:
        logger.info(f"\n--- Training Models (SMOTE: {use_smote}, CV: TimeSeriesSplit(3)) ---")
    
    for name, model in models.items():
        start_time = time.time()
        
        # Determine which data to use and whether to tune
        is_tuned = 'tuned' in name.lower()
        
        if 'LogisticRegression' in name:
            # Logistic Regression: use scaled data
            X = X_train_smote_scaled if use_smote and is_tuned else X_train_scaled
            y = y_train_smote if use_smote and is_tuned else y_train
            
            # RandomizedSearchCV with TimeSeriesSplit for tuned models
            if is_tuned:
                try:
                    rs = RandomizedSearchCV(
                        model,
                        config.models.logreg_param_grid,
                        cv=tscv,
                        scoring='roc_auc',
                        n_iter=10,
                        n_jobs=config.models.n_jobs if config.models.n_jobs != 1 else -1,
                        random_state=42,
                        verbose=0
                    )
                    rs.fit(X, y)
                except (ModuleNotFoundError, ImportError):
                    print("Parallel processing failed, retrying with n_jobs=1")
                    rs = RandomizedSearchCV(
                        model,
                        config.models.logreg_param_grid,
                        cv=tscv,
                        scoring='roc_auc',
                        n_iter=10,
                        n_jobs=1,
                        random_state=42,
                        verbose=0
                    )
                    rs.fit(X, y)
                model = rs.best_estimator_
                models[name] = model
                if verbose:
                    logger.info(f"  → TimeSeriesCV best params: {rs.best_params_}")
                    logger.info(f"  → TimeSeriesCV best AUC: {rs.best_score_:.4f}")
            else:
                model.fit(X, y)
        else:  # Tree-based models (RF, XGBoost)
            # Tree models: use raw data
            X = X_train_smote if use_smote and is_tuned else X_train
            y = y_train_smote if use_smote and is_tuned else y_train
            
            # RandomizedSearchCV with TimeSeriesSplit for tuned models
            if is_tuned and 'RandomForest' in name:
                try:
                    rs = RandomizedSearchCV(
                        model,
                        config.models.rf_param_grid,
                        cv=tscv,
                        scoring='roc_auc',
                        n_iter=10,
                        n_jobs=config.models.n_jobs if config.models.n_jobs != 1 else -1,
                        random_state=42,
                        verbose=0
                    )
                    rs.fit(X, y)
                except (ModuleNotFoundError, ImportError):
                    print("Parallel processing failed, retrying with n_jobs=1")
                    rs = RandomizedSearchCV(
                        model,
                        config.models.rf_param_grid,
                        cv=tscv,
                        scoring='roc_auc',
                        n_iter=10,
                        n_jobs=1,
                        random_state=42,
                        verbose=0
                    )
                    rs.fit(X, y)
                model = rs.best_estimator_
                models[name] = model
                if verbose:
                    logger.info(f"  → TimeSeriesCV best params: {rs.best_params_}")
                    logger.info(f"  → TimeSeriesCV best AUC: {rs.best_score_:.4f}")
            elif is_tuned and 'XGBoost' in name:
                try:
                    rs = RandomizedSearchCV(
                        model,
                        config.models.xgb_param_grid,
                        cv=tscv,
                        scoring='roc_auc',
                        n_iter=10,
                        n_jobs=config.models.n_jobs if config.models.n_jobs != 1 else -1,
                        random_state=42,
                        verbose=0
                    )
                    rs.fit(X, y)
                except (ModuleNotFoundError, ImportError):
                    print("Parallel processing failed, retrying with n_jobs=1")
                    rs = RandomizedSearchCV(
                        model,
                        config.models.xgb_param_grid,
                        cv=tscv,
                        scoring='roc_auc',
                        n_iter=10,
                        n_jobs=1,
                        random_state=42,
                        verbose=0
                    )
                    rs.fit(X, y)
                model = rs.best_estimator_
                models[name] = model
                if verbose:
                    logger.info(f"  → TimeSeriesCV best params: {rs.best_params_}")
                    logger.info(f"  → TimeSeriesCV best AUC: {rs.best_score_:.4f}")
            else:
                model.fit(X, y)
        
        elapsed = time.time() - start_time
        training_metrics[name] = {
            'training_time': elapsed,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'is_tuned': is_tuned,
        }
        
        if verbose:
            logger.info(f"✓ {name:30s} trained in {elapsed:6.2f}s  |  {X.shape[0]:,} samples")
    
    return models, training_metrics


# ============================================================================
# HYPERPARAMETERS REPORT
# ============================================================================

def generate_hyperparameters_report(
    models_base: Dict,
    models_tuned: Dict,
    selected_features: List[str],
    training_metrics: Dict = None,
    verbose: bool = True
) -> str:
    """
    Generate a readable hyperparameters report saved to outputs/reports/
    
    Args:
        models_base: Dictionary of trained base models
        models_tuned: Dictionary of trained tuned models
        selected_features: List of RFE-selected features
        training_metrics: Training stats per model
        verbose: Print report path
    
    Returns:
        Path to saved report
    """
    from datetime import datetime
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BANKRUPTCY PREDICTION - HYPERPARAMETERS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    
    # Global settings
    report_lines.append("-" * 80)
    report_lines.append("GLOBAL SETTINGS (from config.py)")
    report_lines.append("-" * 80)
    report_lines.append(f"  random_state:        {config.models.random_state}")
    report_lines.append(f"  use_smote:           {config.models.use_smote}")
    report_lines.append(f"  smote_ratio_tuned:   {config.models.smote_ratio_tuned}")
    report_lines.append(f"  fp_weight_multiplier:{config.models.fp_weight_multiplier}")
    report_lines.append(f"  rfe_n_features:      {config.models.rfe_n_features}")
    report_lines.append(f"  split_year:          {config.data.split_year}")
    report_lines.append("")
    
    # RFE Features
    report_lines.append("-" * 80)
    report_lines.append(f"RFE SELECTED FEATURES ({len(selected_features)})")
    report_lines.append("-" * 80)
    for i, feat in enumerate(selected_features, 1):
        report_lines.append(f"  {i:2d}. {feat}")
    report_lines.append("")
    
    # Base models
    report_lines.append("-" * 80)
    report_lines.append("BASE MODELS")
    report_lines.append("-" * 80)
    for name, model in models_base.items():
        report_lines.append(f"\n  {name}:")
        params = model.get_params()
        for k, v in sorted(params.items()):
            if v is not None and k not in ['verbose', 'verbosity', 'silent']:
                report_lines.append(f"    {k}: {v}")
        if training_metrics and name in training_metrics:
            tm = training_metrics[name]
            report_lines.append(f"    [training_time: {tm.get('training_time', 0):.2f}s]")
    report_lines.append("")
    
    # Tuned models
    report_lines.append("-" * 80)
    report_lines.append("TUNED MODELS")
    report_lines.append("-" * 80)
    for name, model in models_tuned.items():
        report_lines.append(f"\n  {name}:")
        params = model.get_params()
        for k, v in sorted(params.items()):
            if v is not None and k not in ['verbose', 'verbosity', 'silent']:
                report_lines.append(f"    {k}: {v}")
        if training_metrics and name in training_metrics:
            tm = training_metrics[name]
            report_lines.append(f"    [training_time: {tm.get('training_time', 0):.2f}s]")
    report_lines.append("")
    
    # Search grids used
    report_lines.append("-" * 80)
    report_lines.append("HYPERPARAMETER SEARCH GRIDS (RandomizedSearchCV)")
    report_lines.append("-" * 80)
    report_lines.append("\n  LogisticRegression:")
    for k, v in config.models.logreg_param_grid.items():
        report_lines.append(f"    {k}: {v}")
    report_lines.append("\n  RandomForest:")
    for k, v in config.models.rf_param_grid.items():
        report_lines.append(f"    {k}: {v}")
    report_lines.append("\n  XGBoost:")
    for k, v in config.models.xgb_param_grid.items():
        report_lines.append(f"    {k}: {v}")
    report_lines.append("")
    
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report_lines)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = config.reports_dir / f"hyperparameters_{timestamp}.txt"
    
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    if verbose:
        logger.info(f"✓ Hyperparameters report: {report_path}")
    
    return str(report_path)


# ============================================================================
# MODEL SAVING
# ============================================================================

def save_models(models_base, models_tuned, scaler, metadata: Dict, training_metrics: Dict = None, verbose: bool = True):
    """
    Save all models and metadata
    
    Args:
        models_base: Dictionary of base models
        models_tuned: Dictionary of tuned models
        scaler: Fitted scaler
        metadata: Additional information
        training_metrics: Training time and stats per model
        verbose: Print save paths
    """
    # Save base models
    base_path = config.models_dir / "models_base.pkl"
    with open(base_path, 'wb') as f:
        pickle.dump(models_base, f)
    if verbose:
        logger.info(f"\n✓ Saved base models: {base_path}")
    
    # Save tuned models
    tuned_path = config.models_dir / "models_tuned.pkl"
    with open(tuned_path, 'wb') as f:
        pickle.dump(models_tuned, f)
    if verbose:
        logger.info(f"✓ Saved tuned models: {tuned_path}")
    
    # Save scaler
    scaler_path = config.models_dir / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    if verbose:
        logger.info(f"✓ Saved scaler: {scaler_path}")
    
    # Enrich metadata with training metrics
    if training_metrics:
        metadata['training_metrics'] = training_metrics
    
    # Save metadata
    meta_path = config.models_dir / "metadata.pkl"
    with open(meta_path, 'wb') as f:
        pickle.dump(metadata, f)
    if verbose:
        logger.info(f"✓ Saved metadata: {meta_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_models_pipeline(data: Dict, verbose: bool = True, n_features: int = None) -> Dict:
    """
    Complete model training pipeline with hyperparameter tuning
    
    Pipeline Steps:
    1. RFE Feature Selection (select top N features from config)
    2. Build BASE and TUNED models
    3. Train BASE models (no tuning, original data)
    4. Train TUNED models (TimeSeriesSplit CV + RandomizedSearch)
    5. Save all models and generate hyperparameters report
    
    Args:
        data: Preprocessed data dictionary from module 02
        verbose: Print progress
        n_features: Number of features to select via RFE (default from config)
    
    Returns:
        Dictionary with base and tuned models + metrics
    """
    # Use config value if not specified
    if n_features is None:
        n_features = config.models.rfe_n_features
    
    print("\n" + "="*70)
    print("MODULE 03: MODEL TRAINING")
    print("="*70)
    
    # Extract data
    X_train = data['X_train']
    X_train_scaled = data['X_train_scaled']
    y_train = data['y_train']
    X_test = data.get('X_test')
    X_test_scaled = data.get('X_test_scaled')
    X_train_smote = data.get('X_train_smote')
    X_train_smote_scaled = data.get('X_train_smote_scaled')
    y_train_smote = data.get('y_train_smote')
    class_weights = data['class_weights']
    scale_pos_weight = data['scale_pos_weight']
    scaler = data['scaler']
    
    # ====== STEP 0: RFE Feature Selection ======
    print("\n" + "-"*70)
    print("PHASE 0: RFE FEATURE SELECTION")
    print("-"*70)
    
    selected_features, feature_mask = select_features_rfe(
        X_train, y_train, 
        n_features_to_select=n_features,
        verbose=verbose
    )
    
    # Apply feature selection to all datasets
    if hasattr(X_train, 'columns'):
        X_train = X_train[selected_features]
        if X_test is not None:
            X_test = X_test[selected_features]
        if X_train_smote is not None:
            X_train_smote = X_train_smote[selected_features] if hasattr(X_train_smote, 'columns') else X_train_smote[:, feature_mask]
    else:
        X_train = X_train[:, feature_mask]
        if X_test is not None:
            X_test = X_test[:, feature_mask]
        if X_train_smote is not None:
            X_train_smote = X_train_smote[:, feature_mask]
    
    # Re-scale with selected features
    from sklearn.preprocessing import StandardScaler
    scaler_new = StandardScaler()
    X_train_scaled = scaler_new.fit_transform(X_train)
    if X_test is not None:
        X_test_scaled = scaler_new.transform(X_test)
    if X_train_smote is not None:
        X_train_smote_scaled = scaler_new.transform(X_train_smote)
    
    # Update data dict with selected features
    data['X_train'] = X_train
    data['X_train_scaled'] = X_train_scaled
    data['X_test'] = X_test
    data['X_test_scaled'] = X_test_scaled
    data['selected_features'] = selected_features
    data['scaler'] = scaler_new
    if X_train_smote is not None:
        data['X_train_smote'] = X_train_smote
        data['X_train_smote_scaled'] = X_train_smote_scaled
    
    logger.info(f"✓ Feature selection complete: {len(selected_features)} features retained")
    
    # ====== STEP 1: Build models ======
    models_base = build_base_models(class_weights, scale_pos_weight, verbose=verbose)
    models_tuned = build_tuned_models(class_weights, scale_pos_weight, verbose=verbose)
    
    # ====== STEP 2: Train BASE models ======
    print("\n" + "-"*70)
    print("PHASE 1: Training BASE Models (No Tuning)")
    print("-"*70)
    models_base, training_metrics_base = fit_models(
        models_base,
        X_train, X_train_scaled, y_train,
        X_train_smote, X_train_smote_scaled, y_train_smote,
        verbose=verbose
    )
    
    # ====== STEP 3: Train TUNED models ======
    print("\n" + "-"*70)
    print("PHASE 2: Training TUNED Models (TimeSeriesSplit CV + RandomizedSearch)")
    print("-"*70)
    models_tuned, training_metrics_tuned = fit_models(
        models_tuned,
        X_train, X_train_scaled, y_train,
        X_train_smote, X_train_smote_scaled, y_train_smote,
        verbose=verbose
    )
    
    # Combine training metrics
    training_metrics = {**training_metrics_base, **training_metrics_tuned}
    
    # ====== STEP 4: Save models ======
    metadata = {
        'train_size': len(X_train),
        'n_features': X_train.shape[1],
        'selected_features': selected_features,
        'class_weights': class_weights,
        'scale_pos_weight': float(scale_pos_weight),
        'use_smote': config.models.use_smote,
        'split_year': config.data.split_year,
        'cv_method': 'TimeSeriesSplit(n_splits=3)',
    }
    
    save_models(models_base, models_tuned, scaler_new, metadata, training_metrics, verbose=verbose)
    
    # Generate hyperparameters report
    report_path = generate_hyperparameters_report(
        models_base, models_tuned, selected_features,
        training_metrics, verbose=verbose
    )
    
    print("\n" + "="*70)
    print("MODULE 03 COMPLETED")
    print("="*70)
    logger.info("Training Summary:")
    logger.info(f"   Features: {len(selected_features)} (RFE selected)")
    logger.info(f"   CV method: TimeSeriesSplit (no future leakage)")
    logger.info(f"   Base models: {len(models_base)}")
    logger.info(f"   Tuned models: {len(models_tuned)}")

    
    return {
        'models_base': models_base,
        'models_tuned': models_tuned,
        'training_metrics': training_metrics,
        'selected_features': selected_features,
    }


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Load preprocessed data
    import pickle
    
    data_path = config.models_dir / "preprocessed_data.pkl"
    if not data_path.exists():
        logger.error(f"{data_path} not found. Run module 02 first.")
        sys.exit(1)
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Run training
    models = run_models_pipeline(data, verbose=True)
    
    print(f"\nTraining complete!")
    print(f"  Base models: {list(models['models_base'].keys())}")
    print(f"  Tuned models: {list(models['models_tuned'].keys())}")
