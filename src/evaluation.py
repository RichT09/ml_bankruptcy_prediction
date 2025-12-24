#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 04: Model Evaluation

Author: Richard Tschumi
Institution: HEC Lausanne
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from pathlib import Path
import logging

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)

# Import config and paths
from .config import PLOTS_DIR, DATA_DIR, YEAR_COL, TARGET_COL, FEATURE_COLS, SPLIT_YEAR

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
#  Overfitting Analysis
# -------------------------------------------------------------------

def compute_overfitting_table(all_models, data, selected_features=None, verbose=True):
    """
    Compare Train vs Test accuracy and AUC to detect overfitting
    
    Overfitting indicators:
    - Large gap between train and test accuracy/AUC
    - Train metrics near 100% with lower test metrics
    
    Args:
        all_models: Dictionary of trained models
        data: Dictionary from preprocessing pipeline
        selected_features: List of RFE-selected features
        verbose: Print table
    
    Returns:
        DataFrame with train/test comparison
    """
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    X_train = data['X_train']
    X_test = data['X_test']
    X_train_scaled = data['X_train_scaled']
    X_test_scaled = data['X_test_scaled']
    y_train = data['y_train']
    y_test = data['y_test']
    
    # Apply feature selection if provided
    if selected_features is not None:
        feature_mask = [col in selected_features for col in X_train.columns]
        X_train = X_train.loc[:, feature_mask]
        X_test = X_test.loc[:, feature_mask]
        X_train_scaled = X_train_scaled[:, feature_mask]
        X_test_scaled = X_test_scaled[:, feature_mask]
    
    results = []
    
    for name, model in all_models.items():
        # Determine which data to use
        if 'LogisticRegression' in name:
            X_tr, X_te = X_train_scaled, X_test_scaled
        else:
            X_tr, X_te = X_train, X_test
        
        # Train predictions
        y_pred_train = model.predict(X_tr)
        y_proba_train = model.predict_proba(X_tr)[:, 1]
        
        # Test predictions
        y_pred_test = model.predict(X_te)
        y_proba_test = model.predict_proba(X_te)[:, 1]
        
        # Compute metrics
        acc_train = accuracy_score(y_train, y_pred_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        auc_train = roc_auc_score(y_train, y_proba_train)
        auc_test = roc_auc_score(y_test, y_proba_test)
        
        # Gaps (positive = overfitting)
        acc_gap = acc_train - acc_test
        auc_gap = auc_train - auc_test
        
        results.append({
            'Model': name,
            'Acc_Train': acc_train,
            'Acc_Test': acc_test,
            'Acc_Gap': acc_gap,
            'AUC_Train': auc_train,
            'AUC_Test': auc_test,
            'AUC_Gap': auc_gap,
            'Overfit': 'YES' if (acc_gap > 0.05 or auc_gap > 0.05) else 'NO'
        })
    
    df_overfit = pd.DataFrame(results)
    df_overfit = df_overfit.set_index('Model')
    
    # Save to CSV
    out_path = DATA_DIR / "overfitting_analysis.csv"
    df_overfit.to_csv(out_path)
    
    if verbose:
        logger.info("")
        logger.info("-" * 110)
        logger.info(f"{'Model':<30} {'Acc_Train':>10} {'Acc_Test':>10} {'Acc_Gap':>10} {'AUC_Train':>10} {'AUC_Test':>10} {'AUC_Gap':>10} {'Overfit':>8}")
        logger.info("-" * 110)
        for _, row in df_overfit.iterrows():
            logger.info(f"{row.name:<30} {row['Acc_Train']:>10.4f} {row['Acc_Test']:>10.4f} {row['Acc_Gap']:>+10.4f} {row['AUC_Train']:>10.4f} {row['AUC_Test']:>10.4f} {row['AUC_Gap']:>+10.4f} {row['Overfit']:>8}")
        logger.info("-" * 110)
        logger.info("Overfit threshold: Accuracy or AUC gap > 5%")
        logger.info(f"Saved: {out_path}")
    
    return df_overfit


# -------------------------------------------------------------------
#  Pipeline
# -------------------------------------------------------------------

def run_evaluation_pipeline(models_base, models_tuned, data, verbose=True, selected_features=None):
    """
    Complete evaluation pipeline - Compatible with main.py
    
    - Evaluates all models at threshold 0.5
    - Generates ROC/PR curves, confusion matrices, annual backtest
    - Computes overfitting analysis (train vs test metrics)
    
    Args:
        models_base: Dictionary of base models
        models_tuned: Dictionary of tuned models
        data: Dictionary from preprocessing pipeline
        verbose: Print progress
        selected_features: List of feature names used for training (from RFE)
    
    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print("")
        print("=" * 70)
        print("MODULE 04: EVALUATION")
        print("=" * 70)
    
    # Combine all models and sort by name: LR_base, LR_tuned, RF_base, RF_tuned, XGB_base, XGB_tuned
    all_models = {}
    if models_base:
        all_models.update(models_base)
    if models_tuned:
        all_models.update(models_tuned)
    
    # Sort models: by algorithm name first, then base before tuned
    def model_sort_key(name):
        # Extract algorithm name and type
        if 'LogisticRegression' in name:
            algo = 0
        elif 'RandomForest' in name:
            algo = 1
        elif 'XGBoost' in name:
            algo = 2
        else:
            algo = 3
        # base=0, tuned=1
        is_tuned = 1 if 'tuned' in name.lower() else 0
        return (algo, is_tuned)
    
    sorted_model_names = sorted(all_models.keys(), key=model_sort_key)
    
    # Extract data
    X_test = data['X_test']
    X_test_scaled = data['X_test_scaled']
    y_test = data['y_test']
    df_test = data['df_test']
    scaler = data['scaler']
    
    # Step 1: Predictions
    if verbose:
        logger.info("--- Making Predictions ---")
    y_pred, y_proba = predict_all(all_models, X_test, X_test_scaled)
    if verbose:
        logger.info(f"Predictions generated for {len(all_models)} models")
    
    # Step 2: Evaluate each model (sorted by name)
    if verbose:
        logger.info("--- Evaluating Models ---")
    
    all_metrics = {}
    for name in sorted_model_names:
        # Evaluate at threshold=0.5
        metrics = evaluate_model(name, y_test, y_pred[name], y_proba[name])
        all_metrics[name] = metrics
    
    # Step 3: ROC & PR Curves (BASE)
    if verbose:
        logger.info("--- Generating ROC & PR Curves (BASE models) ---")
    plot_roc_pr_curves(y_test, y_proba, all_metrics)
    
    # Step 3a: ROC & PR Curves (TUNED) - if tuned models exist
    tuned_exists = any('tuned' in k.lower() for k in y_proba.keys())
    if tuned_exists:
        if verbose:
            logger.info("--- Generating ROC & PR Curves (TUNED models) ---")
        plot_roc_pr_curves_tuned(y_test, y_proba)
    
    # Step 3b: Confusion Matrices (BASE + TUNED)
    if verbose:
        logger.info("--- Generating Confusion Matrices (BASE + TUNED) ---")
    plot_confusion_matrices(y_test, y_pred, y_proba)
    
    # Step 3c: Model Comparison Bar Chart (BASE)
    if verbose:
        logger.info("--- Generating Model Comparison Plot (BASE models) ---")
    plot_model_comparison(y_test, y_pred, y_proba)
    
    # Step 3d: Model Comparison Bar Chart (TUNED) - if tuned models exist
    if tuned_exists:
        if verbose:
            logger.info("--- Generating Model Comparison Plot (TUNED models) ---")
        plot_model_comparison_tuned(y_test, y_pred, y_proba)
    
    if verbose:
        logger.info(f"Curves, matrices and comparison saved to {PLOTS_DIR / 'evaluation'}")

    
    # Step 4: Yearly Backtest
    if verbose:
        logger.info("--- Yearly Backtest ---")
    yearly_df = yearly_backtest(df_test, all_models, scaler, selected_features=selected_features)
    if verbose:
        logger.info(f"Yearly performance computed")
    
    # Step 4b: Plot Yearly AUC Stability
    if verbose:
        logger.info("--- Plotting Yearly AUC Stability ---")
    plot_yearly_auc(yearly_df)
    
    # Step 5: BASE vs TUNED Comparison (SKIPPED if no tuned models)
    if verbose:
        logger.info("--- BASE vs TUNED Model Comparison ---")
    compare_base_vs_tuned(all_metrics, verbose=verbose)
    
    # Step 6: Overfitting Analysis (Train vs Test)
    if verbose:
        logger.info("")
        logger.info("="*60)
        logger.info("OVERFITTING ANALYSIS")
        logger.info("="*60)
    df_overfit = compute_overfitting_table(all_models, data, selected_features=selected_features, verbose=verbose)
    
    # Step 7: Save results
    out_metrics = DATA_DIR / "metrics_summary_eval.csv"
    df_metrics = pd.DataFrame(all_metrics).T
    df_metrics.to_csv(out_metrics)
    
    if verbose:
        logger.info(f"Metrics saved: {out_metrics}")
        print("")
        print("=" * 70)
        print("MODULE 04 COMPLETED")
        print("=" * 70)
    
    # Return results compatible with main.py
    return {
        'results': all_metrics,
        'df_results': df_metrics,
        'df_overfit': df_overfit,
        'yearly_df': yearly_df,
        'predictions': y_pred,
        'probabilities': y_proba
    }


# -------------------------------------------------------------------
#  Utilities
# -------------------------------------------------------------------
def time_based_split(df: pd.DataFrame):
    """Time-based train/test split using SPLIT_YEAR."""
    df_train = df[df[YEAR_COL] <= SPLIT_YEAR].copy()
    df_test = df[df[YEAR_COL] > SPLIT_YEAR].copy()

    X_train = df_train[FEATURE_COLS]
    y_train = df_train[TARGET_COL].astype(int)

    X_test = df_test[FEATURE_COLS]
    y_test = df_test[TARGET_COL].astype(int)

    return df_train, df_test, X_train, X_test, y_train, y_test


# -------------------------------------------------------------------
#  Evaluation Functions
# -------------------------------------------------------------------
def compare_base_vs_tuned(all_metrics: Dict, verbose: bool = True) -> Dict:
    """
    Compare BASE models vs TUNED models
    
    Shows average improvement in key metrics:
    - Precision (lower is better for FP reduction)
    - Recall (higher is better for TP capture)
    - F1-score (balanced metric)
    - AUC (ranking quality)
    
    Args:
        all_metrics: Dictionary with metrics for all models
        verbose: Print comparison
    
    Returns:
        Dictionary with BASE vs TUNED comparison
    """
    # Separate BASE and TUNED models
    base_metrics = {k: v for k, v in all_metrics.items() if 'base' in k.lower()}
    tuned_metrics = {k: v for k, v in all_metrics.items() if 'tuned' in k.lower()}
    
    if not base_metrics or not tuned_metrics:
        if verbose:
            logger.warning("⚠️  Cannot compare: need both BASE and TUNED models")
        return {}
    
    # Compute averages
    avg_base = {}
    avg_tuned = {}
    
    for metric_key in ['precision', 'recall', 'f1', 'auc']:
        base_values = [m.get(metric_key, np.nan) for m in base_metrics.values() if metric_key in m]
        tuned_values = [m.get(metric_key, np.nan) for m in tuned_metrics.values() if metric_key in m]
        
        if base_values:
            avg_base[metric_key] = np.nanmean(base_values)
        if tuned_values:
            avg_tuned[metric_key] = np.nanmean(tuned_values)
    
    # Print comparison
    if verbose:
        logger.info("")
        logger.info("="*60)
        logger.info("BASE vs TUNED MODELS - AVERAGE METRICS COMPARISON")
        logger.info("="*60)
        
        logger.info(f"\nBASE MODELS (Average across {len(base_metrics)} models):")
        for key, val in avg_base.items():
            logger.info(f"  {key:20s}: {val:.4f}")
        
        logger.info(f"\nTUNED MODELS (Average across {len(tuned_metrics)} models):")
        for key, val in avg_tuned.items():
            logger.info(f"  {key:20s}: {val:.4f}")
        
        # Show improvement
        logger.info("\nIMPROVEMENT (TUNED vs BASE):")
        for key in avg_base.keys():
            if key in avg_tuned:
                diff = avg_tuned[key] - avg_base[key]
                pct_change = (diff / avg_base[key] * 100) if avg_base[key] != 0 else 0
                
                # Determine direction (positive = improvement for all metrics)
                direction = "↑" if diff > 0 else "↓"
                better = "BETTER" if diff > 0 else "WORSE"
                
                logger.info(f"  {key:20s}: {direction} {diff:+.4f} ({pct_change:+.2f}%) [{better}]")
        
        logger.info("="*70)
    
    return {
        'base_metrics': avg_base,
        'tuned_metrics': avg_tuned,
        'n_base_models': len(base_metrics),
        'n_tuned_models': len(tuned_metrics),
    }


def predict_all(models, X_test, X_test_scaled):
    """
    Generate predictions (y_pred and y_proba) for all models
    
    Strategy:
    - LogisticRegression: Use scaled data (X_test_scaled)
    - RandomForest & XGBoost: Use raw data (X_test)
    
    Handles 3 model types: LogReg, RF, XGBoost
    
    Args:
        models: Dictionary of trained models
        X_test: Raw test features
        X_test_scaled: Scaled test features
    
    Returns:
        (y_pred, y_proba): Dictionaries with predictions and probabilities
    """
    y_pred = {}
    y_proba = {}

    for name, model in models.items():
        try:
            # Determine data to use based on model type
            if 'LogisticRegression' in name:
                # Linear models use scaled data
                X = X_test_scaled
            else:
                # Tree-based models (RF, XGBoost) use raw data
                X = X_test
            
            # sklearn models (LogReg, RF, XGBoost)
            y_proba_temp = model.predict_proba(X)[:, 1]
            y_pred_temp = model.predict(X)
            
            y_pred[name] = y_pred_temp
            y_proba[name] = y_proba_temp
        
        except Exception as e:
            logger.warning(f"⚠️  Error generating predictions for {name}: {str(e)}")
            # Return NaN predictions to indicate error
            y_pred[name] = np.full(len(X_test), np.nan)
            y_proba[name] = np.full(len(X_test), np.nan)

    return y_pred, y_proba


def evaluate_model(name, y_true, y_pred, y_proba) -> Dict[str, float]:
    """
    Model evaluation using standard metrics.
    
    Uses threshold=0.5 for all models (default sklearn behavior).
    
    Args:
        name: Model name
        y_true: True labels
        y_pred: Binary predictions at 0.5 threshold
        y_proba: Probability predictions
    
    Returns:
        Dictionary with metrics
    """
    is_base = 'base' in name.lower()
    is_tuned = 'tuned' in name.lower()
    
    logger.info(f"{'='*60}")
    logger.info(f"MODEL: {name}")
    logger.info(f"{'='*60}")
    logger.info(f"Type: {'BASE' if is_base else 'TUNED' if is_tuned else 'UNKNOWN'}")

    # Confusion matrix and classification report
    logger.info("Confusion matrix:")
    cm = confusion_matrix(y_true, y_pred)
    # Format without outer brackets
    logger.info(f" [{cm[0,0]:>5}  {cm[0,1]:>5}]")
    logger.info(f" [{cm[1,0]:>5}  {cm[1,1]:>5}]")
    
    logger.info("Classification report:")
    logger.info(f"\n{classification_report(y_true, y_pred, digits=4)}")
    
    # Metrics
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    
    logger.info(f"AUC-ROC: {auc:.4f}")

    return {
        "model_type": "BASE" if is_base else "TUNED" if is_tuned else "UNKNOWN",
        "auc": float(auc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "threshold": 0.5,
    }

def plot_roc_pr_curves(y_test, proba_dict, all_metrics=None):
    """
    Plot ROC and PR curves for BASE models only with distinct colors
    """
    # Filter BASE models only
    base_models = {k: v for k, v in proba_dict.items() if 'base' in k.lower()}
    
    if not base_models:
        logger.warning("No BASE models found for ROC/PR curves")
        return
    
    # Create evaluation subfolder
    eval_dir = PLOTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Distinct colors for each model
    colors = {
        'LogisticRegression_base': '#2ecc71',  # Green
        'RandomForest_base': '#3498db',         # Blue
        'XGBoost_base': '#e74c3c',              # Red
    }
    
    # ============ ROC CURVES (BASE ONLY) ============
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, proba in base_models.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_score = roc_auc_score(y_test, proba)
        color = colors.get(name, 'gray')
        
        # Clean name for legend
        display_name = name.replace('_base', '').replace('_', ' ')
        ax.plot(fpr, tpr, linewidth=2.5, color=color, 
                label=f'{display_name} (AUC={auc_score:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - BASE Models Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(eval_dir / "roc_curves_base.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============ PR CURVES (BASE ONLY) ============
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate baseline (random classifier)
    baseline = y_test.sum() / len(y_test)
    
    for name, proba in base_models.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        color = colors.get(name, 'gray')
        
        # Clean name for legend
        display_name = name.replace('_base', '').replace('_', ' ')
        ax.plot(recall, precision, linewidth=2.5, color=color, label=f'{display_name}')
    
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, 
               label=f'Random baseline ({baseline:.2%})')
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curves - BASE Models Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(eval_dir / "pr_curves_base.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ ROC/PR curves (BASE) saved to {eval_dir}")


def plot_roc_pr_curves_tuned(y_test, proba_dict):
    """
    Plot ROC and PR curves for TUNED models only with distinct colors
    """
    # Filter TUNED models only
    tuned_models = {k: v for k, v in proba_dict.items() if 'tuned' in k.lower()}
    
    if not tuned_models:
        logger.warning("No TUNED models found for ROC/PR curves")
        return
    
    # Create evaluation subfolder
    eval_dir = PLOTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Distinct colors for each model (darker shades for tuned)
    colors = {
        'LogisticRegression_tuned': '#1e8449',  # Dark Green
        'RandomForest_tuned': '#2874a6',         # Dark Blue
        'XGBoost_tuned': '#c0392b',              # Dark Red
    }
    
    # ============ ROC CURVES (TUNED ONLY) ============
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for name, proba in tuned_models.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc_score = roc_auc_score(y_test, proba)
        color = colors.get(name, 'gray')
        
        # Clean name for legend
        display_name = name.replace('_tuned', '').replace('_', ' ')
        ax.plot(fpr, tpr, linewidth=2.5, color=color, 
                label=f'{display_name} (AUC={auc_score:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - TUNED Models Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(eval_dir / "roc_curves_tuned.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # ============ PR CURVES (TUNED ONLY) ============
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate baseline (random classifier)
    baseline = y_test.sum() / len(y_test)
    
    for name, proba in tuned_models.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        color = colors.get(name, 'gray')
        
        # Clean name for legend
        display_name = name.replace('_tuned', '').replace('_', ' ')
        ax.plot(recall, precision, linewidth=2.5, color=color, label=f'{display_name}')
    
    ax.axhline(y=baseline, color='gray', linestyle='--', linewidth=1.5, 
               label=f'Random baseline ({baseline:.2%})')
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curves - TUNED Models Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(eval_dir / "pr_curves_tuned.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ ROC/PR curves (TUNED) saved to {eval_dir}")


def plot_confusion_matrices(y_test, y_pred_dict, y_proba_dict):
    """
    Plot clean confusion matrices for ALL models (BASE and TUNED)
    Each model gets a different color palette - counts only, no percentages
    """
    import seaborn as sns
    
    # Separate BASE and TUNED models
    base_models = {k: v for k, v in y_pred_dict.items() if 'base' in k.lower()}
    tuned_models = {k: v for k, v in y_pred_dict.items() if 'tuned' in k.lower()}
    
    if not base_models:
        logger.warning("No BASE models found for confusion matrices")
        return
    
    # Create evaluation subfolder
    eval_dir = PLOTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Color palettes for models
    color_palettes = {
        'LogisticRegression_base': sns.light_palette("#27ae60", as_cmap=True),   # Green
        'RandomForest_base': sns.light_palette("#3498db", as_cmap=True),          # Blue
        'XGBoost_base': sns.light_palette("#e74c3c", as_cmap=True),               # Red
        'LogisticRegression_tuned': sns.light_palette("#1e8449", as_cmap=True),  # Dark Green
        'RandomForest_tuned': sns.light_palette("#2874a6", as_cmap=True),         # Dark Blue
        'XGBoost_tuned': sns.light_palette("#c0392b", as_cmap=True),              # Dark Red
    }
    
    # Determine layout: 2 rows if we have tuned models
    n_base = len(base_models)
    n_tuned = len(tuned_models)
    
    if n_tuned > 0:
        # 2 rows: BASE on top, TUNED on bottom
        n_cols = max(n_base, n_tuned)
        fig, axes = plt.subplots(2, n_cols, figsize=(5.5*n_cols, 10))
        
        # Plot BASE models (row 0)
        for idx, (name, y_pred) in enumerate(base_models.items()):
            ax = axes[0, idx] if n_cols > 1 else axes[0]
            display_name = name.replace('_base', '').replace('_', ' ')
            cmap = color_palettes.get(name, sns.light_palette("#34495e", as_cmap=True))
            cm = confusion_matrix(y_test, y_pred)
            labels = np.array([[f'{val:,}' for val in row] for row in cm])
            
            sns.heatmap(cm, annot=labels, fmt='', cmap=cmap, ax=ax,
                        xticklabels=['Pred: Survive', 'Pred: Bankrupt'],
                        yticklabels=['True: Survive', 'True: Bankrupt'],
                        cbar=False, linewidths=3, linecolor='white',
                        annot_kws={'size': 14, 'weight': 'bold'})
            ax.set_title(f'{display_name} (BASE)', fontsize=14, fontweight='bold', pad=12)
            ax.tick_params(axis='both', labelsize=10)
        
        # Hide unused axes in row 0
        for idx in range(n_base, n_cols):
            if n_cols > 1:
                fig.delaxes(axes[0, idx])
        
        # Plot TUNED models (row 1)
        for idx, (name, y_pred) in enumerate(tuned_models.items()):
            ax = axes[1, idx] if n_cols > 1 else axes[1]
            display_name = name.replace('_tuned', '').replace('_', ' ')
            cmap = color_palettes.get(name, sns.light_palette("#34495e", as_cmap=True))
            cm = confusion_matrix(y_test, y_pred)
            labels = np.array([[f'{val:,}' for val in row] for row in cm])
            
            sns.heatmap(cm, annot=labels, fmt='', cmap=cmap, ax=ax,
                        xticklabels=['Pred: Survive', 'Pred: Bankrupt'],
                        yticklabels=['True: Survive', 'True: Bankrupt'],
                        cbar=False, linewidths=3, linecolor='white',
                        annot_kws={'size': 14, 'weight': 'bold'})
            ax.set_title(f'{display_name} (TUNED)', fontsize=14, fontweight='bold', pad=12)
            ax.tick_params(axis='both', labelsize=10)
        
        # Hide unused axes in row 1
        for idx in range(n_tuned, n_cols):
            if n_cols > 1:
                fig.delaxes(axes[1, idx])
        
        plt.suptitle('Confusion Matrices - BASE vs TUNED Models', fontsize=16, fontweight='bold', y=1.02)
    else:
        # Only BASE models - single row
        n_models = len(base_models)
        fig, axes = plt.subplots(1, n_models, figsize=(5.5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (name, y_pred) in enumerate(base_models.items()):
            ax = axes[idx]
            display_name = name.replace('_base', '').replace('_', ' ')
            cmap = color_palettes.get(name, sns.light_palette("#34495e", as_cmap=True))
            cm = confusion_matrix(y_test, y_pred)
            labels = np.array([[f'{val:,}' for val in row] for row in cm])
            
            sns.heatmap(cm, annot=labels, fmt='', cmap=cmap, ax=ax,
                        xticklabels=['Pred: Survive', 'Pred: Bankrupt'],
                        yticklabels=['True: Survive', 'True: Bankrupt'],
                        cbar=False, linewidths=3, linecolor='white',
                        annot_kws={'size': 14, 'weight': 'bold'})
            ax.set_title(f'{display_name}', fontsize=14, fontweight='bold', pad=12)
            ax.tick_params(axis='both', labelsize=10)
        
        plt.suptitle('Confusion Matrices - BASE Models', fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(eval_dir / "confusion_matrices.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Confusion matrices saved to {eval_dir}")


def plot_model_comparison(y_test, y_pred_dict, y_proba_dict):
    """
    Plot BASE models performance comparison bar chart
    X-axis: Performance metrics, Colors: by Model
    """
    import seaborn as sns
    
    # Filter BASE models only
    base_models_pred = {k: v for k, v in y_pred_dict.items() if 'base' in k.lower()}
    base_models_proba = {k: v for k, v in y_proba_dict.items() if 'base' in k.lower()}
    
    if not base_models_pred:
        logger.warning("No BASE models found for comparison plot")
        return
    
    # Create evaluation subfolder
    eval_dir = PLOTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics for each model
    metrics_data = []
    for name in base_models_pred.keys():
        y_pred = base_models_pred[name]
        y_proba = base_models_proba[name]
        
        display_name = name.replace('_base', '').replace('_', ' ')
        
        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        metrics_data.append({'Model': display_name, 'Metric': 'AUC', 'Value': auc})
        metrics_data.append({'Model': display_name, 'Metric': 'Precision', 'Value': precision})
        metrics_data.append({'Model': display_name, 'Metric': 'Recall', 'Value': recall})
        metrics_data.append({'Model': display_name, 'Metric': 'F1-Score', 'Value': f1})
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create grouped bar chart - X axis = metrics, colors = models
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color palette for models
    model_colors = {
        'LogisticRegression': '#27ae60',  # Green
        'RandomForest': '#3498db',         # Blue
        'XGBoost': '#e74c3c',              # Red
    }
    
    # Get unique models and metrics
    models = df_metrics['Model'].unique()
    metrics = ['AUC', 'Precision', 'Recall', 'F1-Score']
    
    # Plot grouped bars - metrics on X-axis, models as different colors
    x = np.arange(len(metrics))
    width = 0.25
    n_models = len(models)
    
    for i, model in enumerate(models):
        values = [df_metrics[(df_metrics['Model'] == model) & (df_metrics['Metric'] == m)]['Value'].values[0] 
                  for m in metrics]
        color = model_colors.get(model, '#34495e')
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=color, alpha=0.85, edgecolor='white', linewidth=1)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1%}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Performance Metric', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('BASE Models - Performance Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, title='Model', title_fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add horizontal line at 50%
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(eval_dir / "model_comparison_base.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Model comparison plot (BASE) saved to {eval_dir}")


def plot_model_comparison_tuned(y_test, y_pred_dict, y_proba_dict):
    """
    Plot TUNED models performance comparison bar chart
    X-axis: Performance metrics, Colors: by Model
    """
    import seaborn as sns
    
    # Filter TUNED models only
    tuned_models_pred = {k: v for k, v in y_pred_dict.items() if 'tuned' in k.lower()}
    tuned_models_proba = {k: v for k, v in y_proba_dict.items() if 'tuned' in k.lower()}
    
    if not tuned_models_pred:
        logger.warning("No TUNED models found for comparison plot")
        return
    
    # Create evaluation subfolder
    eval_dir = PLOTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute metrics for each model
    metrics_data = []
    for name in tuned_models_pred.keys():
        y_pred = tuned_models_pred[name]
        y_proba = tuned_models_proba[name]
        
        display_name = name.replace('_tuned', '').replace('_', ' ')
        
        auc = roc_auc_score(y_test, y_proba)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        metrics_data.append({'Model': display_name, 'Metric': 'AUC', 'Value': auc})
        metrics_data.append({'Model': display_name, 'Metric': 'Precision', 'Value': precision})
        metrics_data.append({'Model': display_name, 'Metric': 'Recall', 'Value': recall})
        metrics_data.append({'Model': display_name, 'Metric': 'F1-Score', 'Value': f1})
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Create grouped bar chart - X axis = metrics, colors = models
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Color palette for models (darker shades for tuned)
    model_colors = {
        'LogisticRegression': '#1e8449',  # Dark Green
        'RandomForest': '#2874a6',         # Dark Blue
        'XGBoost': '#c0392b',              # Dark Red
    }
    
    # Get unique models and metrics
    models = df_metrics['Model'].unique()
    metrics = ['AUC', 'Precision', 'Recall', 'F1-Score']
    
    # Plot grouped bars - metrics on X-axis, models as different colors
    x = np.arange(len(metrics))
    width = 0.25
    n_models = len(models)
    
    for i, model in enumerate(models):
        values = [df_metrics[(df_metrics['Model'] == model) & (df_metrics['Metric'] == m)]['Value'].values[0] 
                  for m in metrics]
        color = model_colors.get(model, '#34495e')
        offset = (i - n_models/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=color, alpha=0.85, edgecolor='white', linewidth=1)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1%}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize plot
    ax.set_xlabel('Performance Metric', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score', fontsize=13, fontweight='bold')
    ax.set_title('TUNED Models - Performance Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, title='Model', title_fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add horizontal line at 50%
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(eval_dir / "model_comparison_tuned.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  ✓ Model comparison plot (TUNED) saved to {eval_dir}")


# -------------------------------------------------------------------
#  YEARLY AUC STABILITY PLOT (All Models in One Chart)
# -------------------------------------------------------------------
def plot_yearly_auc(yearly_df, save_path=None):
    """
    Plot yearly AUC for all models in one chart to show stability over time.
    
    WHY THIS PLOT?
    - Shows model performance stability across years
    - Identifies if model degrades over time
    - Compares BASE vs TUNED consistency
    
    Args:
        yearly_df: DataFrame from yearly_backtest with columns: year, n_obs, n_fail, {model}_auc
        save_path: Directory to save plot (default: PLOTS_DIR / 'evaluation')
    """
    if yearly_df is None or yearly_df.empty:
        logger.warning("No yearly data available for plotting")
        return
    
    # Get AUC columns
    auc_cols = [col for col in yearly_df.columns if col.endswith('_auc')]
    if not auc_cols:
        logger.warning("No AUC columns found in yearly_df")
        return
    
    # Setup plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color scheme - BASE (solid) vs TUNED (dashed)
    base_colors = {
        'LogisticRegression_base': '#3498db',
        'RandomForest_base': '#2ecc71', 
        'XGBoost_base': '#e74c3c'
    }
    tuned_colors = {
        'LogisticRegression_tuned': '#2980b9',
        'RandomForest_tuned': '#27ae60',
        'XGBoost_tuned': '#c0392b'
    }
    
    years = yearly_df['year'].values
    
    # Plot each model
    for col in auc_cols:
        model_name = col.replace('_auc', '')
        auc_values = yearly_df[col].values
        
        # Determine style
        if 'tuned' in model_name.lower():
            color = tuned_colors.get(model_name, '#9b59b6')
            linestyle = '--'
            marker = 's'
            label = f"{model_name} (TUNED)"
        else:
            color = base_colors.get(model_name, '#34495e')
            linestyle = '-'
            marker = 'o'
            label = f"{model_name} (BASE)"
        
        # Plot line
        ax.plot(years, auc_values, 
                color=color, linestyle=linestyle, marker=marker,
                linewidth=2.5, markersize=8, label=label, alpha=0.85)
        
        # Add value labels
        for x, y in zip(years, auc_values):
            if not np.isnan(y):
                ax.annotate(f'{y:.3f}', (x, y), 
                           textcoords="offset points", xytext=(0, 8),
                           ha='center', fontsize=8, alpha=0.8)
    
    # Add reference line at 0.5 (random classifier)
    ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, label='Random (0.5)')
    
    # Add sample size annotations at bottom
    for i, (year, n_obs, n_fail) in enumerate(zip(years, yearly_df['n_obs'], yearly_df['n_fail'])):
        ax.annotate(f'n={int(n_obs)}\nfail={int(n_fail)}', 
                   (year, ax.get_ylim()[0] + 0.02),
                   ha='center', fontsize=7, alpha=0.6)
    
    # Customize plot
    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('AUC Score', fontsize=13, fontweight='bold')
    ax.set_title('Model Stability Over Time - Yearly AUC Performance\n(Higher = Better, Stable = Consistent)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(years)
    ax.set_xticklabels([int(y) for y in years], rotation=45)
    ax.set_ylim(0.45, 1.02)
    ax.legend(loc='lower right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add shaded region for "good" AUC (>0.7)
    ax.axhspan(0.7, 1.0, alpha=0.1, color='green', label='_Good AUC')
    ax.axhspan(0.5, 0.7, alpha=0.1, color='orange', label='_Moderate AUC')
    
    plt.tight_layout()
    
    # Save plot
    if save_path is None:
        save_path = PLOTS_DIR / 'evaluation'
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(save_path / "yearly_auc_stability.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  Yearly AUC stability plot saved to {save_path / 'yearly_auc_stability.png'}")
    
    # Print summary statistics
    logger.info("")
    logger.info("=" * 60)
    logger.info("YEARLY AUC STABILITY SUMMARY")
    logger.info("=" * 60)
    for col in auc_cols:
        model_name = col.replace('_auc', '')
        values = yearly_df[col].dropna()
        if len(values) > 0:
            logger.info(f"{model_name}:")
            logger.info(f"  Mean AUC: {values.mean():.4f}")
            logger.info(f"  Std AUC:  {values.std():.4f} ({'Stable' if values.std() < 0.05 else 'Variable'})")
            logger.info(f"  Min:      {values.min():.4f} ({int(yearly_df.loc[values.idxmin(), 'year'])})")
            logger.info(f"  Max:      {values.max():.4f} ({int(yearly_df.loc[values.idxmax(), 'year'])})")


def yearly_backtest(df_test, models, scaler, selected_features=None):
    """
    Compute AUC per year on the test period.
    
    Args:
        df_test: Test DataFrame
        models: Dictionary of trained models
        scaler: Fitted scaler
        selected_features: List of feature names (from RFE)
    
    Returns:
        DataFrame with yearly AUC for each model
    """
    results = []
    
    # Use selected features if provided, otherwise all features
    feature_cols = selected_features if selected_features else FEATURE_COLS
    
    for year in sorted(df_test[YEAR_COL].unique()):
        df_y = df_test[df_test[YEAR_COL] == year]
        if df_y[TARGET_COL].nunique() < 2:
            continue

        X_y = df_y[feature_cols]
        y_y = df_y[TARGET_COL].astype(int)
        X_y_scaled = scaler.transform(X_y)

        row = {"year": int(year), "n_obs": int(len(df_y)), "n_fail": int(y_y.sum())}

        for name, model in models.items():
            try:
                Xx = X_y_scaled if 'LogisticRegression' in name else X_y
                proba = model.predict_proba(Xx)[:, 1]
                auc = roc_auc_score(y_y, proba)
                row[f"{name}_auc"] = float(auc)
            except Exception as e:
                logger.warning(f"Error in yearly_backtest for {name}: {e}")
                row[f"{name}_auc"] = np.nan

        results.append(row)

    res_df = pd.DataFrame(results)
    res_df.to_csv(DATA_DIR / "yearly_performance.csv", index=False)
    return res_df