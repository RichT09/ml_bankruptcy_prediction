# 04_evaluation.py

"""
Module 04: Model Evaluation & Comparison
Comprehensive metrics, visualizations, base vs tuned comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
import pickle
import logging
from pathlib import Path
import sys
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Setup
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import config
except ImportError:
    import config

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = config.evaluation.figure_dpi
plt.rcParams['font.size'] = 10


# ============================================================================
# PREDICTIONS
# ============================================================================

def predict_all_models(
    models: Dict, 
    X_test, 
    X_test_scaled,
    verbose: bool = True
) -> Tuple[Dict, Dict]:
    """
    Generate predictions for all models
    
    Args:
        models: Dictionary of trained models
        X_test: Raw test features
        X_test_scaled: Scaled test features
        verbose: Print progress
    
    Returns:
        predictions dict, probabilities dict
    """
    y_pred = {}
    y_proba = {}
    
    if verbose:
        logger.info("\n--- Generating Predictions ---")
    
    for name, model in models.items():
        # Use scaled data for linear models
        if 'LogisticRegression' in name or 'SVM' in name:
            X = X_test_scaled
        else:
            X = X_test
        
        proba = model.predict_proba(X)[:, 1]
        pred = model.predict(X)
        
        y_pred[name] = pred
        y_proba[name] = proba
        
        if verbose:
            logger.info(f"  ✓ {name}")
    
    return y_pred, y_proba


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_metrics(y_true, y_pred, y_proba, model_name: str) -> Dict:
    """
    Compute comprehensive metrics for a single model
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        model_name: Name of the model
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'model': model_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc': roc_auc_score(y_true, y_proba),
        'avg_precision': average_precision_score(y_true, y_proba),
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['tn'] = int(tn)
        metrics['fp'] = int(fp)
        metrics['fn'] = int(fn)
        metrics['tp'] = int(tp)
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


def evaluate_all_models(
    y_test,
    predictions_base: Dict,
    probabilities_base: Dict,
    predictions_tuned: Dict,
    probabilities_tuned: Dict,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluate all models and create comparison DataFrame
    
    Args:
        y_test: True test labels
        predictions_base: Base model predictions
        probabilities_base: Base model probabilities
        predictions_tuned: Tuned model predictions
        probabilities_tuned: Tuned model probabilities
        verbose: Print results
    
    Returns:
        DataFrame with all metrics
    """
    if verbose:
        logger.info("\n--- Computing Metrics ---")
    
    results = []
    
    # Evaluate base models
    for name, pred in predictions_base.items():
        proba = probabilities_base[name]
        metrics = compute_metrics(y_test, pred, proba, name)
        results.append(metrics)
    
    # Evaluate tuned models
    for name, pred in predictions_tuned.items():
        proba = probabilities_tuned[name]
        metrics = compute_metrics(y_test, pred, proba, name)
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    
    if verbose:
        logger.info("\n✓ Metrics computed for all models")
        logger.info(f"\n{df_results[['model', 'accuracy', 'precision', 'recall', 'f1', 'auc']].to_string()}")
    
    return df_results


# ============================================================================
# VISUALIZATIONS - COMPARISON PLOTS
# ============================================================================

def plot_metric_comparison(
    df_results: pd.DataFrame,
    metric: str,
    title: str,
    output_path: Path,
    figsize=(12, 6)
):
    """
    Create bar plot comparing metric across all models
    
    Args:
        df_results: DataFrame with metrics
        metric: Metric column name
        title: Plot title
        output_path: Save path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate base and tuned
    df_base = df_results[df_results['model'].str.contains('_base')]
    df_tuned = df_results[df_results['model'].str.contains('_tuned')]
    
    # Extract model family
    df_base['family'] = df_base['model'].str.replace('_base', '')
    df_tuned['family'] = df_tuned['model'].str.replace('_tuned', '')
    
    x = np.arange(len(df_base))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, df_base[metric], width, label='Base', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x + width/2, df_tuned[metric], width, label='Tuned', alpha=0.8, color='coral')
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df_base['family'], rotation=15, ha='right')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ {output_path.name}")


def create_all_comparison_plots(df_results: pd.DataFrame, plots_dir: Path):
    """
    Create all metric comparison plots
    
    Args:
        df_results: DataFrame with all metrics
        plots_dir: Directory to save plots
    """
    logger.info("\n--- Creating Comparison Plots ---")
    
    comparisons = [
        ('precision', 'Precision Comparison: Base vs Tuned'),
        ('recall', 'Recall Comparison: Base vs Tuned'),
        ('f1', 'F1-Score Comparison: Base vs Tuned'),
        ('auc', 'AUC Comparison: Base vs Tuned'),
        ('accuracy', 'Accuracy Comparison: Base vs Tuned'),
    ]
    
    for metric, title in comparisons:
        output_path = plots_dir / f"{metric}_comparison.png"
        plot_metric_comparison(df_results, metric, title, output_path)


# ============================================================================
# ROC CURVES
# ============================================================================

def plot_roc_curves_comparison(
    y_test,
    probabilities_base: Dict,
    probabilities_tuned: Dict,
    output_path: Path
):
    """
    Plot ROC curves with base vs tuned comparison
    
    Args:
        y_test: True labels
        probabilities_base: Base model probabilities
        probabilities_tuned: Tuned model probabilities
        output_path: Save path
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Base models
    ax = axes[0]
    for name, proba in probabilities_base.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        model_family = name.replace('_base', '')
        ax.plot(fpr, tpr, lw=2, label=f'{model_family} (AUC={roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - BASE Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Tuned models
    ax = axes[1]
    for name, proba in probabilities_tuned.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        model_family = name.replace('_tuned', '')
        ax.plot(fpr, tpr, lw=2, label=f'{model_family} (AUC={roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - TUNED Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ {output_path.name}")


# ============================================================================
# PRECISION-RECALL CURVES
# ============================================================================

def plot_pr_curves_comparison(
    y_test,
    probabilities_base: Dict,
    probabilities_tuned: Dict,
    output_path: Path
):
    """
    Plot Precision-Recall curves with base vs tuned comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Base models
    ax = axes[0]
    for name, proba in probabilities_base.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        avg_prec = average_precision_score(y_test, proba)
        model_family = name.replace('_base', '')
        ax.plot(recall, precision, lw=2, label=f'{model_family} (AP={avg_prec:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curves - BASE Models', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    # Tuned models
    ax = axes[1]
    for name, proba in probabilities_tuned.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        avg_prec = average_precision_score(y_test, proba)
        model_family = name.replace('_tuned', '')
        ax.plot(recall, precision, lw=2, label=f'{model_family} (AP={avg_prec:.3f})')
    
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curves - TUNED Models', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ {output_path.name}")


# ============================================================================
# CONFUSION MATRICES
# ============================================================================

def plot_confusion_matrices(
    y_test,
    predictions_base: Dict,
    predictions_tuned: Dict,
    output_path: Path
):
    """
    Plot confusion matrices grid (2 rows: base, tuned)
    """
    n_models = len(predictions_base)
    fig, axes = plt.subplots(2, n_models, figsize=(4*n_models, 8))
    
    # Base models
    for idx, (name, pred) in enumerate(predictions_base.items()):
        ax = axes[0, idx] if n_models > 1 else axes[0]
        cm = confusion_matrix(y_test, pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        model_family = name.replace('_base', '')
        ax.set_title(f'{model_family}\n(BASE)', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
    
    # Tuned models
    for idx, (name, pred) in enumerate(predictions_tuned.items()):
        ax = axes[1, idx] if n_models > 1 else axes[1]
        cm = confusion_matrix(y_test, pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax, cbar=False)
        model_family = name.replace('_tuned', '')
        ax.set_title(f'{model_family}\n(TUNED)', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ {output_path.name}")


# ============================================================================
# BASE VS TUNED - INDIVIDUAL MODEL COMPARISON
# ============================================================================

def plot_base_vs_tuned_individual(df_results: pd.DataFrame, output_path: Path):
    """
    Create individual comparison plots for each model family
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # model_families = ['LogisticRegression', 'RandomForest', 'XGBoost'] # 'SVM'
    # Détection automatique des modèles présents
    model_families = []
    for model_name in df_results['model'].unique():
        if '_base' in model_name:
            family = model_name.replace('_base', '')
            if family not in model_families:
                model_families.append(family)
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    for idx, family in enumerate(model_families):
        ax = axes[idx]
        
        base_row = df_results[df_results['model'] == f'{family}_base'].iloc[0]
        tuned_row = df_results[df_results['model'] == f'{family}_tuned'].iloc[0]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        base_values = [base_row[m] for m in metrics]
        tuned_values = [tuned_row[m] for m in metrics]
        
        bars1 = ax.bar(x - width/2, base_values, width, label='Base', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, tuned_values, width, label='Tuned', alpha=0.8, color='coral')
        
        ax.set_ylabel('Score', fontsize=11, fontweight='bold')
        ax.set_title(f'{family}: Base vs Tuned', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in metrics], rotation=15, ha='right', fontsize=9)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ {output_path.name}")


# ============================================================================
# YEARLY BACKTEST
# ============================================================================

def yearly_backtest(
    df_test: pd.DataFrame,
    models_base: Dict,
    models_tuned: Dict,
    scaler,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Compute AUC per year for all models
    """
    year_col = config.data.year_col
    target_col = config.data.target_col
    feature_cols = [f for f in config.data.feature_cols if f in df_test.columns]
    
    results = []
    
    years = sorted(df_test[year_col].unique())
    
    if verbose:
        logger.info(f"\n--- Yearly Backtest ---")
        logger.info(f"Years: {years}")
    
    for year in years:
        df_year = df_test[df_test[year_col] == year]
        
        if df_year[target_col].nunique() < 2:
            continue
        
        X_year = df_year[feature_cols]
        y_year = df_year[target_col].astype(int)
        X_year_scaled = scaler.transform(X_year)
        
        row = {
            'year': int(year),
            'n_obs': int(len(df_year)),
            'n_failures': int(y_year.sum())
        }
        
        # Base models
        for name, model in models_base.items():
            X = X_year_scaled if 'LogisticRegression' in name or 'SVM' in name else X_year
            proba = model.predict_proba(X)[:, 1]
            auc_score = roc_auc_score(y_year, proba)
            row[f'{name}_auc'] = float(auc_score)
        
        # Tuned models
        for name, model in models_tuned.items():
            X = X_year_scaled if 'LogisticRegression' in name or 'SVM' in name else X_year
            proba = model.predict_proba(X)[:, 1]
            auc_score = roc_auc_score(y_year, proba)
            row[f'{name}_auc'] = float(auc_score)
        
        results.append(row)
    
    df_yearly = pd.DataFrame(results)
    
    # Save
    out_path = config.metrics_dir / "yearly_performance.csv"
    df_yearly.to_csv(out_path, index=False)
    
    if verbose:
        logger.info(f"✓ Yearly backtest complete")
        logger.info(f"  Saved to: {out_path}")
    
    return df_yearly


def plot_yearly_performance(df_yearly: pd.DataFrame, output_path: Path):
    """
    Plot yearly AUC evolution for all models
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    auc_cols_base = [c for c in df_yearly.columns if '_base_auc' in c]
    auc_cols_tuned = [c for c in df_yearly.columns if '_tuned_auc' in c]
    
    # Base models
    ax = axes[0]
    for col in auc_cols_base:
        model_name = col.replace('_base_auc', '')
        ax.plot(df_yearly['year'], df_yearly[col], marker='o', linewidth=2, label=model_name)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title('Yearly AUC Evolution - BASE Models', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    # Tuned models
    ax = axes[1]
    for col in auc_cols_tuned:
        model_name = col.replace('_tuned_auc', '')
        ax.plot(df_yearly['year'], df_yearly[col], marker='o', linewidth=2, label=model_name)
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC', fontsize=12, fontweight='bold')
    ax.set_title('Yearly AUC Evolution - TUNED Models', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ {output_path.name}")


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_evaluation_results(df_results: pd.DataFrame, df_yearly: pd.DataFrame):
    """
    Save all evaluation results
    """
    # Main metrics
    metrics_path = config.metrics_dir / "model_metrics.csv"
    df_results.to_csv(metrics_path, index=False)
    logger.info(f"\n✓ Saved metrics: {metrics_path}")
    
    # Yearly performance
    yearly_path = config.metrics_dir / "yearly_performance.csv"
    df_yearly.to_csv(yearly_path, index=False)
    logger.info(f"✓ Saved yearly performance: {yearly_path}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_evaluation_pipeline(
    models_base: Dict,
    models_tuned: Dict,
    data: Dict,
    verbose: bool = True
) -> Dict:
    """
    Complete evaluation pipeline
    
    Args:
        models_base: Base models
        models_tuned: Tuned models
        data: Test data dictionary
        verbose: Print progress
    
    Returns:
        Dictionary with results
    """
    logger.info("\n" + "="*70)
    logger.info("MODULE 04: EVALUATION & COMPARISON")
    logger.info("="*70)
    
    # Extract data
    X_test = data['X_test']
    X_test_scaled = data['X_test_scaled']
    y_test = data['y_test']
    df_test = data['df_test']
    scaler = data['scaler']
    
    # 1. Predictions
    pred_base, proba_base = predict_all_models(models_base, X_test, X_test_scaled, verbose)
    pred_tuned, proba_tuned = predict_all_models(models_tuned, X_test, X_test_scaled, verbose)
    
    # 2. Metrics
    df_results = evaluate_all_models(
        y_test, pred_base, proba_base, pred_tuned, proba_tuned, verbose
    )
    
    # 3. Visualizations
    logger.info("\n--- Generating Visualizations ---")
    
    create_all_comparison_plots(df_results, config.plots_dir)
    
    plot_roc_curves_comparison(
        y_test, proba_base, proba_tuned,
        config.plots_dir / "roc_curves_comparison.png"
    )
    
    plot_pr_curves_comparison(
        y_test, proba_base, proba_tuned,
        config.plots_dir / "pr_curves_comparison.png"
    )
    
    plot_confusion_matrices(
        y_test, pred_base, pred_tuned,
        config.plots_dir / "confusion_matrices.png"
    )
    
    plot_base_vs_tuned_individual(
        df_results,
        config.plots_dir / "base_vs_tuned_individual.png"
    )
    
    # 4. Yearly backtest
    df_yearly = yearly_backtest(df_test, models_base, models_tuned, scaler, verbose)
    
    if not df_yearly.empty:
        plot_yearly_performance(df_yearly, config.plots_dir / "yearly_performance.png")
    
    # 5. Save results
    save_evaluation_results(df_results, df_yearly)
    
    logger.info("\n" + "="*70)
    logger.info("✅ MODULE 04 COMPLETED")
    logger.info(f"  Metrics saved: {len(df_results)} models")
    logger.info(f"  Plots created: {len(list(config.plots_dir.glob('*.png')))}")
    logger.info("="*70)
    
    return {
        'df_results': df_results,
        'df_yearly': df_yearly,
        'predictions_base': pred_base,
        'probabilities_base': proba_base,
        'predictions_tuned': pred_tuned,
        'probabilities_tuned': proba_tuned,
    }


if __name__ == "__main__":
    import pickle
    
    # Load models
    with open(config.models_dir / "models_base.pkl", 'rb') as f:
        models_base = pickle.load(f)
    with open(config.models_dir / "models_tuned.pkl", 'rb') as f:
        models_tuned = pickle.load(f)
    
    # Load data
    with open(config.models_dir / "preprocessed_data.pkl", 'rb') as f:
        data = pickle.load(f)
    
    # Run evaluation
    results = run_evaluation_pipeline(models_base, models_tuned, data, verbose=True)
    
    print("\n🎯 Evaluation complete!")
    print(f"  Best model: {results['df_results'].loc[results['df_results']['auc'].idxmax(), 'model']}")
