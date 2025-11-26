#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 05: EDA & Model Interpretation
Exploratory Data Analysis + SHAP analysis

Author: Master Finance Student
HEC Lausanne - Fall 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import logging
from pathlib import Path
import sys
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

# Plot settings
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = config.evaluation.figure_dpi


# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

def plot_target_distribution(df: pd.DataFrame, output_path: Path):
    """
    Plot target variable distribution (BAR CHART SEULEMENT)
    """
    target_col = config.data.target_col
    
    # Une seule figure avec bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count plot
    counts = df[target_col].value_counts()
    colors = ['green', 'red']
    bars = ax.bar(['Survived', 'Failed'], counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax.set_xlabel('Status', fontsize=13, fontweight='bold')
    ax.set_title('Target Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # Ajouter les valeurs au-dessus des barres
    for i, (bar, v) in enumerate(zip(bars, counts.values)):
        percentage = v / counts.sum() * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2, 
            bar.get_height(),
            f'{v:,}\n({percentage:.2f}%)',
            ha='center', 
            va='bottom', 
            fontsize=12, 
            fontweight='bold'
        )
    
    # Améliorer la grille
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Format des nombres sur l'axe Y
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ {output_path.name}")


def plot_temporal_distribution(df: pd.DataFrame, output_path: Path):
    """
    Plot failures over time
    """
    year_col = config.data.year_col
    target_col = config.data.target_col
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Yearly counts
    ax = axes[0]
    yearly = df.groupby([year_col, target_col]).size().unstack(fill_value=0)
    yearly.plot(kind='bar', stacked=False, ax=ax, color=['green', 'red'], alpha=0.7)
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Observations per Year', fontsize=14, fontweight='bold')
    ax.legend(['Survived', 'Failed'], fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Failure rate over time
    ax = axes[1]
    failure_rate = df.groupby(year_col)[target_col].mean() * 100
    ax.plot(failure_rate.index, failure_rate.values, marker='o', linewidth=2, color='darkred')
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Failure Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Annual Failure Rate Evolution', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axhline(y=failure_rate.mean(), color='blue', linestyle='--', label=f'Mean: {failure_rate.mean():.2f}%')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ {output_path.name}")


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path):
    """
    Plot correlation heatmap - Version triangulaire
    """
    feature_cols = [f for f in config.data.feature_cols if f in df.columns][:20]
    
    corr = df[feature_cols].corr()
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(
        corr,
        mask=mask,  
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        annot=True,
        fmt='.1f',
        annot_kws={'size': 8},
        vmin=-1,
        vmax=1,
        ax=ax
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title('Feature Correlation Heatmap (Top 20)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ {output_path.name}")


# def plot_feature_distributions_by_target(df: pd.DataFrame, output_path: Path, n_features: int = 12):
#     """
#     Plot distribution of top features by target class
#     """
#     target_col = config.data.target_col
#     feature_cols = [f for f in config.data.feature_cols if f in df.columns][:n_features]
    
#     n_cols = 3
#     n_rows = (n_features + n_cols - 1) // n_cols
    
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
#     axes = axes.flatten() if n_features > 1 else [axes]
    
#     for idx, feat in enumerate(feature_cols):
#         ax = axes[idx]
        
#         df_survived = df[df[target_col] == 0][feat].dropna()
#         df_failed = df[df[target_col] == 1][feat].dropna()
        
#         ax.hist(df_survived, bins=30, alpha=0.6, label='Survived', color='green', density=True)
#         ax.hist(df_failed, bins=30, alpha=0.6, label='Failed', color='red', density=True)
        
#         ax.set_xlabel(feat, fontsize=10)
#         ax.set_ylabel('Density', fontsize=10)
#         ax.set_title(f'{feat}', fontsize=11, fontweight='bold')
#         ax.legend(fontsize=9)
#         ax.grid(alpha=0.3)
    
#     # Hide extra subplots
#     for idx in range(n_features, len(axes)):
#         fig.delaxes(axes[idx])
#     
#     plt.suptitle('Feature Distributions by Target Class', fontsize=14, fontweight='bold', y=1.00)
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=150, bbox_inches='tight')
#     plt.close()
#     logger.info(f"  ✓ {output_path.name}")


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def plot_feature_importance_rf(model, feature_names: list, output_path: Path, top_n: int = 20):
    """
    Plot Random Forest feature importance
    """
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    importances.head(top_n).plot(kind='barh', ax=ax, color='steelblue', alpha=0.8)
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Random Forest - Top {top_n} Features', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ {output_path.name}")


def plot_feature_importance_xgb(model, feature_names: list, output_path: Path, top_n: int = 20):
    """
    Plot XGBoost feature importance
    """
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    importances.head(top_n).plot(kind='barh', ax=ax, color='darkgreen', alpha=0.8)
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'XGBoost - Top {top_n} Features', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ {output_path.name}")


def plot_logistic_regression_coef(model, feature_names: list, output_path: Path, top_n: int = 20):
    """
    Plot Logistic Regression coefficients
    """
    coef = pd.Series(model.coef_[0], index=feature_names).sort_values()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot top positive and negative
    top_positive = coef.tail(top_n//2)
    top_negative = coef.head(top_n//2)
    top_features = pd.concat([top_negative, top_positive])
    
    colors = ['red' if x < 0 else 'green' for x in top_features.values]
    top_features.plot(kind='barh', ax=ax, color=colors, alpha=0.7)
    
    ax.set_xlabel('Coefficient', fontsize=12, fontweight='bold')
    ax.set_title(f'Logistic Regression - Top {top_n} Coefficients', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ {output_path.name}")


# ============================================================================
# SHAP ANALYSIS
# ============================================================================

def shap_analysis_tree_model(
    model, 
    X_sample: pd.DataFrame, 
    model_name: str,
    output_dir: Path,
    verbose: bool = True
):
    """
    SHAP analysis for tree-based models (RF, XGBoost)
    """
    if verbose:
        logger.info(f"\n  → SHAP analysis: {model_name}")
    
    try:
        # Pour XGBoost, utiliser un workaround
        if 'XGBoost' in model_name:
            # Méthode alternative : Kernel explainer (plus lent mais fonctionne)
            import warnings
            warnings.filterwarnings('ignore')
            
            # Utiliser un petit échantillon pour Kernel (sinon trop lent)
            X_sample_small = X_sample.sample(min(500, len(X_sample)), random_state=42)
            explainer = shap.KernelExplainer(
                model.predict_proba, 
                X_sample_small,
                link="logit"
            )
            shap_values = explainer.shap_values(X_sample_small)
            
            # Si binary classification, prendre classe positive
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            
            X_for_plot = X_sample_small
            
        else:
            # RandomForest : méthode normale
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]
            
            X_for_plot = X_sample
        
        # 1. Summary plot (beeswarm)
        plt.figure()
        shap.summary_plot(shap_values, X_for_plot, show=False, max_display=20)
        plt.title(f'{model_name} - SHAP Summary', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_{model_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Bar plot (mean absolute SHAP)
        plt.figure()
        shap.summary_plot(shap_values, X_for_plot, plot_type="bar", show=False, max_display=20)
        plt.title(f'{model_name} - Feature Importance (SHAP)', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_importance_{model_name}.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            logger.info(f"    ✓ SHAP plots saved")
            
    except Exception as e:
        if verbose:
            logger.warning(f"    ⚠️  SHAP analysis failed: {str(e)}")
            logger.info(f"    → Using feature importance instead")
        
        # Fallback : utiliser feature importance
        try:
            importances = pd.Series(
                model.feature_importances_, 
                index=X_sample.columns
            ).sort_values(ascending=False)
            
            plt.figure(figsize=(10, 8))
            importances.head(20).plot(kind='barh')
            plt.title(f'{model_name} - Feature Importance (Fallback)', fontsize=14, fontweight='bold')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig(output_dir / f"importance_{model_name.lower()}_fallback.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            if verbose:
                logger.info(f"    ✓ Fallback plot saved")
        except:
            pass


def shap_analysis_linear_model(
    model,
    X_sample: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    verbose: bool = True
):
    """
    SHAP analysis for linear models (LogReg)
    """
    if verbose:
        logger.info(f"\n  → SHAP analysis: {model_name}")
    
    # Create explainer
    explainer = shap.LinearExplainer(model, X_sample)
    shap_values = explainer.shap_values(X_sample)
    
    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
    plt.title(f'{model_name} - SHAP Summary', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_summary_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Bar plot
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False, max_display=20)
    plt.title(f'{model_name} - Feature Importance (SHAP)', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_importance_{model_name}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    if verbose:
        logger.info(f"    ✓ SHAP plots saved")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_eda_pipeline(df: pd.DataFrame, plots_dir: Path, verbose: bool = True):
    """
    Complete EDA pipeline
    """
    if verbose:
        logger.info("\n--- Exploratory Data Analysis ---")
    
    plot_target_distribution(df, plots_dir / "eda_target_distribution.png")
    plot_temporal_distribution(df, plots_dir / "eda_temporal_distribution.png")
    plot_correlation_heatmap(df, plots_dir / "eda_correlation_heatmap.png")
    # plot_feature_distributions_by_target(df, plots_dir / "eda_feature_distributions.png")
    
    if verbose:
        logger.info("✓ EDA plots complete")


def run_interpretation_pipeline(
    models_tuned: dict,
    X_sample: pd.DataFrame,
    plots_dir: Path,
    verbose: bool = True
):
    """
    Complete model interpretation pipeline
    """
    if verbose:
        logger.info("\n--- Model Interpretation ---")
    
    feature_names = list(X_sample.columns)
    
    # Feature importance
    if 'RandomForest_tuned' in models_tuned:
        plot_feature_importance_rf(
            models_tuned['RandomForest_tuned'],
            feature_names,
            plots_dir / "importance_rf.png"
        )
    
    if 'XGBoost_tuned' in models_tuned:
        plot_feature_importance_xgb(
            models_tuned['XGBoost_tuned'],
            feature_names,
            plots_dir / "importance_xgb.png"
        )
    
    if 'LogisticRegression_tuned' in models_tuned:
        plot_logistic_regression_coef(
            models_tuned['LogisticRegression_tuned'],
            feature_names,
            plots_dir / "importance_logreg.png"
        )
    
    # SHAP analysis
    if config.evaluation.enable_shap and verbose:
        logger.info("\n--- SHAP Analysis ---")
        
        if 'RandomForest_tuned' in models_tuned:
            shap_analysis_tree_model(
                models_tuned['RandomForest_tuned'],
                X_sample,
                'RandomForest',
                plots_dir,
                verbose
            )
        
        if 'XGBoost_tuned' in models_tuned:
            shap_analysis_tree_model(
                models_tuned['XGBoost_tuned'],
                X_sample,
                'XGBoost',
                plots_dir,
                verbose
            )
    
    if verbose:
        logger.info("\n✓ Model interpretation complete")


def run_eda_interpretation_pipeline(
    df: pd.DataFrame,
    models_tuned: dict,
    verbose: bool = True
):
    """
    Complete EDA + Interpretation pipeline
    
    Args:
        df: Full dataset
        models_tuned: Trained tuned models
        verbose: Print progress
    """
    logger.info("\n" + "="*70)
    logger.info("MODULE 05: EDA & INTERPRETATION")
    logger.info("="*70)
    
    # 1. EDA
    run_eda_pipeline(df, config.plots_dir, verbose)
    
    # 2. Sample for SHAP
    feature_cols = [f for f in config.data.feature_cols if f in df.columns]
    n_sample = min(config.evaluation.shap_sample_size, len(df))
    df_sample = df.sample(n=n_sample, random_state=config.models.random_state)
    X_sample = df_sample[feature_cols]
    
    # 3. Interpretation
    run_interpretation_pipeline(models_tuned, X_sample, config.plots_dir, verbose)
    
    logger.info("\n" + "="*70)
    logger.info("✅ MODULE 05 COMPLETED")
    logger.info("="*70)


if __name__ == "__main__":
    import pickle
    
    # Load data
    df_path = config.data_dir / "dataset_clean.csv"
    df = pd.read_csv(df_path)
    
    # Load tuned models
    with open(config.models_dir / "models_tuned.pkl", 'rb') as f:
        models_tuned = pickle.load(f)
    
    # Run pipeline
    run_eda_interpretation_pipeline(df, models_tuned, verbose=True)
    
    print("\n🎯 EDA & Interpretation complete!")
