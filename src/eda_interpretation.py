#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module 05: EDA & Model Interpretation
Exploratory Data Analysis

Author: Richard Tschumi
Institution: HEC Lausanne
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

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.config import config
except ImportError:
    import config as config_module
    config = config_module.config

# Plot settings
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = config.evaluation.figure_dpi

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================

def plot_target_distribution(df: pd.DataFrame, output_path: Path):
    """
    Plot target distribution
    """
    target_col = config.data.target_col
    
    # Figure with bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count plot
    counts = df[target_col].value_counts()
    colors = ['green', 'red']
    bars = ax.bar(['Survived', 'Failed'], counts.values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Count', fontsize=13, fontweight='bold')
    ax.set_xlabel('Status', fontsize=13, fontweight='bold')
    ax.set_title('Target Distribution', fontsize=16, fontweight='bold', pad=20)
    
    # add value labels on bars
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
    
    # Improve grid styling
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Format y-axis with commas
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


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path, selected_features: list = None):
    """
    Plot correlation heatmap
    Uses RFE selected features if provided, otherwise uses config.data.feature_cols
    """
    if selected_features is not None:
        feature_cols = [f for f in selected_features if f in df.columns]
    else:
        # Use config for number of features
        n_features = config.models.rfe_n_features
        feature_cols = [f for f in config.data.feature_cols if f in df.columns][:n_features]
    
    # Prepare data (convert to numeric)
    X = df[feature_cols].copy()
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    X = X.fillna(0)
    
    corr = X.corr()
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Upper triangular mask (keep lower + diagonal)
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    
    cmap = sns.diverging_palette(250, 10, as_cmap=True)
    
    sns.heatmap(
        corr,
        mask=mask,  # Mask upper triangle only
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
    ax.set_title('Feature Correlation Heatmap (20 RFE Features)', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  ✓ {output_path.name}")


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
# SHAP ANALYSIS - FIXED VERSION
# ============================================================================

def prepare_shap_data(df: pd.DataFrame, selected_features: list) -> pd.DataFrame:
    """
    Prepare data for SHAP: select features and convert to float64.
    Handles strings in scientific notation (e.g., "3.1894583E-1").
    
    Args:
        df: Full DataFrame (may contain non-numeric columns)
        selected_features: List of RFE selected features
    
    Returns:
        DataFrame with only the required numeric features (float64)
    """
    # Select only the features needed for SHAP
    available_features = [f for f in selected_features if f in df.columns]
    X = df[available_features].copy()
    
    # Convert all columns to numeric (float64), coercing errors to NaN
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill NaNs with 0
    X = X.fillna(0)
    
    # Ensure float64 dtype
    X = X.astype(np.float64)
    
    return X


def shap_summary_tree_model(
    model, 
    X_shap: pd.DataFrame, 
    model_name: str,
    output_dir: Path,
    verbose: bool = True
):
    """
    SHAP Summary Plot for tree-based models (RF, XGBoost).
    Generates ONLY ONE plot (beeswarm summary).
    """
    if verbose:
        logger.info(f"\n  → SHAP Summary: {model_name}")
    
    try:
        # Limit the sample for speed
        n_sample = min(500, len(X_shap))
        if len(X_shap) > n_sample:
            X_sample = X_shap.sample(n=n_sample, random_state=42)
        else:
            X_sample = X_shap.copy()
        
        # Convert to numpy array float64 (CRITICAL)
        X_array = X_sample.values.astype(np.float64)
        feature_names = list(X_sample.columns)
        
        # Different approach for XGBoost vs RandomForest
        if 'XGBoost' in model_name:
            # XGBoost: use shap.Explainer with predict_proba (workaround for string conversion bug)
            background = X_array[:min(100, len(X_array))]
            explainer = shap.Explainer(model.predict_proba, background)
            shap_values_obj = explainer(X_array)
            # Get class 1 (bankruptcy) SHAP values
            shap_values = shap_values_obj.values[:, :, 1]
        else:
            # RandomForest: use TreeExplainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_array)
            
            # Handle different SHAP output formats:
            # - RandomForest: returns 3D array (n_samples, n_features, n_classes) or list
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Class 1 (bankruptcy)
            elif len(shap_values.shape) == 3:
                shap_values = shap_values[:, :, 1]  # Class 1 (bankruptcy)
        
        # Verify dimensions match
        if shap_values.shape != X_array.shape:
            raise ValueError(f"Shape mismatch: SHAP {shap_values.shape} vs Data {X_array.shape}")
        
        # Summary Plot (beeswarm) - ONLY ONE PLOT
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, 
            X_array, 
            feature_names=feature_names,
            show=False, 
            max_display=20
        )
        plt.title(f'{model_name} - SHAP Summary Plot', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_{model_name}.png", dpi=150, bbox_inches='tight')
        plt.close('all')
        
        if verbose:
            logger.info(f"    ✓ SHAP summary plot saved: shap_summary_{model_name}.png")
            
    except Exception as e:
        if verbose:
            logger.error(f"    ❌ SHAP failed for {model_name}: {str(e)}")
        _plot_importance_fallback(model, X_shap, model_name, output_dir, verbose)


def shap_summary_linear_model(
    model,
    X_shap: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    scaler=None,
    verbose: bool = True
):
    """
    SHAP Summary Plot for Logistic Regression.
    Applies the scaler if provided (IMPORTANT: LogReg was trained on scaled data).
    Generates ONLY ONE plot (beeswarm summary).
    """
    if verbose:
        logger.info(f"\n  → SHAP Summary: {model_name}")
    
    try:
        # Limit the sample
        n_sample = min(500, len(X_shap))
        if len(X_shap) > n_sample:
            X_sample = X_shap.sample(n=n_sample, random_state=42)
        else:
            X_sample = X_shap.copy()
        
        # Convert to numpy array float64
        X_array = X_sample.values.astype(np.float64)
        feature_names = list(X_sample.columns)
        
        # Apply the scaler if provided (IMPORTANT for LogReg)
        if scaler is not None:
            X_scaled = scaler.transform(X_array)
        else:
            X_scaled = X_array
        
        # Background data for LinearExplainer
        n_background = min(100, len(X_scaled))
        background = X_scaled[:n_background]
        
        # LinearExplainer
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(X_scaled)
        
        # Summary Plot - ONLY ONE PLOT
        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values, 
            X_scaled,
            feature_names=feature_names,
            show=False, 
            max_display=20
        )
        plt.title(f'{model_name} - SHAP Summary Plot', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_{model_name}.png", dpi=150, bbox_inches='tight')
        plt.close('all')
        
        if verbose:
            logger.info(f"    ✓ SHAP summary plot saved: shap_summary_{model_name}.png")
            
    except Exception as e:
        if verbose:
            logger.error(f"    ❌ SHAP failed for {model_name}: {str(e)}")


def _plot_importance_fallback(
    model,
    X_shap: pd.DataFrame,
    model_name: str,
    output_dir: Path,
    verbose: bool = True
):
    """
    Fallback: feature importance when SHAP fails
    """
    try:
        importances = pd.Series(
            model.feature_importances_, 
            index=X_shap.columns
        ).sort_values(ascending=True)
        
        plt.figure(figsize=(12, 8))
        importances.tail(20).plot(kind='barh', color='steelblue', alpha=0.8)
        plt.xlabel('Importance', fontsize=12, fontweight='bold')
        plt.title(f'{model_name} - Feature Importance (Fallback)', fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"shap_summary_{model_name}_fallback.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            logger.info(f"    ✓ Fallback plot saved")
    except Exception as e:
        if verbose:
            logger.error(f"    ❌ Fallback also failed: {str(e)}")


# ============================================================================
# FEATURE IMPORTANCE SUMMARY TABLE
# ============================================================================

def generate_feature_importance_summary(
    models_base: dict,
    models_tuned: dict,
    feature_names: list,
    output_dir: Path,
    X_shap: pd.DataFrame = None,
    scaler=None,
    top_n: int = 5,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate a summary table of top features using ONLY XGBoost (Base) FI and SHAP.
    """
    if verbose:
        logger.info("\n--- Generating Feature Importance Summary Table ---")
        logger.info("    Using ONLY XGBoost (Base) Feature Importance AND SHAP values")

    # Get XGBoost (Base) model
    xgb_base = models_base.get('XGBoost_base')
    if xgb_base is None:
        raise ValueError("XGBoost_base model not found in models_base")

    # Feature Importance
    fi = pd.Series(xgb_base.feature_importances_, index=feature_names)

    # SHAP Importance
    shap_imp = None
    if X_shap is not None:
        try:
            n_sample = min(200, len(X_shap))
            X_sample = X_shap.sample(n=n_sample, random_state=42) if len(X_shap) > n_sample else X_shap
            X_array = X_sample.values.astype(np.float64)
            background = X_array[:min(50, len(X_array))]
            explainer = shap.Explainer(xgb_base.predict_proba, background)
            shap_vals = explainer(X_array).values[:, :, 1]
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)
            shap_imp = pd.Series(mean_abs_shap, index=feature_names)
        except Exception as e:
            if verbose:
                logger.warning(f"    SHAP failed for XGBoost_base: {str(e)[:50]}")
            shap_imp = pd.Series([0]*len(feature_names), index=feature_names)
    else:
        shap_imp = pd.Series([0]*len(feature_names), index=feature_names)

    # Combine and rank
    df = pd.DataFrame({
        'FI_XGB_Base': fi,
        'SHAP_XGB_Base': shap_imp
    })
    df['FI_XGB_Base'] = df['FI_XGB_Base'].apply(lambda x: f"{x:.4f}")
    df['SHAP_XGB_Base'] = df['SHAP_XGB_Base'].apply(lambda x: f"{x:.4f}")
    df['Combined'] = df[['FI_XGB_Base', 'SHAP_XGB_Base']].astype(float).mean(axis=1).apply(lambda x: f"{x:.4f}")
    df_sorted = df.sort_values('Combined', ascending=False).head(top_n)
    df_sorted['Rank'] = range(1, top_n + 1)

    # Get direction from LogReg (base model) if available
    if 'LogisticRegression_base' in models_base:
        logreg = models_base['LogisticRegression_base']
        def get_direction(feature):
            idx = feature_names.index(feature)
            coef = logreg.coef_[0][idx]
            return "↑ Risk" if coef > 0 else "↓ Risk"
        df_sorted['Effect'] = [get_direction(f) for f in df_sorted.index]
    else:
        df_sorted['Effect'] = "N/A"

    # Save to CSV
    csv_path = config.data_dir / "top_features_summary.csv"
    df_sorted[['Rank', 'FI_XGB_Base', 'SHAP_XGB_Base', 'Combined', 'Effect']].to_csv(csv_path, index=True)
    if verbose:
        logger.info(f"✓ Summary table saved: {csv_path}")

    # Visual table (PNG)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('off')
    cell_text = df_sorted[['Rank', 'FI_XGB_Base', 'SHAP_XGB_Base', 'Combined', 'Effect']].reset_index().values.tolist()
    col_labels = ['Feature', 'Rank', 'FI_XGB_Base', 'SHAP_XGB_Base', 'Combined', 'Effect']
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colColours=['#4472C4'] * len(col_labels)
    )
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    # Style header
    for j, label in enumerate(col_labels):
        cell = table[(0, j)]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#4472C4')
    
    # Alternate row colors
    for i in range(1, len(cell_text) + 1):
        for j in range(len(col_labels)):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#D6DCE5')
            else:
                cell.set_facecolor('#FFFFFF')
            # Highlight Effect column
            if j == len(col_labels) - 1:  # Effect column
                if '↑' in str(cell_text[i-1][j]):
                    cell.set_text_props(color='red', weight='bold')
                elif '↓' in str(cell_text[i-1][j]):
                    cell.set_text_props(color='green', weight='bold')
    
    plt.title('Top Features Affecting Bankruptcy Risk\n(Normalized Importance Scores)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    png_path = output_dir / "top_features_table.png"
    plt.savefig(png_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    if verbose:
        logger.info(f"✓ Visual table saved: {png_path}")
    
    # Print to console
    if verbose:
        logger.info("\n" + "="*80)
        logger.info("TOP FEATURES AFFECTING BANKRUPTCY RISK")
        logger.info("="*80)
        logger.info(f"\n{df_sorted[['Rank', 'FI_XGB_Base', 'SHAP_XGB_Base', 'Combined', 'Effect']].to_string(index=True)}")
        logger.info("\n" + "-"*80)
        logger.info("Effect: ↑ Risk = increases bankruptcy probability")
        logger.info("        ↓ Risk = decreases bankruptcy probability")
        logger.info("="*80)

    return df_sorted


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_eda_pipeline(df: pd.DataFrame, plots_dir: Path, verbose: bool = True, selected_features: list = None):
    """
    Complete EDA pipeline - saves to eda/ subfolder
    """
    if verbose:
        logger.info("\n--- Exploratory Data Analysis ---")
    
    # Create EDA subfolder
    eda_dir = plots_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)
    
    plot_target_distribution(df, eda_dir / "target_distribution.png")
    plot_temporal_distribution(df, eda_dir / "temporal_distribution.png")
    plot_correlation_heatmap(df, eda_dir / "correlation_heatmap.png", selected_features)
    
    if verbose:
        logger.info(f"✓ EDA plots saved to {eda_dir}")


def run_interpretation_pipeline(
    models_base: dict,
    X_shap: pd.DataFrame,
    plots_dir: Path,
    verbose: bool = True,
    scaler=None,
    models_tuned: dict = None
):
    """
    Complete model interpretation pipeline - BASE and TUNED models
    Saves to models/ subfolder
    Generates 6 SHAP summary plots (1 per model × 2 versions)
    """
    if verbose:
        logger.info("\n--- Model Interpretation (BASE models) ---")
    
    # Create model_interpretation subfolder
    models_dir = plots_dir / "model_interpretation"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    feature_names = list(X_shap.columns)
    
    # ========== BASE MODELS ==========
    # Feature importance (BASE models)
    if 'RandomForest_base' in models_base:
        plot_feature_importance_rf(
            models_base['RandomForest_base'],
            feature_names,
            models_dir / "importance_rf_base.png"
        )
    
    if 'XGBoost_base' in models_base:
        plot_feature_importance_xgb(
            models_base['XGBoost_base'],
            feature_names,
            models_dir / "importance_xgb_base.png"
        )
    
    if 'LogisticRegression_base' in models_base:
        plot_logistic_regression_coef(
            models_base['LogisticRegression_base'],
            feature_names,
            models_dir / "importance_logreg_base.png"
        )
    
    # SHAP Summary Plots (BASE models)
    if config.evaluation.enable_shap:
        if verbose:
            logger.info("\n--- SHAP Summary Plots (BASE models) ---")
        
        if 'RandomForest_base' in models_base:
            shap_summary_tree_model(
                models_base['RandomForest_base'],
                X_shap,
                'RandomForest_base',
                models_dir,
                verbose
            )
        
        if 'XGBoost_base' in models_base:
            shap_summary_tree_model(
                models_base['XGBoost_base'],
                X_shap,
                'XGBoost_base',
                models_dir,
                verbose
            )
        
        if 'LogisticRegression_base' in models_base:
            shap_summary_linear_model(
                models_base['LogisticRegression_base'],
                X_shap,
                'LogisticRegression_base',
                models_dir,
                scaler=scaler,
                verbose=verbose
            )
    
    if verbose:
        logger.info(f"\n✓ BASE model interpretation plots saved to {models_dir}")
    
    # ========== TUNED MODELS ==========
    if models_tuned is not None:
        if verbose:
            logger.info("\n--- Model Interpretation (TUNED models) ---")
        
        # Feature importance (TUNED models)
        if 'RandomForest_tuned' in models_tuned:
            plot_feature_importance_rf(
                models_tuned['RandomForest_tuned'],
                feature_names,
                models_dir / "importance_rf_tuned.png"
            )
        
        if 'XGBoost_tuned' in models_tuned:
            plot_feature_importance_xgb(
                models_tuned['XGBoost_tuned'],
                feature_names,
                models_dir / "importance_xgb_tuned.png"
            )
        
        if 'LogisticRegression_tuned' in models_tuned:
            plot_logistic_regression_coef(
                models_tuned['LogisticRegression_tuned'],
                feature_names,
                models_dir / "importance_logreg_tuned.png"
            )
        
        # SHAP Summary Plots (TUNED models)
        if config.evaluation.enable_shap:
            if verbose:
                logger.info("\n--- SHAP Summary Plots (TUNED models) ---")
            
            if 'RandomForest_tuned' in models_tuned:
                shap_summary_tree_model(
                    models_tuned['RandomForest_tuned'],
                    X_shap,
                    'RandomForest_tuned',
                    models_dir,
                    verbose
                )
            
            if 'XGBoost_tuned' in models_tuned:
                shap_summary_tree_model(
                    models_tuned['XGBoost_tuned'],
                    X_shap,
                    'XGBoost_tuned',
                    models_dir,
                    verbose
                )
            
            if 'LogisticRegression_tuned' in models_tuned:
                shap_summary_linear_model(
                    models_tuned['LogisticRegression_tuned'],
                    X_shap,
                    'LogisticRegression_tuned',
                    models_dir,
                    scaler=scaler,
                    verbose=verbose
                )
        
        if verbose:
            logger.info(f"\n✓ TUNED model interpretation plots saved to {models_dir}")
    
    # ========== FEATURE IMPORTANCE SUMMARY TABLE ==========
    generate_feature_importance_summary(
        models_base=models_base,
        models_tuned=models_tuned,
        feature_names=feature_names,
        output_dir=models_dir,
        X_shap=X_shap,
        scaler=scaler,
        top_n=10,
        verbose=verbose
    )


def run_eda_interpretation_pipeline(
    df: pd.DataFrame,
    models_base: dict,
    verbose: bool = True,
    selected_features: list = None,
    scaler=None,
    models_tuned: dict = None
):
    """
    Complete EDA + Interpretation pipeline (BASE and TUNED models)
    
    Args:
        df: Full dataset
        models_base: Trained BASE models
        verbose: Print progress
        selected_features: List of RFE selected features (required for correct SHAP)
        scaler: StandardScaler used for LogisticRegression (optional but recommended)
        models_tuned: Trained TUNED models (optional, if provided will generate plots for them too)
    """
    print("\n" + "="*70)
    print("MODULE 05: EDA & INTERPRETATION (BASE + TUNED MODELS)")
    print("="*70)
    
    # 1. EDA (with RFE features for correlation heatmap)
    run_eda_pipeline(df, config.plots_dir, verbose, selected_features)
    
    # 2. Prepare SHAP data with RFE features
    if selected_features is not None:
        feature_cols = selected_features
    else:
        feature_cols = [f for f in config.data.feature_cols if f in df.columns]
    
    if verbose:
        logger.info(f"\n→ Using {len(feature_cols)} features for SHAP analysis")
    
    # Sample the dataset
    n_sample = min(config.evaluation.shap_sample_size, len(df))
    df_sample = df.sample(n=n_sample, random_state=config.models.random_state)
    
    # Prepare SHAP data (numeric conversion with prepare_shap_data)
    X_shap = prepare_shap_data(df_sample, feature_cols)
    
    if verbose:
        logger.info(f"→ SHAP sample: {X_shap.shape[0]} rows × {X_shap.shape[1]} features")
        logger.info(f"→ Features: {list(X_shap.columns)[:5]}...\n showing first 5")
    
    # 3. Interpretation (BASE + TUNED models) with scaler for LogReg
    run_interpretation_pipeline(models_base, X_shap, config.plots_dir, verbose, scaler, models_tuned)
    
    print("")
    print("\n" + "="*70)
    print("MODULE 05 COMPLETED")
    print("="*70)


if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    import pickle
    import joblib
    
    # Load data
    df_path = config.data_dir / "dataset_clean.csv"
    df = pd.read_csv(df_path)
    
    # Load BASE models
    with open(config.models_dir / "models_base.pkl", 'rb') as f:
        models_base = pickle.load(f)
    
    # Load TUNED models
    models_tuned = None
    tuned_path = config.models_dir / "models_tuned.pkl"
    if tuned_path.exists():
        with open(tuned_path, 'rb') as f:
            models_tuned = pickle.load(f)
        logger.info("Loaded TUNED models for interpretation")
    
    # Load metadata to get selected_features
    metadata_path = config.models_dir / "metadata.pkl"
    selected_features = None
    if metadata_path.exists():
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        selected_features = metadata.get('selected_features')
        if selected_features:
            logger.info(f"Loaded {len(selected_features)} RFE features from metadata")
    
    # Load scaler for LogisticRegression
    scaler_path = config.models_dir / "scaler.pkl"
    scaler = None
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        logger.info("Loaded scaler for LogisticRegression SHAP")
    
    # Run pipeline with selected_features, scaler, and tuned models
    run_eda_interpretation_pipeline(
        df, 
        models_base, 
        verbose=True,
        selected_features=selected_features,
        scaler=scaler,
        models_tuned=models_tuned
    )
    
    print("\nEDA & Interpretation complete!")
