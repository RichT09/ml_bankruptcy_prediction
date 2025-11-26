# congig.py

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any


# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]  # bkty_ml parent
RAW_DATA_PATH = BASE_DIR / "us-bkty-final.csv"
PROJECT_DIR = Path(__file__).resolve().parents[1]  # bkty_ml/

DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
METRICS_DIR = OUTPUT_DIR / "metrics"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Create directories
for dir_path in [DATA_DIR, OUTPUT_DIR, MODELS_DIR, METRICS_DIR, PLOTS_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)


# ============================================================================
# DATA CONFIGURATION
# ============================================================================

@dataclass
class DataConfig:
    """Data configuration"""
    
    # Temporal configuration
    year_col: str = "fyear"
    year_start: int = 1995
    year_end: int = 2023
    split_year: int = 2014
    
    # Key columns
    id_col: str = "gvkey"
    ticker_col: str = "tic"
    name_col: str = "conm"
    status_col: str = "dlrsn"
    target_col: str = "failed_next_year"
    
    # Financial ratios
    ratio_cols: List[str] = field(default_factory=lambda: [
        "current_ratio", "quick_ratio", "cash_ratio",
        "debt_to_assets", "debt_to_equity", "int_coverage",
        "roa_approx", "roe_approx", "gross_margin", "ebit_margin",
        "cf_to_debt", "fcf_margin", "fcf_to_assets", 
        "re_assets", "asset_turnover",
    ])
    
    # Raw financial variables
    raw_fin_cols: List[str] = field(default_factory=lambda: [
        "total_assets", "total_liabilities", "current_assets",
        "current_liabilities", "current_debt", "longterm_debt",
        "revenue", "ebit", "ebitda", "interest_exp",
        "ocf", "fcf", "icf", "bookvalue_share", "cash_equiv",
        "inventories", "retained_earnings", "shareholder_equity",
        "cogs", "capx",
    ])
    
    @property
    def feature_cols(self) -> List[str]:
        """All features for modeling"""
        return self.raw_fin_cols + self.ratio_cols
    
    @property
    def meta_cols(self) -> List[str]:
        """Metadata columns"""
        return [
            self.id_col, self.ticker_col, self.name_col, 
            self.year_col, self.status_col, "datadate", 
            "cusip", self.target_col
        ]


# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

@dataclass
class PreprocessingConfig:
    """Preprocessing parameters"""
    
    outlier_method: str = "winsorize"  # "winsorize", "clip", "remove"
    outlier_lower_pct: float = 1.0
    outlier_upper_pct: float = 99.0
    
    scaling_method: str = "standard"  # "standard", "minmax", "robust"


# ============================================================================
# MODEL CONFIGURATION - BASE & TUNED
# ============================================================================

@dataclass
class ModelConfig:
    """Model configuration - Base and Tuned versions"""
    
    random_state: int = 42
    use_smote: bool = True
    n_jobs: int = -1
    
    # ==================== LOGISTIC REGRESSION ====================
    logreg_base: Dict[str, Any] = field(default_factory=lambda: {
        'max_iter': 1000,
        'class_weight': 'balanced',
        'solver': 'lbfgs',
        'penalty': 'l2',
        'C': 1.0,
        'random_state': 42,
    })
    
    logreg_tuned: Dict[str, Any] = field(default_factory=lambda: {
        'max_iter': 2000,
        'class_weight': 'balanced',
        'solver': 'saga',  # Better for large datasets
        'penalty': 'elasticnet',
        'C': 0.5,
        'l1_ratio': 0.5,  # Elastic net mixing
        'random_state': 42,
    })
    
    logreg_param_grid: Dict[str, List] = field(default_factory=lambda: {
        'C': [0.1, 0.5, 1.0, 2.0],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'l1_ratio': [0.3, 0.5, 0.7],
        'solver': ['saga'],
    })
    
    # ==================== RANDOM FOREST ====================
    rf_base: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 300,
        'max_depth': 10,
        'min_samples_split': 20,
        'min_samples_leaf': 10,
        'class_weight': 'balanced_subsample',
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
    })
    
    rf_tuned: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 500,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'class_weight': 'balanced_subsample',
        'max_features': 'sqrt',
        'min_impurity_decrease': 0.0001,
        'random_state': 42,
        'n_jobs': -1,
    })
    
    rf_param_grid: Dict[str, List] = field(default_factory=lambda: {
        'n_estimators': [300, 500, 700],
        'max_depth': [10, 15, 20],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [2, 5, 10],
    })
    
    # ==================== XGBOOST ====================
    xgb_base: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 3,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'gamma': 0,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss',
    })
    
    xgb_tuned: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 500,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'min_child_weight': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 2.0,
        'gamma': 0.1,
        'random_state': 42,
        'n_jobs': -1,
        'eval_metric': 'logloss',
    })
    
    xgb_param_grid: Dict[str, List] = field(default_factory=lambda: {
        'n_estimators': [300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8],
        'colsample_bytree': [0.7, 0.8],
        'min_child_weight': [3, 5, 7],
        'gamma': [0, 0.1, 0.2],
    })
    
    # ==================== SVM ====================
    svm_base: Dict[str, Any] = field(default_factory=lambda: {
        'kernel': 'rbf',
        'C': 1.0,
        'gamma': 'scale',
        'class_weight': 'balanced',
        'probability': True,
        'random_state': 42,
    })
    
    svm_tuned: Dict[str, Any] = field(default_factory=lambda: {
        'kernel': 'rbf',
        'C': 10.0,
        'gamma': 0.01,
        'class_weight': 'balanced',
        'probability': True,
        'random_state': 42,
    })
    
    svm_param_grid: Dict[str, List] = field(default_factory=lambda: {
        'C': [0.1, 1.0, 10.0, 100.0],
        'gamma': [0.001, 0.01, 0.1, 'scale'],
        'kernel': ['rbf', 'poly'],
    })


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

@dataclass
class EvaluationConfig:
    """Evaluation settings"""
    
    metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision'
    ])
    
    # SHAP analysis
    enable_shap: bool = True
    shap_sample_size: int = 2000
    shap_top_features: int = 20
    
    # Visualization
    figure_dpi: int = 150
    figure_format: str = "png"


# ============================================================================
# MASTER CONFIGURATION
# ============================================================================

class Config:
    """Master configuration aggregating all sub-configs"""
    
    def __init__(self):
        self.data = DataConfig()
        self.preprocessing = PreprocessingConfig()
        self.models = ModelConfig()
        self.evaluation = EvaluationConfig()
        
        # Paths
        self.base_dir = BASE_DIR
        self.raw_data_path = RAW_DATA_PATH
        self.data_dir = DATA_DIR
        self.output_dir = OUTPUT_DIR
        self.models_dir = MODELS_DIR
        self.metrics_dir = METRICS_DIR
        self.plots_dir = PLOTS_DIR
    
    def __repr__(self):
        return (
            f"Config(\n"
            f"  split_year={self.data.split_year},\n"
            f"  n_features={len(self.data.feature_cols)},\n"
            f"  use_smote={self.models.use_smote}\n"
            f")"
        )


# ============================================================================
# GLOBAL INSTANCE
# ============================================================================

config = Config()


# ============================================================================
# CONVENIENCE EXPORTS (for backward compatibility)
# ============================================================================

# Paths
DATA_DIR = config.data_dir
PLOTS_DIR = config.plots_dir
MODELS_DIR = config.models_dir
METRICS_DIR = config.metrics_dir

# Data
YEAR_COL = config.data.year_col
SPLIT_YEAR = config.data.split_year
ID_COL = config.data.id_col
TARGET_COL = config.data.target_col
FEATURE_COLS = config.data.feature_cols

# Models
RANDOM_STATE = config.models.random_state
USE_SMOTE = config.models.use_smote


if __name__ == "__main__":
    print("="*60)
    print("BANKRUPTCY PREDICTION - CONFIGURATION")
    print("="*60)
    print(config)
    print(f"\n Data path: {config.raw_data_path}")
    print(f"Features: {len(config.data.feature_cols)}")
    print(f"Target: {config.data.target_col}")
    print("\n Config loaded successfully!")
