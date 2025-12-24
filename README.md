# ML Bankruptcy Prediction

Bankruptcy Prediction Using Machine Learning

**Author:** Richard Tschumi  
**Institution:** HEC Lausanne  
**Date:** 2025

## Overview
This project implements a complete machine learning pipeline for predicting corporate bankruptcy using financial statements derived from Compustat data.

## Environment Setup
You can set up the Python environment for this project in two ways:

### 1. Using `environment.yml` (Recommended for Conda/Miniconda users)
```bash
# Create the environment from the YAML file
conda env create -f environment.yml

# Activate the environment
conda activate ml_bankruptcy_prediction
```

### 2. Using `requirements.txt` (For pip/virtualenv users)
If you prefer pip or do not use Conda, you can install dependencies as follows:

```bash
# (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run full pipeline
python main.py

# Skip SHAP analysis (faster)
python main.py --skip-shap

# Skip EDA & interpretation
python main.py --skip-eda

# Override split year
python main.py --split-year 2012
```

Expected Outputs : performance comparison between three models and top features leading bankruptcy

## Models
- Logistic Regression
- Random Forest
- XGBoost

Each model is trained in two variants:
- **BASE**: Conservative hyperparameters + class weighting
- **TUNED**: Hyperparameter optimisation via TimeSeriesSplit CV + SMOTE

## Project Structure
```
ml_bankruptcy_prediction/
├── main.py                     # Pipeline orchestrator
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment specification
├── data/                       # Input data and intermediate outputs
│   └── us-bkty-final.csv       # Raw Compustat dataset
├── outputs/
│   ├── models/                 # Serialised trained models (.pkl)
│   ├── plots/                  # Diagnostic figures and SHAP plots
│   ├── logs/                   # Execution logs
│   └── reports/                # Hyperparameter reports
└── src/                        # Source modules
    ├── config.py               # Centralised configuration
    ├── data_features.py        # Data loading and feature engineering
    ├── preprocessing.py        # Cleaning, splitting, scaling, SMOTE
    ├── models.py               # Model training (base + tuned)
    ├── evaluation.py           # Metrics, curves, backtesting
    └── eda_interpretation.py   # EDA and SHAP analysis
```

## Restults
INFO - ============================================================
INFO - TOP FEATURES AFFECTING BANKRUPTCY RISK
INFO - ============================================================
INFO - 
                    Rank FI_XGB_Base SHAP_XGB_Base Combined  Effect
int_coverage           1      0.1614        0.0174   0.0894  ↓ Risk
roa_approx             2      0.1357        0.0249   0.0803  ↑ Risk
ocf                    3      0.0804        0.0383   0.0593  ↓ Risk
quick_ratio            4      0.0613        0.0437   0.0525  ↓ Risk
ebit                   5      0.0615        0.0363   0.0489  ↓ Risk
inventories            6      0.0649        0.0284   0.0466  ↓ Risk
shareholder_equity     7      0.0567        0.0355   0.0461  ↓ Risk
current_assets         8      0.0392        0.0384   0.0388  ↓ Risk
retained_earnings      9      0.0630        0.0100   0.0365  ↓ Risk
interest_exp          10      0.0382        0.0274   0.0328  ↑ Risk

INFO - Effect: ↑ Risk = increases bankruptcy probability
INFO -         ↓ Risk = decreases bankruptcy probability

Best Models by AUC:
  Best BASE:  XGBoost_base                   AUC = 0.9248
  Best TUNED: XGBoost_tuned                  AUC = 0.9085
  Improvement: -1.63% points