# ML Bankruptcy Prediction

Bankruptcy Prediction Using Machine Learning

**Author:** Richard Tschumi  
**Institution:** HEC Lausanne  
**Date:** 2025

## Overview

This project implements a complete machine learning pipeline for predicting corporate bankruptcy using financial statements derived from Compustat data.

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

## Models

- Logistic Regression
- Random Forest
- XGBoost

Each model is trained in two variants:
- **BASE**: Conservative hyperparameters + class weighting
- **TUNED**: Hyperparameter optimisation via TimeSeriesSplit CV + SMOTE

## Environment Setup

You can set up the Python environment for this project in two ways:

### 1. Using `environment.yml` (Recommended for Conda/Miniconda users)

This ensures full reproducibility and installs all dependencies, including the correct Python version.

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

- The `environment.yml` file is recommended for full reproducibility, as it pins the Python version and uses the conda-forge channel.
- The `requirements.txt` file is suitable for standard pip-based workflows.