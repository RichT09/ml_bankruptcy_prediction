# ML Bankruptcy Prediction

Bankruptcy Prediction using Machine Learning - Master Finance Thesis Project

**Author:** Richard Tschumi  
**Institution:** HEC Lausanne  
**Date:** 2025

## Overview

This project implements a complete machine learning pipeline for predicting corporate bankruptcy using financial ratios derived from Compustat data.

## Project Structure

```
ml_bankruptcy_prediction/
├── main.py                     # Pipeline orchestrator
├── requirements.txt            # Python dependencies
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

- Logistic Regression (L2 regularised)
- Random Forest
- XGBoost

Each model is trained in two variants:
- **BASE**: Conservative hyperparameters + class weighting
- **TUNED**: Hyperparameter optimisation via TimeSeriesSplit CV + SMOTE