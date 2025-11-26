Project Proposal – Bankruptcy Prediction Using Machine Learning

This project aims to develop a machine learning model capable of estimating whether a U.S. company is at risk of going bankrupt in the following year (T+1). The objective is both predictive and analytical: (1) produce a probability of future bankruptcy, and (2) identify which financial indicators contribute the most to that risk.

!!!(CHANGE PERIOD SPLIT WHEN OPTIMIZED)!!!
The analysis will rely exclusively on a structured U.S. public company dataset, covering the period from 1995 to 2024. To ensure a robust and realistic evaluation, I will use a time-based split: training from XXXX-XXXX and testing from XXXX-XXXX. 
!!!(CHANGE PERIOD SPLIT WHEN OPTIMIZED)!!!

The methodology is deliberately simple and focused on four well-established models: Logistic Regression, Support Vector Machine, Random Forest, and XGBoost. These models provide complementary strengths—interpretability, non-linear learning, and high predictive performance. The goal is to compare different models based on their metrics (AUC, precision, recall, F1-score) and their ability to handle imbalanced databases.. SMOTE and class-weighting will be used when necessary to mitigate class imbalance.

Finally, model interpretability is a key part of the project. Using SHAP values, I will identify which financial ratios most strongly increase bankruptcy risk. This provides useful economic insights in addition to predictive accuracy.

This project follows a modular pipeline structure, as recommended in the course materials. 