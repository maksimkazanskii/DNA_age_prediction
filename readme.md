# DNA Age Prediction from Ancient DNA Damage Profiles

This repository contains the implementation of a machine learning
pipeline for predicting the chronological age of ancient biological
samples using DNA damage patterns extracted from BAM files.

The approach is based on statistical features derived from post-mortem
DNA degradation and evaluates whether molecular damage signatures alone
can be used to estimate sample age.

------------------------------------------------------------------------

## Overview

The pipeline operates on damage statistics obtained from DamageProfiler
output files and consists of:

1.  Feature extraction from damage profiles\
2.  Row-wise normalization of mutation statistics\
3.  Batch-aware dataset partitioning\
4.  Dimensionality reduction via PCA\
5.  Supervised regression models\
6.  Grouped cross-validation by publication\
7.  Model selection by cross-validated MAE\
8.  Inference on unseen datasets

The trained model is saved together with the preprocessing pipeline
(**Scaler + PCA**) for reproducible inference.

------------------------------------------------------------------------

## Project Structure

    src_final/
    ├── experiments.py        # Model training pipeline
    ├── inference.py          # Inference on new BAM files
    ├── inference_test.py     # Inference on held-out test set
    ├── visualization.py      # Plotting and reporting utilities

    data/
    ├── labels/
    └── full_data/

    config/
    └── config_harvard_60_cv5.json

------------------------------------------------------------------------

## Requirements

Python ≥ 3.8

Install dependencies:

``` bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn joblib
```

------------------------------------------------------------------------

## Training the Model

Run:

``` bash
python src_final/experiments.py
```

This script performs:

-   Extraction of damage statistics from DamageProfiler output
    (`misincorporation.txt`)
-   Metadata loading (age and publication batch)
-   Row-wise normalization of mutation statistics
-   Flattening of positional mutation features
-   Batch-aware dataset grouping
-   Train/Test split using `GroupShuffleSplit`
-   PCA dimensionality reduction
-   Hyperparameter optimization via `GridSearchCV`
-   Grouped 5-fold cross-validation (`GroupKFold`)
-   Model selection by minimum MAE

All regression models are trained using the following pipeline:

    StandardScaler → PCA → Regressor

------------------------------------------------------------------------

## Evaluated Models

-   Linear Regression\
-   Ridge Regression\
-   Lasso\
-   ElasticNet\
-   Random Forest\
-   Gradient Boosting\
-   Support Vector Regression\
-   K-Nearest Neighbors\
-   Bayesian Ridge\
-   Decision Tree\
-   XGBoost\
-   Mean (Baseline) Predictor

------------------------------------------------------------------------

## Saved Artifacts

After training, the best-performing configuration is saved to:

    exp_folder/best_model/

Including:

-   `best_model.pkl` -- trained regression model\
-   `best_scaler.pkl` -- fitted StandardScaler\
-   `best_pca.pkl` -- fitted PCA\
-   `best_model_params.json` -- selected model and hyperparameters

------------------------------------------------------------------------

## Inference on New BAM Files

To perform inference on new DamageProfiler outputs:

``` bash
python src_final/inference.py
```

Predictions are generated using:

    Scaler → PCA → Model.predict()

------------------------------------------------------------------------

## Inference on Test Dataset

To evaluate model performance on the held-out grouped test set:

``` bash
python src_final/inference_test.py
```

Metrics:

-   MAE -- Mean Absolute Error\
-   RMSE -- Root Mean Squared Error\
-   Baseline MAE -- Error of mean predictor

Predictions are saved to:

    exp_folder/test/test_predictions.csv

------------------------------------------------------------------------

## Visualization

To generate figures for model comparison and reporting:

``` bash
python src_final/visualization.py
```

Outputs are written to:

    exp_folder/FINAL_REPORT/
