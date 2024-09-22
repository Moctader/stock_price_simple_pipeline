# EUR/USD Forecasting Pipeline

This repository contains a pipeline for predicting EUR/USD prices using a machine learning model. The pipeline covers various stages, from data fetching and preprocessing to feature engineering, model training, validation, deployment, and drift detection.

## Table of Contents

- [Overview](#overview)
- [Pipeline Stages](#pipeline-stages)
- [Setup](#setup)
- [Usage](#usage)
- [Configuration](#configuration)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

This pipeline is designed to:

1. Fetch EUR/USD price data.
2. Preprocess and engineer features for training.
3. Train and validate a machine learning model (e.g., ElasticNet, RandomForest).
4. Deploy the trained model for production use.
5. Monitor performance and detect drift in the prediction process.
6. Automate retraining based on updated data.

The primary goal is to automate the forecasting process and manage the lifecycle of machine learning models, ensuring continuous improvement and monitoring.

## Pipeline Stages

### 1. **Fetch Data**
The `fetch_eurusd_data` function retrieves historical EUR/USD data, which serves as the foundation for the entire pipeline.

### 2. **Preprocess Data**
The `data_preprocessing` function cleans the data, handles missing values, and applies any necessary transformations based on the configuration file.

### 3. **Feature Engineering**
In this step, `create_features` generates new features, such as moving averages, to enhance the predictive power of the model.

### 4. **Train Model**
The `train_model` function splits the data into training and testing sets, applies model training (ElasticNet, RandomForest, etc.), and produces predictions. The model type and hyperparameters are specified in the configuration.

### 5. **Evaluate Model**
`evaluate_model` assesses the model's performance using metrics such as Mean Squared Error (MSE) and R².

### 6. **Model Validation**
The `model_validation` function validates the model's performance on a holdout validation set, ensuring it generalizes well to unseen data.

### 7. **Model Deployment**
`model_deployment` deploys the trained model using BentoML. The model is stored and tagged for production use.

### 8. **Prediction Service**
Once the model is deployed, the `prediction_service` function uses it to generate predictions for new data points.

### 9. **Post Processing**
In this step, `post_processing` creates buy/sell/hold decisions based on the predicted and actual prices.

### 10. **Prepare Current Data**
`process_current_data` is responsible for preprocessing and engineering features for the current dataset (e.g., real-time EUR/USD data) to make predictions and evaluate drift.

### 11. **Drift Detection**
The `suite.model_performance_and_Target_drift_analysis` function compares the new data (predictions and features) to the historical data used during training, detecting any significant shifts in data distribution.

### 12. **Automated Retraining**
The `automated_retraining` function periodically retrains the model based on new data to ensure that the model stays up-to-date and adapts to changes in the data distribution.

## Setup

### Prerequisites

- Python 3.8 or higher
- Required Python libraries can be installed using the provided `requirements.txt` file.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Moctader/stock_price_simple_pipeline/blob/yaml_based_pipeline/pipeline.py
   cd stock-pipeline
