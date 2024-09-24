import pandas as pd
import numpy as np
import yaml
import os
import requests
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import pandas as pd
import joblib
import bentoml
from drift_suite import suite




# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 1. Load YAML configuration
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


# 1. Original data_ingestion function
def data_ingestion(api_url: str) -> pd.DataFrame:

    try:
        # Timeout set to handle cases where the API is unresponsive
        response = requests.get(api_url, timeout=10)

        # Raise exception for unsuccessful requests (4xx and 5xx)
        response.raise_for_status()

        # Check if the response is empty or None
        if not response.text.strip():
            logger.error("API response is empty")
            raise ValueError("Received empty response from the API")

        # Try converting the response to JSON, handle case where it's not valid JSON
        try:
            data = pd.DataFrame(response.json())
            print(data)
        except ValueError as e:
            logger.error("Failed to parse JSON from API response")
            raise ValueError("Invalid JSON response from the API") from e

        # Edge case: Ensure the response contains data
        if data.empty:
            logger.error("Data ingestion failed: No data returned from the API")
            raise ValueError("No data returned from API")

        # Log success
        logger.info("Data ingestion completed successfully")
        return data

    except requests.exceptions.Timeout:
        logger.error("The request timed out")
        raise

    except requests.exceptions.RequestException as e:
        logger.error(f"An error occurred while fetching data: {e}")
        raise

# 2. Data Ingestion based on config
def fetch_eurusd_data(config):
  
    # API endpoint setup (with date range parameters if needed)
    api_base_url = config['data_ingestion']['api_url']
    start_date = pd.Timestamp(config['data_ingestion']['start_date']).timestamp()
    end_date = pd.Timestamp(config['data_ingestion']['end_date']).timestamp()

    # Construct the API URL dynamically based on dates from config
    api_url = f"{api_base_url}?api_token={config['data_ingestion']['api_token']}&fmt=json&from={int(start_date)}&to={int(end_date)}"

    try:
        # Try to fetch real data using the data_ingestion function
        data = data_ingestion(api_url)

        # Log that real data is used
        logger.info(" data fetched successfully from the API")

    except Exception as e:
        logger.warning(f"Failed to fetch real data from API: {e}")

    return data


def data_preprocessing(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    try:
        # Fetch the required settings from the config
        datetime_column = config['preprocessing']['datetime_column']
        required_columns = config['preprocessing']['required_columns']

        # Ensure essential columns are present (as defined in the config)
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Required columns are missing: {missing_columns}")
            raise KeyError(f"Missing columns: {missing_columns}")

        # Convert datetime column to pandas datetime format, coerce invalid entries to NaT
        data[datetime_column] = pd.to_datetime(data[datetime_column], errors='coerce')

        # Drop rows where datetime is NaT (invalid dates)
        if data[datetime_column].isnull().any():
            logger.warning(f"Dropping {data[datetime_column].isnull().sum()} rows with invalid datetime values")
            data.dropna(subset=[datetime_column], inplace=True)

        # Set datetime column as the DataFrame index
        data.set_index(datetime_column, inplace=True)

        # Handle missing values by filling with the median value for each column
        logger.info("Filling missing values with the median value for each column")
        data.fillna(data.median(), inplace=True)

        # Corner case: Ensure no NaNs remain in the essential columns after filling
        remaining_nans = data[required_columns].isnull().sum().sum()
        if remaining_nans > 0:
            logger.error("Data preprocessing failed: Some columns still contain NaN values after filling")
            raise ValueError(f"NaN values remain after processing: {remaining_nans}")

        # Log success and return the preprocessed DataFrame
        logger.info("Data preprocessing completed successfully")
        return data[required_columns].copy()

    except KeyError as e:
        logger.error(f"Data preprocessing failed due to missing columns: {e}")
        raise KeyError("Required columns are missing from the data") from e
    except Exception as e:
        logger.error(f"Data preprocessing encountered an unexpected error: {e}")
        raise


def create_features(data: pd.DataFrame, config: dict) -> pd.DataFrame:
  
    try:
        # Fetch the moving average window size from the config
        window_size = config['feature_engineering']['moving_average_window']
        target_column = config['feature_engineering']['target_column']
        ma_column_name = f"MA{window_size}"

        # Ensure we have enough data for the rolling window
        if len(data) < window_size:
            logger.error(f"Feature engineering failed: Not enough data points for {window_size}-period MA")
            raise ValueError(f"Not enough data points for {window_size}-period MA feature")

        # Calculate the moving average for the target column (e.g., 'close' prices)
        data[ma_column_name] = data[target_column].rolling(window=window_size).mean()

        # Log the number of NaN values generated by rolling window
        num_nans = data[ma_column_name].isna().sum()
        logger.info(f"Rolling window created {num_nans} NaN values for {ma_column_name}")

        # Forward-fill or back-fill to handle NaN values caused by rolling
        data[ma_column_name].ffill(inplace=True)  # Forward fill the NaN values
        data[ma_column_name].bfill(inplace=True)

        # Fallback: If forward-fill still results in all NaNs, apply a fallback strategy
        if data[ma_column_name].isnull().all():
            logger.warning(f"All values in {ma_column_name} are still NaN after forward-filling. Applying fallback strategy.")

            # Fallback strategy: Fill with mean or a predefined constant value
            if len(data[target_column].dropna()) > 0:
                fallback_value = data[target_column].mean()
                data[ma_column_name].fillna(fallback_value, inplace=True)
                logger.info(f"Filled NaN values in {ma_column_name} with mean: {fallback_value:.2f}")
            else:
                # In case the target column itself has no valid values
                fallback_value = 0  # Use a suitable fallback depending on your context
                data[ma_column_name].fillna(fallback_value, inplace=True)
                logger.info(f"Filled NaN values in {ma_column_name} with fallback value: {fallback_value:.2f}")

        # Ensure the DataFrame still contains data after the transformation
        if data.empty:
            logger.error(f"Feature engineering failed: No rows remaining after forward-filling the {ma_column_name} values")
            raise ValueError(f"No rows remaining after forward-filling the {window_size}-period moving average")

        # Log success and return the DataFrame with all relevant columns + the moving average column
        logger.info(f"Feature engineering completed successfully with {ma_column_name} feature added")
        return data

    except KeyError as e:
        logger.error(f"Feature engineering failed due to missing column: {e}")
        raise KeyError(f"Required column '{target_column}' is missing from the data") from e
    except ValueError as e:
        logger.error(f"Feature engineering failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Feature engineering encountered an unexpected error: {e}")
        raise



def train_model(data: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, object]:

    try:
        # Ensure required columns are present for training
        required_columns = ['open', 'high', 'low', 'close', 'MA3']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            logger.error(f"Model training failed due to missing columns: {missing_columns}")
            raise KeyError(f"Missing columns: {missing_columns}")

        # Shift 'close' values to create the target variable and store as 'shifted_close'
        data['shifted_close'] = data['close'].shift(-1)

        # Prepare features (X) and target (y)
        X = data[required_columns].copy()
        X.ffill(inplace=True)
        X.fillna(0, inplace=True)  # Ensure no NaNs remain
        y = data['shifted_close'].values  # Predict the next time step's 'close' value

        # Remove last row due to shift creating NaN
        X = X[:-1]
        y = y[:-1]

        if np.isnan(y).any():
            logger.error("Target variable 'y' contains NaN values after shifting.")
            raise ValueError("Target variable 'y' contains NaN values.")

        # Time-based train-test split
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Get the model type and hyperparameters from the config
        model_type = config['model']['type']
        param_grid = config['model']['param_grid']
        model_name = config['model']['name']  # Name to save the model

        if model_type == 'ElasticNet':
            model = ElasticNet(max_iter=10000)
        elif model_type == 'RandomForest':
            model = RandomForestRegressor()
        else:
            logger.error(f"Unknown model type '{model_type}' specified in config.")
            raise ValueError(f"Unknown model type: {model_type}")

        # Perform GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(model, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Get the best model from GridSearchCV
        best_model = grid_search.best_estimator_

        # Make predictions
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Evaluate the model
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # Log model performance
        logger.info(f"Model training completed. Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}")
        logger.info(f"Model training completed. Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")

        # Add predictions to the original DataFrame
        data['predicted'] = np.nan
        data.loc[data.index[:-1], 'predicted'] = best_model.predict(X)

        # Save the trained model
        model_file_path = f"{model_name}.joblib"
        joblib.dump(best_model, model_file_path)
        logger.info(f"Model saved as {model_file_path}")

        # Return updated data and the best model
        return data, best_model

    except KeyError as e:
        logger.error(f"Model training failed due to missing columns: {e}")
        raise KeyError(f"Required columns are missing from the data") from e
    except ValueError as e:
        logger.error(f"Model training failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Model training encountered an unexpected error: {e}")
        raise

def evaluate_model(data: pd.DataFrame, config: dict) -> dict:

    try:
        # Ensure the 'predicted' and 'close' columns are present

        if 'predicted' not in data.columns or 'close' not in data.columns:
            logger.error("Model evaluation failed due to missing 'predicted' or 'close' columns")
            raise KeyError("Required columns 'predicted' or 'close' are missing for evaluation")

        # Ensure there is enough data for evaluation (at least 1 row of predictions)
        if data['predicted'].isnull().all():
            logger.error("Model evaluation failed: No predictions available for evaluation")
            raise ValueError("No predictions available for evaluation")

        # Align predicted and actual values by ensuring non-NaN values are used
        predicted = data['predicted'].dropna().values
        actual = data['close'].shift(-1).dropna().values  # Compare with shifted actuals

        # Ensure the length of predicted and actual values match
        if len(predicted) != len(actual):
            logger.error("Model evaluation failed: Mismatch between predicted and actual lengths")
            raise ValueError("Mismatch between predicted and actual lengths for evaluation")

        # Initialize an empty dictionary to store the evaluation results
        evaluation_results = {}

        # Get the list of metrics to calculate from the config
        metrics_to_calculate = config.get('evaluation', {}).get('metrics', ['mse', 'r2', 'mae'])

        # Compute metrics based on the config
        if 'mse' in metrics_to_calculate:
            mse = mean_squared_error(actual, predicted)
            evaluation_results['mse'] = mse
            logger.info(f"Mean Squared Error (MSE): {mse:.4f}")
        
        if 'r2' in metrics_to_calculate:
            r2 = r2_score(actual, predicted)
            evaluation_results['r2'] = r2
            logger.info(f"R-squared (R²): {r2:.4f}")
        
        if 'mae' in metrics_to_calculate:
            mae = mean_absolute_error(actual, predicted)
            evaluation_results['mae'] = mae
            logger.info(f"Mean Absolute Error (MAE): {mae:.4f}")

  

        # Return the dictionary with all the computed metrics
        logger.info("Model evaluation completed successfully with selected metrics.")
        return evaluation_results

    except KeyError as e:
        logger.error(f"Model evaluation failed due to missing column(s): {e}")
        raise KeyError("Required columns are missing for model evaluation") from e
    except ValueError as e:
        logger.error(f"Model evaluation failed due to data inconsistency: {e}")
        raise ValueError(f"Data inconsistency during model evaluation: {e}") from e
    except Exception as e:
        logger.error(f"Model evaluation encountered an unexpected error: {e}")
        raise



from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd

def model_validation(data: pd.DataFrame, model: ElasticNet, config: dict) -> dict:
   
    try:
        # Get required columns and validation split from config
        required_columns = config['validation']['features']
        validation_split_ratio = config['validation']['split_ratio']

        # Prepare the features (X) and target (y) for validation
        X = data[required_columns].fillna(0).values
        y = data['close'].shift(-1).values  # Shift 'close' to predict x(t+1) at time step t

        # Remove the last row from X and y, as y[-1] will be NaN after shifting
        X = X[:-1]
        y = y[:-1]

        # Time-based train-validation split
        split_index = int(len(X) * (1 - validation_split_ratio))
        X_train, X_val = X[:split_index], X[split_index:]
        y_train, y_val = y[:split_index], y[split_index:]

        # ElasticNet model already trained, use it for prediction on the validation set
        y_pred_val = model.predict(X_val)

        # Calculate evaluation metrics: MSE, R², MAE for validation
        mse_val = mean_squared_error(y_val, y_pred_val)
        r2_val = r2_score(y_val, y_pred_val)
        mae_val = mean_absolute_error(y_val, y_pred_val)

        # Log the validation metrics
        logger.info(f"Model validation completed. Validation MSE: {mse_val:.4f}, R²: {r2_val:.4f}, MAE: {mae_val:.4f}")

        # Return all metrics in a dictionary
        return {"mse": mse_val, "r2": r2_val, "mae": mae_val}

    except KeyError as e:
        logger.error(f"Model validation failed due to missing column(s): {e}")
        raise KeyError("Required columns are missing for model validation") from e
    except ValueError as e:
        logger.error(f"Model validation failed due to data error: {e}")
        raise ValueError("Data error during model validation") from e
    except Exception as e:
        logger.error(f"Model validation encountered an unexpected error: {e}")
        raise



def model_deployment(model, config) -> str:
    try:
        # Set BENTOML_HOME to the current directory
        os.environ["BENTOML_HOME"] = os.getcwd()
        bentoml_home = os.environ["BENTOML_HOME"]
        logger.info(f"BENTOML_HOME set to: {bentoml_home}")

        # Save the ElasticNet model using BentoML
        bento_svc = bentoml.sklearn.save_model("elasticnet_model", model)
        model_tag = str(bento_svc.tag)

        # Log the deployment success
        logger.info(f"ElasticNet model deployed successfully with BentoML. Model tag: {model_tag}")

        # Print the save location for debugging
        # save_location = bentoml.models.get("elasticnet_model").path
        # logger.info(f"Model saved at: {save_location}")

        return model_tag

    except Exception as e:
        logger.error(f"Model deployment failed: {e}")
        raise



def prediction_service(model_tag: str, engineered_data: pd.DataFrame, config: dict) -> pd.Series:
  
    try:
        # Load the best model from the BentoML model store using the model tag
        model_runner = bentoml.sklearn.get(model_tag).to_runner()
        model_runner.init_local()  # For debugging and local testing only

        # Load required columns from the config
        required_columns = config['prediction']['required_columns']

        # Forward-fill missing values in required columns
        engineered_data[required_columns] = engineered_data[required_columns].ffill()
        # backward-fill missing values in required columns
        engineered_data[required_columns] = engineered_data[required_columns].bfill()

        # If there are still NaNs, fill them with 0 as a last resort
        engineered_data[required_columns] = engineered_data[required_columns].fillna(0)

        # Extract the most recent row of features for prediction
        last_row_features = engineered_data[required_columns].iloc[-1].values.reshape(1, -1)

        # Make the prediction using the trained and deployed model
        predicted_value = model_runner.predict.run(last_row_features)

        # Log the prediction success
        logger.info(f"Prediction completed successfully. Predicted value: {predicted_value}")

        # Handle timestamp for the next prediction
        last_timestamp = engineered_data.index[-1]
        if isinstance(last_timestamp, pd.Timestamp):
            # Assuming daily frequency for time series, increment the timestamp by 1 day
            next_timestamp = last_timestamp + pd.DateOffset(days=1)
        else:
            # For non-Timestamp indices, increment by 1
            next_timestamp = last_timestamp + 1

        # Add the predicted value to the 'predicted' column for the next timestamp
        engineered_data['predicted'] = np.nan  # Initialize the 'predicted' column with NaNs
        engineered_data.at[next_timestamp, 'predicted'] = predicted_value[0]  # Add the prediction

        return engineered_data['predicted']

    except Exception as e:
        logger.error(f"Prediction service failed: {e}")
        raise



def post_processing(data: pd.DataFrame, config: dict) -> pd.DataFrame:

    try:
        # 1. Ensure both 'predicted' and 'close' columns are present
        if 'predicted' not in data.columns or 'close' not in data.columns:
            logger.error("Post-processing failed due to missing 'predicted' or 'close' columns")
            raise KeyError("Required columns 'predicted' or 'close' are missing for post-processing")

        # 2. Define a tolerance for deciding to hold from the config file
        tolerance = config['post_processing'].get('tolerance', 0.001)  # Default to 0.001 if not specified


        # 3. Handle missing values using backward fill and forward fill
        data['predicted'].fillna(method='bfill', inplace=True)
        data['predicted'].fillna(method='ffill', inplace=True)
        data['close'].fillna(method='bfill', inplace=True)
        data['close'].fillna(method='ffill', inplace=True)

        # 3. Initialize a 'decision' column with 'hold' as the default
        data['decision'] = 'hold'

        # 4. Buy if the predicted price is higher than the actual price by more than the tolerance
        data.loc[data['predicted'] > data['close'] + tolerance, 'decision'] = 'buy'

        # 5. Sell if the predicted price is lower than the actual price by more than the tolerance
        data.loc[data['predicted'] < data['close'] - tolerance, 'decision'] = 'sell'

        # 6. Log the completion of post-processing
        logger.info("Post-processing completed. Decisions made based on predicted vs actual prices.")

        # 7. Return the DataFrame with the new 'decision' column
        return data[['close', 'predicted', 'decision']]

    except KeyError as e:
        # 8. Log and raise KeyError if required columns are missing
        logger.error(f"Post-processing failed due to missing column(s): {e}")
        raise KeyError("Required columns are missing for post-processing") from e
    except Exception as e:
        # 9. Log and raise any unexpected errors
        logger.error(f"Post-processing encountered an unexpected error: {e}")
        raise



import bentoml
import pandas as pd
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def load_latest_model(model_name: str):
    try:
        # Get the latest model from BentoML model store
        model_info = bentoml.models.get(model_name)
        model_runner = model_info.to_runner()
        model_runner.init_local()  # Initialize the model runner for local predictions
        logger.info(f"Loaded latest model: {model_info.tag}")
        return model_runner
    except Exception as e:
        logger.error(f"Failed to load the latest model: {e}")
        raise

def prepare_data_for_prediction(data: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Preprocess the data and create features
    data = data_preprocessing(data, config)
    data = create_features(data, config)
    return data

def make_predictions(model_runner, data: pd.DataFrame, config: dict) -> pd.DataFrame:
    try:
        # Load required columns from the config
        required_columns = config['prediction']['required_columns']

        # Forward-fill missing values in required columns
        data[required_columns] = data[required_columns].ffill()
        # Backward-fill missing values in required columns
        data[required_columns] = data[required_columns].bfill()

        # If there are still NaNs, fill them with 0 as a last resort
        data[required_columns] = data[required_columns].fillna(0)

        # Extract the features for prediction
        features = data[required_columns].values

        # Make predictions using the trained and deployed model
        predictions = model_runner.run(features)

        # Store the predictions in the DataFrame
        data['predicted'] = predictions

        logger.info("Predictions completed successfully.")
        return data

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

def process_current_data(current_data_with_feature: pd.DataFrame, config: dict) -> pd.DataFrame:
    try:
        # Load the latest model
        model_runner = load_latest_model("elasticnet_model")

        # Prepare the data for prediction
        prepared_data = prepare_data_for_prediction(current_data_with_feature, config)

        # Shift 'close' values to create the target variable and store as 'shifted_close'
        prepared_data['shifted_close'] = prepared_data['close'].shift(-1)

        # Make predictions and store them in the DataFrame
        data_with_predictions = make_predictions(model_runner, prepared_data, config)

        # Return the DataFrame with shifted_close and predictions
        return data_with_predictions

    except Exception as e:
        logger.error(f"Processing current data failed: {e}")
        raise


def automated_retraining(rmse: float, threshold: float = 0.1) -> bool:
    """
    Trigger automated retraining if RMSE exceeds a predefined threshold.
    """
    try:
        # 1. Check if the RMSE exceeds the threshold
        if rmse > threshold:
            # 2. Log that retraining is needed
            logger.info(f"RMSE of {rmse:.4f} exceeded threshold of {threshold:.4f}. Triggering retraining...")
            return True
        else:
            # 3. Log that no retraining is needed
            logger.info(f"RMSE of {rmse:.4f} is within acceptable limits. No retraining needed.")
            return False

    except Exception as e:
        # 4. Log and raise any unexpected errors
        logger.error(f"Automated retraining check encountered an error: {e}")
        raise

############################################################################################################

import logging
from scipy.stats import ks_2samp

def ks_test(reference_data, current_data, target_col):
    statistic, p_value = ks_2samp(reference_data[target_col], current_data[target_col])
    return {
        'ks_statistic': statistic,
        'p_value': p_value
    }

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib.gridspec import GridSpec

# Define colors
# Define colors
GREY = '#808080'
RED = '#FF0000'

def plot_example(ref: pd.Series, curr: pd.Series):
  fig = plt.figure(constrained_layout=True, figsize=(15,7))

  gs = GridSpec(2, 3, figure=fig)
  ax1 = fig.add_subplot(gs[0, :])
  ax2 = fig.add_subplot(gs[1, 0])
  ax3 = fig.add_subplot(gs[1, 1])
  ax4 = fig.add_subplot(gs[1, 2])

  # plot feature in time
  ref_points = int(np.round(150 * len(ref) /(len(ref) + len(curr))))
  curr_points = 150 - ref_points

  ref_in_time = [np.mean(x) for x in np.array_split(ref, ref_points)]
  curr_in_time = [np.mean(x) for x in np.array_split(curr, curr_points)]

  ax1.plot(range(ref_points), ref_in_time, color=GREY)
  ax1.plot(range(ref_points, ref_points + curr_points), curr_in_time, color=RED)

  # plot referense distr
  sns.histplot(ref, color=GREY, ax=ax2)
  # plot current distr
  sns.histplot(curr, color=RED, ax=ax3)
  # plot two distr
  sns.histplot(ref, color=GREY, ax=ax4)
  sns.histplot(curr, color=RED, ax=ax4)
  plt.show()


import pandas as pd
from evidently.metrics import ColumnDriftMetric
from evidently.report import Report

def evaluare_drift(ref: pd.Series, curr: pd.Series):
    report = Report(metrics=[
        ColumnDriftMetric(column_name="value", stattest="ks", stattest_threshold=0.05),
        ColumnDriftMetric(column_name="value", stattest="psi", stattest_threshold=0.1),
        ColumnDriftMetric(column_name="value", stattest="kl_div", stattest_threshold=0.1),
        ColumnDriftMetric(column_name="value", stattest="jensenshannon", stattest_threshold=0.1),
        ColumnDriftMetric(column_name="value", stattest="wasserstein", stattest_threshold=0.1)
    ])
    report.run(reference_data=pd.DataFrame({"value": ref}), current_data=pd.DataFrame({"value": curr}))
    results = report.as_dict()
  
    
    drift_report = pd.DataFrame(columns=['stat_test', 'drift_score', 'is_drifted'])
    for i, metric in enumerate(results['metrics']):
        stat_test_name = metric['result'].get('stattest_name', 'Unknown')
        drift_report.loc[i, 'stat_test'] = stat_test_name
        drift_report.loc[i, 'drift_score'] = metric['result']['drift_score']
        drift_report.loc[i, 'is_drifted'] = metric['result']['drift_detected']
    
    return drift_report

def run_pipeline(config):
    # 1. Fetch Data
    data = fetch_eurusd_data(config)
    
    # 2. Preprocess Data
    data = data_preprocessing(data, config)
    
    # 3. Feature Engineering
    training_data_with_features = create_features(data, config)
    
    # 4. Train Model
    data_with_prediction_shifted_close, model = train_model(training_data_with_features, config)
    
    # 10. Preparing Current Data
    current_data = pd.read_csv('current.csv')
    current_data_predict_shifted_close = process_current_data(current_data, config)
    
    # Drop rows with NaN values in 'shifted_close' and 'predicted' columns
    data_with_prediction_shifted_close_cleaned = data_with_prediction_shifted_close.dropna(subset=['shifted_close', 'predicted'])
    current_data_predict_shifted_close_cleaned = current_data_predict_shifted_close.dropna(subset=['shifted_close', 'predicted'])
    
    target_col = 'shifted_close'
    drift_report = evaluare_drift(data_with_prediction_shifted_close_cleaned[target_col], current_data_predict_shifted_close_cleaned[target_col])
    print(drift_report)

# Main function to run everything
if __name__ == "__main__":
    config = load_config("config.yaml")
    run_pipeline(config)



def run_pipeline(config):
    # 1. Fetch Data
    data = fetch_eurusd_data(config)
    
    # 2. Preprocess Data
    data = data_preprocessing(data, config)
    #print(data)
    
    # 3. Feature Engineering
    training_data_with_features = create_features(data, config)
    # print(training_data_with_features)

    
    # 4. Train Model
    data_with_prediction_shifted_close, model = train_model(training_data_with_features, config)

    
    # # 5. Evaluate Model
    # evaluation_results = evaluate_model(data_with_prediction_shifted_close, config)

    # # 6. Model Validation
    # model_validation(data_with_prediction_shifted_close, model, config)

    # # 7. Deploy Model
    # model_tag=model_deployment(model, config)

    # # 8. Prediction Service
    # prediction_service(model_tag, data, config)

    # # 9. post processing    
    # post_processed_data = post_processing(training_data_with_features, config)

    
    # 10. preparing current data
    current_data=pd.read_csv('current.csv')
 
    current_data_predict_shifted_close = process_current_data(current_data, config)
    # Drop rows with NaN values in 'shifted_close' and 'predicted' columns
    data_with_prediction_shifted_close_cleaned = data_with_prediction_shifted_close.dropna(subset=['shifted_close', 'predicted'])
    current_data_predict_shifted_close_cleaned = current_data_predict_shifted_close.dropna(subset=['shifted_close', 'predicted'])
    target_col='shifted_close'
    ks_test_result = ks_test(data_with_prediction_shifted_close_cleaned, current_data_predict_shifted_close_cleaned, target_col)

    if ks_test_result['p_value'] < 0.05:
        logging.info("Drift detected: The distributions are significantly different.")
    else:
        logging.info("No significant drift detected: The distributions are similar.")
 
    # 11. Drift Detection
    #suite.model_performance_and_Target_drift_analysis(current_data_predict_shifted_close,data_with_prediction_shifted_close)

    
    # 7. Deploy Model
    # automated_retraining(data, config)
    print(data_with_prediction_shifted_close[target_col])
    drift_report=evaluare_drift(data_with_prediction_shifted_close_cleaned[target_col], current_data_predict_shifted_close_cleaned[target_col])
    print(drift_report)
    plot_example(data_with_prediction_shifted_close_cleaned[target_col], current_data_predict_shifted_close_cleaned[target_col])


# Main function to run everything
if __name__ == "__main__":
    config = load_config("config.yaml")
    run_pipeline(config)