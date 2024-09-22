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
        nan_handling_method = config['preprocessing']['nan_handling']  # 'ffill' or 'bfill'

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

        # Handle missing values based on the config
        if nan_handling_method == 'ffill':
            logger.info("Applying forward fill for missing values")
            data.ffill(inplace=True)
        elif nan_handling_method == 'bfill':
            logger.info("Applying backward fill for missing values")
            data.bfill(inplace=True)
        else:
            logger.error(f"Unknown NaN handling method: {nan_handling_method}")
            raise ValueError(f"Unknown NaN handling method: {nan_handling_method}")

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

        # Prepare features (X) and target (y)
        X = data[required_columns].copy()
        X.ffill(inplace=True)
        X.fillna(0, inplace=True)  # Ensure no NaNs remain
        y = data['close'].shift(-1).values  # Predict the next time step's 'close' value

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
        logger.error(f"Model training failed due to missing column(s): {e}")
        raise KeyError("Required columns are missing for model training") from e
    except ValueError as e:
        logger.error(f"Model training encountered a data error: {e}")
        raise ValueError("Data error during model training") from e
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

        # Add more metrics as needed based on config
        # Example: if 'rmse' in metrics_to_calculate:
        #   rmse = np.sqrt(mse)
        #   evaluation_results['rmse'] = rmse
        #   logger.info(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

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



def model_deployment(model) -> str:
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
        save_location = bentoml.models.get("elasticnet_model").path
        logger.info(f"Model saved at: {save_location}")

        return model_tag

    except Exception as e:
        logger.error(f"Model deployment failed: {e}")
        raise


def run_pipeline(config):
    # 1. Fetch Data
    data = fetch_eurusd_data(config)
    
    # 2. Preprocess Data
    data = data_preprocessing(data, config)
    #print(data)
    
    # 3. Feature Engineering
    data = create_features(data, config)
    
    # 4. Train Model
    data, model = train_model(data, config)

    # 5. Evaluate Model
    evaluation_results = evaluate_model(data, config)

    # 6. Model Validation
    model_validation(data, model, config)

    # 7. Deploy Model
    model_deployment(model)

    
    # # 6. Drift Detection
    # detect_drift(data, config)
    
    # # 7. Deploy Model
    # deploy_model(data, config)

# Main function to run everything
if __name__ == "__main__":
    config = load_config("config.yaml")
    run_pipeline(config)