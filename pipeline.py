import pandas as pd
import numpy as np
import yaml
import os
import requests
import logging

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



def run_pipeline(config):
    # 1. Fetch Data
    data = fetch_eurusd_data(config)
    
    # 2. Preprocess Data
    data = data_preprocessing(data, config)
    #print(data)
    
    # 3. Feature Engineering
    data = create_features(data, config)
    
    # # 4. Train Model
    # data = train_model(data, config)
    
    # # 5. Evaluate Model
    # data = evaluate_model(data, config)
    
    # # 6. Drift Detection
    # detect_drift(data, config)
    
    # # 7. Deploy Model
    # deploy_model(data, config)

# Main function to run everything
if __name__ == "__main__":
    config = load_config("config.yaml")
    run_pipeline(config)