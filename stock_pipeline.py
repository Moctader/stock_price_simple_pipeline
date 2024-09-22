# # Install necessary libraries in Colab
# !pip install "zenml[server]" "bentoml" "fastapi" "uvicorn" scikit-learn
# !pip install requests pandas pyparsing==2.4.2 matplotlib

# # Optional: Clean up any existing ZenML configuration to ensure a fresh start
# !rm -rf .zen

# # Initialize ZenML
# !zenml init

# Restart the kernel to apply changes
# import IPython
# IPython.Application.instance().kernel.do_shutdown(restart=True)
from zenml import pipeline, step
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
import logging
import bentoml
import matplotlib.pyplot as plt

# Setup logging for debugging and tracking pipeline steps
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define an acceptable threshold for MSE
MSE_THRESHOLD = 1e-4
# Step 1: Data Ingestion - Fetch prices for EURUSD
@step
def data_ingestion() -> pd.DataFrame:
    """
    Fetch EURUSD prices from an external API and return as a DataFrame.
    Handles connection timeouts, empty responses, and malformed data.
    """
    try:
        api_url = 'https://eodhd.com/api/intraday/EURUSD.FOREX?api_token=658e841fa1a6f9.30546411&fmt=json&from=1683615062&to=1726815062'

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

    except requests.Timeout:
        logger.error("Data ingestion failed due to a timeout")
        raise TimeoutError("The API request timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"Data ingestion failed due to network issue: {e}")
        raise RuntimeError("Failed to ingest data from the API") from e
    except Exception as e:
        logger.error(f"Data ingestion encountered an unexpected error: {e}")
        raise


    # Step 2: Data Preprocessing -> Check for NaN and handle missing values for all relevant columns
@step
def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the raw data by handling missing values and setting a datetime index.
    Ensures that all relevant columns are passed forward for further processing.
    """
    try:
        # Ensure essential columns are present (datetime, close, open, high, low, volume)
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            logger.error(f"Required columns are missing: {missing_columns}")
            raise KeyError(f"Missing columns: {missing_columns}")

        # Convert 'datetime' to pandas datetime format, coerce invalid entries to NaT
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')

        # Drop rows where 'datetime' is NaT (invalid dates)
        if data['datetime'].isnull().any():
            logger.warning(f"Dropping {data['datetime'].isnull().sum()} rows with invalid datetime values")
            data.dropna(subset=['datetime'], inplace=True)

        # Set 'datetime' as the DataFrame index
        data.set_index('datetime', inplace=True)

        # Handle missing values for all columns, using ffill (forward fill)
        data.ffill(inplace=True)

        # Corner case: Ensure no NaNs remain in any of the essential columns
        required_non_null_columns = ['open', 'high', 'low', 'close']
        remaining_nans = data[required_non_null_columns].isnull().sum().sum()

        if remaining_nans > 0:
            logger.error("Data preprocessing failed: Some columns still contain NaN values after filling")
            raise ValueError(f"NaN values remain after forward filling: {remaining_nans}")

        # Log success and pass the entire relevant dataset forward for further feature engineering
        logger.info("Data preprocessing completed successfully")
        return data[required_columns].copy()

    except KeyError as e:
        logger.error(f"Data preprocessing failed due to missing columns: {e}")
        raise KeyError("Required columns are missing from the data") from e
    except Exception as e:
        logger.error(f"Data preprocessing encountered an unexpected error: {e}")
        raise


# Step 3: Feature Engineering -> Create a MA3 feature (3-period moving average) and pass other columns
@step
def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a 3-period moving average (MA3) for the 'close' prices.
    Retain all relevant columns for future use in model training and analysis.
    Handle missing values created by the rolling window.
    """
    try:
        # Ensure we have enough data for a 3-period rolling window
        if len(data) < 3:
            logger.error("Feature engineering failed: Not enough data points for 3-period MA")
            raise ValueError("Not enough data points for 3-period MA feature")

        # Calculate the 3-period moving average for 'close' prices
        data['MA3'] = data['close'].rolling(window=3).mean()

        # Log the number of NaN values generated by rolling window
        num_nans = data['MA3'].isna().sum()
        logger.info(f"Rolling window created {num_nans} NaN values")

        # Forward-fill or back-fill to handle NaN values caused by rolling
        data['MA3'].ffill(inplace=True)  # Forward fill the NaN values

        # Fallback: If forward-fill still results in all NaNs, apply a fallback strategy
        if data['MA3'].isnull().all():
            logger.warning("All values in MA3 are still NaN after forward-filling. Applying fallback strategy.")

            # Fallback strategy: Fill with mean or a predefined constant value (e.g., 0 or a small default)
            if len(data['close'].dropna()) > 0:
                fallback_value = data['close'].mean()
                data['MA3'].fillna(fallback_value, inplace=True)
                logger.info(f"Filled NaN values in MA3 with mean: {fallback_value:.2f}")
            else:
                # In case 'close' column itself has no valid values
                fallback_value = 0  # Use another suitable fallback depending on your context
                data['MA3'].fillna(fallback_value, inplace=True)
                logger.info(f"Filled NaN values in MA3 with fallback value: {fallback_value:.2f}")

        # Ensure the DataFrame still contains data after the transformation
        if data.empty:
            logger.error("Feature engineering failed: No rows remaining after forward-filling the MA3 values")
            raise ValueError("No rows remaining after forward-filling the 3-period moving average")

        # Log success and return the DataFrame with all relevant columns + MA3
        logger.info("Feature engineering completed successfully with MA3 feature added")
        return data

    except KeyError as e:
        logger.error(f"Feature engineering failed due to missing column: {e}")
        raise KeyError("Required column 'close' is missing from the data") from e
    except ValueError as e:
        logger.error(f"Feature engineering failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Feature engineering encountered an unexpected error: {e}")
        raise



# Step 4: Model Training -> Train a model using relevant features and GridSearchCV for hyperparameter tuning
@step
def model_training(data: pd.DataFrame) -> tuple[pd.DataFrame, ElasticNet]:
    """
    Train a model using relevant features like open, high, low, close, volume, and MA3.
    Use a time-based split to avoid leakage, and perform hyperparameter tuning with GridSearchCV.
    Returns the updated data with predicted values and the trained ElasticNet model.
    """
    try:
        # Ensure that all required columns are present for training
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'MA3']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            logger.error(f"Model training failed due to missing columns: {missing_columns}")
            raise KeyError(f"Missing columns: {missing_columns}")

        # Prepare the features (X) and target (y)
        X = data[required_columns].copy()

        # Fill missing values in the features (forward-fill missing data, then fill remaining NaNs with 0)
        X.ffill(inplace=True)
        X.fillna(0, inplace=True)  # Ensuring no NaNs are left

        y = data['close'].shift(-1).values  # Shift 'close' to predict x(t+1) at time step t

        # Remove the last row from X and y, as y[-1] will be NaN after shifting
        X = X[:-1]
        y = y[:-1]

        # Handle any remaining NaNs in y
        if np.isnan(y).any():
            logger.error(f"Target variable 'y' contains NaN values after shifting.")
            raise ValueError("Target variable 'y' contains NaN values.")

        # Time-based train-test split (80% training, 20% testing)
        split_index = int(len(X) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Define the parameter grid for GridSearchCV
        param_grid = {
            'alpha': [0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9]
        }

        # Define the model with GridSearchCV
        grid_search = GridSearchCV(ElasticNet(max_iter=10000), param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

        # Train the model with GridSearchCV
        grid_search.fit(X_train, y_train)

        # Get the best estimator (ElasticNet) from GridSearchCV
        best_model = grid_search.best_estimator_

        # Predict for the training and test sets using the best model
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        # Evaluate the model
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # Log model training metrics
        logger.info(f"Model training completed. Train MSE: {mse_train:.4f}, Test MSE: {mse_test:.4f}")
        logger.info(f"Model training completed. Train R²: {r2_train:.4f}, Test R²: {r2_test:.4f}")

        # Add predictions to the DataFrame
        data['predicted'] = np.nan  # Initialize the 'predicted' column with NaNs
        data.loc[data.index[:-1], 'predicted'] = best_model.predict(X)  # Set the predictions for all but the last row

        # Return the modified data and the best ElasticNet model (NOT the GridSearchCV object)
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



# Step 5: Model Evaluation -> Evaluate the model using multiple metrics (MSE, R²)
@step
def model_evaluation(data: pd.DataFrame) -> dict:
    """
    Evaluate the trained model by calculating various metrics between
    predicted and actual 'close' prices. Returns a dictionary of evaluation metrics.
    """
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

        # Calculate evaluation metrics
        mse = mean_squared_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        mae = mean_absolute_error(actual, predicted)

        # Log the evaluation success and metrics
        logger.info(f"Model evaluation completed. MSE: {mse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")

        # Return all metrics in a dictionary
        return {"mse": mse, "r2": r2, "mae": mae}

    except KeyError as e:
        logger.error(f"Model evaluation failed due to missing column(s): {e}")
        raise KeyError("Required columns are missing for model evaluation") from e
    except ValueError as e:
        logger.error(f"Model evaluation failed due to data inconsistency: {e}")
        raise ValueError(f"Data inconsistency during model evaluation: {e}") from e
    except Exception as e:
        logger.error(f"Model evaluation encountered an unexpected error: {e}")
        raise



# Step 6: Model Validation -> Validate the model using a time-based split
@step
def model_validation(data: pd.DataFrame, model: ElasticNet) -> dict:
    """
    Validate the trained ElasticNet model using a time-based split for validation.
    Return a dictionary with MSE, R², and MAE for the validation set.
    The model is an ElasticNet model, already trained with the best parameters.
    """
    try:
        # Prepare the features (X) and target (y) for validation
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'MA3']
        X = data[required_columns].fillna(0).values
        y = data['close'].shift(-1).values  # Shift 'close' to predict x(t+1) at time step t

        # Remove the last row from X and y, as y[-1] will be NaN after shifting
        X = X[:-1]
        y = y[:-1]

        # Time-based train-test split (80% training, 20% validation)
        split_index = int(len(X) * 0.8)
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



# Step 7: Model Deployment/Serving -> Deploy the best trained model using BentoML
@step
def model_deployment(model: ElasticNet) -> str:
    """
    Deploy the best trained model using BentoML.
    Save the ElasticNet model and return the model tag for serving.
    """
    try:
        # Save the ElasticNet model using BentoML
        bento_svc = bentoml.sklearn.save_model("forex_elasticnet_model", model)
        model_tag = str(bento_svc.tag)

        # Log the deployment success
        logger.info(f"ElasticNet model deployed successfully with BentoML. Model tag: {model_tag}")
        return model_tag

    except Exception as e:
        logger.error(f"Model deployment failed: {e}")
        raise


# Step 8: Prediction Service -> Use the best-trained model to predict the next time step's 'close' price
@step
def prediction_service(model_tag: str, engineered_data: pd.DataFrame) -> pd.Series:
    """
    Use the deployed model to predict the next 'close' price using the full feature set
    (open, high, low, close, volume, MA3). The prediction is based on the latest available data.
    """
    try:
        # Load the best model from the BentoML model store using the model tag
        model_runner = bentoml.sklearn.get(model_tag).to_runner()
        model_runner.init_local()  # For debugging and local testing only

        # Ensure there are no NaN values in the features, handle missing data properly
        required_columns = ['open', 'high', 'low', 'close', 'volume', 'MA3']

        # Forward-fill missing values in required columns
        engineered_data[required_columns] = engineered_data[required_columns].ffill()

        # If there are still NaNs, fill them with 0 as a last resort
        engineered_data[required_columns] = engineered_data[required_columns].fillna(0)

        # Extract the most recent row of features (this will be used for prediction)
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




# Step 9: Drift Detection -> Compare predicted vs. actual values over time to detect model drift
@step
def drift_detection(data: pd.DataFrame) -> dict:
    """
    Detect model drift by comparing predicted 'close' prices with actual 'close' prices.
    Return multiple metrics (RMSE, MAE, and R²) to assess drift.
    """
    try:
        # Step 1: Ensure both 'predicted' and 'close' columns are present
        required_columns = ['predicted', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Drift detection failed due to missing columns: {missing_columns}")
            raise KeyError(f"Required columns {missing_columns} are missing for drift detection")

        # Step 2: Clean the predicted and actual 'close' columns, remove NaN values
        predicted_cleaned = data['predicted'].dropna()
        actual_cleaned = data['close'].iloc[1:].dropna()  # Align with predictions (shifted by one step)

        # Step 3: Ensure there are enough data points to perform drift detection
        if len(predicted_cleaned) < 2:
            logger.error("Drift detection failed: Not enough data points for drift detection")
            raise ValueError("Not enough data points for drift detection")

        # Step 4: Ensure predicted and actual values have matching lengths
        if len(predicted_cleaned) != len(actual_cleaned):
            logger.error("Drift detection failed: Mismatch between predicted and actual values")
            raise ValueError("Mismatch between predicted and actual values for drift detection")

        # Step 5: Calculate drift detection metrics
        mse = mean_squared_error(actual_cleaned, predicted_cleaned)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_cleaned, predicted_cleaned)
        r2 = r2_score(actual_cleaned, predicted_cleaned)

        # Step 6: Log the drift detection results
        logger.info(f"Drift detection completed. RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")

        # Step 7: Return all metrics in a dictionary
        return {"rmse": rmse, "mae": mae, "r2": r2}

    except KeyError as e:
        # Handle missing columns error
        logger.error(f"Drift detection failed due to missing column(s): {e}")
        raise KeyError("Required columns are missing for drift detection") from e

    except ValueError as e:
        # Handle insufficient data points or mismatch in lengths
        logger.error(f"Drift detection failed: {e}")
        raise ValueError(f"Drift detection error: {e}") from e

    except Exception as e:
        # Catch any other unexpected exceptions
        logger.error(f"Drift detection encountered an unexpected error: {e}")
        raise



# Step 10: Monitoring (Profit/Loss) -> Track the cumulative profit/loss based on predictions and decisions
@step
def monitor_profitability(data: pd.DataFrame) -> pd.DataFrame:
    """
    Monitor profit/loss based on predicted vs. actual 'close' prices.
    Track the cumulative profit or loss based on buy/sell decisions inferred from the model's predictions.
    """
    try:
        # 1. Ensure both 'predicted' and 'close' columns are present
        required_columns = ['predicted', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            logger.error(f"Profitability monitoring failed due to missing columns: {missing_columns}")
            raise KeyError(f"Required columns {missing_columns} are missing for monitoring")

        # 2. Initialize profit/loss and cumulative_profit_loss columns
        data['profit_loss'] = 0.0
        data['cumulative_profit_loss'] = 0.0

        # 3. Use vectorized operations for better performance
        # Decision logic: If predicted price is higher than actual, buy; if lower, sell
        data['decision'] = np.where(data['predicted'] > data['close'], 'buy', 'sell')

        # Calculate profit or loss based on the decision
        data['profit_loss'] = np.where(
            data['decision'].shift(1) == 'buy',
            data['close'] - data['close'].shift(1),  # Profit for buy decision
            np.where(
                data['decision'].shift(1) == 'sell',
                data['close'].shift(1) - data['close'],  # Profit for sell decision
                0  # No profit/loss if no valid decision
            )
        )

        # 4. Calculate cumulative profit/loss
        data['cumulative_profit_loss'] = data['profit_loss'].cumsum()

        # 5. Log the final cumulative profit/loss
        total_profit_loss = data['cumulative_profit_loss'].iloc[-1]
        logger.info(f"Monitoring completed with total cumulative profit/loss: {total_profit_loss}")

        # 6. Return the DataFrame with the new columns
        return data[['close', 'predicted', 'profit_loss', 'cumulative_profit_loss']]

    except KeyError as e:
        # 7. Log and raise KeyError if required columns are missing
        logger.error(f"Monitoring profitability failed due to missing column(s): {e}")
        raise KeyError("Required columns are missing for profitability monitoring") from e

    except Exception as e:
        # 8. Log and raise any unexpected errors
        logger.error(f"Monitoring profitability encountered an unexpected error: {e}")
        raise


# Step 11: Automated Retraining -> Trigger retraining if drift is detected
@step
def automated_retraining(rmse: float, threshold: float = 0.1) -> bool:
    """
    Trigger automated retraining if RMSE exceeds a predefined threshold.
    """
    try:
        # 1. Check if the RMSE exceeds the threshold
        print("------------------")
        print(type(rmse))
        print(type(threshold))  
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




# Step 12: Post-Processing -> Create decisions (buy/sell/hold) based on the predictions
@step
def post_processing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create buy/sell/hold decisions based on the difference between predicted and actual 'close' prices.
    The decisions are:
    - 'buy' if the predicted price is higher than the actual price (price expected to rise).
    - 'sell' if the predicted price is lower than the actual price (price expected to fall).
    - 'hold' if the predicted price is close to the actual price (within a specified tolerance).
    """
    try:
        # 1. Ensure both 'predicted' and 'close' columns are present
        if 'predicted' not in data.columns or 'close' not in data.columns:
            logger.error("Post-processing failed due to missing 'predicted' or 'close' columns")
            raise KeyError("Required columns 'predicted' or 'close' are missing for post-processing")

        # 2. Define a tolerance for deciding to hold (adjust as needed)
        tolerance = 0.001  # Example: 0.1% tolerance

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



# Define the complete pipeline following MLOps best practices
@pipeline(enable_cache=False)
def stock_pipeline():
    # Step 1: Data Ingestion
    data = data_ingestion()

    # Step 2: Data Preprocessing
    preprocessed_data = data_preprocessing(data)

    # Step 3: Feature Engineering
    engineered_data = feature_engineering(preprocessed_data)

    # Step 4: Model Training
    trained_data, trained_model = model_training(engineered_data)

    # Step 5: Model Evaluation
    mse = model_evaluation(trained_data)

    # Step 6: Model Validation
    validation_metrics = model_validation(trained_data, trained_model)

    # Step 7: Model Deployment
    model_tag = model_deployment(trained_model)

    # Step 8: Prediction Service
    # We predict using engineered data to maintain consistency
    prediction = prediction_service(model_tag, engineered_data)

    # Step 9: Post-Processing (Decision Making)
    # Post-processing works on trained data which includes the predictions
    post_processed_data = post_processing(trained_data)

    # Step 10: Drift Detection
    # Drift detection should happen on post-processed data to detect drift based on the decisions made
    drift_score = drift_detection(post_processed_data)

    # Step 11: Monitoring Profitability
    # Profitability monitoring based on post-processed data including decisions
    profit_data = monitor_profitability(post_processed_data)
    # Step 12: Automated Retraining (optional, based on drift detection)
    profit_data = monitor_profitability(post_processed_data)

    retrain_needed = automated_retraining(drift_score)

# Instantiate and run the pipeline
stock_pipeline_svc = stock_pipeline()

# Run the pipeline, manage if retraining is triggered
# stock_pipeline_svc.run(unlisted=True)
