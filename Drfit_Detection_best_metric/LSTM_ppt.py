import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error

# Step 1: Download Apple stock data from Yahoo Finance
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Step 2: Preprocess the data
# Use the 'Close' price as the target
close_prices = data['Close'].values.reshape(-1, 1)

# Normalize the data (scale between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Function to create a dataset with a sliding window for time series forecasting
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Function to build, train, and predict using an LSTM model
def build_and_train_lstm(X, y):
    # Reshape the input to be 3D (samples, time steps, features) for LSTM
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    # Compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Train the LSTM model
    model.fit(X, y, epochs=20, batch_size=64, verbose=0)

    # Predict stock prices for the entire dataset
    predicted_stock_price = model.predict(X)
    
    return predicted_stock_price

# List of time steps to compare
time_steps = [30, 60, 90, 120, 150, 180]

# Create subplots for visual comparison
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
axes = axes.flatten()  # Flatten the 2D axes array

mae_values = []  # Store MAE values for each time step

# Iterate over each time step
for i, time_step in enumerate(time_steps):
    # Step 3: Create datasets for the current time step
    X, y = create_dataset(scaled_data, time_step)

    # Step 4: Train the LSTM model and get predictions
    predicted_stock_price = build_and_train_lstm(X, y)

    # Inverse transform to get the actual stock price predictions
    predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

    # Step 5: Calculate drift (difference between actual and predicted prices)
    actual_prices = close_prices[time_step:].flatten()
    predicted_prices = predicted_stock_price.flatten()
    drift = actual_prices - predicted_prices

    # Step 6: Calculate MAE for the current time step
    mae = mean_absolute_error(actual_prices, predicted_prices)
    mae_values.append(mae)

    # Create date range for the predicted data
    dates = data.index[time_step:]

    # Step 7: Plot actual vs predicted stock prices with drift highlighted
    axes[i].plot(data.index, data['Close'], label='Actual AAPL Stock Price', color='blue', linewidth=1)
    axes[i].plot(dates, predicted_stock_price, label='Predicted AAPL Stock Price', color='orange', linestyle='--', linewidth=1)

    # Fill the area for drift
    axes[i].fill_between(dates, predicted_prices, actual_prices, 
                         where=(actual_prices > predicted_prices), color='green', alpha=0.3, 
                         label='Positive Drift (Actual > Predicted)')
    axes[i].fill_between(dates, predicted_prices, actual_prices, 
                         where=(actual_prices < predicted_prices), color='red', alpha=0.3, 
                         label='Negative Drift (Actual < Predicted)')

    # Set title and MAE annotation
    axes[i].set_title(f'Timestep: {time_step} | MAE: {mae:.4f}', fontsize=12)
    axes[i].set_xlabel('Date', fontsize=10)
    axes[i].set_ylabel('Stock Price (USD)', fontsize=10)
    axes[i].legend(loc='upper left')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plot
plt.show()

# Step 8: Print out the MAE values for each time step
for time_step, mae in zip(time_steps, mae_values):
    print(f"Time Step: {time_step} | MAE: {mae:.4f}")
