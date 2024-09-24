import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Step 1: Download Apple stock data from Yahoo Finance
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Step 2: Preprocess the data
# Use the 'Close' price as the target
close_prices = data['Close'].values.reshape(-1, 1)

# Normalize the data (scale between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Step 3: Create a dataset with a sliding window for time series forecasting
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Use a 60-day window to predict the next day's stock price
time_step = 60
X, y = create_dataset(scaled_data, time_step)

# Reshape the input to be 3D (samples, time steps, features) for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Step 4: Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Step 5: Train the LSTM model
model.fit(X, y, epochs=20, batch_size=64, verbose=1)

# Step 6: Predict stock prices for the entire dataset
predicted_stock_price = model.predict(X)

# Inverse transform to get the actual stock price predictions
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

# Step 7: Calculate the drift (difference between actual and predicted prices)
drift = close_prices[time_step:] - predicted_stock_price

# Create a date range for the predicted data
dates = data.index[time_step:]

# Step 8: Plot the actual vs predicted stock prices and drift
plt.figure(figsize=(12, 8))

# Plot actual and predicted prices
plt.plot(data.index, data['Close'], label='Actual AAPL Stock Price', color='blue', linewidth=2)
plt.plot(dates, predicted_stock_price, label='Predicted AAPL Stock Price (LSTM)', color='orange', linestyle='--', linewidth=2)

# Fill the area for drift
mask = close_prices[time_step:] > predicted_stock_price
idx = np.nonzero(mask[:-1] != mask[1:])[0]

plt.fill_between(dates, predicted_stock_price.flatten(), close_prices[time_step:].flatten(), 
                 where=mask.flatten(), color='green', alpha=0.3, 
                 label='Positive Drift (Actual > Predicted)')
plt.fill_between(dates, predicted_stock_price.flatten(), close_prices[time_step:].flatten(), 
                 where=~mask.flatten(), color='red', alpha=0.3, 
                 label='Negative Drift (Actual < Predicted)')

# Add titles and labels
plt.title('AAPL Stock Prices: Actual vs Predicted with Drift (LSTM)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Stock Price (USD)', fontsize=12)

# Show legend
plt.legend()

# Show plot
plt.show()