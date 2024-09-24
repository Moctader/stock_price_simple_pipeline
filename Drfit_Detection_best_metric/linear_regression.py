import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Download Apple stock data from Yahoo Finance
data = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Prepare the data for linear regression
# Convert the date to ordinal for linear regression
data['Date_Ordinal'] = data.index.map(pd.Timestamp.toordinal)
X = data['Date_Ordinal'].values.reshape(-1, 1)  # Feature (date)
y = data['Close'].values  # Target (closing price)

# Step 2: Train a linear regression model on the entire dataset
model = LinearRegression()
model.fit(X, y)

# Step 3: Predict stock prices for the entire dataset
y_pred = model.predict(X)

# Step 4: Calculate the drift (difference between actual and predicted prices)
drift = y - y_pred

# Step 5: Plot everything in a single plot
plt.figure(figsize=(12, 8))

# Plot the actual and predicted prices
plt.plot(data.index, data['Close'], label='Actual AAPL Stock Price', color='blue', linewidth=2)
plt.plot(data.index, y_pred, label='Predicted AAPL Stock Price (Linear Regression)', color='orange', linestyle='--', linewidth=2)

# Fill the area for drift
plt.fill_between(data.index, y_pred, y, where=(y > y_pred), color='green', alpha=0.3, interpolate=True, label='Positive Drift (Actual > Predicted)')
plt.fill_between(data.index, y_pred, y, where=(y < y_pred), color='red', alpha=0.3, interpolate=True, label='Negative Drift (Actual < Predicted)')

# Add titles and labels
plt.title('AAPL Stock Prices: Actual vs Predicted with Drift (2020-2023)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Stock Price (USD)', fontsize=12)

# Show legend
plt.legend()

# Show plot
plt.show()
