import yfinance as yf
import pandas as pd

# Fetch historical stock data for Apple (AAPL)
apple_data = yf.Ticker('AAPL')

# Get the historical stock data, which includes stock splits
history_data = apple_data.history(period='max')

# Filter rows where a stock split occurred and the split ratio is >= 1
splits = history_data[history_data['Stock Splits'] >= 1]

# Define the number of days before and after the split to fetch
days_before_after = 1

# Initialize a list to store the results
split_surrounding_data_list = []

# Iterate over each split date
for split_date in splits.index:
    # Get the date range for the surrounding data
    start_date = split_date - pd.Timedelta(days=days_before_after)
    end_date = split_date + pd.Timedelta(days=days_before_after)
    
    # Fetch the data for the date range
    surrounding_data = history_data.loc[start_date:end_date]
    
    # Append the surrounding data to the list
    split_surrounding_data_list.append(surrounding_data)

# Concatenate all the DataFrames in the list into a single DataFrame
split_surrounding_data = pd.concat(split_surrounding_data_list)
# Select specific columns: Open, Close, and Stock Splits
selected_columns = split_surrounding_data[['Open','High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]

# Print the selected columns
print(selected_columns)
# Optionally, save the results to a CSV file
split_surrounding_data.to_csv('split_surrounding_data.csv')








