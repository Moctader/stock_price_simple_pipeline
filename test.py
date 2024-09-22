import requests
import pandas as pd
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Fetch data from API
def fetch_data(api_url, start_timestamp, end_timestamp):
    try:
        # Construct the complete API URL
        url = f"{api_url}&from={start_timestamp}&to={end_timestamp}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise exception for unsuccessful requests

        # Parse the JSON response
        data = response.json()
        df = pd.DataFrame(data)
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Data fetching failed: {e}")
        raise

# Save DataFrame to CSV
def save_to_csv(df, file_path):
    df.to_csv(file_path, index=False)
    logger.info(f"Data saved to {file_path}")

# Main function
def main():
    api_url = "https://eodhd.com/api/intraday/EURUSD.FOREX?api_token=658e841fa1a6f9.30546411&fmt=json&"
    start_date = "2024-01-01"
    end_date = "2024-05-07"

    # Convert dates to timestamps
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    df = fetch_data(api_url, start_timestamp, end_timestamp)
    save_to_csv(df, 'current.csv')
    return df

if __name__ == "__main__":
    df = main()
    print(df.head())  # Print the first few rows of the DataFrame to verify