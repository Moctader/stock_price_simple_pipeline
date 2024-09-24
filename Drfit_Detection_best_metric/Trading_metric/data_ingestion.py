import os
import pandas as pd
import json
import time
import asyncio
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dict_namespace import DICT_NAMESPACE   
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator




load_dotenv()

config = DICT_NAMESPACE({
    'api_url': os.getenv('API_URL'),
    'api_token': os.getenv('API_TOKEN'),
    'api_tickers': os.getenv('API_TICKERS'),
    'api_date_from': os.getenv('API_DATE_FROM'),
    'api_date_to': os.getenv('API_DATE_TO'),
})

async def fetch_data_from_api(symbol, start_date, end_date, is_real_data=True):
    try:
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        api_url = (
            f"{config.api_url}/{symbol}"
            f"?api_token={config.api_token}&fmt=json"
            f"&from={start_timestamp}&to={end_timestamp}"
        )

        response = await asyncio.to_thread(requests.get, api_url)
        response.raise_for_status()

        data = response.json()

        if not data:
            return pd.DataFrame()

        data = pd.DataFrame(data)

        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
            data = data.dropna(subset=['datetime'])
        else:
            return pd.DataFrame()


        return data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching real-time data for {symbol}: {e}")
        return pd.DataFrame()
   

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
        # Drop rows with invalid datetime
        data = data.dropna(subset=['datetime'])
        data.set_index('datetime', inplace=True)

        # Handle missing values using forward fill or time interpolation
        data.fillna(method='ffill', inplace=True)

        # Remove duplicates based on datetime
        data.drop_duplicates(inplace=True)

        # Normalize and rescale numerical columns using Min-Max Scaling
        columns_to_scale = ['open', 'high', 'low', 'close', 'volume']
        scaler = MinMaxScaler()
        data.loc[:, columns_to_scale] = scaler.fit_transform(
            data[columns_to_scale])

        # normalization: x - ema(x)
        for col in columns_to_scale:
            ema_col = data[col].ewm(span=12, adjust=False).mean()
            data[col] = data[col] - ema_col

        # Ensure timestamp is filled from datetime if missing
        data['timestamp'] = data.apply(
            lambda row: int(row.name.timestamp()) if pd.isna(row['timestamp']) else row['timestamp'], axis=1
        )
        data['timestamp'] = data['timestamp'].fillna(0).astype(int)
        data['volume'] = data['volume'].fillna(0).astype(int)

        return data.reset_index()
    except Exception as e:
        print(f"Error processing data: {e}")
        return pd.DataFrame()
    

def calculate_trading_metrics(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data['close'] = data['close'].replace([np.inf, -np.inf], np.nan)
        data['close'] = data['close'].fillna(
            method='ffill').fillna(method='bfill')
        data['returns'] = data['close'].pct_change().fillna(0)
        data['cumulative_return'] = (1 + data['returns']).cumprod()
        trading_days = 252
        data['annualized_return'] = data['cumulative_return'].pow(
            1 / trading_days) - 1
        data['annualized_volatility'] = data['returns'].rolling(
            window=trading_days).std() * np.sqrt(trading_days)
        data['sharpe_ratio'] = data['annualized_return'] / \
            data['annualized_volatility']
        data['cum_return'] = (1 + data['returns']).cumprod()
        data['cum_max'] = data['cum_return'].cummax()
        data['drawdown'] = data['cum_max'] - data['cum_return']
        data['max_drawdown'] = data['drawdown'].max()
        data['equally_weighted_return'] = data[['close']].mean(axis=1)
        return data
    except Exception as e:
        print(f"Error calculating trading metrics: {e}")
        return data

def compute_ema(data, span):
    try:
        return data.ewm(span=span).mean()
    except Exception as e:
        print(f"Error computing EMA for span {span}: {e}")
        return pd.Series()


def generate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data['EMA_10'] = compute_ema(data['close'], span=10)
        data['EMA_30'] = compute_ema(data['close'], span=30)
        indicator_bb = BollingerBands(
            close=data['close'], window=20, window_dev=2)
        data['bb_mavg'] = indicator_bb.bollinger_mavg()
        data['bb_hband'] = indicator_bb.bollinger_hband()
        data['bb_lband'] = indicator_bb.bollinger_lband()
        macd = MACD(close=data['close'])
        data['MACD'] = macd.macd()
        data['MACD_signal'] = macd.macd_signal()
        data['MACD_diff'] = macd.macd_diff()
        rsi = RSIIndicator(close=data['close'], window=14)
        data['RSI'] = rsi.rsi()
        return data
    except Exception as e:
        print(f"Error generating technical indicators: {e}")
        return data

def clean_data(data):
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    return data



if __name__ == "__main__":
    # Use asyncio.run to execute the coroutine
    data = asyncio.run(fetch_data_from_api('AAPL', datetime(2024, 1, 1), datetime(2024, 5, 7), is_real_data=True))
    process_data=process_data(data)
    clean_data=clean_data(process_data)
    # print(type(process_data))    
    trading_metrics=calculate_trading_metrics(data)
    generate_technical_indicators=generate_technical_indicators=generate_technical_indicators(clean_data)
    print(generate_technical_indicators.head())
   