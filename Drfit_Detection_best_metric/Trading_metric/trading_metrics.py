import os
import pandas as pd
import numpy as np
import vectorbt as vbt
from dotenv import load_dotenv
from datetime import datetime
import requests
import asyncio
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

load_dotenv()

# Load environment variables
API_URL = os.getenv('API_URL')
API_TOKEN = os.getenv('API_TOKEN')

async def fetch_data_from_api(symbol, start_date, end_date):
    try:
        start_timestamp = int(start_date.timestamp())
        end_timestamp = int(end_date.timestamp())

        api_url = f"{API_URL}/{symbol}?api_token={API_TOKEN}&from={start_timestamp}&to={end_timestamp}&fmt=json"

        response = await asyncio.to_thread(requests.get, api_url)
        response.raise_for_status()

        data = response.json()

        if not data:
            return pd.DataFrame()

        data = pd.DataFrame(data)

        if 'datetime' in data.columns:
            data['datetime'] = pd.to_datetime(data['datetime'], errors='coerce')
            data = data.dropna(subset=['datetime'])
            return data.set_index('datetime')

        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    try:
        data.dropna(inplace=True)
        columns_to_scale = ['open', 'high', 'low', 'close', 'volume']

        scaler = MinMaxScaler()
        data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

        return data
    except Exception as e:
        print(f"Error processing data: {e}")
        return pd.DataFrame()


def generate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    try:
        close_price = data['close']

        # Use vectorbt to calculate technical indicators
        ema_10 = vbt.MA.run(close_price, window=10, short_name='EMA_10')
        ema_30 = vbt.MA.run(close_price, window=30, short_name='EMA_30')
        rsi = vbt.RSI.run(close_price, window=14)
        macd = vbt.MACD.run(close_price)

        # Combine all indicators into the dataframe
        data['EMA_10'] = ema_10.ma
        data['EMA_30'] = ema_30.ma
        data['RSI'] = rsi.rsi
        data['MACD'] = macd.macd
        data['MACD_signal'] = macd.signal

        return data
    except Exception as e:
        print(f"Error generating technical indicators: {e}")
        return pd.DataFrame()


def run_backtest(data):
    close = data['close']

    if close.isna().any():
        print("NaN values found in close prices. Filling NaN values with the previous value.")
        close = close.fillna(method='ffill').fillna(method='bfill')
    
    close = close[close > 0]
    close = close.astype(float)
    short_ma = vbt.MA.run(close, window=10)
    long_ma = vbt.MA.run(close, window=50)
    rsi = vbt.RSI.run(close, window=14)
    macd = vbt.MACD.run(close)
    entries = (short_ma.ma_crossed_above(long_ma)) 
    exits = (short_ma.ma_crossed_below(long_ma)) 

    pf = vbt.Portfolio.from_signals(close, entries, exits, init_cash=100, freq='D')
    return pf, short_ma, long_ma, entries, exits, close, rsi, macd


# Calculate trading metrics using vectorbt
def calculate_trading_metrics(data: pd.DataFrame) -> pd.DataFrame:
    try:
        pf, short_ma, long_ma, entries, exits, close, rsi, macd = run_backtest(data)

        returns = pf.returns()
        metrics = {
            'total_return': pf.total_return(),
            'cumulative_return': pf.total_return(),
            'max_drawdown': pf.drawdowns.max_drawdown(),
            'volatility': returns.std(),
            'sharpe_ratio': pf.sharpe_ratio(),
            'sortino_ratio': pf.sortino_ratio(),
        }

        return pd.DataFrame([metrics])
    except Exception as e:
        print(f"Error calculating trading metrics: {e}")
        return pd.DataFrame()
    
    

if __name__ == "__main__":
    symbol = 'AAPL'
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 5, 7)

    data = asyncio.run(fetch_data_from_api(symbol, start_date, end_date))
    data = process_data(data)
    data_with_indicators = generate_technical_indicators(data)
    metrics = calculate_trading_metrics(data_with_indicators)
    print(metrics)
    pf, short_ma, long_ma, entries, exits, close, rsi, macd = run_backtest(data_with_indicators)
    
    
    # Plot the backtest results using Plotly
    fig = pf.plot()
    fig.add_trace(go.Scatter(x=short_ma.ma.index, y=short_ma.ma, mode='lines', name='Short MA'))
    fig.add_trace(go.Scatter(x=long_ma.ma.index, y=long_ma.ma, mode='lines', name='Long MA'))
    fig.add_trace(go.Scatter(x=entries.index, y=close[entries], mode='markers', name='Entries', marker=dict(color='green', symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=exits.index, y=close[exits], mode='markers', name='Exits', marker=dict(color='red', symbol='triangle-down')))

    # Add RSI to the plot
    fig.add_trace(go.Scatter(x=rsi.rsi.index, y=rsi.rsi, mode='lines', name='RSI'))
    
    # Add MACD to the plot
    fig.add_trace(go.Scatter(x=macd.macd.index, y=macd.macd, mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=macd.signal.index, y=macd.signal, mode='lines', name='MACD Signal'))
    
    fig.show()