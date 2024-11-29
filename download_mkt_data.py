import pandas as pd
from datetime import datetime
import concurrent.futures
import logging
import os
import requests
from time import sleep

# def get_binance_klines(symbol="BTCUSDT", interval="1s", limit=1000):
#     url = "https://api.binance.com/api/v3/klines"
#     params = {
#         "symbol": symbol,
#         "interval": interval,
#         "limit": limit
#     }
#     response = requests.get(url, params=params)
#     return response.json()

# def process_klines(klines):
#     # Extract relevant fields and create a DataFrame
#     df = pd.DataFrame(klines, columns=[
#         'open_time', 'open', 'high', 'low', 'close', 'volume', 
#         'close_time', 'quote_asset_volume', 'number_of_trades', 
#         'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
#     ])
#     df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
#     df.set_index('timestamp', inplace=True)
#     df = df[['open', 'high', 'low', 'close', 'volume']]
#     return df

# klines = get_binance_klines()
# df_klines = process_klines(klines)
# df_klines.to_csv('binance_klines.csv')  # Save DataFrame to a CSV file
# print(df_klines.head())

def interval_to_seconds(interval):
    interval_mapping = {
        '1s': 1,
        '1m': 60,
        '3m': 3 * 60,
        '5m': 5 * 60,
        '15m': 15 * 60,
        '30m': 30 * 60,
        '1h': 60 * 60,
        '2h': 2 * 60 * 60,
        '4h': 4 * 60 * 60,
        '6h': 6 * 60 * 60,
        '8h': 8 * 60 * 60,
        '12h': 12 * 60 * 60,
        '1d': 24 * 60 * 60,
        '3d': 3 * 24 * 60 * 60,
        '1w': 7 * 24 * 60 * 60,
        '1M': 30 * 24 * 60 * 60  # Assuming 30 days in a month
    }
    
    return interval_mapping.get(interval, None)

from utilities import fetch_binance_klines
from parameters import selected_params

# 80 market_data/ICX
# 73 market_data/TRX
# 61 market_data/ETC
# 49 market_data/NULS
# 45 market_data/ONT
# 36 market_data/USDC
# 29 market_data/VET
# 14 market_data/LINK

selected_params['limit'] = 200
selected_params['interval'] = '1d'
selected_params['symbols'] = [
    "LTC",
    "DOGE",
    "SHIB",
    "PEOPLE",
    "FLOKI",
    "PEPE",
    "MEME",
    "BONK",
    "WIF",
    "BOME"
] # 16 mars 2024
# selected_params['symbols'] = ["ICX", "TRX", "ETC", "NULS", "VET", "USDC", "LINK", "ONT"]
# selected_params['symbols'] = ['BTC', 'ETH', 'BNB', 'NEO', 'LTC', 'QTUM', 'ADA', 'XRP', 'EOS', 'TUSD', 'IOTA', 'XLM', 'ONT', 'TRX', 'ETC', 'ICX', 'NULS', 'VET', 'USDC', 'LINK']
selected_params['end_time'] = '2024-11-22'

def process_symbol(symbol):
    # Create a directory for the symbol if it doesn't exist
    symbol_dir = f'market_data/{symbol}'
    os.makedirs(symbol_dir, exist_ok=True)
    
    # Calculate the total time to subtract based on limit and interval
    interval_seconds = interval_to_seconds(selected_params['interval'])
    
    # Set current_end_time to the start of the day of the end_time
    last_day_time = pd.to_datetime(selected_params['end_time']).normalize()
    first_day_time = last_day_time - pd.Timedelta(days=selected_params['limit'])
    
    # Set start_time to the beginning of the day
    start_time = first_day_time
    
    logging.info(f'Processing {symbol} until {last_day_time}')
    
    try:
        for _ in range(selected_params['limit']):  # Loop for the number of limits
            # Set end_time to the end of the current day
            current_end_time = start_time + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            
            # Determine the format based on the interval
            if 'd' in selected_params['interval']:
                format = '%Y%m%d'
            elif 'h' in selected_params['interval']:
                format = '%Y%m%d-%H'
            elif 'm' in selected_params['interval']:
                format = '%Y%m%d-%H%M'
            else:
                format = '%Y%m%d-%H%M%S'  # Default to including seconds
            date_str = start_time.strftime(format)
            
            # Check if the file already exists
            file_path = f'market_data/{symbol}/{date_str}.csv'
            if os.path.exists(file_path):
                logging.info(f'Skipping {symbol} data for {date_str} as it already exists')
            else:
                try:
                    df_klines, start_time, end_time = fetch_binance_klines(
                        symbol, 
                        '1s', 
                        interval_seconds, 
                        current_end_time
                    )
                    
                    logging.info(f'Dumping {symbol} data for {date_str}')
                    df_klines.to_csv(file_path)  # Save DataFrame to a CSV file
                except Exception as e:
                    logging.error(f'Error fetching or saving data for {symbol} on {date_str}: {e}')
                    continue  # Skip to the next iteration
            
            # Move to the next day
            start_time += pd.Timedelta(days=1)
            
            # Debugging: Log the current_end_time and selected_params['end_time']
            logging.info(f'Current end time: {current_end_time}, Selected end time: {selected_params["end_time"]}')
            
            # Break the loop if the current_end_time exceeds the selected_params['end_time']
            if current_end_time > pd.to_datetime(selected_params['end_time']):
                logging.info(f'Stopping because {current_end_time} > {selected_params["end_time"]}')
                break
    except Exception as e:
        logging.error(f'Error processing symbol {symbol}: {e}')

# Use ThreadPoolExecutor to parallelize the process
with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(process_symbol, selected_params['symbols'])
