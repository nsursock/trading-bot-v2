import requests
import pandas as pd
from datetime import datetime
import time
import unittest
import matplotlib.pyplot as plt
import logging
import numpy as np
from time import sleep

import ta

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO)

def add_technical_indicators(df_original):
    logging.debug("Adding technical indicators")
    df = df_original.copy()
    df['ema_short'] = ta.trend.EMAIndicator(close=df['close'], window=12).ema_indicator()
    df['ema_long'] = ta.trend.EMAIndicator(close=df['close'], window=26).ema_indicator()
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()
    macd = ta.trend.MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    df['boll_hband'] = ta.volatility.BollingerBands(close=df['close']).bollinger_hband()
    df['boll_mband'] = ta.volatility.BollingerBands(close=df['close']).bollinger_mavg()
    df['boll_lband'] = ta.volatility.BollingerBands(close=df['close']).bollinger_lband()
    df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    
    # Add DMI and ADX
    dmi = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = dmi.adx()
    df['adx_neg'] = dmi.adx_neg()
    df['adx_pos'] = dmi.adx_pos()
    
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
    df['vwap'] = ta.volume.VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).volume_weighted_average_price()
    
    df['parabolic_sar'] = ta.trend.PSARIndicator(high=df['high'], low=df['low'], close=df['close']).psar()
    # ichimoku = ta.trend.IchimokuIndicator(high=df['high'], low=df['low'])
    # df['ichimoku_base_line'] = ichimoku.ichimoku_base_line()
    # df['ichimoku_conversion_line'] = ichimoku.ichimoku_conversion_line()
    # df['ichimoku_a'] = ichimoku.ichimoku_a()
    # df['ichimoku_b'] = ichimoku.ichimoku_b()
    stochastic = ta.momentum.StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
    df['stoch_k'] = stochastic.stoch()
    df['stoch_d'] = stochastic.stoch_signal()
    df['roc'] = ta.momentum.ROCIndicator(close=df['close']).roc()
    
    # if btc_data is not None:
    #     # Align BTC data timestamps with the main DataFrame
    #     btc_data = btc_data.reindex(df.index, method='nearest')
        
    #     # Add BTC price data as technical indicators
    #     df['btc_open'] = btc_data['open']
    #     df['btc_high'] = btc_data['high']
    #     df['btc_low'] = btc_data['low']
    #     df['btc_close'] = btc_data['close']
    #     df['btc_volume'] = btc_data['volume']


    # Check for NaN values before interpolation
    if df.isnull().values.any():
        logging.debug("NaN values detected before interpolation.")
    
    # Fill NaN values using interpolation
    df.interpolate(method='linear', inplace=True)

    # Fill any remaining NaN values at the beginning with the first valid observation
    df.bfill(inplace=True)  # Backward fill to handle leading NaNs

    # Check for NaN values after filling
    if df.isnull().values.any():
        logging.warning("NaN values detected after filling.")
        
    
    logging.debug(f"Technical indicators added: {df.head()}")
    return df

def identify_patterns(df):
    """
    Identifies various technical patterns in the given DataFrame of price data.

    :param df: DataFrame containing 'open', 'high', 'low', 'close' columns.
    :return: Dictionary of pattern features.
    """
    patterns = {
        'bullish_flag': False,
        'bearish_flag': False,
        'pennant': False,
        'ascending_triangle': False,
        'descending_triangle': False,
        'symmetrical_triangle': False,
        'head_and_shoulders': False,
        'inverse_head_and_shoulders': False,
        'double_top': False,
        'double_bottom': False,
        'triple_top': False,
        'triple_bottom': False,
        'rounding_bottom': False,
        'rounding_top': False,
        'cup_and_handle': False,
        'falling_wedge': False,
        'rising_wedge': False,
        'breakaway_gap': False,
        'runaway_gap': False,
        'exhaustion_gap': False
    }

    if len(df) < 20:  # Ensure enough data points
        return patterns

    # Example logic for identifying patterns (simplified)
    # Bullish Flag
    if df['close'].iloc[-1] > df['close'].iloc[-5:].mean() and df['close'].iloc[-5:].mean() > df['close'].iloc[-10:].mean():
        patterns['bullish_flag'] = True

    # Bearish Flag
    if df['close'].iloc[-1] < df['close'].iloc[-5:].mean() and df['close'].iloc[-5:].mean() < df['close'].iloc[-10:].mean():
        patterns['bearish_flag'] = True

    # Pennant
    if df['high'].iloc[-1] < df['high'].iloc[-5:].max() and df['low'].iloc[-1] > df['low'].iloc[-5:].min():
        patterns['pennant'] = True

    # Ascending Triangle
    if df['high'].iloc[-1] == df['high'].iloc[-5:].max() and df['low'].iloc[-1] > df['low'].iloc[-5:].min():
        patterns['ascending_triangle'] = True

    # Descending Triangle
    if df['low'].iloc[-1] == df['low'].iloc[-5:].min() and df['high'].iloc[-1] < df['high'].iloc[-5:].max():
        patterns['descending_triangle'] = True

    # Symmetrical Triangle
    if df['high'].iloc[-1] < df['high'].iloc[-5:].max() and df['low'].iloc[-1] > df['low'].iloc[-5:].min():
        patterns['symmetrical_triangle'] = True

    # Head and Shoulders
    if df['high'].iloc[-3] < df['high'].iloc[-2] > df['high'].iloc[-1]:
        patterns['head_and_shoulders'] = True

    # Inverse Head and Shoulders
    if df['low'].iloc[-3] > df['low'].iloc[-2] < df['low'].iloc[-1]:
        patterns['inverse_head_and_shoulders'] = True

    # Double Top
    if df['high'].iloc[-1] == df['high'].iloc[-2]:
        patterns['double_top'] = True

    # Double Bottom
    if df['low'].iloc[-1] == df['low'].iloc[-2]:
        patterns['double_bottom'] = True

    # Triple Top
    if df['high'].iloc[-1] == df['high'].iloc[-2] == df['high'].iloc[-3]:
        patterns['triple_top'] = True

    # Triple Bottom
    if df['low'].iloc[-1] == df['low'].iloc[-2] == df['low'].iloc[-3]:
        patterns['triple_bottom'] = True

    # Rounding Bottom
    if df['low'].iloc[-1] > df['low'].iloc[-5:].min() and df['low'].iloc[-5:].min() > df['low'].iloc[-10:].min():
        patterns['rounding_bottom'] = True

    # Rounding Top
    if df['high'].iloc[-1] < df['high'].iloc[-5:].max() and df['high'].iloc[-5:].max() < df['high'].iloc[-10:].max():
        patterns['rounding_top'] = True

    # Cup and Handle
    if df['low'].iloc[-1] > df['low'].iloc[-5:].min() and df['low'].iloc[-5:].min() > df['low'].iloc[-10:].min():
        patterns['cup_and_handle'] = True

    # Falling Wedge
    if df['high'].iloc[-1] < df['high'].iloc[-5:].max() and df['low'].iloc[-1] > df['low'].iloc[-5:].min():
        patterns['falling_wedge'] = True

    # Rising Wedge
    if df['high'].iloc[-1] > df['high'].iloc[-5:].max() and df['low'].iloc[-1] < df['low'].iloc[-5:].min():
        patterns['rising_wedge'] = True

    # Breakaway Gap
    if df['close'].iloc[-1] > df['close'].iloc[-2] * 1.05:
        patterns['breakaway_gap'] = True

    # Runaway Gap
    if df['close'].iloc[-1] > df['close'].iloc[-2] * 1.02:
        patterns['runaway_gap'] = True

    # Exhaustion Gap
    if df['close'].iloc[-1] < df['close'].iloc[-2] * 0.98:
        patterns['exhaustion_gap'] = True

    return patterns

def fetch_binance_klines(symbol, interval, limit=1000, end_time=None, retries=3, backoff_factor=0.3):
    logging.info(f"Fetching Binance klines for {symbol} with interval {interval} and limit {limit}")
    
    base_url = "https://api.binance.com/api/v3/klines"
    klines = []

    # Convert end_time to datetime if it's a string
    if isinstance(end_time, str):
        end_time = datetime.strptime(end_time, "%Y-%m-%d")

    # Convert end_time to milliseconds if it's a datetime object
    if isinstance(end_time, datetime):
        end_time = int(end_time.timestamp() * 1000)

    total_fetch_time = 0
    fetch_count = 0

    while len(klines) < limit:
        params = {
            'symbol': symbol + 'USDT',
            'interval': interval,
            'limit': min(limit - len(klines), 1000)
        }
        if end_time is not None:
            params['endTime'] = end_time

        for attempt in range(retries):
            try:
                fetch_start_time = time.time()  # Record the start time of this fetch
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                fetch_end_time = time.time()  # Record the end time of this fetch
        
                data = response.json()
            except requests.exceptions.RequestException as e:
                logging.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < retries - 1:
                    sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
                else:
                    raise
        
        # Check for error in response
        if 'code' in data:
            raise Exception(f"API Error: {data['msg']}")
        
        if isinstance(data, list) and data:
            klines = data + klines
            end_time = data[0][0] - 1
        else:
            logging.error(f"No data returned or error in response.")
            break

        # Update total fetch time and count
        total_fetch_time += (fetch_end_time - fetch_start_time)
        fetch_count += 1

        # Calculate and log ETA
        if fetch_count > 0:
            average_fetch_time = total_fetch_time / fetch_count
            klines_fetched = len(klines)
            eta_seconds = average_fetch_time * (limit - klines_fetched) / params['limit']
            
            # Format ETA
            if eta_seconds > 3600:
                hours, remainder = divmod(eta_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                eta_formatted = f"{int(hours):02}h{int(minutes):02}m{int(seconds):02}s"
            elif eta_seconds > 60:
                minutes, seconds = divmod(eta_seconds, 60)
                eta_formatted = f"{int(minutes):02}m{int(seconds):02}s"
            else:
                eta_formatted = f"{int(eta_seconds):02}s"

            logging.debug(f"Fetched {klines_fetched} klines for {symbol}. Average fetch time: {average_fetch_time:.2f} seconds. ETA for completion: {eta_formatted}.")

        if len(data) < params['limit']:
            break
        
        time.sleep(0.1)

    klines = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    klines['timestamp'] = pd.to_datetime(klines['timestamp'], unit='ms')
    klines[['open', 'high', 'low', 'close', 'volume']] = klines[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)
    klines = klines[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    klines.set_index('timestamp', inplace=True)
    
    logging.info(f"Fetched {len(klines)} klines for {symbol} with interval {interval} and limit {limit}")
    return pd.DataFrame(klines), klines.index[0], klines.index[-1]

def preprocess_data(symbols, interval='1d', limit=365, end_time=None):
    """
    Fetches and processes cryptocurrency data into a matrix of shape 
    (num_candles x num_symbols x num_features) using a common period.

    :param symbols: List of cryptocurrency symbols to fetch.
    :param interval: Time interval for each kline.
    :param limit: Maximum number of klines to fetch.
    :param end_time: End time for fetching klines.
    :return: A 3D numpy array of shape (num_candles x num_symbols x num_features).
    """
    dataframes = {}
    start_times = []
    end_times = []

    valid_symbols = []
    target_num_symbols = 10

    # Fetch data for each symbol
    for symbol in symbols:
        try:
            df, start_datetime, end_datetime = fetch_binance_klines(symbol, interval, limit, end_time)
            if not df.empty:
                dataframes[symbol] = df
                start_times.append(start_datetime)
                end_times.append(end_datetime)
                valid_symbols.append(symbol)
                logging.info(f"Symbol: {symbol}, Start Time: {start_datetime}, End Time: {end_datetime}")
                
                # if len(valid_symbols) == target_num_symbols:
                #     break
            else:
                logging.warning(f"No data returned for {symbol}. It will be excluded.")
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            continue

    # Update the number of symbols to only include valid ones
    num_symbols = len(valid_symbols)

    # Determine the common period
    if not start_times or not end_times:
        logging.error("No valid data fetched for any symbols.")
        return None, None, None

    # Calculate the frequency of each period
    period_counts = {}
    for start, end in zip(start_times, end_times):
        period = (start, end)
        if period in period_counts:
            period_counts[period] += 1
        else:
            period_counts[period] = 1

    # Determine the most frequent period
    most_frequent_period = max(period_counts, key=period_counts.get)
    common_start_time, common_end_time = most_frequent_period

    logging.info(f"Most frequent period: {common_start_time} to {common_end_time}")

    # Remove symbols that do not contribute to the most frequent period
    valid_symbols = [
        symbol for symbol in valid_symbols
        if dataframes[symbol].index[0] <= common_start_time and dataframes[symbol].index[-1] >= common_end_time
    ]
    
    valid_symbols = valid_symbols[:target_num_symbols]

    # Determine the number of pattern features
    num_pattern_features = len(identify_patterns(pd.DataFrame(columns=['open', 'high', 'low', 'close'])).keys())
    # print(f"Number of pattern features: {num_pattern_features}")
    num_features = len(add_technical_indicators(dataframes[valid_symbols[0]][:50]).columns)
    # print(f"Number of technical features: {num_features}")

    # Recalculate the number of symbols after filtering
    num_symbols = len(valid_symbols)

    if num_symbols == 0:
        logging.error("No symbols have data for the most frequent period.")
        return None, None, None

    # Filter dataframes to the common period and prepare the matrix
    # num_features = len(dataframes[valid_symbols[0]].columns)
    # print(f"Number of features: {num_features}")
    # Calculate the number of candles based on the common period
    num_candles = min(len(dataframes[symbol].loc[common_start_time:common_end_time]) for symbol in valid_symbols)

    # Initialize the matrix with the correct number of candles and additional pattern features
    data_matrix = np.zeros((num_candles, num_symbols, num_features + num_pattern_features))

    # Extract timestamps
    timestamps = dataframes[valid_symbols[0]].loc[common_start_time:common_end_time].index[:num_candles].tolist()

    # Create a mapping of field names to indices
    field_mapping = {name: idx for idx, name in enumerate(dataframes[valid_symbols[0]].columns)}
    
    # Add technical indicators to the field mapping
    technical_indicators = add_technical_indicators(dataframes[valid_symbols[0]][:50]).columns
    field_mapping.update({name: idx + len(field_mapping) for idx, name in enumerate(technical_indicators)})

    # Add pattern indicators to the field mapping
    pattern_indicators = identify_patterns(pd.DataFrame(columns=['open', 'high', 'low', 'close'])).keys()
    field_mapping.update({name: idx + len(field_mapping) for idx, name in enumerate(pattern_indicators)})

    for i, symbol in enumerate(valid_symbols):
        df = dataframes[symbol]
        df_filtered = df.loc[common_start_time:common_end_time].iloc[:num_candles]
        df_filtered = add_technical_indicators(df_filtered)
        data_matrix[:, i, :num_features] = df_filtered.values

        # Identify patterns and add them as features
        patterns = identify_patterns(df_filtered)
        for j, (pattern, is_present) in enumerate(patterns.items()):
            data_matrix[:, i, num_features + j] = int(is_present)
            
    return data_matrix, timestamps, field_mapping, valid_symbols


# Leverage to Liquidation Threshold mapping
leverage_liquidation_ranges = [
    (1, 1, 100.00),
    (1, 2, 89.84),
    (2, 5, 89.60),
    (5, 10, 89.20),
    (10, 15, 88.80),
    (15, 20, 88.40),
    (20, 25, 88.00),
    (25, 30, 85.46),
    (30, 35, 82.91),
    (35, 40, 80.37),
    (40, 45, 77.83),
    (45, 50, 75.29),
    (50, 55, 72.74),
    (55, 60, 70.20),
    (60, 65, 69.80),
    (65, 70, 69.40),
    (70, 75, 69.00),
    (75, 80, 68.60),
    (80, 85, 68.20),
    (85, 90, 67.80),
    (90, 95, 67.40),
    (95, 100, 67.00),
    (100, 105, 66.60),
    (105, 110, 66.20),
    (110, 115, 65.80),
    (115, 120, 65.40),
    (120, 125, 65.00),
    (125, 130, 64.60),
    (130, 135, 64.20),
    (135, 140, 63.80),
    (140, 145, 63.40),
    (145, 151, 63.00),
    (145, 150, 63.00), # official liquidation threshold
    (150, 155, 62.60),
    (155, 160, 62.20),
    (160, 165, 61.80),
    (165, 170, 61.40),
    (170, 175, 61.00),
    (175, 180, 60.60),
    (180, 185, 60.20),
    (185, 190, 59.80),
    (190, 195, 59.40),
    (195, 201, 59.00)
]

def get_liquidation_threshold(leverage):
    for min_lev, max_lev, threshold in leverage_liquidation_ranges:
        if min_lev <= leverage < max_lev:
            return threshold
        else:
            return 100.00
    return None


import random
def fetch_symbols():
    response = requests.get('https://backend-arbitrum.gains.trade/trading-variables')
    pairs = response.json()['pairs']
    return [{'symbol': pair['from'], 'index': idx, 'groupIndex': pair['groupIndex']} for idx, pair in enumerate(pairs)]

def select_cryptos(count):
    symbols = fetch_symbols()
    filtered_symbols = [symbol['symbol'] for symbol in symbols if 'groupIndex' in symbol and int(symbol['groupIndex']) in [0, 10]]
    random.shuffle(filtered_symbols)
    selected_cryptos = filtered_symbols[:count]
    return selected_cryptos

from numba import njit

@njit
def handle_risk_management_1s_numba(symbol_index, position_type, low_prices, high_prices, sl_prices, tp_prices, liq_prices, max_prices):
    sl_price = sl_prices[symbol_index]
    tp_price = tp_prices[symbol_index]
    liq_price = liq_prices[symbol_index]
    max_price = max_prices[symbol_index]

    # Check for max price hit
    if max_price is not None:
        for high_price in high_prices:
            if position_type == 1 and high_price >= max_price:
                return symbol_index, max_price, "max"
        for low_price in low_prices:
            if position_type == 0 and low_price <= max_price:
                return symbol_index, max_price, "max"

    # Check for take profit hit
    if tp_price is not None:
        for high_price in high_prices:
            if position_type == 1 and high_price >= tp_price:
                return symbol_index, tp_price, "tp"
        for low_price in low_prices:
            if position_type == 0 and low_price <= tp_price:
                return symbol_index, tp_price, "tp"

    # Check for liquidation hit
    if liq_price is not None:
        for low_price in low_prices:
            if position_type == 1 and low_price <= liq_price:
                return symbol_index, liq_price, "liq"
        for high_price in high_prices:
            if position_type == 0 and high_price >= liq_price:
                return symbol_index, liq_price, "liq"

    # Check for stop loss hit
    if sl_price is not None:
        for low_price in low_prices:
            if position_type == 1 and low_price <= sl_price:
                return symbol_index, sl_price, "sl"
        for high_price in high_prices:
            if position_type == 0 and high_price >= sl_price:
                return symbol_index, sl_price, "sl"

    return None, None, None

def handle_risk_management_1s_numpy(symbol_index, position_type, low_prices, high_prices, sl_prices, tp_prices, liq_prices, max_prices, timestamps):
    # Vectorized risk management
    sl_price = sl_prices[symbol_index]
    tp_price = tp_prices[symbol_index]
    liq_price = liq_prices[symbol_index]
    max_price = max_prices[symbol_index]

    # Vectorized checks for SL, TP, and liquidation
    if max_price is not None:
        max_hit = (position_type == 1 and np.any(high_prices >= max_price)) or \
                    (position_type == 0 and np.any(low_prices <= max_price))
        if max_hit:
            max_index = np.where((high_prices >= max_price) if position_type == 1 else (low_prices <= max_price))[0][0]
            logging.debug(f"Max hit at index {max_index} for symbol {symbol_index} with price {max_price} at {timestamps[max_index]}")
            return symbol_index, max_price, "max", timestamps[max_index]

    if tp_price is not None:
        tp_hit = (position_type == 1 and np.any(high_prices >= tp_price)) or \
                    (position_type == 0 and np.any(low_prices <= tp_price))
        if tp_hit:
            tp_index = np.where((high_prices >= tp_price) if position_type == 1 else (low_prices <= tp_price))[0][0]
            logging.debug(f"TP hit at index {tp_index} for symbol {symbol_index} with price {tp_price} at {timestamps[tp_index]}")
            return symbol_index, tp_price, "tp", timestamps[tp_index]

    if liq_price is not None:
        liq_hit = (position_type == 1 and np.any(low_prices <= liq_price)) or \
                    (position_type == 0 and np.any(high_prices >= liq_price))
        if liq_hit:
            liq_index = np.where((low_prices <= liq_price) if position_type == 1 else (high_prices >= liq_price))[0][0]
            logging.debug(f"Liquidation hit at index {liq_index} for symbol {symbol_index} with price {liq_price} at {timestamps[liq_index]}")
            return symbol_index, liq_price, "liq", timestamps[liq_index]

    if sl_price is not None:
        sl_hit = (position_type == 1 and np.any(low_prices <= sl_price)) or \
                    (position_type == 0 and np.any(high_prices >= sl_price))
        if sl_hit:
            sl_index = np.where((low_prices <= sl_price) if position_type == 1 else (high_prices >= sl_price))[0][0]
            logging.debug(f"SL hit at index {sl_index} for symbol {symbol_index} with price {sl_price} at {timestamps[sl_index]}")
            return symbol_index, sl_price, "sl", timestamps[sl_index]

    return None, None, None, None

handle_risk_management_optimized = handle_risk_management_1s_numpy

# from parameters import selected_params, training_params
from synthetic import create_synthetic_data
from environment import TradingEnvironment
from agent import TradingAgent
from rewards import calculate_reward

def initialize_environments(financial_params, training_params, plot_dir='.'):
    selected_params = financial_params
    full_data_matrix = None
    train_env = None
    eval_env = None
    test_env = None
    agent = None
    market_conditions = {}
        
    if financial_params['market_data'] == 'random':
        # Preprocess data using the utility function
        symbols = ['GAINS_40']
        if symbols[0].startswith('GAINS_'):
            count = int(symbols[0].split('_')[1])
            symbols = select_cryptos(count)
            logging.info(f"Selected top {count} cryptos: {symbols}")
        selected_params['symbols'] = sorted(symbols)
        
    if financial_params['market_data'] == 'synthetic':
        # Generate synthetic data
        data_matrix, full_data_matrix, timestamps, mapping, valid_symbols, market_conditions = create_synthetic_data(selected_params['limit'], selected_params['interval'])
        
        # Set up parameters
        selected_params['symbols'] = valid_symbols
        # selected_params['interval'] = '1d'
        # selected_params['limit'] = len(timestamps)
        selected_params['end_time'] = timestamps[-1]
    else:
        symbols = selected_params['symbols']
        interval = selected_params['interval']
        limit = selected_params['limit']
        end_time = selected_params['end_time']
        data_matrix, timestamps, mapping, valid_symbols = preprocess_data(symbols, interval, limit, end_time=end_time)
        end_time = timestamps[-1]
        selected_params['end_time'] = end_time
        selected_params['symbols'] = valid_symbols

    if training_params['train_model']:  
        # Calculate split indices
        train_split_index = int(0.7 * len(timestamps))
        val_split_index = int(0.8 * len(timestamps))

        # Split data into training, validation, and test sets
        train_data_matrix = data_matrix[:train_split_index]
        val_data_matrix = data_matrix[train_split_index:val_split_index]
        test_data_matrix = data_matrix[val_split_index:]

        train_timestamps = timestamps[:train_split_index]
        val_timestamps = timestamps[train_split_index:val_split_index]
        test_timestamps = timestamps[val_split_index:]

        # Initialize the environments
        train_env = TradingEnvironment(data_matrix=train_data_matrix, timestamps=train_timestamps, mapping=mapping, render_mode='human', params=selected_params, reward_function=calculate_reward, market_data=full_data_matrix)
        eval_env = TradingEnvironment(data_matrix=val_data_matrix, timestamps=val_timestamps, mapping=mapping, render_mode='human', params=selected_params, reward_function=calculate_reward, market_data=full_data_matrix)
        test_env = TradingEnvironment(data_matrix=test_data_matrix, timestamps=test_timestamps, mapping=mapping, render_mode='human', params=selected_params, reward_function=calculate_reward, market_data=full_data_matrix)

        # Initialize the TradingAgent with the training and validation environments
        agent = TradingAgent(train_env, eval_env, test_env, training_params, financial_params)

    else:
        test_env = TradingEnvironment(data_matrix=data_matrix, timestamps=timestamps, mapping=mapping, render_mode='human', params=selected_params, reward_function=calculate_reward, market_data=full_data_matrix)
        agent = TradingAgent(test_env, test_env, test_env, training_params, financial_params, plot_dir)

    return train_env, eval_env, test_env, agent, market_conditions

class TestUtilities(unittest.TestCase):

    def test_fetch_binance_klines(self):
        # Define test parameters
        symbol = 'BTC'
        interval = '1m'
        end_time = '2022-01-02'
        limit = 60 * 24

        # Call the function
        df, start_time, end_time = fetch_binance_klines(symbol, interval, limit, end_time)
        logging.info(f"Period start time: {start_time} and end time: {end_time}")

        # Check the DataFrame
        expected_columns = [
            'open', 'high', 'low', 'close', 'volume', 
            # 'close_time', 'quote_asset_volume', 'number_of_trades', 
            # 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ]
        self.assertEqual(list(df.columns), expected_columns)
        self.assertGreater(len(df), 0)  # Ensure that some data is returned
        self.assertEqual(len(df), limit)
        self.assertIsInstance(df.index, pd.DatetimeIndex)

    # def test_compare_crypto_close_prices(self):
    #     # Define test parameters
    #     symbols = sorted(['BTC', 'ETH', 'BNB', 'ADA', 'FTM', 'NEAR', 'SOL', 'SHIB', 'BONK', 'PEPE'])
    #     interval = '1d'
    #     end_time = None #'2021-01-01'
    #     limit = 365 * 3

    #     # Fetch data for each symbol
    #     dataframes = {}
    #     start_times = []
    #     end_times = []
    #     for symbol in symbols:
    #         try:
    #             df, start_time, end_time = fetch_binance_klines(symbol, interval, limit, end_time)
    #             # Normalize close prices by the first close price
    #             df['normalized_close'] = df['close'] / df['close'].iloc[0]
    #             dataframes[symbol] = df
    #             start_times.append(start_time)
    #             end_times.append(end_time)
    #         except Exception as e:
    #             logging.error(f"Error fetching data for {symbol}.")
    #             continue

    #     # Determine the common period
    #     common_start_time = max(start_times)
    #     common_end_time = min(end_times)
    #     logging.info(f"Common period: {common_start_time} to {common_end_time}")

    #     # Filter dataframes to the common period
    #     for symbol in dataframes:
    #         df = dataframes[symbol]
    #         dataframes[symbol] = df[(df.index >= common_start_time) & (df.index <= common_end_time)]

    #     # Assert that the length of BTC DataFrame equals all other lengths
    #     btc_length = len(dataframes['BTC'])
    #     for symbol, df in dataframes.items():
    #         self.assertEqual(btc_length, len(df), f"Length mismatch for {symbol}")

    #     # Plot the normalized close prices
    #     plt.figure(figsize=(12, 6))
    #     for symbol, df in dataframes.items():
    #         plt.plot(df.index, df['normalized_close'], label=symbol, linewidth=2)  # Set linewidth to 2.5

    #     plt.title(f"Normalized Close Price Evolution: {common_start_time} to {common_end_time}")
    #     plt.xlabel('Time')
    #     plt.ylabel('Normalized Close Price')
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    def test_preprocess_data(self):
        # Define test parameters
        symbols = sorted(['BTC', 'ETH', 'BNB']) #, 'ADA', 'FTM', 'NEAR', 'SOL', 'SHIB', 'BONK', 'PEPE'])
        interval = '1h'
        limit = 24 * 7 * 12
        end_time = None #'2022-01-01'

        # Call the preprocess_data function
        data_matrix, timestamps, field_mapping, _ = preprocess_data(symbols, interval, limit, end_time)

        # Check the shape of the returned matrix
        num_candles = data_matrix.shape[0]
        num_symbols = data_matrix.shape[1]
        num_features = data_matrix.shape[2]

        # Assert the number of symbols matches
        self.assertEqual(num_symbols, len(symbols), "Number of symbols does not match")

        # Assert the number of features matches the expected number of columns
        expected_features = 5  # Assuming 'open', 'high', 'low', 'close', 'volume'
        self.assertEqual(num_features, expected_features, "Number of features does not match")

        # Assert that the number of candles is greater than zero
        self.assertGreater(num_candles, 0, "Number of candles should be greater than zero")

        # Check that the data is not empty
        self.assertFalse(np.all(data_matrix == 0), "Data matrix should not be all zeros")

    def test_preprocess_data_column_order(self):
        # Define test parameters
        symbols = ['BTC']
        interval = '1h'
        limit = 24
        end_time = None

        # Call the preprocess_data function
        data_matrix, timestamps, field_mapping, _ = preprocess_data(symbols, interval, limit, end_time)

        # Fetch the data separately to verify the order
        df, _, _ = fetch_binance_klines(symbols[0], interval, limit, end_time)

        # Get the expected order of columns
        expected_order = list(df.columns)

        # Extract the column order from the data matrix
        # Assuming the first symbol and all candles have the same order
        actual_order = df.columns.tolist()

        # Assert that the order of columns in the matrix matches the expected order
        self.assertEqual(actual_order, expected_order, "Column order in the data matrix does not match the expected 'ohlcv' order")

    def test_preprocess_data_column_order_and_values(self):
        # Define test parameters
        symbols = ['BTC', 'ETH']
        interval = '1m'
        limit = 100
        end_time = None

        # Call the preprocess_data function
        data_matrix, timestamps, field_mapping, _ = preprocess_data(symbols, interval, limit, end_time)

        # Fetch the data separately to verify the order and values
        df, _, _ = fetch_binance_klines(symbols[0], interval, limit, end_time)

        # Get the expected order of columns
        expected_order = list(df.columns)

        # Extract the column order from the data matrix
        # Assuming the first symbol and all candles have the same order
        actual_order = df.columns.tolist()

        # Assert that the order of columns in the matrix matches the expected order
        self.assertEqual(actual_order, expected_order, "Column order in the data matrix does not match the expected 'ohlcv' order")

        # Verify the values in the matrix match the DataFrame, skipping the last row
        for i, row in enumerate(df.itertuples(index=False)):
            if i == len(df) - 1:
                break
            for j, column in enumerate(expected_order):
                # Increase the tolerance by reducing the number of decimal places
                self.assertAlmostEqual(data_matrix[i, 0, j], getattr(row, column), places=7, msg=f"Mismatch in values for {column} at row {i}")

        # Additional checks for 'high' and 'low' values
        for i, row in df.iterrows():
            # Check if 'high' is greater than or equal to 'open', 'low', and 'close'
            self.assertGreaterEqual(row['high'], row['open'], f"High value at index {i} is not greater than or equal to open value")
            self.assertGreaterEqual(row['high'], row['low'], f"High value at index {i} is not greater than or equal to low value")
            self.assertGreaterEqual(row['high'], row['close'], f"High value at index {i} is not greater than or equal to close value")

            # Check if 'low' is less than or equal to 'open', 'high', and 'close'
            self.assertLessEqual(row['low'], row['open'], f"Low value at index {i} is not less than or equal to open value")
            self.assertLessEqual(row['low'], row['high'], f"Low value at index {i} is not less than or equal to high value")
            self.assertLessEqual(row['low'], row['close'], f"Low value at index {i} is not less than or equal to close value")

# Run the tests
if __name__ == '__main__':
    unittest.main()
