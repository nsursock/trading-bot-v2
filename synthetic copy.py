import numpy as np
from datetime import datetime, timedelta
# from utilities import identify_patterns
import pandas as pd

def generate_gbm_prices(S0, mu, sigma, T, dt, noise_level=0.01):
    N = int(T / dt)  # Number of time steps
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)  # Brownian motion
    X = (mu - 0.5 * sigma**2) * t + sigma * W
    S = S0 * np.exp(X)  # Geometric Brownian Motion
    
    # Add noise to the GBM prices
    noise = np.random.normal(0, noise_level, N)
    S = S * (1 + noise)
    
    # Adjusted stochastic volatility model
    vol = sigma * np.exp(np.random.normal(0, 0.01, N))  # Reduced volatility change

    # Adjusted time-of-day volatility pattern
    tod_volatility = 0.002 * (1 + np.sin(2 * np.pi * t / T))  # Reduced amplitude

    # Adjusted news events impact
    news_impact = np.random.choice([0, 0.005, -0.005], size=N, p=[0.995, 0.0025, 0.0025])
    
    # Generate intra-candle volatility
    high = S * (1 + np.random.normal(0, vol + tod_volatility, N) + news_impact)
    low = S * (1 - np.random.normal(0, vol + tod_volatility, N) + news_impact)
    close = S * (1 + np.random.normal(0, vol / 8, N) + news_impact)

    # Match open and close prices to nearby candles
    open_prices = np.roll(close, 1)  # Shift close prices to get open prices
    open_prices[0] = S0  # Set the first open price to the initial price

    return open_prices, high, low, close

def generate_realistic_volumes(N, base_volume=500, volatility=0.1, market_condition_factor=1.0):
    """
    Generate realistic volumes based on time-of-day patterns, volatility, and market conditions.
    """
    # Time-of-day pattern: higher volume at the start and end of the day
    time_of_day_pattern = np.sin(np.linspace(0, np.pi, N)) + 1  # Sinusoidal pattern

    # Volatility effect: higher volume with higher volatility
    volatility_effect = 1 + np.random.normal(0, volatility, N)

    # Adjusted random spikes
    random_spikes = np.random.choice([1, 1.05, 1.2], size=N, p=[0.98, 0.015, 0.005])

    # Combine effects using a log-normal distribution
    volumes = np.random.lognormal(mean=np.log(base_volume), sigma=0.3, size=N) * time_of_day_pattern * volatility_effect * market_condition_factor * random_spikes

    # Ensure volumes are positive integers
    volumes = np.maximum(volumes, 1).astype(int)

    return volumes

def create_synthetic_data(limit=100, interval='1d'):
    # market_conditions = {
    #     'TUP': {'description': 'trending up', 'mu': 0.05, 'sigma': 0.1},
    #     'TDO': {'description': 'trending down', 'mu': -0.05, 'sigma': 0.1},
    #     'RANG': {'description': 'ranging', 'mu': 0.0, 'sigma': 0.05},
    #     'VOLH': {'description': 'high vol', 'mu': 0.0, 'sigma': 0.3},
    #     'VOLL': {'description': 'low vol', 'mu': 0.0, 'sigma': 0.01},
    #     # 'CONS': {'description': 'consolidating', 'mu': 0.0, 'sigma': 0.02},
    #     'BULL': {'description': 'bullish', 'mu': 0.03, 'sigma': 0.1},
    #     'BEAR': {'description': 'bearish', 'mu': -0.03, 'sigma': 0.1},
    #     'SIDE': {'description': 'sideways', 'mu': 0.0, 'sigma': 0.05},
    #     'XBULL': {'description': 'extreme bullish', 'mu': 0.1, 'sigma': 0.2},
    #     'XBEAR': {'description': 'extreme bearish', 'mu': -0.1, 'sigma': 0.2},
    #     # 'BEARM': {'description': 'bear market', 'mu': -0.1, 'sigma': 0.3},
    #     # 'BULLM': {'description': 'bull market', 'mu': 0.1, 'sigma': 0.3},
    #     # 'STABM': {'description': 'stable market', 'mu': 0.02, 'sigma': 0.1},
    #     # 'VOLM': {'description': 'volatile market', 'mu': 0.05, 'sigma': 0.5}
    # }
    
    expected_return = 0.1
    volatility_max = 0.5
    volatility_min = 0
    price_min = 100
    price_max = 1000
    
    num_cryptos = 10
    
    # Create 10 synthetic cryptos with random mu and sigma
    market_conditions = {}
    for i in range(num_cryptos):
        # Randomly generate mu, sigma, and S0
        mu = round(np.random.uniform(-expected_return, expected_return), 2)
        sigma = round(np.random.uniform(volatility_min, volatility_max), 2)
        S0 = round(np.random.uniform(price_min, price_max), 2)
        
        # Determine category based on mu and sigma using the given parameters
        if mu > expected_return / 2 and sigma < volatility_max / 2:
            category = 'BULL'
            description = 'bullish'
        elif mu < -expected_return / 2 and sigma < volatility_max / 2:
            category = 'BEAR'
            description = 'bearish'
        elif abs(mu) <= expected_return / 2 and sigma < volatility_max / 3:
            category = 'STAB'
            description = 'stable'
        elif mu > expected_return / 2 and sigma >= volatility_max / 2:
            category = 'XBULL'
            description = 'volatile bullish'
        elif mu < -expected_return / 2 and sigma >= volatility_max / 2:
            category = 'XBEAR'
            description = 'volatile bearish'
        else:
            category = 'NEUT'
            description = 'neutral'
        
        # Generate a crypto name based on the category
        symbol = f"{category}{i+1}"
        
        market_conditions[symbol] = {'description': description, 'mu': mu, 'sigma': sigma, 'S0': S0}
        print(f"Generated market condition: {symbol} - {description}: expected={mu}, volatility={sigma}, price={S0}")

    # S0 = 100  # Initial price
    T = limit  # Total time for 1 day (or adjust as needed)
    dt = 1 / (24 * 60 * 60)  # Time step for one-second data
    N = int(T / dt)  # Number of time steps
    timestamps = [datetime.now() + timedelta(seconds=i) for i in range(N)]

    full_data_matrix = []  # To store full 1-second interval data for all cryptos
    resampled_data_matrix = []  # To store resampled data
    mapping = {'open': 0, 'high': 1, 'low': 2, 'close': 3, 'volume': 4}
    valid_symbols = list(market_conditions.keys())

    for symbol, params in market_conditions.items():
        open_prices, high_prices, low_prices, close_prices = generate_gbm_prices(round(np.random.uniform(1, 100), 2), params['mu'], params['sigma'], T, dt)
        volumes = generate_realistic_volumes(N)

        # Generate synthetic OHLC data for full 1-second interval
        full_symbol_data = np.column_stack((open_prices, high_prices, low_prices, close_prices, volumes))
        
        # Append the full_symbol_data to the full_data_matrix
        full_data_matrix.append(full_symbol_data)

        # Resample data to the specified interval
        resampled_symbol_data, _ = pick_interval_data(full_symbol_data, interval)
        
        # Ensure resampled_symbol_data is a 2D array before appending
        if resampled_symbol_data.ndim == 1:
            resampled_symbol_data = resampled_symbol_data[np.newaxis, :]
        
        resampled_data_matrix.append(resampled_symbol_data)

    # Convert full_data_matrix to a 3D numpy array
    full_data_matrix = np.array(full_data_matrix)

    # Check if the array is 3D before transposing
    if full_data_matrix.ndim == 3:
        full_data_matrix = full_data_matrix.transpose(1, 0, 2)
    else:
        raise ValueError("Full data matrix is not 3D, cannot transpose")

    # Convert resampled_data_matrix to a 3D numpy array
    resampled_data_matrix = np.array(resampled_data_matrix)

    # Check if the array is 3D before transposing
    if resampled_data_matrix.ndim == 3:
        resampled_data_matrix = resampled_data_matrix.transpose(1, 0, 2)
    else:
        raise ValueError("Resampled data matrix is not 3D, cannot transpose")

    # Adjust timestamps for the resampled data
    resampled_timestamps = resample_timestamps(timestamps, interval)

    print(f"Resampled data matrix shape: {resampled_data_matrix.shape}")
    print(f"Full data matrix shape: {full_data_matrix.shape}")

    return resampled_data_matrix, full_data_matrix, resampled_timestamps, mapping, valid_symbols, market_conditions

def pick_interval_data(data, interval):
    # Convert interval to seconds
    interval_seconds = convert_interval_to_seconds(interval)
    
    # Initialize lists to store resampled data
    resampled_data = []
    
    # Iterate over the data in chunks of interval_seconds
    for start in range(0, len(data), interval_seconds):
        end = min(start + interval_seconds, len(data))
        
        # Extract the chunk of data for the current interval
        interval_data = data[start:end]
        
        # Calculate open, high, low, close, and volume for the interval
        open_price = interval_data[0, 0]  # Open price of the first candle in the interval
        high_price = np.max(interval_data[:, 1])  # Highest high in the interval
        low_price = np.min(interval_data[:, 2])  # Lowest low in the interval
        close_price = interval_data[-1, 3]  # Close price of the last candle in the interval
        volume = np.sum(interval_data[:, 4])  # Sum of volumes in the interval
        
        # Append the aggregated data to the resampled list
        resampled_data.append([open_price, high_price, low_price, close_price, volume])
    
    return np.array(resampled_data), None

def resample_timestamps(timestamps, interval):
    # Convert interval to seconds
    interval_seconds = convert_interval_to_seconds(interval)
    
    # Resample timestamps by picking every interval_seconds-th timestamp
    return [timestamps[i] for i in range(0, len(timestamps), interval_seconds)]

def convert_interval_to_seconds(interval):
    # Convert interval string to seconds
    if interval.endswith('d'):
        return int(interval[:-1]) * 24 * 60 * 60
    elif interval.endswith('h'):
        return int(interval[:-1]) * 60 * 60
    elif interval.endswith('m'):
        return int(interval[:-1]) * 60
    elif interval.endswith('s'):
        return int(interval[:-1])
    else:
        raise ValueError("Unsupported interval format")