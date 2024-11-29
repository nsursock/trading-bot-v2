import numpy as np
from datetime import datetime, timedelta
# from utilities import identify_patterns
import pandas as pd
import matplotlib.pyplot as plt

def create_synthetic_data(limit=100, interval='1d', mode='testing'):
    if mode == 'training':
        market_conditions = {
            'TUP': {'description': 'trending up', 'mu': 0.05, 'sigma': 0.1},
            'TDO': {'description': 'trending down', 'mu': -0.05, 'sigma': 0.1},
            'RANG': {'description': 'ranging', 'mu': 0.0, 'sigma': 0.05},
            'VOLH': {'description': 'high vol', 'mu': 0.0, 'sigma': 0.3},
            'VOLL': {'description': 'low vol', 'mu': 0.0, 'sigma': 0.01},
            # 'CONS': {'description': 'consolidating', 'mu': 0.0, 'sigma': 0.02},
            'BULL': {'description': 'bullish', 'mu': 0.03, 'sigma': 0.1},
            'BEAR': {'description': 'bearish', 'mu': -0.03, 'sigma': 0.1},
            'SIDE': {'description': 'sideways', 'mu': 0.0, 'sigma': 0.05},
            'XBULL': {'description': 'extreme bullish', 'mu': 0.1, 'sigma': 0.2},
            'XBEAR': {'description': 'extreme bearish', 'mu': -0.1, 'sigma': 0.2},
            # 'BEARM': {'description': 'bear market', 'mu': -0.1, 'sigma': 0.3},
            # 'BULLM': {'description': 'bull market', 'mu': 0.1, 'sigma': 0.3},
            # 'STABM': {'description': 'stable market', 'mu': 0.02, 'sigma': 0.1},
            # 'VOLM': {'description': 'volatile market', 'mu': 0.05, 'sigma': 0.5}
        }
    elif mode == 'testing':
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
    else:
        raise ValueError("Invalid mode. Please use 'training' or 'testing'.")

    # S0 = 100  # Initial price
    T = limit  # Total time for 1 day (or adjust as needed)
    dt = 1 / (24 * 60 * 60)  # Time step for one-second data
    N = int(T / dt)  # Number of time steps
    timestamps = [datetime.now() + timedelta(seconds=i) for i in range(N)]

    full_data_matrix = []  # To store DataFrames for each crypto
    resampled_data_matrix = []  # To store resampled data
    mapping = {'open': 0, 'high': 1, 'low': 2, 'close': 3, 'volume': 4}
    valid_symbols = list(market_conditions.keys())

    for symbol, params in market_conditions.items():
        print(f"Generating market condition: {symbol} - {params['description']}: expected={params['mu']}, volatility={params['sigma']}")
        open_prices, high_prices, low_prices, close_prices = generate_gbm_prices(round(np.random.uniform(1, 100), 2), params['mu'], params['sigma'], T, dt)
        
        # Calculate intra_volatility as the absolute price changes
        price_changes = np.diff(close_prices) / close_prices[:-1]
        intra_volatility = np.abs(price_changes)
        
        # Fix: Pad intra_volatility to match the length of coupled_volumes
        intra_volatility = np.insert(intra_volatility, 0, 0)  # Insert a zero at the beginning

        volumes = generate_realistic_volumes(N, mu=7, sigma=0.5, theta=0.1, long_term_mean=1000, intra_volatility=intra_volatility)

        # print(f"Generated {symbol} | Length: {len(open_prices)}")

        # Generate synthetic OHLC data for full 1-second interval
        full_symbol_data = np.column_stack((open_prices, high_prices, low_prices, close_prices, volumes))
        
        # Convert full_symbol_data to a DataFrame
        full_symbol_df = pd.DataFrame(
            full_symbol_data,
            columns=['open', 'high', 'low', 'close', 'volume']
        )
        
        # Add timestamps to the DataFrame
        full_symbol_df['timestamp'] = timestamps
        
        # Convert 'timestamp' to datetime and set it as the index
        full_symbol_df['timestamp'] = pd.to_datetime(full_symbol_df['timestamp'])
        full_symbol_df.set_index('timestamp', inplace=True)
        
        # Append the DataFrame to the full_data_matrix
        full_data_matrix.append(full_symbol_df)

        # Resample data to the specified interval
        resampled_symbol_data, _ = pick_interval_data(full_symbol_data, interval)
        
        # Ensure resampled_symbol_data is a 2D array before appending
        if resampled_symbol_data.ndim == 1:
            resampled_symbol_data = resampled_symbol_data[np.newaxis, :]
        
        resampled_data_matrix.append(resampled_symbol_data)

    # Convert resampled_data_matrix to a 3D numpy array
    resampled_data_matrix = np.array(resampled_data_matrix)

    # Check if the array is 3D before transposing
    if resampled_data_matrix.ndim == 3:
        resampled_data_matrix = resampled_data_matrix.transpose(1, 0, 2)
    else:
        raise ValueError("Resampled data matrix is not 3D, cannot transpose")

    # Adjust timestamps for the resampled data
    resampled_timestamps = resample_timestamps(timestamps, interval)

    # print(f"Resampled data matrix shape: {resampled_data_matrix.shape}")
    # print(f"Full data matrix shape: {full_data_matrix.shape}")

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
        volume = np.max(interval_data[:, 4])  # Use max volume to capture spikes

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

def generate_gbm_prices(S0, mu, sigma, T, dt):
    """
    Generate prices using the Geometric Brownian Motion model.
    
    :param S0: Initial stock price
    :param mu: Expected return
    :param sigma: Volatility
    :param T: Total time
    :param dt: Time step
    :return: Arrays of open, high, low, close prices
    """
    N = int(T / dt)  # Number of time steps
    prices = np.zeros(N)
    prices[0] = S0
    for t in range(1, N):
        prices[t] = prices[t-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * np.random.normal())
    
    # For simplicity, let's assume open, high, low, close are the same
    open_prices = prices
    high_prices = prices
    low_prices = prices
    close_prices = prices
    
    return open_prices, high_prices, low_prices, close_prices

def generate_realistic_volumes(N, mu=7, sigma=0.5, theta=0.1, long_term_mean=1000, intra_volatility=None):
    """
    Generate realistic trading volumes using different models.
    
    :param N: Number of time steps
    :param mu: Mean for log-normal distribution
    :param sigma: Standard deviation for log-normal distribution
    :param theta: Mean-reversion speed for OU process
    :param long_term_mean: Long-term mean for OU process
    :param intra_volatility: Array of intra-candle volatilities
    :return: Array of volumes
    """
    # Log-Normal Distribution
    log_normal_volumes = np.exp(mu + sigma * np.random.normal(size=N))
    
    # Mean-Reverting Process (Ornstein-Uhlenbeck)
    ou_volumes = np.zeros(N)
    ou_volumes[0] = long_term_mean
    for t in range(1, N):
        ou_volumes[t] = ou_volumes[t-1] + theta * (long_term_mean - ou_volumes[t-1]) + sigma * np.random.normal()
    
    # Volume-Volatility Coupling (example with log-normal)
    # Assuming sigma_t is the simulated price volatility
    sigma_t = np.random.uniform(0.1, 0.5, N)  # Example volatility
    coupled_volumes = np.exp(mu + sigma_t * np.random.normal(size=N))
    
    # Check if intra_volatility is iterable
    if intra_volatility is not None:
        if isinstance(intra_volatility, (list, np.ndarray)):
            # Increase volume where intra-candle volatility is high
            volume_spike_factor = 1 + 20 * intra_volatility  # Further increase the spike factor
            coupled_volumes *= volume_spike_factor

            # Debugging: Print some values
            # print("Intra-volatility sample:", intra_volatility[:10])
        else:
            print("Warning: intra_volatility is not iterable. Received:", intra_volatility)

    # Alternative method: Add random spikes
    num_spikes = int(0.01 * N)  # 1% of the data points will have spikes
    spike_indices = np.random.choice(N, num_spikes, replace=False)
    
    # Adjust the spike factor 
    spike_factor = np.random.uniform(2, 5, size=num_spikes)
    coupled_volumes[spike_indices] *= spike_factor  # Apply the spike factor

    # Ensure no negative volumes
    volumes = np.maximum(coupled_volumes, 0)

    return volumes