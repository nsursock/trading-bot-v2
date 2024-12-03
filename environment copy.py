import gymnasium as gym
from gymnasium import spaces
import numpy as np
import logging
from rewards import calculate_reward
import matplotlib.pyplot as plt
from utilities import get_liquidation_threshold, handle_risk_management_optimized, ACTIONS_AVAILABLE
import pandas as pd
import glob

class TradingEnvironment(gym.Env):
    def __init__(self, data_matrix, timestamps, mapping, render_mode='human', params=None, reward_function=None, market_data=None, live_mode=False):
        super(TradingEnvironment, self).__init__()
        
        self.min_max_values = {
            'net_profit': [float('inf'), float('-inf')],
            'sharpe_ratio': [float('inf'), float('-inf')],
            'max_drawdown': [float('inf'), float('-inf')]
        }
        
        self.params = params if params else { # TODO: finish params
            'symbols': ['BTC', 'ETH'],
            'initial_balance': 10000,
            'leverage_min': 1,
            'leverage_max': 200,
            'collateral_min': 50,
            'collateral_max': 200,
            'risk_per_trade': 0.01,
            'trading_fee': 0.001,  # 0.1% trading fee
            'slippage': 0.0005,    # 0.05% slippage
            'bid_ask_spread': 0.0002,  # 0.02% bid-ask spread
            'borrowing_fee_per_hour': 0.0001,  # Example: 0.01% per hour
            'trailing_stop_percent': 0.02,  # Default to 2% trailing stop
        }
        
        self.previous_returns = [0] * len(self.params['symbols'])  # Initialize previous returns to zero for each symbol
        self.actions_available = ACTIONS_AVAILABLE
        
        # Ensure the environment supports rendering
        self.render_mode = render_mode  # or 'rgb_array' if preferred
        self.history = []
        self.reward_function = reward_function
        
        self.data_matrix = data_matrix
        self.timestamps = timestamps
        self.mapping = mapping
        # self.symbols = self.params.symbols if self.params else ['BTC', 'ETH']

        # Define action and observation space
        self.num_symbols = data_matrix.shape[1]
        self.num_features = data_matrix.shape[2]
        self.action_space = spaces.MultiDiscrete([len(self.actions_available)] * self.data_matrix.shape[1])  # Hold, Long, Short, Close, Hedge
        obs_shape = (self.num_symbols * self.num_features + 2 + self.num_symbols, )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        
        # Initialize state
        self.current_step = 0
        self.balance = self.params['initial_balance']
        self.positions = [{} for _ in range(self.num_symbols)]  # Initialize as a list of dictionaries
        self.net_worth = self.params['initial_balance']

        self.cooldown_period = self.params['cooldown_period']  # Define the cooldown period in steps
        self.cooldowns = [0] * self.num_symbols  # Initialize cooldowns for each symbol
        
        # Initialize counters for fractal and percentage calculations
        self.fractal_counter = 0
        self.percentage_counter = 0
        self.fallback_counter = 0

        # Check if market_data is None or empty
        if not self.params['basic_risk_mgmt']:
            if market_data is None or len(market_data) == 0:
                self.market_data = self.fetch_market_data()
            else:
                self.market_data = market_data

        self.trailing_stop_percent = self.params.get('trailing_stop_percent', 0.02)  # Default to 2% trailing stop

        # Initialize episode counter
        self.episode_counter = 1
        self.num_episodes = self.params.get('num_episodes', 1)
        
        self.live_mode = live_mode  # Add a flag to indicate live or backtesting mode
        
    def update_data(self, data_matrix, timestamps):
        self.data_matrix = data_matrix
        self.timestamps = timestamps

    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Check if data_matrix is empty
        if self.data_matrix.size == 0:
            raise ValueError("Data matrix is empty. Ensure data is loaded correctly before resetting the environment.")
        
        # Set current_step based on live_mode
        if self.live_mode:
            self.current_step = len(self.data_matrix) - 1  # Start at the last step for live mode
        else:
            self.current_step = 0  # Start at the beginning for backtesting

        self.balance = self.params['initial_balance']
        self.positions = [{} for _ in range(self.num_symbols)]  # Initialize as a list of dictionaries
        self.net_worth = self.params['initial_balance']
        return self.next_observation(), {}
    
    # def latest_observation(self):
    #     # Ensure the observation includes all features for all symbols
    #     observation = self.data_matrix[len(self.data_matrix) - 1, :, :]  # matrix of shape (num_symbols, num_features)
        
    #     # Add balance, net worth, and positions to the observation
    #     balance_array = np.array([self.balance])
    #     net_worth_array = np.array([self.net_worth])
    #     positions_array = np.array([pos.get('position_size', 0) for pos in self.positions])
        
    #     # Concatenate the additional information to the observation
    #     extended_observation = np.concatenate((observation.flatten(), balance_array, net_worth_array, positions_array))
        
    #     return extended_observation

    def next_observation(self):
        # Ensure the observation includes all features for all symbols
        observation = self.data_matrix[self.current_step, :, :]  # matrix of shape (num_symbols, num_features)
        
        # Add balance, net worth, and positions to the observation
        balance_array = np.array([self.balance])
        net_worth_array = np.array([self.net_worth])
        positions_array = np.array([pos.get('position_size', 0) for pos in self.positions])
        
        # Concatenate the additional information to the observation
        extended_observation = np.concatenate((observation.flatten(), balance_array, net_worth_array, positions_array))
        
        return extended_observation
    
    def calculate_fractal_high(self, step, symbol_index, window=5):
        """
        Calculate the fractal high for a given step and symbol based on past data.
        
        :param step: The current step in the data matrix.
        :param symbol_index: The index of the symbol to check.
        :param window: The number of past steps to consider, including the current step.
        :return: The fractal high price for the specified symbol.
        """
        for i in range(step, window - 1, -1):
            # Extract the high prices for the window of past steps for the specific symbol
            high_prices = self.data_matrix[i-window+1:i+1, symbol_index, self.mapping['high']]

            # Check if the middle point is a fractal high
            middle_index = window // 2
            if np.all(high_prices[middle_index] > high_prices[:middle_index]) and np.all(high_prices[middle_index] > high_prices[middle_index+1:]):
                return high_prices[middle_index]
        return None

    def calculate_fractal_low(self, step, symbol_index, window=5):
        """
        Calculate the fractal low for a given step and symbol based on past data.
        
        :param step: The current step in the data matrix.
        :param symbol_index: The index of the symbol to check.
        :param window: The number of past steps to consider, including the current step.
        :return: The fractal low price for the specified symbol.
        """
        for i in range(step, window - 1, -1):
            # Extract the low prices for the window of past steps for the specific symbol
            low_prices = self.data_matrix[i-window+1:i+1, symbol_index, self.mapping['low']]

            # Check if the middle point is a fractal low
            middle_index = window // 2
            if np.all(low_prices[middle_index] < low_prices[:middle_index]) and np.all(low_prices[middle_index] < low_prices[middle_index+1:]):
                return low_prices[middle_index]
        return None
    
    def compute_prices(self, symbol_index, type, current_price, leverage):
        if self.params['risk_mgmt'] == 'fractals':
            sl_price, tp_price = self.compute_fractal_prices(symbol_index, type, current_price, leverage)
        else:
            sl_price, tp_price = self.compute_percentage_prices(type, current_price, leverage)
            
        liq_percentage = get_liquidation_threshold(leverage) / 100 / leverage  # Convert to percentage
        max_percentage = 9 / leverage
        if type == 'long':
            liq_price = current_price * (1 - liq_percentage)
            max_price = current_price * (1 + max_percentage)
        elif type == 'short':
            liq_price = current_price * (1 + liq_percentage)
            max_price = current_price * (1 - max_percentage)
        else:
            liq_price = None
            max_price = None
        
        # Add coherence check for SL and TP prices
        sl_percentage = self.params['sl_mult_perc'] / leverage  # Adjust SL by leverage
        tp_percentage = self.params['tp_mult_perc'] / leverage  # Adjust TP by leverage

        if type == 'long':
            if sl_price >= current_price and tp_price <= current_price:
                self.fallback_counter += 1
                self.fractal_counter -= 1
            if sl_price >= current_price:
                logging.debug(f"SL price {sl_price} is not coherent for a long position. Adjusting to below current price.")
                sl_price = current_price * (1 - sl_percentage)
            if tp_price <= current_price:
                logging.debug(f"TP price {tp_price} is not coherent for a long position. Adjusting to above current price.")
                tp_price = current_price * (1 + tp_percentage)
        elif type == 'short':
            if sl_price <= current_price and tp_price >= current_price:
                self.fallback_counter += 1
                self.fractal_counter -= 1
            if sl_price <= current_price:
                logging.debug(f"SL price {sl_price} is not coherent for a short position. Adjusting to above current price.")
                sl_price = current_price * (1 + sl_percentage)
            if tp_price >= current_price:
                logging.debug(f"TP price {tp_price} is not coherent for a short position. Adjusting to below current price.")
                tp_price = current_price * (1 - tp_percentage)
                
        # if type == 'long':
        #     tp_price = current_price * (1 + liq_percentage) # TODO: bug that can be profitable, analyze this
        # elif type == 'short':
        #     tp_price = current_price * (1 - liq_percentage)
        
        return sl_price, tp_price, liq_price, max_price
    
    def compute_fractal_prices(self, symbol_index, type, current_price, leverage):
        # Calculate fractal levels
        fractal_high = self.calculate_fractal_high(self.current_step, symbol_index)
        fractal_low = self.calculate_fractal_low(self.current_step, symbol_index)
        
        # Determine SL and TP based on fractals
        if fractal_low is not None and fractal_high is not None:
            self.fractal_counter += 1  # Increment fractal counter
            logging.debug(f"Using fractals for SL and TP: Fractal High: {fractal_high}, Fractal Low: {fractal_low}")
            if type == 'long':
                sl_price = fractal_low
                tp_price = fractal_high
            elif type == 'short':
                sl_price = fractal_high
                tp_price = fractal_low
            else:
                sl_price, tp_price = None, None
        else:
            logging.debug(f"Using percentage for SL and TP: Fractal High: {fractal_high}, Fractal Low: {fractal_low}")
            sl_price, tp_price = self.compute_percentage_prices(type, current_price, leverage)
        
        # Add logging for debugging
        logging.debug(f"Fractal High: {fractal_high}, Fractal Low: {fractal_low}, SL Price: {sl_price}, TP Price: {tp_price}")
        
        return sl_price, tp_price
    
    def compute_percentage_prices(self, type, current_price, leverage):
        self.percentage_counter += 1  # Increment percentage counter
        sl_percentage = self.params['sl_mult_perc'] / leverage  # Adjust SL by leverage
        tp_percentage = self.params['tp_mult_perc'] / leverage  # Adjust TP by leverage
        if type == 'long':
            sl_price = current_price * (1 - sl_percentage)
            tp_price = current_price * (1 + tp_percentage)
        elif type == 'short':
            sl_price = current_price * (1 + sl_percentage)
            tp_price = current_price * (1 - tp_percentage)
        else:
            sl_price, tp_price = None, None
        return sl_price, tp_price

    def calculate_kelly_fraction(self, win_probability, win_loss_ratio, fraction=0.75):
        kelly_fraction = (win_probability * (win_loss_ratio + 1) - 1) / win_loss_ratio if win_loss_ratio != 0 else 0
        scaled_kelly_fraction = max(0, min(fraction * kelly_fraction, 1))
        
        # Add logging for debugging
        logging.debug(f"Win Probability: {win_probability}, Win/Loss Ratio: {win_loss_ratio}, Kelly Fraction: {kelly_fraction}, Scaled Kelly Fraction: {scaled_kelly_fraction}")
        
        return scaled_kelly_fraction

    def calculate_trade_statistics(self):
        """
        Calculate win probability and win/loss ratio from the trade history.
        
        :return: Tuple containing win probability and win/loss ratio.
        """
        if not self.history:
            return 0, 0  # No trades have been made

        wins = [trade for trade in self.history if trade['pnl'] > 0]
        losses = [trade for trade in self.history if trade['pnl'] < 0]

        num_wins = len(wins)
        num_losses = len(losses)
        total_trades = num_wins + num_losses

        win_probability = num_wins / total_trades if total_trades > 0 else 0

        average_win = np.mean([trade['pnl'] for trade in wins]) if num_wins > 0 else 0
        average_loss = np.mean([trade['pnl'] for trade in losses]) if num_losses > 0 else 0

        win_loss_ratio = average_win / abs(average_loss) if average_loss != 0 else 0

        return win_probability, win_loss_ratio

    def open_position(self, symbol_index, type, current_price):
        if self.cooldowns[symbol_index] > 0:
            logging.debug(f"Cannot open position for symbol {symbol_index} due to cooldown.")
            return None, None, None, None  # Ensure a tuple is returned

        # Calculate win probability and win/loss ratio from history
        win_probability, win_loss_ratio = self.calculate_trade_statistics()

        # Calculate the Kelly fraction
        kelly_fraction = self.calculate_kelly_fraction(win_probability, win_loss_ratio, self.params['kelly_fraction'])

        # Adjust risk per trade using the Kelly fraction
        risk_per_trade = kelly_fraction if win_loss_ratio != 0 else self.params['risk_per_trade']
        
        # Progressive collateral calculation: start with a lower percentage of the balance
        initial_collateral_factor = 0.1  # Start with 10% of the balance
        collateral = max(self.params['collateral_min'], min(initial_collateral_factor * self.balance, self.params['collateral_max']))

        leverage = self.calculate_leverage(symbol_index, collateral)[self.current_step]
        if self.params['adjust_leverage']:
            leverage = self.adjust_leverage(leverage, self.params['boost_factor'])
        position_size = collateral / current_price

        # Compute SL, TP, and liquidation prices
        sl_price, tp_price, liq_price, max_price = self.compute_prices(symbol_index, type, current_price, leverage)

        # Adjust current price for slippage and bid-ask spread
        if type == 'long':
            adjusted_price = current_price * (1 + self.params['slippage'] + self.params['bid_ask_spread'])
        elif type == 'short':
            adjusted_price = current_price * (1 - self.params['slippage'] - self.params['bid_ask_spread'])
        
        # Calculate fees
        fee = adjusted_price * self.params['trading_fee'] * position_size

        # Update balance with fees
        self.balance -= fee

        # Calculate borrowing fees for the initial hour
        borrowing_fee = collateral * self.params['borrowing_fee_per_hour']
        self.balance -= borrowing_fee

        self.positions[symbol_index] = {
            'open_time': self.timestamps[self.current_step],
            'type': type,
            'entry_price': adjusted_price,
            'leverage': round(leverage),
            'collateral': round(collateral),
            'position_size': round(position_size, 3),
            'sl_price': sl_price,
            'tp_price': tp_price,
            'liq_price': liq_price,
            'max_price': max_price,
            'borrowing_fee': borrowing_fee,  # Added borrowing_fee to position data
            'trailing_stop': None  # Initialize trailing stop
        }
        self.balance -= collateral
        self.cooldowns[symbol_index] = self.cooldown_period  # Set cooldown after opening a position
        
        return collateral, leverage, tp_price, sl_price  # Ensure a tuple is returned
        
    def close_position(self, symbol_index, exit_price, exit_reason="std", exit_time=None):
        position = self.positions[symbol_index]
        if not position:
            return
        
        entry_price = position['entry_price']
        position_size = position['position_size']
        leverage = position['leverage']
        collateral = position['collateral']
        
        # Adjust exit price for slippage and bid-ask spread
        if position['type'] == 'long':
            adjusted_exit_price = exit_price * (1 - self.params['slippage'] - self.params['bid_ask_spread'])
        elif position['type'] == 'short':
            adjusted_exit_price = exit_price * (1 + self.params['slippage'] + self.params['bid_ask_spread'])

        # Calculate PnL
        pnl = (adjusted_exit_price - entry_price) * position_size * leverage
        if position['type'] == 'short':
            pnl = -pnl  # Reverse PnL for short positions

        # Calculate fees
        fee = adjusted_exit_price * self.params['trading_fee'] * position_size

        # Update balance with PnL and deduct fees
        self.balance += position_size * adjusted_exit_price + pnl - fee

        # Calculate return
        return_on_investment = pnl / collateral if collateral != 0 else 0
        
        # Ensure close_time is later than open_time
        close_time = exit_time if exit_time is not None else self.timestamps[self.current_step]
        # Convert both times to the same precision
        close_time = pd.to_datetime(close_time).floor('us')
        open_time = pd.to_datetime(position['open_time']).floor('us')
    
        if close_time <= open_time:
            # close_time = position['open_time'] + pd.Timedelta(seconds=1)  # Adjust to be slightly later
            logging.debug(f"Close time {close_time} is not later than open time {open_time}. Exit reason: {exit_reason}")

        # Format times to exclude microseconds
        close_time = close_time.strftime('%Y-%m-%d %H:%M:%S')
        open_time = open_time.strftime('%Y-%m-%d %H:%M:%S')
        
        close_time = pd.to_datetime(close_time)
        open_time = pd.to_datetime(open_time)

        # Calculate total borrowing fees
        hours_open = (close_time - open_time).total_seconds() / 3600
        total_borrowing_fee = position['borrowing_fee'] * hours_open

        # Deduct total borrowing fees from balance
        self.balance -= total_borrowing_fee

        # Add borrowing fees and episode number to history
        self.history.append({
            'episode': self.episode_counter,  # Add the current episode number
            'open_time': open_time,
            'close_time': close_time,
            'symbol': self.params['symbols'][symbol_index],
            'type': position['type'],
            'entry_price': entry_price,
            'exit_price': adjusted_exit_price,
            'position_size': position_size,
            'leverage': leverage,
            'collateral': round(collateral),
            'sl_price': position['sl_price'],
            'tp_price': position['tp_price'],
            'liq_price': position['liq_price'],
            'max_price': position['max_price'],
            'exit_reason': exit_reason,
            'pnl': round(pnl, 2),  # Add PnL to history
            'return': round(return_on_investment, 2),  # Add return to history
            'borrowing_fee': round(total_borrowing_fee, 2),
        })
        
        # Reset position to an empty dictionary
        self.positions[symbol_index] = {}
        self.cooldowns[symbol_index] = self.cooldown_period  # Set cooldown after closing a position
        
    def adjust_leverage(self, base_leverage, boost_factor=3):
        # Define volatility-based adjustment factors for each interval
        interval_volatility_factors = {
            '1s': 0.2,
            '1m': 0.3,   # Low volatility for short intervals
            '3m': 0.5,
            '5m': 0.7,
            '15m': 0.9,
            '30m': 1.1,
            '1h': 1.3,
            '2h': 1.5,
            '4h': 1.8,
            '6h': 2.0,
            '8h': 2.3,
            '12h': 2.6,
            '1d': 3.0,
            '3d': 4.0,   # Higher volatility for long intervals
            '1w': 5.0,
            '1M': 6.0    # Highest volatility for very long intervals
        }
        
        interval = self.params.get('interval', '1d')  # Default to '1d' if no interval is provided
        
        # Check if the interval is in the dictionary, with a default warning
        if interval not in interval_volatility_factors:
            logging.warning(f"Interval {interval} is not supported.")
            return base_leverage  # Return base leverage if unsupported interval is encountered

        # Inverse volatility factor: increase leverage when volatility is low, decrease when high
        volatility_factor = interval_volatility_factors[interval]
        adjusted_leverage = base_leverage * boost_factor / volatility_factor  # Inverse adjustment based on volatility
        
        # Clamp adjusted leverage between minimum and maximum limits
        min_leverage = self.params.get('leverage_min', 1)  # Default minimum leverage to 1
        max_leverage = self.params.get('leverage_max', 200)  # Default maximum leverage to 150
        adjusted_leverage = max(min_leverage, min(adjusted_leverage, max_leverage))

        return adjusted_leverage
        
    def calculate_leverage(self, symbol_index, collateral, volume_factor_base=10000):
        volumes = self.data_matrix[:, symbol_index, self.mapping['volume']]
        closes = self.data_matrix[:, symbol_index, self.mapping['close']]
        volatilities = np.std(closes)
        
        # Handle cases where volatility is NaN or zero
        valid_volatilities = np.where((~np.isnan(volatilities)) & (volatilities != 0), volatilities, np.inf)
    
        # Expand dimensions of position_sizes to match volumes
        position_sizes = (collateral / valid_volatilities)
        adjusted_position_sizes = position_sizes * (volumes / volume_factor_base)
        position_values = adjusted_position_sizes * closes
        leverages = position_values / collateral
        
        # Normalize leverages
        leverage_min = np.min(leverages)
        leverage_max = np.max(leverages)
        
        if leverage_max == leverage_min:
            normalized_leverages = np.full_like(leverages, self.params['leverage_min'])
        else:
            normalized_leverages = self.params['leverage_min'] + (leverages - leverage_min) * (self.params['leverage_max'] - self.params['leverage_min']) / (leverage_max - leverage_min)
    
        # Handle NaN values
        normalized_leverages = np.where(np.isnan(normalized_leverages), self.params['leverage_min'], normalized_leverages)
        
        return np.round(normalized_leverages).astype(int) 
    
    def interval_to_seconds(self,interval):
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

    def fetch_market_data(self):
        """Fetch and concatenate market data from CSV files within the specified date range."""
        start_time = pd.to_datetime(self.timestamps[0])
        end_time = pd.to_datetime(self.timestamps[-1])
        
        market_data = [None] * len(self.params['symbols'])  # Initialize market data storage as a list
        
        logging.info(f"Fetching 1s (more granular) market data for risk management")

        for symbol_index, symbol in enumerate(self.params['symbols']):
            logging.info(f"Fetching market data for symbol: {symbol}")
            # Construct the directory path for the symbol
            directory_path = f"market_data/{symbol}/"
            
            # Use glob to find files matching the date range
            files = glob.glob(f"{directory_path}*.csv")
            
            # Debugging: Print the list of files found
            logging.debug(f"Files found for symbol {symbol}: {files}")
            
            # Adjust the filtering logic to only consider the date part
            relevant_files = [
                file for file in files
                if start_time.date() <= pd.to_datetime(file.split('/')[-1].replace('.csv', ''), format='%Y%m%d').date() <= end_time.date()
            ]
            
            # Debugging: Print the list of relevant files
            logging.debug(f"Relevant files for symbol {symbol}: {relevant_files}")
            
            if not relevant_files:
                raise ValueError(f"No files found for symbol {symbol} in the specified date range: {start_time} to {end_time}")
            
            data_frames = []
            for file in relevant_files:
                try:
                    df = pd.read_csv(file)
                    if df.empty:
                        logging.warning(f"File {file} is empty. Skipping.")
                        continue
                    data_frames.append(df)
                except pd.errors.EmptyDataError:
                    logging.warning(f"File {file} is empty or has no columns to parse. Skipping.")
                    continue
            
            if not data_frames:
                raise ValueError(f"No valid data found for symbol {symbol} in the specified date range.")
            
            concatenated_df = pd.concat(data_frames)
            
            # Set the 'timestamp' column as the index
            concatenated_df['timestamp'] = pd.to_datetime(concatenated_df['timestamp'])
            concatenated_df.set_index('timestamp', inplace=True)
            
            # Sort the DataFrame by the index
            concatenated_df.sort_index(inplace=True)
            
            # Store the DataFrame in the market_data list
            market_data[symbol_index] = concatenated_df
            
            # Debugging: Print the head and tail of the concatenated DataFrame
            logging.debug(f"Fetched market data for symbol {symbol}:")
            logging.debug(concatenated_df.head())
            logging.debug(concatenated_df.tail())
            
        return market_data
    
    def handle_risk_management_basic(self, symbol_index, low_price, high_price):
        position = self.positions[symbol_index]
        if not position:
            return

        sl_price = position['sl_price']
        tp_price = position['tp_price']
        liq_price = position['liq_price']
        max_price = position['max_price']

        logging.debug(f"SL Price: {sl_price}, TP Price: {tp_price}, Liquidation Price: {liq_price}")

        # Check if SL, TP, or liquidation is hit using low and high prices
        if max_price is not None and (
            (position['type'] == 'long' and high_price >= max_price) or
            (position['type'] == 'short' and low_price <= max_price)
        ):
            logging.debug(f"Max price hit for symbol {symbol_index}. Closing position.")
            self.close_position(symbol_index, max_price, exit_reason="max")
        if tp_price is not None and (
            (position['type'] == 'long' and high_price >= tp_price) or
            (position['type'] == 'short' and low_price <= tp_price)
        ):
            logging.debug(f"Take-profit hit for symbol {symbol_index}. Closing position.")
            self.close_position(symbol_index, tp_price, exit_reason="tp")
        if liq_price is not None and (
            (position['type'] == 'long' and low_price <= liq_price) or
            (position['type'] == 'short' and high_price >= liq_price)
        ):
            logging.debug(f"Liquidation hit for symbol {symbol_index}. Closing position.")
            self.close_position(symbol_index, liq_price, exit_reason="liq")
        if sl_price is not None and (
            (position['type'] == 'long' and low_price <= sl_price) or
            (position['type'] == 'short' and high_price >= sl_price)
        ):
            logging.debug(f"Stop-loss hit for symbol {symbol_index}. Closing position.")
            self.close_position(symbol_index, sl_price, exit_reason="sl")

    def handle_risk_management_1s(self, symbol_index):
        position = self.positions[symbol_index]
        if not position:
            return
        
        # Ensure timestamps are in datetime format
        self.timestamps = pd.to_datetime(self.timestamps)
        
        # print("Environment Timestamps:", self.timestamps[:5])
        # print("Market Data Timestamps for Symbol 5:", self.market_data[5].index[:5])

        # Check if market data is still None after fetching
        if self.market_data[symbol_index] is None:
            logging.warning(f"Market data for symbol {symbol_index} is not available.")
            return

        # Prepare data for Numba function
        sl_prices = np.array([pos.get('sl_price', 0) for pos in self.positions])
        tp_prices = np.array([pos.get('tp_price', 0) for pos in self.positions])
        liq_prices = np.array([pos.get('liq_price', 0) for pos in self.positions])
        max_prices = np.array([pos.get('max_price', 0) for pos in self.positions])

        # Check if the position has a 'type' key
        if 'type' in position:
            end_time = self.timestamps[self.current_step]
            
            # Convert market data index to datetime if necessary
            market_data_index = pd.to_datetime(self.market_data[symbol_index].index)

            # Adjust the slicing logic to ensure it captures the correct range
            start_time = self.timestamps[self.current_step - 1] if self.current_step > 0 else self.timestamps[0]
            mkt_data = self.market_data[symbol_index].loc[
                (market_data_index >= start_time) & 
                (market_data_index < end_time)
            ]
            
            logging.debug(f"Market data length for symbol {symbol_index}: {len(mkt_data)}")
            
            low_prices = mkt_data['low'].values
            high_prices = mkt_data['high'].values

            # Determine the position type as an integer
            position_type = 1 if position['type'] == 'long' else 0 if position['type'] == 'short' else None

            # Call the optimized function
            symbol_index, price, reason, exit_time = handle_risk_management_optimized(
                symbol_index, position_type, low_prices, high_prices, sl_prices, tp_prices, liq_prices, max_prices, mkt_data.index.values
            )

            if symbol_index is not None:
                self.close_position(symbol_index, price, exit_reason=reason, exit_time=exit_time)

    def update_trailing_stop(self, symbol_index, current_price):
        position = self.positions[symbol_index]
        if not position:
            return

        if position['type'] == 'long':
            new_sl_price = current_price * (1 - self.trailing_stop_percent)
            if position['sl_price'] is None or new_sl_price > position['sl_price']:
                position['sl_price'] = new_sl_price
                logging.debug(f"Updated trailing stop for long position: {new_sl_price}")
        elif position['type'] == 'short':
            new_sl_price = current_price * (1 + self.trailing_stop_percent)
            if position['sl_price'] is None or new_sl_price < position['sl_price']:
                position['sl_price'] = new_sl_price
                logging.debug(f"Updated trailing stop for short position: {new_sl_price}")

    def step(self, action):
        # Get the current low and high prices for all symbols
        low_prices = self.data_matrix[self.current_step, :, self.mapping['low']]
        high_prices = self.data_matrix[self.current_step, :, self.mapping['high']]
        current_prices = self.data_matrix[self.current_step, :, self.mapping['close']]
        
        infos = {}
        
        # Handle SL and TP for each position before processing actions
        if self.params['basic_risk_mgmt']:
            for i in range(self.num_symbols):
                self.handle_risk_management_basic(i, low_prices[i], high_prices[i])
        else:
            for i in range(self.num_symbols):
                self.handle_risk_management_1s(i)

        # Ensure action is a 1D array
        # if len(action.shape) > 1:
        #     logging.warning(f"Action is malformed for step {self.current_step}, it is a matrix. Flattening it: {action}")
        action = action.flatten()

        logging.debug(f"Step {self.current_step} called with action: {action} (type: {type(action)})")

        # Execute action for each symbol
        for i in range(self.num_symbols):
            if self.cooldowns[i] > 0:
                self.cooldowns[i] -= 1
                continue
            if action[i] == self.actions_available['hold'] or (self.params['reverse_actions'] and action[i] == self.actions_available['close']):  # Hold
                pass
            elif action[i] == self.actions_available['long'] or (self.params['reverse_actions'] and action[i] == self.actions_available['short']):  # Long
                if self.positions[i]:
                    self.close_position(i, current_prices[i], 'long')
                collateral, leverage, tp_price, sl_price = self.open_position(i, 'long', current_prices[i])
                infos[self.params['symbols'][i]] = { 'type': 'open_long', 'collateral': collateral, 'leverage': leverage, 'tp_price': tp_price, 'sl_price': sl_price }
            elif action[i] == self.actions_available['short'] or (self.params['reverse_actions'] and action[i] == self.actions_available['long']):  # Short
                if self.positions[i]:
                    self.close_position(i, current_prices[i], 'short')
                collateral, leverage, tp_price, sl_price = self.open_position(i, 'short', current_prices[i])
                infos[self.params['symbols'][i]] = { 'type': 'open_short', 'collateral': collateral, 'leverage': leverage, 'tp_price': tp_price, 'sl_price': sl_price }
            elif action[i] == self.actions_available['close'] or (self.params['reverse_actions'] and action[i] == self.actions_available['hold']):  # Close
                if self.positions[i]:
                    self.close_position(i, current_prices[i], 'close')
                    infos[self.params['symbols'][i]] = { 'type': 'close', 'exit_price': current_prices[i] }
            elif action[i] == self.actions_available['hedge']:  # Hedge
                if self.positions[i]:
                    current_type = 'long' if self.positions[i]['type'] == 'short' else 'short'
                    self.close_position(i, current_prices[i], current_type)
                    collateral, leverage, tp_price, sl_price = self.open_position(i, current_type, current_prices[i])
                    infos[self.params['symbols'][i]] = { 'type': f'open_{current_type}', 'collateral': collateral, 'leverage': leverage, 'tp_price': tp_price, 'sl_price': sl_price }
            elif action[i] == self.actions_available['trail']:  # Trailing Stop
                self.update_trailing_stop(i, current_prices[i])

        # Update net worth
        self.net_worth = self.balance + sum(
            pos.get('position_size', 0) * price for pos, price in zip(self.positions, current_prices)
        )

        # Move to the next step
        if not self.live_mode:
            self.current_step += 1
        
        # if self.params['include_risk_management']:
        #     # Handle SL and TP for each position before processing actions
        #     for i in range(self.num_symbols):
        #         self.handle_risk_management_1s(i) #low_prices[i], high_prices[i])

        # Check if the episode is done
        max_steps_reached = self.current_step >= self.data_matrix.shape[0] - 1
        balance_below_min = self.balance < self.params['collateral_min']
        done = (max_steps_reached or balance_below_min) and not self.live_mode

        # Increment episode counter if done
        if done and not self.live_mode:
            logging.debug(f"Episode {self.episode_counter} completed.")
            logging.debug(f"max_steps_reached: {max_steps_reached}, balance_below_min: {balance_below_min}")
            self.episode_counter += 1

        # Calculate reward
        if self.reward_function:
            reward = self.reward_function(self, action)
        else:
            reward = self.net_worth - self.params['initial_balance']

        # info = {'balance': self.balance, 'net_worth': self.net_worth}

        # Return next observation, reward, done, and info
        return self.next_observation(), reward, done, balance_below_min, infos

    def render(self):
        # Implement rendering logic based on self.render_mode
        if self.render_mode == 'human':
            # Example rendering logic
            logging.debug(f"info: Current Step: {self.current_step}, Length of data_matrix: {self.data_matrix.shape[0]}, Balance: {self.balance}, Net Worth: {self.net_worth}")
            
            # # Plotting balance and net worth as a bar graph
            # plt.figure(figsize=(5, 3))
            # plt.bar(['Balance', 'Net Worth'], [self.balance, self.net_worth], color=['blue', 'green'])
            # plt.title('Balance and Net Worth')
            # plt.ylabel('Amount')
            # plt.show()
        else:
            # Handle other render modes if necessary
            pass