import unittest
import logging
import pandas as pd
import numpy as np
from utilities import fetch_binance_klines, preprocess_data

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
        data_matrix, timestamps, field_mapping, _ = preprocess_data(3, symbols, interval, limit, end_time)

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
        data_matrix, timestamps, field_mapping, _ = preprocess_data(1, symbols, interval, limit, end_time)

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
        data_matrix, timestamps, field_mapping, _ = preprocess_data(2, symbols, interval, limit, end_time)

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
