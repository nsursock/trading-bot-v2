import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class TestSyntheticData(unittest.TestCase):

    def test_generate_gbm_prices(self):
        from synthetic import generate_gbm_prices
        
        S0 = 100
        mu = 0.05
        sigma = 0.2
        T = 1  # 1 day
        dt = 1 / (24 * 60)  # 1 minute intervals

        open_prices, high_prices, low_prices, close_prices = generate_gbm_prices(S0, mu, sigma, T, dt)
        
        self.assertEqual(len(open_prices), len(high_prices))
        self.assertEqual(len(low_prices), len(close_prices))
        self.assertTrue(all(price > 0 for price in open_prices), "All open prices should be positive")

    def test_generate_realistic_volumes(self):
        from synthetic import generate_realistic_volumes
        
        N = 1440  # Number of minutes in a day
        volumes = generate_realistic_volumes(N)

        self.assertEqual(len(volumes), N, "Volume array length should match the number of time steps")
        self.assertTrue(all(volume > 0 for volume in volumes), "All volumes should be positive")

    def test_create_synthetic_data(self):
        from synthetic import create_synthetic_data
        
        limit = 1  # 1 day
        interval = '1h'  # 1-hour interval

        resampled_data_matrix, full_data_matrix, resampled_timestamps, mapping, valid_symbols, market_conditions = create_synthetic_data(limit, interval)

        self.assertGreater(len(resampled_timestamps), 0, "Resampled timestamps should not be empty")
        self.assertGreater(len(valid_symbols), 0, "There should be valid symbols generated")
        self.assertGreater(len(market_conditions), 0, "Market conditions should not be empty")

    def test_visualize_synthetic_data(self):
        from synthetic import create_synthetic_data
        from reporting import plot_symbol
        
        # Use the create_synthetic_data function to generate data
        limit = 4  # 4 days
        interval = '1h'  # 1-hour interval
        resampled_data_matrix, full_data_matrix, resampled_timestamps, mapping, valid_symbols, market_conditions = create_synthetic_data(limit, interval, 'training')

        # Visualize the first symbol's data using plot_symbol
        for index, symbol in enumerate(valid_symbols):
            symbol_data = resampled_data_matrix[:, index, :]  # Directly use the numpy array
            trade_history = None  # Empty trade history for visualization
            title = 'Synthetic Data Visualization'
            save_path = 'synthetic_data_visualization'
            plot_symbol(symbol, mapping, resampled_timestamps, symbol_data, trade_history, title, save_path)

if __name__ == '__main__':
    unittest.main()