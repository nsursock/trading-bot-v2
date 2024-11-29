from reporting import *
import unittest
import pandas as pd
import numpy as np

class TestReportingFunctions(unittest.TestCase):

    def setUp(self):
        # Create a fictive history DataFrame with all necessary columns
        np.random.seed(0)  # For reproducibility
        self.history = pd.DataFrame({
            'symbol': ['BTC', 'BTC', 'ETH', 'ETH', 'ETH'],
            'return': [0.05, -0.02, 0.03, -0.01, 0.04],
            'pnl': [500, -200, 300, -100, 400],
            'exit_reason': ['tp', 'sl', 'tp', 'liq', 'tp'],
            'sl_price': [10000 + np.random.uniform(-50, 50) for _ in range(5)],
            'tp_price': [10500 + np.random.uniform(-50, 50) for _ in range(5)],
            'liq_price': [9500 + np.random.uniform(-50, 50) for _ in range(5)]
        })

    def test_win_rate(self):
        returns = self.history['return']
        self.assertEqual(win_rate(returns), 60.0)

    def test_avg_profit(self):
        pnls = self.history['pnl']
        self.assertEqual(avg_profit(pnls), 400.0)

    def test_avg_loss(self):
        pnls = self.history['pnl']
        self.assertEqual(avg_loss(pnls), -150.0)

    def test_net_profit(self):
        pnls = self.history['pnl']
        self.assertEqual(net_profit(pnls), 900)

    def test_sharpe(self):
        returns = self.history['return']
        self.assertAlmostEqual(sharpe(returns), 0.287, places=3)

    def test_max_drawdown(self):
        returns = self.history['return']
        self.assertAlmostEqual(max_drawdown(returns), -0.02, places=3)

    def test_risk_return_ratio(self):
        returns = self.history['return']
        self.assertAlmostEqual(risk_return_ratio(returns), 0.646, places=3)

    def test_display_stats(self):
        # This test is more about ensuring no exceptions are raised
        # and the function runs with the given data.
        try:
            display_stats(self.history)
        except Exception as e:
            self.fail(f"display_stats raised an exception {e}")

if __name__ == '__main__':
    unittest.main()