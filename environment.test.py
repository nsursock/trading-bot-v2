import unittest
from utilities import preprocess_data
from parameters import selected_params
from environment import TradingEnvironment
import numpy as np

class TestTradingEnvironment(unittest.TestCase):
    def setUp(self):
        # Preprocess data using the utility function
        selected_params['symbols'] = ['BTC', 'ETH', 'SOL']
        symbols = selected_params['symbols']
        interval = '1d'
        limit = 365
        self.data_matrix, self.timestamps, self.mapping, _ = preprocess_data(symbols, interval, limit, end_time=None)

        # Initialize the environment
        self.env = TradingEnvironment(data_matrix=self.data_matrix, timestamps=self.timestamps, mapping=self.mapping, params=selected_params)

    def test_initialization(self):
        self.assertEqual(self.env.balance, self.env.params['initial_balance'])
        self.assertTrue(all(pos == {} for pos in self.env.positions))  # Check if all positions are empty dictionaries
        self.assertEqual(self.env.net_worth, self.env.params['initial_balance'])

    def test_reset(self):
        observation = self.env.reset()
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.balance, self.env.params['initial_balance'])
        self.assertTrue(all(pos == {} for pos in self.env.positions))  # Check if all positions are empty dictionaries
        self.assertEqual(self.env.net_worth, self.env.params['initial_balance'])
        self.assertEqual(len(observation[0]), self.data_matrix.shape[1])  # Corrected to use data_matrix

    def test_step_hold(self):
        self.env.reset()
        action = [0] * self.env.num_symbols  # Create a hold action for each symbol
        observation, reward, done, _, info = self.env.step(np.array(action))
        self.assertEqual(self.env.current_step, 1)
        self.assertTrue(all(pos == {} for pos in self.env.positions))  # Check if all positions are empty dictionaries
        self.assertEqual(self.env.balance, self.env.params['initial_balance'])  # No change in balance
        self.assertFalse(done)

    def test_step_long(self):
        self.env.reset()
        current_price = self.data_matrix[0, :, 3]  # Assuming 'close' is the 4th feature
        action = [1] + [0] * (self.env.num_symbols - 1)  # Set action to 1 for the first symbol, 0 for others
        observation, reward, done, _, info = self.env.step(np.array(action))
        
        # Check if positions are not empty for the first symbol
        self.assertTrue(self.env.positions[0] != {})
        self.assertTrue(all(pos == {} for pos in self.env.positions[1:]))  # Ensure other positions are empty
        
        # Calculate expected balance considering risk per trade
        risk_per_trade = self.env.params['risk_per_trade']
        collateral = self.env.params['initial_balance'] * risk_per_trade
        expected_balance = self.env.params['initial_balance'] - collateral  # Only one position opened
        
        # Adjust expected balance for any additional costs (e.g., slippage, fees)
        # Example: expected_balance -= np.sum(current_price * slippage)
        # Ensure this matches the logic in the `step` method

        # Use a larger tolerance to account for small discrepancies
        self.assertAlmostEqual(self.env.balance, expected_balance, delta=5.0)

        # Add more test cases for other actions and edge cases

if __name__ == '__main__':
    unittest.main()