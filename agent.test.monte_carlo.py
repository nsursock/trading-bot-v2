import unittest
import logging
import traceback
from parameters import selected_params, training_params
from utilities import initialize_environments

class TestTradingAgent(unittest.TestCase):
    
    def setUp(self):
        
        self.train_env, self.eval_env, self.test_env, self.agent, self.market_conditions = initialize_environments(selected_params, training_params) 

    def generate_random_market_conditions(self):
        # Implement logic to generate random market conditions
        # This could involve randomizing price movements, volatility, etc.
        pass

    def test_monte_carlo_analysis(self):
        num_simulations = 100  # Number of Monte Carlo simulations
        initial_balance = 100  # Low initial balance for testing

        results = []

        for _ in range(num_simulations):
            # Generate random market conditions
            random_conditions = self.generate_random_market_conditions()
            
            # Initialize environments with random conditions
            self.train_env, self.eval_env, self.test_env, self.agent, _ = initialize_environments(selected_params, training_params, random_conditions)
            
            # Set a low initial balance
            self.test_env.initial_balance = initial_balance
            
            try:
                # Evaluate the agent
                _, _, _, _, net_worths = self.agent.evaluate(episodes=training_params['num_episodes'])
                
                # Calculate net profit
                net_profit = sum(net_worths) / training_params['num_episodes'] - initial_balance
                results.append(net_profit)
                
            except Exception as e:
                logging.error(f"Simulation failed with exception: {e}")
                logging.error(traceback.format_exc())
                results.append(None)

        # Analyze results
        successful_runs = [result for result in results if result is not None]
        average_profit = sum(successful_runs) / len(successful_runs) if successful_runs else 0
        print(f"Average Net Profit over {num_simulations} simulations: {average_profit:.2f}")

        # Assert that the strategy is profitable on average
        self.assertTrue(average_profit > 0, "Strategy should be profitable on average")
