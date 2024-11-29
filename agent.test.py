import os
import sys
import traceback
import pandas as pd
from datetime import datetime
import logging
from tabulate import tabulate
import unittest

from parameters import selected_params, training_params, log_parameters
from reporting import *

class TestTradingAgent(unittest.TestCase):
    
    def setUp(self):
        from utilities import initialize_environments
        self.train_env, self.eval_env, self.test_env, self.agent, self.market_conditions = initialize_environments(selected_params, training_params) 

    # def test_train(self):
    #     # Test if the train method runs without errors
    #     try:
    #         self.agent.train(timesteps=3000)
    #         success = True
    #     except Exception as e:
    #         success = False
    #         logging.error(f"Training failed with exception: {e}")
        
    #     self.assertTrue(success, "Training should complete without exceptions")

    def test_evaluate(self):
        # Test if the evaluate method runs without errors
        try:
            # Calculate and print the period from the timestamps
            start_time = self.test_env.timestamps[0]
            end_time = self.test_env.timestamps[-1]
            period = end_time - start_time
            days_covered = period.days
            
            # Create directory for saving plots
            end_date = pd.to_datetime(end_time).strftime('%Y-%m-%d')
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            plot_dir = f"eval_{timestamp}_{end_date}"
            os.makedirs(plot_dir, exist_ok=True)
            
            # Train the agent
            if training_params['train_model']:
                self.agent.train(timesteps=training_params['timesteps'], output_dir=plot_dir)

            # Evaluate on the test set
            actions_history, rewards_history, episode_durations, balances, net_worths = self.agent.evaluate(episodes=training_params['num_episodes'])

            # Redirect print to both console and file
            sys.stdout = DualOutput(os.path.join(plot_dir, "output_recap.log"))
            
            # Reporting (Dual output: screen and file)
            history = self.test_env.history
            log_parameters(selected_params)
            log_parameters(training_params)
            portfolio_stats = display_stats(history, self.market_conditions, self.test_env, save_path=os.path.join(plot_dir, "net_profits.png"))

            # Save plots to the directory
            strategy_return_3 = plot_combined_metrics(history, save_path=os.path.join(plot_dir, "combined_metrics.png"))
            plot_financial_metrics(history, save_path=os.path.join(plot_dir, "financial_metrics.png"))
            plot_histograms(history, rewards_history, actions_history, save_path=os.path.join(plot_dir, "histograms.png"))
            plot_environment_counters(self.test_env.fractal_counter, self.test_env.percentage_counter, self.test_env.fallback_counter, save_path=os.path.join(plot_dir, "environment_counters.png"))
            plot_returns_heatmap(history, save_path=os.path.join(plot_dir, "returns_heatmap.png"))
            plot_combined_correlation_heatmap(history, save_path=os.path.join(plot_dir, "metrics_correlation_heatmap.png"))
            # plot_additional_metrics_correlation_heatmap(history, save_path=os.path.join(plot_dir, "additional_metrics_correlation_heatmap.png"))
            for symbol_index, symbol in enumerate(self.test_env.params['symbols']):
                plot_symbol(symbol, self.test_env.mapping, self.test_env.timestamps, self.test_env.data_matrix[:, symbol_index, :], self.test_env.history, title=symbol, save_path=plot_dir)

            # Log outputs to the directory
            log_trade_history(history, save_path=os.path.join(plot_dir, "trade_history.csv"))
            log_cumulative_returns(history, save_path=os.path.join(plot_dir, "cumulative_returns.png"))

            # Use the new function to calculate strategy return
            initial_balance = selected_params['initial_balance']
            net_profit_1 = float(portfolio_stats['net_profit'].replace(',', '')) / training_params['num_episodes']  # Remove thousands separator # self.test_env.balance - initial_balance
            net_profit_2 = sum(net_worths) / training_params['num_episodes'] - initial_balance
            num_trades = portfolio_stats['num_trades'] / training_params['num_episodes']
            
            # Prepare data for tabulation
            data = [
                ["Start Date", start_time.date()],
                ["Start Time", start_time.time()],
                ["End Date", end_time.date()],
                ["End Time", end_time.time()],
                ["Days Covered", days_covered],
                ["Avg Net Profit", f"{net_profit_1:,.2f}"],
                ["Avg Num Trades", f"{num_trades:,.2f}"],
                ["Strategy Return", f"{net_profit_1 / initial_balance * 100:.2f}%"],
                # ["Strategy Return 2", f"{net_profit_2 / initial_balance * 100:.2f}%"],
                # ["Strategy Return 3", f"{strategy_return_3:.2f}%"]
            ]

            # Print the table
            print(tabulate(data, headers=["Metric", "Value"], tablefmt="pretty"))

            # # At the end of your script, ensure the file is closed
            # sys.stdout.close()

            success = True
        except Exception as e:
            success = False
            logging.error(f"Evaluation failed with exception: {e}")
            logging.error("Traceback details:")
            logging.error(traceback.format_exc())
        
        self.assertTrue(success, "Evaluation should complete without exceptions")

if __name__ == '__main__':
    unittest.main()