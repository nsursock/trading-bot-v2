from bayes_opt import BayesianOptimization
import os
import sys
import pandas as pd
from datetime import datetime
from parameters import log_parameters
from reporting import display_stats, DualOutput, plot_combined_correlation_heatmap,plot_combined_metrics, plot_financial_metrics, plot_histograms, plot_environment_counters, plot_returns_heatmap
import hashlib

def print_best_params(results, top_n):
    for i, result in enumerate(results[:top_n]):
        # Convert parameter indices to real values
        real_values = get_real_values(result['params'])
        
        # Merge real values with original params, removing replaced items
        merged_params = {**result['params'], **real_values}
        
        # Remove the index-based keys from merged_params
        index_keys = ['batch_size_index', 'n_steps_index', 'interval_index', 'clip_range_index', 'learning_rate_index']
        for key in index_keys:
            merged_params.pop(key, None)
        
        # Sort the merged parameters by key
        sorted_params = dict(sorted(merged_params.items()))
        
        # Create a config ID for these sorted params
        config_id = create_config_id(sorted_params)
        
        # Find the output directory
        output_dir = output_dirs.get(config_id, "N/A")
        
        # Print the object with target, sorted params, and output_dir
        result_object = {
            'target': result['target'],
            'params': sorted_params,
            'output_dir': output_dir
        }
        print(f"params_{i+1} = {result_object}")

# Global dictionary to store the output directory for each configuration
output_dirs = {}

# Define the list of n_steps and batch sizes (powers of 2)
possible_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
n_steps_list = [512, 1024, 2048, 4096]
batch_sizes = [32, 64, 128, 256]
schedule_types = ['linear', 'exponential', 'step']

def get_real_values(params):
    # Convert indices to real values
    real_values = {
        'batch_size': batch_sizes[int(round(params['batch_size_index']))],
        'n_steps': n_steps_list[int(round(params['n_steps_index']))],
        'interval': possible_intervals[int(round(params['interval_index']))],
        'clip_range_schedule': schedule_types[int(round(params['clip_range_index']))],
        'learning_rate_schedule': schedule_types[int(round(params['learning_rate_index']))],
    }
    
    return real_values

def create_config_id(params):
    """Create a unique identifier for the given parameters."""
    params_str = str(sorted((k, v) for k, v in params.items()))
    return hashlib.md5(params_str.encode()).hexdigest()

# Define the objective function
def objective(timesteps, num_episodes, n_steps_index, batch_size_index, n_epochs, learning_rate, clip_range, clip_range_index, learning_rate_index, gae_lambda, ent_coef, vf_coef, max_grad_norm, initial_balance, collateral_min, collateral_max, boost_factor, leverage_min, leverage_max, risk_per_trade, tp_mult_perc, sl_mult_perc, interval_index, limit, cooldown_period, trading_penalty, kelly_fraction):
    global output_dirs  # Declare the global dictionary
    original_stdout = sys.stdout  # Store the original sys.stdout

    # Calculate the maximum allowable limit based on the interval
    interval = possible_intervals[int(round(interval_index))]
    interval_days = {
        '1m': 1/1440,  # 1 minute
        '3m': 1/480,   # 3 minutes
        '5m': 1/288,   # 5 minutes
        '15m': 1/96,   # 15 minutes
        '30m': 1/48,   # 30 minutes
        '1h': 1/24,    # 1 hour
        '2h': 1/12,    # 2 hours
        '4h': 1/6,     # 4 hours
        '6h': 1/4,     # 6 hours
        '8h': 1/3,     # 8 hours
        '12h': 1/2,    # 12 hours
        '1d': 1        # 1 day
    }
    max_limit = int(200 / interval_days[interval])
    limit = min(limit, max_limit)

    bayesian_params = {
        'interval': interval,
        'limit': int(limit),
        'boost_factor': float(boost_factor),
        'initial_balance': int(initial_balance),
        'leverage_min': int(leverage_min),
        'leverage_max': int(leverage_max),
        'collateral_min': int(collateral_min),
        'collateral_max': int(collateral_max),
        'risk_per_trade': float(risk_per_trade),
        'tp_mult_perc': float(tp_mult_perc),
        'sl_mult_perc': float(sl_mult_perc),
        'cooldown_period': int(cooldown_period),
        'trading_penalty': float(trading_penalty),
        'kelly_fraction': float(kelly_fraction),
    }

    constant_params = {
        # 'symbols': sorted(['BTC', 'ETH', 'BNB', 'SOL', 'NEAR', 'LINK', 'ADA', 'SHIB', 'BONK', 'PEPE']),
        'symbols': sorted(['ADA', 'BNB', 'EOS', 'ETH', 'IOTA', 'LTC', 'NEO', 'QTUM', 'XLM', 'XRP']),
        'end_time': '2021-08-31',
        'adjust_leverage': True,
        'risk_mgmt': 'fractals',
        'reverse_actions': False,
        'trading_fee': 0.008,  # 0.1% trading fee
        'slippage': 0.0005,    # 0.05% slippage 
        'bid_ask_spread': 0.0002,  # 0.02% bid-ask spread
        'borrowing_fee_per_hour': 0.0001,  # 0.01% per hour
        'market_data': 'original', # 'original' or 'random' or 'synthetic'
        'model_name': 'model_ppo_crypto_trading',
        'basic_risk_mgmt': False
    }
    
    training_params = {
        'train_model': True,
        'training_mode': 'custom',
        'timesteps': int(timesteps),
        'num_episodes': int(num_episodes),
        'n_steps': n_steps_list[int(round(n_steps_index))],
        'batch_size': batch_sizes[int(round(batch_size_index))],
        'n_epochs': int(n_epochs),
        'learning_rate': float(learning_rate),
        'clip_range': float(clip_range),
        'gae_lambda': float(gae_lambda),
        'ent_coef': float(ent_coef),
        'vf_coef': float(vf_coef),
        'max_grad_norm': float(max_grad_norm),
        'clip_range_schedule': schedule_types[int(round(clip_range_index))],
        'learning_rate_schedule': schedule_types[int(round(learning_rate_index))],
    }

    financial_params = constant_params.copy()
    financial_params.update(bayesian_params)

    all_params = financial_params.copy()
    all_params.update(training_params)

    # Filter all_params to keep only the specified keys
    allowed_keys = {'batch_size', 'boost_factor', 'clip_range', 'clip_range_schedule', 'collateral_max', 'collateral_min', 'cooldown_period', 'ent_coef', 'gae_lambda', 'initial_balance', 'interval', 'kelly_fraction', 'learning_rate', 'learning_rate_schedule', 'leverage_max', 'leverage_min', 'limit', 'max_grad_norm', 'n_epochs', 'n_steps', 'num_episodes', 'risk_per_trade', 'sl_mult_perc', 'timesteps', 'tp_mult_perc', 'trading_penalty', 'vf_coef'}
    all_params = {k: v for k, v in all_params.items() if k in allowed_keys}

    config_id = create_config_id(all_params)
    print(f"Generated config_id: {config_id}")  # Debug print

    # Create directory for saving plots
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    plot_dir = f"optim_{timestamp}"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Created plot directory: {plot_dir}")  # Debug print

    from utilities import initialize_environments
    log_parameters(financial_params)
    log_parameters(training_params)
    train_env, eval_env, test_env, agent, market_conditions = initialize_environments(financial_params, training_params, plot_dir) 
    
    # Train the agent
    # if training_params['train_model']:
    agent.train(timesteps=training_params['timesteps'], output_dir=plot_dir)

    # Evaluate on the test set
    actions_history, rewards_history, episode_durations, balances, net_worths = agent.evaluate(episodes=training_params['num_episodes'])

    # Redirect print to both console and file
    with DualOutput(os.path.join(plot_dir, "output_recap.log")) as dual_output:
        sys.stdout = dual_output

        # Reporting (Dual output: screen and file)
        history = test_env.history
        log_parameters(financial_params)
        log_parameters(training_params)

        try:
            # Save plots to the directory
            portfolio_stats = display_stats(history, market_conditions, test_env, save_path=os.path.join(plot_dir, "net_profits.png"))
            plot_combined_metrics(history, save_path=os.path.join(plot_dir, "combined_metrics.png"))
            plot_financial_metrics(history, save_path=os.path.join(plot_dir, "financial_metrics.png"))
            plot_histograms(history, rewards_history, actions_history, save_path=os.path.join(plot_dir, "histograms.png"))
            plot_environment_counters(test_env.fractal_counter, test_env.percentage_counter, test_env.fallback_counter, save_path=os.path.join(plot_dir, "environment_counters.png"))
            plot_returns_heatmap(history, save_path=os.path.join(plot_dir, "returns_heatmap.png"))
            plot_combined_correlation_heatmap(history, save_path=os.path.join(plot_dir, "combined_correlation_heatmap.png"))
        except Exception as e:
            print(f"Error in reporting: {e}")
            return 0
        finally:
            # Ensure sys.stdout is restored even if an error occurs
            sys.stdout = original_stdout

    # Restore the original sys.stdout
    sys.stdout = original_stdout

    # Use the new function to calculate strategy return
    initial_balance = financial_params['initial_balance']
    avg_net_profit = float(portfolio_stats['net_profit'].replace(',', '')) / training_params['num_episodes']
    output_dirs[config_id] = plot_dir
    print(f"Stored plot directory in output_dirs: {output_dirs[config_id]}")  # Debug print

    perf_score = avg_net_profit / initial_balance * 100
    return perf_score

# Define the parameter bounds
pbounds = {
    # Financial parameters
    'boost_factor': (1.0, 50.0),
    'leverage_min': (1, 20),
    'leverage_max': (50, 150),
    'risk_per_trade': (0.01, 0.2),
    'tp_mult_perc': (0.1, 1.0),
    'sl_mult_perc': (0.1, 1.0),
    'initial_balance': (1000, 5000),
    'cooldown_period': (5, 20),
    'trading_penalty': (0.01, 5),
    'kelly_fraction': (0.1, 0.75),
    'interval_index': (0, 11),
    'limit': (250, 1000),
    'collateral_min': (50, 1000),
    'collateral_max': (1000, 1_000_000),
    # Training parameters
    'timesteps': (20_000, 50_000),
    'num_episodes': (10, 50),
    'n_steps_index': (0, 3),  # Index for n_steps
    'batch_size_index': (0, 3),  # Index for batch sizes
    'n_epochs': (3, 30),
    'learning_rate': (1e-5, 1e-2),
    'clip_range': (0.1, 0.4),
    'gae_lambda': (0.8, 1.0),
    'ent_coef': (0.0, 0.1),
    'vf_coef': (0.1, 1.0),
    'max_grad_norm': (0.3, 1.0),
    'clip_range_index': (0, 2),
    'learning_rate_index': (0, 2),
}

def main():

    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(
        f=objective,
        pbounds=pbounds,
        random_state=1,
        verbose=2,
        allow_duplicate_points=True,  # Allow duplicate points for categorical
    )

    # # Perform optimization
    # optimizer.maximize(
    #     init_points=5,
    #     n_iter=45
    # )


    # Perform optimization
    optimizer.maximize(
        init_points=5,
        n_iter=20
    )

    # Sort and list the best parameters by performance score
    results = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)
    top_n = 10  # Number of top configurations to list

    print_best_params(results, top_n)

if __name__ == "__main__":
    main()