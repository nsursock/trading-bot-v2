import pandas as pd
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import os

# import quantstats as qs

# # extend pandas functionality with metrics, etc.
# qs.extend_pandas()

import sys

# Custom print function
class DualOutput:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None

    def __enter__(self):
        self.file = open(self.filepath, 'w')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()

    def write(self, message):
        # Write to both the file and the console
        sys.__stdout__.write(message)
        if self.file:
            self.file.write(message)

    def flush(self):
        # Flush both the file and the console
        sys.__stdout__.flush()
        if self.file:
            self.file.flush()


def win_rate(returns):
    """Calculate the win rate of a series of returns."""
    wins = np.sum(returns > 0)
    total = len(returns)
    return round(wins / total * 100, 2) if total > 0 else 0

def avg_profit(pnls):
    """Calculate the average profit of a series of pnl values."""
    profits = pnls[pnls > 0]
    return round(np.mean(profits), 2) if profits.size > 0 else 0

def avg_loss(pnls):
    """Calculate the average loss of a series of pnl values."""
    losses = pnls[pnls < 0]
    return round(np.mean(losses), 2) if losses.size > 0 else 0

def net_profit(pnls):
    """Calculate the net profit from a series of pnl values."""
    return np.sum(pnls)


def sharpe(returns):
    returns = np.array(returns)
    if len(returns) < 2:
        return 0  # Not enough data to calculate Sharpe ratio
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Assuming risk-free rate is 0 for simplicity
    sharpe_ratio = mean_return / std_return * np.sqrt(252) if std_return != 0 else 0
    return sharpe_ratio

def max_drawdown(returns):
    returns = np.array(returns)
    if len(returns) < 2:
        return 0  # Not enough data to calculate max drawdown
    
    cumulative_returns = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative_returns)
    
    # Handle cases where peak is zero by setting drawdown to zero
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = np.where(peak == 0, 0, (peak - cumulative_returns) / peak)
    
    # Replace NaN values with zero
    drawdown = np.nan_to_num(drawdown, nan=0.0)
    
    max_drawdown = np.max(drawdown)
    return max_drawdown

def risk_return_ratio(returns):
    # Convert to numpy array if it's a list
    returns = np.array(returns)
    
    gains = returns[returns > 0]
    losses = -returns[returns < 0]  # Convert losses to positive values
    
    if len(gains) == 0 or len(losses) == 0:
        return 0  # Avoid division by zero
    
    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)
    
    return avg_gain / avg_loss if avg_loss != 0 else 0


# def sharpe(returns, risk_free_rate=0.01, trading_days=252):
#     """Calculate the annualized Sharpe ratio of a series of returns."""
#     excess_returns = returns - risk_free_rate
#     mean_excess_return = np.mean(excess_returns)
#     std_dev = np.std(excess_returns)
#     # Annualize the Sharpe ratio by multiplying by sqrt(trading_days)
#     annualized_sharpe = (mean_excess_return / std_dev) * np.sqrt(trading_days) if std_dev != 0 else 0
#     return round(annualized_sharpe, 3)

# def max_drawdown(returns):
#     """Calculate the maximum drawdown of a series of returns with filtering."""
#     # Filter out zero or extreme negative returns
#     returns = returns[returns > -1 + 1e-10]  # Exclude extreme negatives that can cause log issues
    
#     # Calculate log returns only on filtered values
#     log_returns = np.log1p(returns)
#     cumulative = np.exp(np.cumsum(log_returns))  # Cumulative product in exponential scale
#     peak = np.maximum.accumulate(cumulative)     # Running maximum of cumulative returns
#     drawdown = (cumulative - peak) / peak        # Calculate drawdown
    
#     return round(np.nanmin(drawdown), 3)  # Use nanmin for robustness

# def risk_return_ratio(returns):
#     """Calculate the risk to reward ratio of a series of returns."""
#     mean_return = np.mean(returns)
#     std_dev = np.std(returns)
#     return round(mean_return / std_dev, 3) if std_dev != 0 else 0


def count_exit_reasons(history_df, reason):
    """Count the number of occurrences of a specific exit reason."""
    return np.sum(history_df['exit_reason'] == reason)

def log_trade_history(history, save_path='trade_history.csv'):
    
     # Convert history to a DataFrame
    history_df = pd.DataFrame(history)
    
    # Write history_df to a CSV file
    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        if not history_df.empty:
            # Write header
            writer.writerow(history_df.columns)
            # Write data
            for _, record in history_df.iterrows():
                writer.writerow(record.tolist())

def log_cumulative_returns(history, save_path='cumulative_returns.png'):
     # Convert history to a DataFrame
    history_df = pd.DataFrame(history)
    
    # Calculate cumulative returns per symbol
    cumulative_returns = {}
    
    # Group by symbol and calculate cumulative returns
    for symbol, group in history_df.groupby('symbol'):
        cumulative_returns[symbol] = group['return'].cumsum()

    # Create a new figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot cumulative returns
    for symbol, cum_returns in cumulative_returns.items():
        ax.plot(cum_returns, label=f'Cumulative Return: {symbol}')
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Cumulative Return per Symbol')
    ax.legend()

    # Adjust layout and show the plot
    # plt.tight_layout()
    # plt.show()
    plt.savefig(save_path)  # Save the figure to a file
    plt.close()  # Close the figure to free up memory
    
def compute_market_conditions(data_matrix, symbols, feature_index):
    """
    Compute market conditions for each symbol based on mu and sigma.

    :param data_matrix: The data matrix containing features for each symbol.
    :param feature_index: The index of the feature to compute statistics for (e.g., 'close' price).
    :return: Dictionary with market conditions for each symbol.
    """
    # Extract the feature data for all symbols
    feature_data = data_matrix[:, :, feature_index]

    # Compute mean and standard deviation for each symbol
    mu = np.round(np.mean(feature_data, axis=0), 3)
    sigma = np.round(np.std(feature_data, axis=0), 3)

    # Initialize the market conditions dictionary
    market_conditions = {}

    # Define thresholds for categorization
    for i, (mean, std) in enumerate(zip(mu, sigma)):
        if mean > 1500 and std < 200:
            description = "Bullish 5"
        elif mean > 1000 and std < 300:
            description = "Bullish 4"
        elif mean > 500:
            description = "Bullish 3"
        elif mean > 100:
            description = "Bullish 2"
        elif 50 < mean <= 100:
            description = "Bullish 1"
        elif -50 <= mean <= 50:
            description = "Neutral"
        elif -100 < mean <= -50:
            description = "Bearish 1"
        elif mean < -500:
            description = "Bearish 2"
        elif mean < -1000 and std < 300:
            description = "Bearish 3"
        elif mean < -1500 and std < 200:
            description = "Bearish 4"
        else:
            description = "Bearish 5"
        
        # Assuming symbols are indexed as 'Symbol_0', 'Symbol_1', etc.
        symbol = f"{symbols[i]}"
        market_conditions[symbol] = {
            'description': description,
            'mu': mean,
            'sigma': std
        }

    return market_conditions

def display_stats(history, market_conditions, environment, save_path='net_profits.png'):
    # Convert history to a DataFrame
    history_df = pd.DataFrame(history)
    
    if history_df.empty:
        raise ValueError("History data is empty.")
    
    if not market_conditions:
        market_conditions = compute_market_conditions(environment.data_matrix, environment.params['symbols'], environment.mapping['close'])
    
    cols = ['entry_price', 'exit_price', 'sl_price', 'tp_price', 'liq_price', 'max_price']
    
    # Specify the columns to exclude
    columns_to_exclude = []  # Add column names you want to exclude
    
    # Filter out columns that exist in the DataFrame
    columns_to_exclude = [col for col in columns_to_exclude if col in history_df.columns]
    
    for col in cols:
        if col in history_df.columns:
            # Ensure the column is of float type
            history_df[col] = history_df[col].astype(float)
            history_df[col] = history_df[col].apply(lambda x: f"{x:.6e}" if x < 0.0001 else f"{x:.6f}")
    
    # Calculate average trade duration
    if 'open_time' in history_df.columns and 'close_time' in history_df.columns:
        history_df['trade_duration'] = (history_df['close_time'] - history_df['open_time']).dt.total_seconds() / 3600  # Convert to hours
        avg_trade_duration = history_df['trade_duration'].mean()
    else:
        avg_trade_duration = 'N/A'
    
    # Print the history
    print(tabulate(history_df.drop(columns=columns_to_exclude).head(10), headers="keys", tablefmt="pretty", showindex=False))
    
    # Get 5 largest losses and 5 largest gains
    largest_losses = history_df.nsmallest(10, 'return')
    largest_gains = history_df.nlargest(10, 'return')
    
    for col in cols:
        if col in largest_losses.columns:
            # Ensure the column is of float type
            largest_losses[col] = largest_losses[col].astype(float)
            largest_losses[col] = largest_losses[col].apply(lambda x: f"{x:.6e}" if x < 0.0001 else f"{x:.6f}")

    for col in cols:
        if col in largest_gains.columns:
            # Ensure the column is of float type
            largest_gains[col] = largest_gains[col].astype(float)
            largest_gains[col] = largest_gains[col].apply(lambda x: f"{x:.6e}" if x < 0.0001 else f"{x:.6f}")

    # Print largest losses and gains
    print("\nTop 5 Largest Losses:")
    print(tabulate(largest_losses.drop(columns=columns_to_exclude), headers="keys", tablefmt="pretty", showindex=False))
    
    print("\nTop 5 Largest Gains:")
    print(tabulate(largest_gains.drop(columns=columns_to_exclude), headers="keys", tablefmt="pretty", showindex=False))
    
    headers = ["Symbol", "Num Trades", "Win Rate (%)", "Avg Profit", "Avg Loss", "Net Profit", "Sharpe Ratio", "Max Drawdown", "Risk to Reward", "Num Tps", "Num Sls", "Num Liqs", "Num Maxs", "Avg Lev", "Avg Coll", "Avg Dur"]
    colalign = ("left", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right", "right")

    if market_conditions:
        headers.extend(["Mu", "Sigma", "Condition"])
        colalign += ("right", "right", "center")

    # Calculate and display stats for each crypto without using groupby
    unique_symbols = history_df['symbol'].unique()
    crypto_stats = []
    for symbol in sorted(unique_symbols):
        symbol_data = history_df[history_df['symbol'] == symbol]
        stats = {
            'symbol': symbol,
            'num_trades': len(symbol_data),
            'win_rate': f"{win_rate(symbol_data['return']):.2f}",
            'avg_profit': f"{avg_profit(symbol_data['pnl']):,.2f}",
            'avg_loss': f"{avg_loss(symbol_data['pnl']):,.2f}",
            'net_profit': f"{net_profit(symbol_data['pnl']):,.2f}",
            'sharpe_ratio': f"{sharpe(symbol_data['return']):.3f}",
            'max_drawdown': f"{max_drawdown(symbol_data['return']):.3f}",
            'risk_to_reward': f"{risk_return_ratio(symbol_data['return']):.3f}",
            'num_tps': count_exit_reasons(symbol_data, 'tp'),
            'num_sls': count_exit_reasons(symbol_data, 'sl'),
            'num_liqs': count_exit_reasons(symbol_data, 'liq'),
            'num_max': count_exit_reasons(symbol_data, 'max'),  # Assuming 'max' is the reason for max price hits
            'avg_leverage': f"{symbol_data['leverage'].mean():.2f}" if 'leverage' in symbol_data.columns else 'N/A',
            'avg_collateral': f"{symbol_data['collateral'].mean():,.2f}" if 'collateral' in symbol_data.columns else 'N/A',
            'avg_trade_duration': f"{symbol_data['trade_duration'].mean():.2f}" if 'trade_duration' in symbol_data.columns else 'N/A',
        }
        crypto_stats.append(stats)
        
        if market_conditions:
            for symbol, details in market_conditions.items():
                description = details['description']
                mu = details['mu']
                sigma = details['sigma']
                
                for stats in crypto_stats:
                    if stats['symbol'] == symbol:
                        stats['mu'] = mu
                        stats['sigma'] = sigma
                        stats['condition'] = description

    crypto_stats_df = pd.DataFrame(crypto_stats)
    
    print("\nCrypto Stats:")
    print(tabulate(crypto_stats_df, headers=headers, tablefmt="pretty", colalign=colalign, showindex=False))
    
    # Calculate and display stats for the portfolio using custom functions
    portfolio_stats = pd.Series({
        'symbol': 'PF',
        'num_trades': len(history_df),
        'win_rate': f"{win_rate(history_df['return']):.2f}",
        'avg_profit': f"{avg_profit(history_df['pnl']):,.2f}",
        'avg_loss': f"{avg_loss(history_df['pnl']):,.2f}",
        'net_profit': f"{net_profit(history_df['pnl']):,.2f}",
        'sharpe_ratio': f"{sharpe(history_df['return']):.3f}",
        'max_drawdown': f"{max_drawdown(history_df['return']):.3f}",
        'risk_to_reward': f"{risk_return_ratio(history_df['return']):.3f}",
        'num_tps': count_exit_reasons(history_df, 'tp'),
        'num_sls': count_exit_reasons(history_df, 'sl'),
        'num_liqs': count_exit_reasons(history_df, 'liq'),
        'num_max': count_exit_reasons(history_df, 'max'),  # Assuming 'max' is the reason for max price hits
        'avg_leverage': f"{history_df['leverage'].mean():.2f}" if 'leverage' in history_df.columns else 'N/A',
        'avg_collateral': f"{history_df['collateral'].mean():,.2f}" if 'collateral' in history_df.columns else 'N/A',
        'avg_trade_duration': f"{avg_trade_duration:.2f}" if avg_trade_duration != 'N/A' else 'N/A',
    })
    
    if market_conditions:
        mu = 0
        sigma = 0
        for symbol, details in market_conditions.items():
            description = details['description']
            mu += details['mu']
            sigma += details['sigma']
        portfolio_stats['mu'] = round(mu / len(market_conditions), 3)
        portfolio_stats['sigma'] = round(sigma / len(market_conditions), 3)
        portfolio_stats['condition'] = 'portfolio'
    
    print("\nPortfolio Stats:")
    print(tabulate(portfolio_stats.to_frame().T, headers=headers, tablefmt="pretty", colalign=colalign, showindex=False))
    
    # Calculate net profit for each symbol
    net_profits = history_df.groupby('symbol')['pnl'].sum()

    # Calculate the total net profit
    total_net_profit_value = net_profits.sum()

    # Plot net profit against symbol
    plt.figure(figsize=(14, 8))
    net_profits.plot(kind='bar', color='skyblue')

    # Annotate each bar with the percentage of total net profit
    for i, (symbol, net_profit_value) in enumerate(net_profits.items()):
        percentage = (net_profit_value / total_net_profit_value) * 100
        plt.text(i, net_profit_value, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10, color='black')

    plt.title('Net Profit by Symbol')
    plt.xlabel('Symbol')
    plt.ylabel('Net Profit')
    plt.xticks(rotation=45)
    
    plt.savefig(save_path)  # Save the figure to a file
    plt.close()  # Close the figure to free up memory

    return portfolio_stats

def plot_histograms(trade_history, rewards, actions_per_symbol, save_path='histograms.png'):
    collaterals = [trade['collateral'] for trade in trade_history]
    leverages = [trade['leverage'] for trade in trade_history]
    
    # Ensure all actions are 1-dimensional
    actions = [np.ravel(action) for action in actions_per_symbol]
    actions = np.concatenate(actions).tolist()

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))

    # Collaterals
    sns.histplot(collaterals, bins=10, kde=True, ax=axs[0, 0], color='blue')
    axs[0, 0].set_title('Collaterals')
    axs[0, 0].set_xlabel('Collateral')
    axs[0, 0].set_ylabel('Frequency')
    axs[0, 0].set_yscale('log')  # Set y-axis to log scale

    # Annotate with percentages
    for p in axs[0, 0].patches:
        percentage = f'{100 * p.get_height() / len(collaterals):.1f}%'
        axs[0, 0].annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), 
                           textcoords='offset points')

    # Leverages
    sns.histplot(leverages, bins=10, kde=True, ax=axs[0, 1], color='green')
    axs[0, 1].set_title('Leverages')
    axs[0, 1].set_xlabel('Leverage')
    axs[0, 1].set_ylabel('Frequency')
    axs[0, 1].set_yscale('log')  # Set y-axis to log scale

    # Annotate with percentages
    for p in axs[0, 1].patches:
        percentage = f'{100 * p.get_height() / len(leverages):.1f}%'
        axs[0, 1].annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), 
                           textcoords='offset points')

    # Rewards
    sns.histplot(rewards, bins=5, kde=True, ax=axs[1, 0], color='red')
    axs[1, 0].set_title('Rewards')
    axs[1, 0].set_xlabel('Reward')
    axs[1, 0].set_ylabel('Frequency')
    axs[1, 0].set_yscale('log')  # Set y-axis to log scale

    # Annotate with percentages
    for p in axs[1, 0].patches:
        percentage = f'{100 * p.get_height() / len(rewards):.1f}%'
        axs[1, 0].annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), 
                           textcoords='offset points')

    # Actions
    from utilities import ACTIONS_AVAILABLE
    sns.histplot(actions, bins=len(ACTIONS_AVAILABLE), kde=False, ax=axs[1, 1], color='purple')
    axs[1, 1].set_title('Actions')
    axs[1, 1].set_xlabel('Action Type')
    axs[1, 1].set_ylabel('Frequency')
    axs[1, 1].set_yscale('log')  # Set y-axis to log scale

    # Annotate with percentages
    for p in axs[1, 1].patches:
        percentage = f'{100 * p.get_height() / len(actions):.1f}%'
        axs[1, 1].annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha='center', va='center', fontsize=10, color='black', xytext=(0, 5), 
                           textcoords='offset points')

    # plt.tight_layout()
    plt.savefig(save_path)  # Save the figure to a file
    plt.close()  # Close the figure to free up memory

def plot_financial_metrics(trade_history, save_path='financial_metrics.png'):
    symbol = 'Evolution of'
    returns = np.array([trade['return'] for trade in trade_history])
    cumulative_profits = np.cumsum(np.array([trade['pnl'] for trade in trade_history]))
    sharpe_ratios = []
    max_drawdowns = []
    risk_to_reward_ratios = []

    for i in range(1, len(cumulative_profits)):
        sharpe_ratios.append(sharpe(returns[:i+1]))
        max_drawdowns.append(max_drawdown(returns[:i+1]))
        risk_to_reward_ratios.append(risk_return_ratio(returns[:i+1]))

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(16, 12), sharex=True)

    # Cumulative Returns
    axs[0, 0].plot(cumulative_profits, label='Cumulative Returns')
    axs[0, 0].set_title(f'{symbol} Cumulative Returns')
    axs[0, 0].set_ylabel('Cumulative Returns')
    axs[0, 0].legend()

    # Sharpe Ratio
    axs[0, 1].plot(sharpe_ratios, label='Sharpe Ratio', color='orange')
    axs[0, 1].set_title(f'{symbol} Sharpe Ratio')
    axs[0, 1].set_ylabel('Sharpe Ratio')
    axs[0, 1].legend()

    # Max Drawdown
    axs[1, 0].plot(max_drawdowns, label='Max Drawdown', color='red')
    axs[1, 0].set_title(f'{symbol} Max Drawdown')
    axs[1, 0].set_ylabel('Max Drawdown')
    axs[1, 0].legend()

    # Risk to Reward Ratio
    axs[1, 1].plot(risk_to_reward_ratios, label='Risk to Reward Ratio', color='green')
    axs[1, 1].set_title(f'{symbol} Risk to Reward Ratio')
    axs[1, 1].set_ylabel('Risk to Reward Ratio')
    axs[1, 1].set_xlabel('Trade Number')
    axs[1, 1].legend()

    # plt.tight_layout()
    plt.savefig(save_path)  # Save the figure to a file
    plt.close()  # Close the figure to free up memory
    
def plot_training_metrics(metrics, save_path='training_metrics.png'):
    # Create a figure with 2 columns and 5 rows
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D for easy iteration

    # Iterate over each metric and its corresponding subplot
    for ax, (metric, values) in zip(axes, metrics.items()):
        ax.plot(values)
        # Make metric name human-readable
        human_readable_metric = metric.replace('_', ' ').title()
        ax.set_title(metric)
        ax.set_xlabel('Timesteps')
        ax.set_ylabel(metric)

    # Adjust layout to prevent overlap
    # plt.tight_layout()
    plt.savefig(save_path)  # Save the figure to a file
    # plt.show()  # Comment out or remove this line if you don't want to display the plot

def plot_environment_counters(fractal_counter, percentage_counter, fallback_counter, save_path='environment_counters.png'):
    """Plot a histogram of environment counters with percentages on top of each bar."""
    counters = [fractal_counter, percentage_counter, fallback_counter]
    counter_labels = ['Fractal Counter', 'Percentage Counter', 'Fallback Counter']
    
    total = sum(counters)
    percentages = [(count / total) * 100 for count in counters]
    
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=counter_labels, y=counters, hue=counter_labels, palette='pastel', legend=False)
    plt.title('Histogram of Environment Counters')
    plt.xlabel('Counter Type')
    plt.ylabel('Count')
    plt.yscale('log')  # Set y-axis to log scale for better visualization
    
    # Annotate each bar with the percentage
    for i, (count, percentage) in enumerate(zip(counters, percentages)):
        ax.text(i, count, f'{percentage:.1f}%', ha='center', va='bottom', fontsize=10, color='black')
    
    # plt.tight_layout()
    plt.savefig(save_path)  # Save the figure to a file
    plt.close()  # Close the figure to free up memory

def plot_combined_metrics(trade_history, save_path='combined_metrics.png'):
    # Extract returns and cumulative returns
    returns = np.array([trade['return'] for trade in trade_history])
    cumulative_returns = np.cumsum(returns)

    # Create a figure with a specific layout
    fig = plt.figure(figsize=(16, 12))
    grid = plt.GridSpec(2, 2, wspace=0.4, hspace=0.3)

    # Cumulative Returns (Top Left)
    ax_cum_returns = fig.add_subplot(grid[0, 0])
    ax_cum_returns.plot(cumulative_returns, label='Cumulative Returns', color='blue')
    ax_cum_returns.set_title('Cumulative Returns')
    ax_cum_returns.set_xlabel('Trade Number')
    ax_cum_returns.set_ylabel('Cumulative Return')
    ax_cum_returns.legend()

    # Scatter Plot of Returns (Bottom Left)
    ax_scatter = fig.add_subplot(grid[1, 0])
    ax_scatter.scatter(range(len(returns)), returns, color='orange', alpha=0.6)
    ax_scatter.set_title('Scatter Plot of Returns')
    ax_scatter.set_xlabel('Trade Number')
    ax_scatter.set_ylabel('Return')

    # Distribution of Returns (Right, spanning both rows)
    ax_dist = fig.add_subplot(grid[:, 1])
    ax_dist.hist(returns, bins=20, color='green', alpha=0.7)
    ax_dist.set_title('Distribution of Returns')
    ax_dist.set_xlabel('Return')
    ax_dist.set_ylabel('Frequency')

    # Save the figure to a file
    # plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    return cumulative_returns[-1]

def plot_returns_heatmap(history, save_path='returns_heatmap.png'):
    """
    Plots a heatmap showing returns in percentage.
    X-axis: Month, Y-axis: Day of Month.

    Parameters:
    history (list of dict): List containing trade history with 'return' and 'open_time' keys.
    """
    history_df = pd.DataFrame(history)
    
    # Convert 'return' to percentage
    history_df['return'] = history_df['return'] * 100

    # Extract month and day from the open_time
    history_df['month'] = history_df['open_time'].dt.month
    history_df['day'] = history_df['open_time'].dt.day

    # Create a pivot table for the heatmap
    pivot_table = history_df.pivot_table(values='return', index='day', columns='month', aggfunc='mean')

    # Define a custom colormap
    colors = [(1, 0, 0), (0, 1, 0)]  # Red to Green
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_red_green', colors)

    # Define a custom normalization with specified vmin and vmax
    norm = mcolors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=1000)

    # Create a heatmap with the custom colormap and normalization
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(pivot_table, cmap=custom_cmap, norm=norm, annot=True, fmt=".1f", cbar_kws={'label': 'Return (%)'})
    plt.title('Returns Heatmap by Month and Day')
    plt.xlabel('Month')
    plt.ylabel('Day of Month')

    # Add indicators for negative returns
    for text in ax.texts:
        value = float(text.get_text())
        if value < 0:
            # text.set_color('red')  # Change color to blue for negative values
            text.set_weight('bold')  # Make the text bold for emphasis
        if value >= 0:
            # text.set_color('green')  # Change color to blue for negative values
            text.set_weight('bold')  # Make the text bold for emphasis


    # plt.tight_layout()
    plt.savefig(save_path)  # Save the figure to a file
    plt.close()  # Close the figure to free up memory
    
def plot_symbol(symbol, mapping, timestamps, symbol_data, history, title, save_path):
    trade_history = history
    # Convert trade_history to a DataFrame if it's not already
    if not isinstance(trade_history, pd.DataFrame):
        trade_history = pd.DataFrame(trade_history)
    
    # Convert timestamps to pandas DatetimeIndex
    timestamp = pd.to_datetime(timestamps)

    start_date = timestamp[0].strftime('%Y-%m-%d')
    end_date = timestamp[-1].strftime('%Y-%m-%d')

    # Group trade history by episode
    episodes = trade_history['episode'].unique() if 'episode' in trade_history.columns else [None]

    for episode in episodes:
        fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]}, figsize=(15, 10))
        
        close = symbol_data[:, mapping['close']]
        open = symbol_data[:, mapping['open']]
        high = symbol_data[:, mapping['high']]
        low = symbol_data[:, mapping['low']]

        up = close >= open
        down = ~up

        # Ensure width calculation uses datetime64
        width = 0.6 * (timestamp[1] - timestamp[0]) / np.timedelta64(1, 'D')
        width2 = 0.2 * width

        ax1.bar(timestamp[up], close[up]-open[up], width, bottom=open[up], color='green', edgecolor='black')
        ax1.bar(timestamp[up], high[up]-close[up], width2, bottom=close[up], color='green', edgecolor='black')
        ax1.bar(timestamp[up], low[up]-open[up], width2, bottom=open[up], color='green', edgecolor='black')
        ax1.bar(timestamp[down], close[down]-open[down], width, bottom=open[down], color='red', edgecolor='black')
        ax1.bar(timestamp[down], high[down]-open[down], width2, bottom=open[down], color='red', edgecolor='black')
        ax1.bar(timestamp[down], low[down]-close[down], width2, bottom=close[down], color='red', edgecolor='black')

        if history is not None:
            symbol_trades = trade_history[(trade_history['symbol'] == symbol) & 
                                          (trade_history['episode'] == episode if episode is not None else True)].copy()
            symbol_trades['open_time'] = pd.to_datetime(symbol_trades['open_time'], unit='ms')
            
            interval = (timestamp[1] - timestamp[0]) / np.timedelta64(1, 'D')
            
            buys = symbol_trades[symbol_trades['type'] == 'long']
            sells = symbol_trades[symbol_trades['type'] == 'short']

            y_shift = 0.05
            for _, trade in buys.iterrows():
                if interval >= 1:
                    trade_time = trade['open_time'].floor('D')
                    indices = np.where(timestamp.floor('D') == trade_time)[0]
                else:
                    trade_time = trade['open_time'].floor('min')
                    indices = np.where(timestamp.floor('min') == trade_time)[0]

                if indices.size > 0:
                    trade_price = low[indices[0]] * (1 - y_shift)
                    ax1.plot([trade_time, trade_time], [trade_price, low[indices[0]]], 
                             color='g', linewidth=1, linestyle='-')
                    ax1.scatter(trade_time, trade_price, marker='^', color='g', s=100)

            for _, trade in sells.iterrows():
                if interval >= 1:
                    trade_time = trade['open_time'].floor('D')
                    indices = np.where(timestamp.floor('D') == trade_time)[0]
                else:
                    trade_time = trade['open_time'].floor('min')
                    indices = np.where(timestamp.floor('min') == trade_time)[0]
                if indices.size > 0:
                    trade_price = high[indices[0]] * (1 + y_shift)
                    ax1.plot([trade_time, trade_time], [trade_price, high[indices[0]]], 
                             color='r', linewidth=1, linestyle='-')
                    ax1.scatter(trade_time, trade_price, marker='v', color='r', s=100)

        volume_width = 0.6 * width
        ax2.bar(timestamp, symbol_data[:, mapping['volume']], width=volume_width, color=np.where(close >= open, 'green', 'red'), alpha=0.6)

        ax1.set_xticklabels([])
        ax1.set_title(f"{title} ({symbol}) - Episode {episode}\nPeriod: {start_date} to {end_date}")
        ax1.set_ylabel("Price")
        ax2.set_ylabel("Volume")

        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=30, ha='right')
        # plt.tight_layout()
        
        if save_path:
            events_output_dir = os.path.join(save_path, 'events')
            os.makedirs(events_output_dir, exist_ok=True)
            episode_output_dir = os.path.join(events_output_dir, f'episode_{episode}')
            os.makedirs(episode_output_dir, exist_ok=True)
            plt.savefig(os.path.join(episode_output_dir, f'{symbol}.png'))
        else:
            plt.show()

        plt.close()
        
def plot_combined_correlation_heatmap(history, group_by='symbol', save_path='combined_correlation_heatmap.png'):
    """
    Plots a correlation heatmap for combined financial metrics.

    Parameters:
    history (list of dict): The trade history data.
    group_by (str): The column name to group by (e.g., 'symbol').
    save_path (str): The path where the heatmap image will be saved.
    """
    # Convert history to a DataFrame
    history_df = pd.DataFrame(history)

    # Group by the specified column and calculate metrics for each group
    grouped_metrics = []
    for name, group in history_df.groupby(group_by):
        metrics_data = {
            'win_rate': win_rate(group['return']),
            'sharpe_ratio': sharpe(group['return']),
            'max_drawdown': max_drawdown(group['return']),
            'net_profit': net_profit(group['pnl']),
            'risk_to_reward': risk_return_ratio(group['return']),
            'num_tps': count_exit_reasons(group, 'tp'),
            'num_sls': count_exit_reasons(group, 'sl'),
            'num_liqs': count_exit_reasons(group, 'liq'),
            'avg_leverage': group['leverage'].mean() if 'leverage' in group.columns else np.nan,
            'avg_collateral': group['collateral'].mean() if 'collateral' in group.columns else np.nan,
            'avg_duration': (group['close_time'] - group['open_time']).dt.total_seconds().mean() / 3600 if 'close_time' in group.columns and 'open_time' in group.columns else np.nan
        }
        grouped_metrics.append(metrics_data)

    # Convert the list of metrics dictionaries to a DataFrame
    metrics_df = pd.DataFrame(grouped_metrics)

    # Calculate the correlation matrix
    corr = metrics_df.corr()

    # Create a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Correlation Coefficient'})

    # Add title and labels
    plt.title('Correlation Heatmap of Combined Financial Metrics')
    plt.xlabel('Metrics')
    plt.ylabel('Metrics')

    # Save the heatmap to a file
    plt.savefig(save_path)
    plt.close()  # Close the figure to free up memory