import json
import ccxt
from dotenv import load_dotenv
import os
from web3 import Web3
import requests
import logging
import argparse

# from interactions import fetch_open_trades

# # Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
# load_dotenv('.env')

# Initialize ccxt Binance exchange
binance = ccxt.binance()

# # Initialize Web3 with Alchemy URL
# web3 = Web3(Web3.HTTPProvider(os.getenv('ALCHEMY_URL')))

# alchemy_url = os.getenv('ALCHEMY_URL')

# # Contract and Wallet details
# contract_address = os.getenv('CONTRACT_ADDRESS')
# private_key = os.getenv('PRIVATE_KEY')
# wallet_address = web3.eth.account.from_key(private_key).address

# # Define the contract ABI
# with open('./abi.json') as f:
#     contract_abi = json.load(f)
# contract = web3.eth.contract(address=contract_address, abi=contract_abi)

def fetch_open_trades(symbol=None):
    try:
        logging.info('Fetching open trades...')

        # Fetch trading history from the provided URL
        url = os.getenv('GAINS_NETWORK_HISTORY')
        print(url)
        response = requests.get(url)
        trading_history = response.json()
        
        # Log the total number of trades fetched
        logging.info(f'Total trades fetched: {len(trading_history)}')

        # Determine if the URL contains 'backend-sepolia'
        is_test_environment = 'backend-sepolia' in url
        historyStartIndex = int(os.getenv('HISTORY_START')) # if is_test_environment else 0

        # Filter for open trades with tradeIndex >= 200
        open_trades_history = [
#            {k: v for k, v in trade.items() if k not in ['address', 'tx']}  # Exclude 'address' and 'tx' keys
            trade for trade in trading_history 
            if trade['tradeIndex'] is not None and trade['tradeIndex'] >= historyStartIndex and
            trade['action'].startswith('TradeOpened') and 
            not any(closed_trade['tradeIndex'] == trade['tradeIndex'] and closed_trade['action'].startswith('TradeClosed') for closed_trade in trading_history)
        ]
        
        # Print trading_history using tabulate
        # print(tabulate(open_trades_history, headers="keys"))
        
        # Filter by symbol if provided
        if symbol:
            open_trades_history = [trade for trade in open_trades_history if trade['pair'].split('/')[0] == symbol]
        
        logging.info(f'Found {len(open_trades_history)} open trades from the trading history.')

        # Log details of the open trades for debugging
        for trade in open_trades_history:
            logging.debug(f'Open trade: {trade}')

        # Extract pair (symbol) and orderId (index) from final open positions
        trade_details = [(position['pair'].split('/')[0], position['tradeIndex'], position['long'], position['price'], position['size'], position['leverage']) for position in open_trades_history]
        logging.debug(f'Trade details: {trade_details}')

        logging.info('Completed fetching open trades.')
        return trade_details
    except Exception as error:
        logging.error('Error fetching open trades:', exc_info=True)
        raise

def check_wallet_balance():
    balance = web3.eth.get_balance(wallet_address)
    logging.info(f'Wallet balance: {web3.from_wei(balance, "ether")} ETH')
    return balance

def update_stop_loss(trade_index, new_sl):
    try:
        # Check wallet balance
        balance = check_wallet_balance()
        # if balance < web3.to_wei(0.002, 'ether'):  # Example threshold
        #     logging.error('Insufficient funds for gas. Please add more ETH to the wallet.')
        #     return

        # Convert new_sl to the required precision (1e10)
        new_sl_precise = int(new_sl * 1e10)
        logging.info(f'New stop-loss price: {new_sl_precise}')

        # Get the current base fee and set max fee per gas
        base_fee = web3.eth.get_block('latest')['baseFeePerGas']
        max_fee_per_gas = base_fee + web3.to_wei(2, 'gwei')  # Add a buffer to the base fee
        max_priority_fee_per_gas = web3.to_wei(1, 'gwei')  # Set a priority fee

        logging.debug(f'Base fee: {base_fee}, Max fee per gas: {max_fee_per_gas}, Max priority fee per gas: {max_priority_fee_per_gas}')

        # Prepare the transaction
        transaction = contract.functions.updateSl(trade_index, new_sl_precise).build_transaction({
            'from': wallet_address,
            'nonce': web3.eth.get_transaction_count(wallet_address),
            'maxFeePerGas': max_fee_per_gas,
            'maxPriorityFeePerGas': max_priority_fee_per_gas,
        })

        # Estimate the gas limit
        estimated_gas = web3.eth.estimate_gas(transaction)
        transaction['gas'] = estimated_gas

        logging.debug(f'Transaction details: {transaction}')

        # Sign the transaction
        signed_txn = web3.eth.account.sign_transaction(transaction, private_key=private_key)

        # Send the transaction
        txn_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)

        # Wait for the transaction to be mined
        txn_receipt = web3.eth.wait_for_transaction_receipt(txn_hash)

        logging.info(f'Stop-loss updated for trade index {trade_index}. Transaction hash: {txn_hash.hex()}')
        return txn_receipt
    except Exception as error:
        logging.error('Error updating stop-loss:', exc_info=True)
        raise

def fetch_latest_ticker(symbol):
    try:
        logging.info(f'Fetching latest ticker for {symbol}')
        if symbol.startswith('RNDR'):
            symbol = 'RENDER/USDT'
        ticker = binance.fetch_ticker(symbol)
        return ticker['last']
    except Exception as error:
        logging.error(f'Error fetching ticker for {symbol}: {str(error)}', exc_info=True)
        raise

def update_stop_loss_for_profitable_trades(stop_loss_percentage=None, pnl_threshold=None):
    try:
        open_trades = fetch_open_trades()
        for trade in open_trades:
            symbol, trade_index, is_long, entry_price, size, leverage = trade

            # Ensure entry_price is not None
            if entry_price is None:
                logging.warning(f'Skipping trade index {trade_index} due to missing entry price.')
                continue

            # Fetch the latest ticker price
            latest_price = fetch_latest_ticker(f"{symbol}/USDT")
            
            # Ensure latest_price is not None
            if latest_price is None:
                logging.warning(f'Skipping trade index {trade_index} due to missing latest price.')
                continue

            # Calculate the PnL
            if is_long:
                pnl = (latest_price - entry_price)
            else:
                pnl = (entry_price - latest_price)

            # Calculate the PnL percentage
            pnl_percentage = (pnl / entry_price) * leverage * 100

            # Determine the stop-loss percentage to use
            sl_percentage = stop_loss_percentage if stop_loss_percentage is not None else 2

            # Apply stop-loss if no threshold is specified or if PnL percentage exceeds the threshold
            if pnl_threshold is None or pnl_percentage >= pnl_threshold:
                # Calculate the new stop-loss price based on the specified percentage and leverage
                if is_long:
                    new_sl_price = latest_price - (latest_price * sl_percentage / 100 / leverage)
                else:
                    new_sl_price = latest_price + (latest_price * sl_percentage / 100 / leverage)
                
                logging.info(f'New stop-loss price: {new_sl_price}')

                # Update the stop-loss
                update_stop_loss(trade_index, new_sl_price)
                logging.info(f'Updated stop-loss for trade index {trade_index} to {new_sl_price} with PnL percentage {pnl_percentage}')
            else:
                logging.info(f'Skipped updating stop-loss for trade index {trade_index} as PnL percentage {pnl_percentage} is below threshold {pnl_threshold}')
    except Exception as error:
        logging.error('Error updating stop-loss for profitable trades:', exc_info=True)
        raise

def main():
    global web3, contract_address, wallet_address, private_key, contract
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Update stop-loss for profitable trades.')
    parser.add_argument('-e', '--env', type=str, help='Environment to use (sepolia or arbitrum).')
    parser.add_argument('-s', '--stop_loss_percentage', type=float, default=3, help='Stop-loss percentage to use.')
    parser.add_argument('-p', '--pnl_threshold', type=float, default=None, help='PnL threshold percentage.')

    # Parse the arguments
    args = parser.parse_args()

    # Load the appropriate environment file
    if args.env:
        env_file = f".env.{args.env}"
    else:
        env_file = ".env"

    # Load environment variables from the specified file
    load_dotenv(env_file)

    # Initialize Web3 with Alchemy URL after loading environment variables
    web3 = Web3(Web3.HTTPProvider(os.getenv('ALCHEMY_URL')))

    # Contract and Wallet details
    contract_address = os.getenv('CONTRACT_ADDRESS')
    private_key = os.getenv('PRIVATE_KEY')
    wallet_address = web3.eth.account.from_key(private_key).address

    # Define the contract ABI
    with open('./abi.json') as f:
        contract_abi = json.load(f)
    contract = web3.eth.contract(address=contract_address, abi=contract_abi)

    open_trades = fetch_open_trades()
    print(open_trades)

    # Use the parsed arguments
    update_stop_loss_for_profitable_trades(stop_loss_percentage=args.stop_loss_percentage, pnl_threshold=args.pnl_threshold)

if __name__ == "__main__":
    main()

