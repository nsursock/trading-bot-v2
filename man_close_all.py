import json
import ccxt
from dotenv import load_dotenv
import os
from web3 import Web3
import requests
import logging
import argparse

# from man_adjust_sl import fetch_open_trades
# from interactions import ContractLogicError
# from interactions import fetch_open_trades

# # Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
# load_dotenv('.env')

# Initialize ccxt Binance exchange
binance = ccxt.binance()

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

def close_all_open_trades():
    try:
        trade_details = fetch_open_trades()
        print('Number of open trades:', len(trade_details))
        
        for trade in trade_details:
            symbol, order_id, is_long = trade[:3]
            ticker = binance.fetch_ticker(symbol + '/USDT')
            latest_price = ticker['last']
            
            if latest_price is None:
                logging.error(f"Failed to fetch latest price for symbol {symbol}. Skipping trade with order_id: {order_id}")
                continue  # Skip this trade if the price is None
            
            logging.info(f"Fetched latest price for symbol {symbol} ({'long' if is_long == 1 else 'short'}) and market order {order_id}: {latest_price}")
            
            try:
                close_trade(order_id, latest_price, is_long, 0.05)
                logging.info(f"Successfully closed trade with order_id: {order_id}")
            except Exception as e:
                logging.error(f"Failed to close trade with order_id: {order_id}. Error: {e}")

    except Exception as error:
        logging.error('Error in main function: %s', error)
    
def close_trade(trade_index, expected_price, is_long, slippage_tolerance=0.01):
    try:
        logging.info(f"Attempting to close trade with trade_index: {trade_index}, expected_price: {expected_price}, is_long: {is_long}")

        # Debugging addresses
        logging.debug(f"Contract address: {contract_address}")
        logging.debug(f"Wallet address: {wallet_address}")

        # Check if addresses are valid
        if not Web3.is_address(contract_address):
            logging.error(f"Invalid contract address: {contract_address}")
            return
        if not Web3.is_address(wallet_address):
            logging.error(f"Invalid wallet address: {wallet_address}")
            return

        nonce = web3.eth.get_transaction_count(wallet_address)
        logging.debug(f"Nonce for transaction: {nonce}")

        gas_price = web3.eth.gas_price
        logging.debug(f"Current gas price: {gas_price}")

        trade_index = int(trade_index)  # Ensure trade_index is an integer
        expected_price = int(expected_price * 1e10)  # Convert expected_price to uint64 by scaling

        # Adjust expected_price for slippage tolerance based on long/short position
        if is_long:
            min_expected_price = int(expected_price * (1 - slippage_tolerance))
            logging.debug(f"Long position: Min expected_price with slippage: {min_expected_price}")
        else:
            max_expected_price = int(expected_price * (1 + slippage_tolerance))
            logging.debug(f"Short position: Max expected_price with slippage: {max_expected_price}")

        # Build the transaction data
        adjusted_expected_price = min_expected_price if is_long else max_expected_price
        tx_data = contract.functions.closeTradeMarket(trade_index, adjusted_expected_price).build_transaction({
            'from': wallet_address,
            'gas': 300000,  # Temporary gas limit for estimation
            'gasPrice': gas_price,
            'nonce': nonce,
            'chainId': web3.eth.chain_id
        })
        logging.debug(f"Transaction data built: {tx_data}")

        # Estimate gas limit
        try:
            gas_limit = web3.eth.estimate_gas({
                'to': contract_address,
                'from': wallet_address,
                'data': tx_data['data']
            })
            logging.debug(f'Estimated gas limit: {gas_limit}')
        except Exception as e:
            logging.error(f"Gas estimation failed: {e}")
            # decode_error(e, tx_data)
            return

        tx = {
            'to': contract_address,
            'data': tx_data['data'],
            'gas': int(gas_limit * 2),
            'gasPrice': gas_price,
            'nonce': nonce,
            'chainId': web3.eth.chain_id
        }
        logging.debug(f"Final transaction data: {tx}")

        signed_tx = web3.eth.account.sign_transaction(tx, private_key)
        logging.debug(f"Signed transaction: {signed_tx}")

        tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        logging.debug(f"Transaction sent, hash: {tx_hash.hex()}")

        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        logging.debug(f"Transaction receipt: {receipt}")

        logging.info(f"Closed trade with trade_index: {trade_index}, Transaction hash: {receipt.transactionHash.hex()}")
    # except ContractLogicError as custom_error:
    #     logging.error(f"Contract logic error occurred: {custom_error}")
    #     # decode_error(custom_error, tx_data)
    #     print(f"An error occurred during trade closing for orderId {trade_index}: {custom_error}")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
        
def main():
    global web3, contract_address, wallet_address, private_key, contract
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Update stop-loss for profitable trades.')
    parser.add_argument('-e', '--env', type=str, help='Environment to use (sepolia or arbitrum).')

    # Parse the arguments
    args = parser.parse_args()

    # Load the appropriate environment file
    if args.env:
        env_file = f".env.{args.env}"
    else:
        env_file = ".env"

    # Log the environment file being loaded
    logging.info(f"Loading environment file: {env_file}")

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
    close_all_open_trades()

if __name__ == "__main__":
    main()
