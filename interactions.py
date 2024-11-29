import unittest
import json
import ccxt
import pandas as pd
from dotenv import load_dotenv
import os
from web3 import Web3
import requests
import random
from datetime import datetime
import pytz
import logging
from web3.exceptions import ContractLogicError, ContractCustomError

# # Configure logging
# logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Initialize ccxt Binance exchange
binance = ccxt.binance()

# Initialize Web3 with Alchemy URL
web3 = Web3(Web3.HTTPProvider(os.getenv('ALCHEMY_URL')))

# Contract and Wallet details
contract_address = os.getenv('SEPOLIA_URL')
private_key = os.getenv('PRIVATE_KEY')
wallet_address = web3.eth.account.from_key(private_key).address

# Define the contract ABI
with open('./abi.json') as f:
    contract_abi = json.load(f)
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# Initialize data storage
data = {}

def fetch_open_trades(symbol=None):
    try:
        logging.info('Fetching open trades...')

        # Fetch trading history from the provided URL
        url = os.getenv('GAINS_NETWORK_HISTORY')
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
        trade_details = [(position['pair'].split('/')[0], position['tradeIndex'], position['long']) for position in open_trades_history]
        logging.debug(f'Trade details: {trade_details}')

        logging.info('Completed fetching open trades.')
        return trade_details
    except Exception as error:
        logging.error('Error fetching open trades:', exc_info=True)
        raise
    
    
from eth_utils import decode_hex
from eth_abi.abi import decode

import base64

def send_post_request_to_blocktorch(transaction):
    url = "http://localhost:8000/api/internal/transaction/decode"  # Updated to local URL
    
    username = "nsursock"
    password = "aVeaatlc24*@"
    credentials = f"{username}:{password}"
    encoded_credentials = base64.b64encode(credentials.encode()).decode()
    
    payload = {
        "abis": {
            contract_address: {
                "contractAbi": contract_abi
            }
        },
        "transaction": transaction
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Basic " + encoded_credentials  # Base64 encoded credentials
    }

    # logging.info(f"Sending POST request to {url} with payload: {json.dumps(payload, indent=2)}")
    response = requests.post(url, json=payload, headers=headers)
    logging.info(f"POST request sent to {url}, status code: {response.status_code}")
    
    try:
        response_json = response.json()
        logging.info(f"Response from blocktorch: {response_json}")
        return response_json
    except ValueError:
        logging.error(f"Failed to decode JSON response: {response.text}")
        return None
    
def decode_error(error, tx_data):
    if hasattr(error, 'data'):
        raw_data = error.data
        # Send the error data to blocktorch.xyz
        transaction = {
            "to": contract_address,
            "input": tx_data['data'],
            "value": "0x0",  # Assuming no value is sent with the transaction
            "error": {
                "data": raw_data
            }
        }
        response = send_post_request_to_blocktorch(transaction)
        if response:
            logging.info(f"Blocktorch response: Transaction function: {response['transaction']['functionFragment']['name']}, Error: {response['error']['name']}")
        else:
            logging.error("Failed to get a valid response from blocktorch")
            
def close_all_open_trades():
    try:
        trade_details = fetch_open_trades()
        print('Number of open trades:', len(trade_details))
        
        # trade_info = []
        for symbol, order_id, is_long in trade_details:
            ticker = binance.fetch_ticker(symbol + '/USDT')
            latest_price = ticker['last']
            # trade_info.append((order_id, latest_price, is_long))
            logging.info(f"Fetched latest price for symbol {symbol} ({'long' if is_long == 1 else 'short'}) and market order {order_id}: {latest_price}")
            
            # Attempt to close the trade
            try:
                close_trade(order_id, latest_price, is_long, 0.05)
                logging.info(f"Successfully closed trade with order_id: {order_id}")
            except Exception as e:
                logging.error(f"Failed to close trade with order_id: {order_id}. Error: {e}")

    except Exception as error:
        logging.error('Error in main function:', error)
    
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
            decode_error(e, tx_data)
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
    except ContractLogicError as custom_error:
        logging.error(f"Contract logic error occurred: {custom_error}")
        decode_error(custom_error, tx_data)
        print(f"An error occurred during trade closing for orderId {trade_index}: {custom_error}")
    except Exception as e:
        logging.error(f"Unexpected error occurred: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
        
def fetch_symbols():
    response = requests.get(os.getenv('GAINS_NETWORK_URL'))
    pairs = response.json()['pairs']
    return [{'symbol': pair['from'], 'index': idx, 'groupIndex': pair['groupIndex']} for idx, pair in enumerate(pairs)]

def compute_prices(latest_close_price, action, tp_percentage, sl_percentage):
    # latest_close_price = data.iloc[-1]['close']  # Use .iloc for positional indexing
    # logging.info(f'Latest close price: {latest_close_price}')  # Debug print for price
    
    tp = latest_close_price * (1 + tp_percentage) if tp_percentage != 0 else 0
    sl = latest_close_price * (1 - sl_percentage) if sl_percentage != 0 else 0

    if action == 'sell':
        tp, sl = sl, tp  # Swap tp and sl for sell action
        
    return tp, sl

from web3.auto import w3
from web3 import Web3
from eth_utils import to_bytes, is_hex

def decode_revert_reason(revert_code):
    try:
        # Check if revert_code is a ContractCustomError instance
        if isinstance(revert_code, ContractCustomError):
            # Use the second element of the tuple as the revert reason
            revert_bytes = to_bytes(hexstr=revert_code.message)  # Access the message directly
            logging.info("revert_bytes = to_bytes")
            revert_msg = w3.eth.call({
                'to': contract_address,
                'data': revert_bytes
            })
            logging.info("revert_msg = w3.eth.call")
            # Decode the revert message using the ABI decoder
            decoded_message = w3.codec.decode_single('string', revert_msg)
            logging.error("Revert message decoded: %s", decoded_message)
    except Exception as e:
        logging.error("Failed to decode revert reason: %s", str(e))

def open_trade(latest_close_price, pairs, symbol, action, collateral=200, leverage=10, tp_price=0, sl_price=0):
    # leverageExperiment = 5
    try: 
        tx_data = None
        logging.warning(f"Attempting to open trade with symbol: {symbol}, action: {action}, collateral: {collateral}, leverage: {leverage}")
        
        pairObject = next(pair for pair in pairs if pair['symbol'] == symbol)
        logging.debug('Found index of symbol')
            
        nonce = web3.eth.get_transaction_count(wallet_address)
        logging.debug(f'Got transaction count {nonce}')
        gas_price = web3.eth.gas_price
        logging.debug(f'Got gas price {gas_price}')

        # latest_close_price = data.iloc[-1]['close']  # Use .iloc for positional indexing
        logging.debug(f'Latest close price: {latest_close_price}')  # Debug print for price
        
        # tp_percentage = 0.25
        # sl_percentage = 0.1
        
        # # Validate TP and SL values
        # if tp_price < 0 or sl_price < 0:
        #     logging.warning("TP and SL prices must be greater than 0.")
        #     if action == 'open_long':
        #         tp_price = latest_close_price * (1 + tp_percentage)
        #         sl_price = latest_close_price * (1 - sl_percentage) 
        #     else:
        #         sl_price = latest_close_price * (1 + sl_percentage)
        #         tp_price = latest_close_price * (1 - tp_percentage)
        
        # if action == 'open_long':
        #     if tp_price <= latest_close_price:
        #         logging.warning("TP price must be greater than the latest close price for a long position.")
        #         tp_price = latest_close_price * (1 + tp_percentage)
        #     if sl_price >= latest_close_price:
        #         logging.warning("SL price must be less than the latest close price for a long position.")
        #         sl_price = latest_close_price * (1 - sl_percentage)
        # elif action == 'open_short':
        #     if tp_price >= latest_close_price:
        #         logging.warning("TP price must be less than the latest close price for a short position.")
        #         tp_price = latest_close_price * (1 - tp_percentage)
        #     if sl_price <= latest_close_price:
        #         logging.warning("SL price must be greater than the latest close price for a short position.")
        #         sl_price = latest_close_price * (1 + sl_percentage)

        _trade = {
            'user': wallet_address,
            'index': 0,  # uint32
            'pairIndex': int(pairObject['index']),  # uint16
            'leverage': int(leverage * 1000),  # uint24 (50x leverage)
            'long': action == 'open_long',  # bool
            'isOpen': True,  # bool
            'collateralIndex': 3,  # uint8
            'tradeType': 0,  # enum ITradingStorage.TradeType
            'collateralAmount': int(web3.to_wei(collateral, 'mwei')),  # uint120, BigNumber to string
            'openPrice': int(latest_close_price * 1e10),  # uint64, converting price to BigNumber
            'tp': int(tp_price * 1e10),  # uint64
            'sl': int(sl_price * 1e10),  # uint64
            '__placeholder': 0  # uint192, BigNumber to string
        }

        # Convert all values to native Python types
        _tradeForJsonDump = {k: int(v) if isinstance(v, (int, float)) else v for k, v in _trade.items()}

        logging.debug('Trade object: %s', json.dumps(_tradeForJsonDump, indent=2))  # Debug print for trade object 

        _max_slippage_p = 5000  # uint32  It's specified in parts per thousand (e.g., 1000 for 1%, 500 for 0.5%).
        _referrer = '0x0000000000000000000000000000000000000000'  # address
        
        
        # max_retries = 3  # Set the maximum number of retries
        # for attempt in range(max_retries):
        #     try:
        #         gas_estimate = contract.functions.openTrade(_trade, _max_slippage_p, _referrer).estimate_gas({'from': wallet_address})
        #         logging.debug(f'Gas estimate: {gas_estimate}')  # Debug print for gas estimate
        #         break  # Exit loop if successful
        #     except Exception as e:
        #         logging.warning(f"Gas estimation failed on attempt {attempt + 1}: {e}")
        #         if attempt == max_retries - 1:
        #             raise  # Raise the exception if the last attempt fails

        gas_estimate = contract.functions.openTrade(_trade, _max_slippage_p, _referrer).estimate_gas({'from': wallet_address})
        logging.debug(f'Gas estimate: {gas_estimate}')  # Debug print for gas estimate
        
        # gas_estimate = 3_000_000 #2500000

        tx_data = contract.functions.openTrade(_trade, _max_slippage_p, _referrer).build_transaction({
                'gas': int(gas_estimate * 2),
                'gasPrice': gas_price,
                'nonce': nonce
            })
        logging.debug(f"Transaction data built: {tx_data}")

        tx = {
            'to': contract_address,
            'data': tx_data['data'],
            'gas': int(gas_estimate * 2),
            'gasPrice': gas_price,
            'nonce': nonce
        }

        signed_tx = web3.eth.account.sign_transaction(tx, private_key)
        logging.debug('Signed transaction')
        tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)  # Corrected attribute name
        logging.debug(f'Transaction hash: {web3.to_hex(tx_hash)}')

        # Check for market cancellation
        receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt['status'] == 0:
            logging.debug('Market was canceled. Handling cancellation...')
            # Add your cancellation handling logic here
            # return -1  # Indicate failure due to market cancellation
            
        
        logging.info(f"Opened trade for symbol: {symbol}, Transaction hash: {receipt.transactionHash.hex()}")
        return web3.to_hex(tx_hash)  # Return transaction hash on success

    except Exception as error:
        logging.error(f'An error occurred: {error}')
        if tx_data:
            decode_error(error, tx_data)
        else:
            decode_revert_reason(error)
        # return -1  # Indicate failure due to exception