import argparse
import json
import os
from dotenv import load_dotenv
from web3 import Web3
import logging
import ccxt
import requests
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
        logging.debug(f"Response from blocktorch: {response_json}")
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

    # except ContractLogicError as error:
    #     logging.error(f'Contract logic error occurred: {error}')
    #     if tx_data:
    #         decode_error(error, tx_data)
    #     else:
    #         decode_revert_reason(error)
    except Exception as error:
        logging.error(f'An unexpected error occurred: {error}')
        if tx_data:
            decode_error(error, tx_data)
        # decode_revert_reason(error)
        
def fetch_symbols():
    response = requests.get(os.getenv('GAINS_NETWORK_URL'))
    pairs = response.json()['pairs']
    return [{'symbol': pair['from'], 'index': idx, 'groupIndex': pair['groupIndex']} for idx, pair in enumerate(pairs)]

import random
def open_trades_for_symbols_with_risk_and_direction(symbols, collateral_shares, total_collateral, risk_levels, tp_percentage, sl_percentage, direction):
    """
    Opens trades for multiple symbols with specified collateral shares, risk levels, tp/sl percentages, and direction.

    :param symbols: List of symbols to trade.
    :param collateral_shares: List of collateral shares corresponding to each symbol.
    :param total_collateral: Total collateral amount to be distributed among symbols.
    :param risk_levels: List of risk levels corresponding to each symbol (0 to 1).
    :param tp_percentage: Take-profit percentage.
    :param sl_percentage: Stop-loss percentage.
    :param direction: Portfolio direction ('bullish' for long, 'bearish' for short).
    """
    # Initialize the exchange
    exchange = ccxt.binance()  # Replace 'binance' with your preferred exchange
    
    pairs = fetch_symbols()

    for symbol, share, risk_level in zip(symbols, collateral_shares, risk_levels):
        # Fetch the latest close price using ccxt
        ticker = exchange.fetch_ticker(symbol + '/USDT')
        latest_close_price = ticker['last']
        
        collateral = round(total_collateral * share)
        leverage = max(2, min(150, round(2 + (risk_level * (150 - 2)) * random.uniform(0.9, 1.1))))  # Calculate leverage based on risk level
        
        if tp_percentage:
            tp_price = latest_close_price * (1 + (tp_percentage / 100) / leverage) if direction == 'bullish' else latest_close_price * (1 - (tp_percentage / 100) / leverage)
        else:
            tp_price = 0
        
        if sl_percentage:
            sl_price = latest_close_price * (1 - (sl_percentage / 100) / leverage) if direction == 'bullish' else latest_close_price * (1 + (sl_percentage / 100) / leverage)
        else:
            sl_price = 0
        
        action = 'open_long' if direction == 'bullish' else 'open_short'
        
        # Assuming 'pairs' is available in the scope
        open_trade(latest_close_price, pairs, symbol, action, collateral, leverage, tp_price, sl_price)

def main():
    global web3, contract_address, wallet_address, private_key, contract, contract_abi
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Open trades for symbols with specified collateral shares, risk levels, tp/sl percentages, and direction.')
    parser.add_argument('-e', '--env', type=str, help='Environment to use (sepolia or arbitrum).')
    parser.add_argument('-s', '--stop_loss_percentage', type=float, default=None, help='Stop-loss percentage to use.')
    parser.add_argument('-t', '--take_profit_percentage', type=float, default=None, help='Take-profit percentage to use.')
    parser.add_argument('-d', '--direction', type=str, default='bullish', help='Direction to use.')

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
    
    symbols = ['BTC', 'ETH', 'SOL', 'ARB', 'DOGE', 'GRT']
    collateral_shares = [0.3, 0.2, 0.2, 0.1, 0.1, 0.1]  # 60% for BTC, 40% for ETH
    total_collateral = 100
    risk_levels = [0.1, 0.1, 0.2, 0.3, 0.4, 0.4]  # Risk level for BTC and ETH (0 to 1 scale)
    
    tp_percentage = args.take_profit_percentage
    sl_percentage = args.stop_loss_percentage
    direction = args.direction

    open_trades_for_symbols_with_risk_and_direction(symbols, collateral_shares, total_collateral, risk_levels, tp_percentage, sl_percentage, direction)

if __name__ == "__main__":
    main()
