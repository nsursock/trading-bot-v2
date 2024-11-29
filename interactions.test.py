import unittest
import logging
from interactions import fetch_symbols, open_trade, close_trade, compute_prices, fetch_open_trades
import ccxt

class TestExecuteTrade(unittest.TestCase):

    def setUp(self):
        
        # Initialize the exchange
        self.exchange = ccxt.binance()

        self.symbol = 'SHIB'
        self.interval = '1m'
        self.pairs = fetch_symbols()
                
    # def test_decode_error(self):
    #     decode_revert_reason(('0x5508e5c6', '0x5508e5c6'))
    
    def test_close_all_open_trades(self):
        try:
            # Fetch open trades
            open_trades = fetch_open_trades()
            
            # Log the number of open trades
            logging.info(f"Number of open trades: {len(open_trades)}")
            
            # Ensure there is at least one open trade
            self.assertGreater(len(open_trades), 0, "No open trades found to close.")
            
            # Attempt to close all open trades
            for symbol, trade_index, is_long in open_trades:
                # Fetch the latest price for the symbol using CCXT
                ticker = self.exchange.fetch_ticker(symbol + '/USDT')
                latest_price = ticker['last']  # Get the last price from the ticker

                try:
                    close_trade(trade_index, latest_price, is_long)
                    logging.info(f"Successfully closed trade with trade_index: {trade_index}")
                except Exception as e:
                    logging.error(f"Failed to close trade with trade_index: {trade_index}. Error: {e}")
            
            # Verify all trades are closed
            remaining_open_trades = fetch_open_trades()
            self.assertEqual(len(remaining_open_trades), 0, f"Expected all trades to be closed, but {len(remaining_open_trades)} trades remain open.")
            
        except Exception as error:
            logging.error('Error in test_close_all_open_trades:', exc_info=True)
            self.fail(f"Test failed due to an unexpected error: {error}")

    def test_open_trade_minimal_collateral(self):
        action = 'open_long'
        leverage = 50
        
        # Fetch the latest price for the symbol using CCXT
        ticker = self.exchange.fetch_ticker(self.symbol + '/USDT')
        latest_price = ticker['last']  # Get the last price from the ticker

        tp_price, sl_price = compute_prices(latest_price, action, 0.02, 0.01)
        open_trade(latest_price, self.pairs, self.symbol, action, 50, leverage, tp_price, sl_price)
        
    def test_open_trade_debug_prod(self):
#         WARNING - Attempting to open trade with symbol: NEAR, action: open_long, collateral: 198.49247999999997, leverage: 8
# 2024-10-02 13:18:14,888 - INFO - An error occurred: ('0x5508e5c6', '0x5508e5c6')
# 2024-10-02 13:18:14,888 - INFO - Opened open_long for NEAR with info {'type': 'open_long', 'collateral': 198.49247999999997, 'leverage': 8, 'tp_price': 4.878717928369065, 'sl_price': 4.820641035815467}
# 2024-10-02 13:18:14,888 - INFO - Executed trade for NEAR with type open_long and info {'type': 'open_long', 'collateral': 198.49247999999997, 'leverage': 8, 'tp_price': 4.878717928369065, 'sl_price': 4.820641035815467}
        action = 'open_long'
        collateral = 200
        leverage = 8
        
        # Fetch the latest price for the symbol using CCXT
        ticker = self.exchange.fetch_ticker(self.symbol + '/USDT')
        latest_price = ticker['last']  # Get the last price from the ticker
                
        tp_price, sl_price = compute_prices(latest_price, action, 0.02, 0.01)
        open_trade(latest_price, self.pairs, self.symbol, action, collateral, leverage, tp_price, sl_price)
        
    def test_close_latest_trade(self):
        try:
            # Fetch open trades
            open_trades = fetch_open_trades()
            
            # Ensure there is at least one open trade
            self.assertGreater(len(open_trades), 0, "No open trades found to close.")
            
            # Get the latest trade
            latest_trade = open_trades[0]
            symbol, trade_index, is_long = latest_trade
            
            # Fetch the latest price for the symbol using CCXT
            ticker = self.exchange.fetch_ticker(self.symbol + '/USDT')
            latest_price = ticker['last']  # Get the last price from the ticker
                
            # Attempt to close the latest trade
            try:
                close_trade(trade_index, latest_price, is_long)
                print(f"Successfully closed trade with trade_index: {trade_index}")
            except Exception as e:
                self.fail(f"Failed to close trade with trade_index: {trade_index}. Error: {e}")
        except Exception as error:
            logging.error('Error in test_close_latest_trade:', exc_info=True)
            self.fail(f"Test failed due to an unexpected error: {error}")

if __name__ == "__main__":
    unittest.main()
    