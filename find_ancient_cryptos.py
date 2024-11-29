import requests
from datetime import datetime

# Binance API base URL
base_url = "https://api.binance.com"

num_symbols = 20

# Get exchange information
exchange_info = requests.get(f"{base_url}/api/v3/exchangeInfo").json()

# Print all available symbols for debugging
all_symbols = [symbol['symbol'] for symbol in exchange_info['symbols']]
# print("All Available Symbols:", all_symbols)

symbols = [symbol['symbol'] for symbol in exchange_info['symbols'] if symbol['status'] == 'TRADING' and symbol['symbol'].endswith('USDT')][:num_symbols]

# List of specific cryptocurrencies to check
specific_cryptos = [
    "APU", "BOME", "BONK", "BRETT", "CAT", "CATI",
    "DOGE", "DOGS", "FLOKI", "GOAT", "HIPPO", "HMSTR",
    "LTC", "MEME", "MEW", "MOG", "MOODENG", "MOTHER",
    "MYRO", "NEIRO", "NEIROCTO", "PEOPLE", "PEPE",
    "PONKE", "POPCAT", "RATS", "SHIB", "SLERF",
    "SUNDOG", "TURBO", "WIF"
]

# Filter symbols to only include the specific cryptocurrencies
symbols = [f"{crypto}USDT" for crypto in specific_cryptos] # if f"{crypto}USDT" in symbols]

# Debugging: Print the filtered symbols
print("Filtered Symbols:", symbols)

# Function to get the earliest date of available data for a symbol
def get_earliest_date(symbol):
    # Fetch the earliest available kline (1-day interval) to find the starting date
    params = {
        "symbol": symbol,
        "interval": "1d",
        "limit": 1,
        "startTime": 0  # 0 to get the earliest data available
    }
    print(f"Fetching earliest date for {symbol}")
    response = requests.get(f"{base_url}/api/v3/klines", params=params)
    if response.status_code == 200 and len(response.json()) > 0:
        first_kline = response.json()[0]
        timestamp = first_kline[0]
        return datetime.fromtimestamp(timestamp / 1000)
    else:
        return None

# Dictionary to store the earliest date for each symbol
earliest_dates = {}

# Iterate over symbols and find their start dates
for symbol in symbols:
    date = get_earliest_date(symbol)
    if date:
        earliest_dates[symbol] = date

# Sort symbols by the earliest available date
sorted_symbols = sorted(earliest_dates.items(), key=lambda x: x[1])

# Display the most ancient trading date for the specified cryptocurrencies
for symbol, date in sorted_symbols:
    print(f"Symbol: {symbol}, Start Date: {date}")

# Create an array of the specified cryptocurrencies with their start dates
oldest_symbols = [symbol.replace('USDT', '') for symbol, date in sorted_symbols]

# Print the array of symbols
print("Oldest Symbols:", oldest_symbols)