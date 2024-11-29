from utilities import fetch_symbols

symbols = fetch_symbols()
filtered_symbols = [symbol['symbol'] for symbol in symbols if 'groupIndex' in symbol and int(symbol['groupIndex']) in [0, 10]]

print(sorted(filtered_symbols))
