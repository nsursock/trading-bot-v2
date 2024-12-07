from collections import defaultdict
from datetime import datetime
import time
import requests
import logging
from time import sleep
import pandas as pd

end_time = None
interval = '1m'
limit = 10000
symbol = 'BTC'
retries = 1
backoff_factor = 0.3

base_url = "https://api.binance.com/api/v3/klines"
klines = []

# Convert end_time to datetime if it's a string
if isinstance(end_time, str):
    end_time = datetime.strptime(end_time, "%Y-%m-%d")

# Convert end_time to milliseconds if it's a datetime object
if isinstance(end_time, datetime):
    end_time = int(end_time.timestamp() * 1000)

total_fetch_time = 0
fetch_count = 0

while len(klines) < limit:
    params = {
        'symbol': symbol + 'USDT',
        'interval': interval,
        'limit': min(limit - len(klines), 1000)
    }
    if end_time is not None:
        params['endTime'] = end_time

    for attempt in range(retries):
        try:
            fetch_start_time = time.time()  # Record the start time of this fetch
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            fetch_end_time = time.time()  # Record the end time of this fetch
    
            data = response.json()
        except requests.exceptions.RequestException as e:
            logging.debug(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                sleep(backoff_factor * (2 ** attempt))  # Exponential backoff
            else:
                raise
    
    # Check for error in response
    if 'code' in data:
        raise Exception(f"API Error: {data['msg']}")
    
    if isinstance(data, list) and data:
        klines = data + klines
        end_time = data[0][0] - 1
    else:
        logging.error(f"No data returned or error in response.")
        break

    # Update total fetch time and count
    total_fetch_time += (fetch_end_time - fetch_start_time)
    fetch_count += 1

    # Calculate and log ETA
    if fetch_count > 0:
        average_fetch_time = total_fetch_time / fetch_count
        klines_fetched = len(klines)
        eta_seconds = average_fetch_time * (limit - klines_fetched) / params['limit']
        
        # Format ETA
        if eta_seconds > 3600:
            hours, remainder = divmod(eta_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            eta_formatted = f"{int(hours):02}h{int(minutes):02}m{int(seconds):02}s"
        elif eta_seconds > 60:
            minutes, seconds = divmod(eta_seconds, 60)
            eta_formatted = f"{int(minutes):02}m{int(seconds):02}s"
        else:
            eta_formatted = f"{int(eta_seconds):02}s"

        logging.debug(f"Fetched {klines_fetched} klines for {symbol}. Average fetch time: {average_fetch_time:.2f} seconds. ETA for completion: {eta_formatted}.")

    if len(data) < params['limit']:
        break
    
    time.sleep(0.1)

klines = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume', 
    'close_time', 'quote_asset_volume', 'number_of_trades', 
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
])
klines['open_time'] = pd.to_datetime(klines['timestamp'], unit='ms')
klines['close_time'] = pd.to_datetime(klines['close_time'], unit='ms')
klines[['open', 'high', 'low', 'close', 'volume']] = klines[['open', 'high', 'low', 'close', 'volume']].apply(pd.to_numeric)


# Create dictionaries to store volume by day and by hour
volume_by_day = defaultdict(float)
volume_by_hour = defaultdict(float)

# Aggregate volume by day and hour
for entry in data:
    open_time = datetime.fromtimestamp(entry[0] / 1000)  # Assuming open_time is the first element
    day = open_time.strftime('%A')  # Get the day of the week
    hour = open_time.hour
    volume = float(entry[5])  # Convert volume to float

    volume_by_day[day] += volume
    volume_by_hour[hour] += volume

# Rank the days by trading volume
ranked_days = sorted(volume_by_day.items(), key=lambda x: x[1], reverse=True)

# Rank the hours by trading volume
ranked_hours = sorted(volume_by_hour.items(), key=lambda x: x[1], reverse=True)

# Print the ranked days
print("Trading days ranked by volume:")
for day, volume in ranked_days:
    print(f"{day} - Volume: {volume}")

# Print the ranked hours
print("\nTrading hours ranked by volume:")
for hour, volume in ranked_hours:
    print(f"{hour}:00 - Volume: {volume}")

