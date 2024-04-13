# imports
import pandas as pd
import numpy as np

from datetime import datetime, timezone
import time

import csv, os, joblib

from binance.client import Client
from binance.enums import KLINE_INTERVAL_3MINUTE, SIDE_BUY, SIDE_SELL, FUTURE_ORDER_TYPE_MARKET, FUTURE_ORDER_TYPE_LIMIT, FUTURE_ORDER_TYPE_STOP_MARKET
from binance.exceptions import BinanceAPIException


'''------------------------------'''


# keys for macbook wifi home
keys = np.load('keys.npy')
API_Key = keys[0]
Secret_Key = keys[1]

#  client
client = Client(api_key=API_Key, api_secret=Secret_Key)

model = joblib.load('Models/RFC_model_80.pkl')
scaler = joblib.load('Scalers/StandardScaler_80.pkl')

'''------------------------------'''


# Convert ISO 8601 date strings to Unix timestamp (milliseconds)
def iso_to_unix(iso_str):
    dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)  # Convert to milliseconds

def unix_to_iso(unix_timestamp_ms):
    # Convert milliseconds to seconds
    unix_timestamp_s = unix_timestamp_ms / 1000
    # Create a datetime object from the Unix timestamp
    dt = datetime.utcfromtimestamp(unix_timestamp_s)
    # Format the datetime object as an ISO 8601 date string
    iso_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return iso_str


'''------------------------------'''


def log_data(data, filename):
    """
    Appends the given data to a CSV file.
    Creates a 'trade_log' directory if it doesn't exist.

    :param data: List of data to log.
    :param filename: Name of the file to which the data will be logged.
    """
    os.makedirs('trade_log', exist_ok=True)
    file_path = f'trade_log/{filename}.csv'

    try:
        with open(file_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            # Adding a timestamp to each log entry
            log_entry = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + data
            csvwriter.writerow(log_entry)
    except IOError as e:
        print(f"IOError while writing to {file_path}: {e}")

def prepare_log_data(output):
    market_condition, prediction, message, direction, balance, balance_on_trade, entry_price, take_profit, stop_loss = output
    market_condition_flat = np.reshape(np.array(market_condition), -1)
    return [message, direction, balance, balance_on_trade, entry_price, take_profit, stop_loss] + list(market_condition_flat) + list(prediction)


'''------------------------------'''


# Convert ISO 8601 date strings to Unix timestamp (milliseconds)
def iso_to_unix(iso_str):
    dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)  # Convert to milliseconds

def unix_to_iso(unix_timestamp_ms):
    # Convert milliseconds to seconds
    unix_timestamp_s = unix_timestamp_ms / 1000
    # Create a datetime object from the Unix timestamp
    dt = datetime.utcfromtimestamp(unix_timestamp_s)
    # Format the datetime object as an ISO 8601 date string
    iso_str = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return iso_str


'''------------------------------'''


def get_binance_server_time_unix(client):
    try:
        # Fetch server time
        server_time = client.get_server_time()
        # Extract the Unix timestamp
        unix_time = server_time['serverTime']
        return unix_time
    except BinanceAPIException as e:
        return f"An error occurred: {e}"
    
def fetch_and_prepare_data(client):
    # Fetch historical candle data
    candles = client.get_klines(symbol='BTCUSDT', interval=KLINE_INTERVAL_3MINUTE, limit=321)

    # Create DataFrame
    df = pd.DataFrame(candles, columns=['Open time', 'Open', 'High', 'Low', 'Close', 'Volume',
                                        'Close time', 'Quote asset volume', 'Number of trades',
                                        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'])

    # Convert to numeric and drop unnecessary columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote asset volume', "Number of trades"]:
        df[col] = pd.to_numeric(df[col])
    df.drop(['Open time', 'Close time', 'Ignore'], axis=1, inplace=True)
    # df['Open time'] = df['Open time'].apply(lambda x: unix_to_iso(x))

    # Calculate Moving Averages
    for ma in [5, 10, 20, 30, 60, 120, 240]:
        df[f'MA_{ma}'] = df['Close'].rolling(window=ma).mean()

    # Calculate Bollinger Bands
    df['BB_moving_avg'] = df['Close'].rolling(window=90).mean()
    df['BB_std'] = df['Close'].rolling(window=90).std()
    df['BB_upper_band'] = df['BB_moving_avg'] + (df['BB_std'] * 1)
    df['BB_lower_band'] = df['BB_moving_avg'] - (df['BB_std'] * 1)

    # Drop rows with NaN values
    df.drop(['BB_std'], axis=1, inplace=True)
    df.dropna(inplace=True)

    df = df.tail(80)
    
    df = np.array(df)

    # Return as matrix
    return df


'''------------------------------'''


def get_usdt_balance(client):
    try:
        # Fetch Futures account information
        futures_account_info = client.futures_account_balance()

        # Search for USDT balance in the Futures account balances
        usdt_balance = next((item for item in futures_account_info if item["asset"] == "USDT"), None)

        if usdt_balance:
            return float(usdt_balance['balance'])  # Returns total USDT balance in Futures account
        else:
            return 0.0  # Returns 0 if USDT is not found in the Futures account
    
    except BinanceAPIException as e:
        return f"An error occurred: {e}"
    
def set_leverage(client, symbol, leverage):
    try:
        # Set the leverage for the specified symbol
        response = client.futures_change_leverage(symbol=symbol, leverage=leverage)
        return response
    except BinanceAPIException as e:
        # Handle potential exceptions from the Binance API
        return {"error": str(e)}
    
def set_margin_type_to_isolated(client, symbol):
    try:
        # Change margin type to 'ISOLATED' for the specified symbol
        response = client.futures_change_margin_type(symbol=symbol, marginType='ISOLATED')
        return response
    except BinanceAPIException as e:
        return f"An error occurred: {e}"
    
def get_current_price(client, symbol):
    try:
        # Fetch the current price for the specified symbol
        current_price = client.get_symbol_ticker(symbol=symbol)

        return float(current_price['price'])

    except BinanceAPIException as e:
        return f"An error occurred: {e}"
    
def how_much_amount(balance, current_price, leverage):
    amount = (balance/current_price)*leverage//0.01*0.01
    return amount
    

'''------------------------------'''


def get_current_position(client, symbol):
    try:
        # Fetch current positions
        positions = client.futures_account()['positions']

        # Find the position for the specified symbol
        position_for_symbol = next((position for position in positions if position['symbol'] == symbol), None)

        if position_for_symbol:
            return position_for_symbol
            # return float(position_for_symbol['positionAmt'])
        else:
            return 0.0  # Returns 0 if no position found for the symbol
    
    except BinanceAPIException as e:
        return f"An error occurred: {e}"

def get_current_futures_open_orders(client, symbol=None):
    try:
        # Fetch open futures orders
        if symbol:
            open_orders = client.futures_get_open_orders(symbol=symbol)
        else:
            # Fetch all open futures orders if no symbol is specified
            open_orders = client.futures_get_open_orders()

        return open_orders
    
    except BinanceAPIException as e:
        return f"An error occurred: {e}"
    
def cancel_futures_order(client, symbol, order_id):
    try:
        # Cancel the specified futures order
        result = client.futures_cancel_order(symbol=symbol, orderId=order_id)
        return result
    except BinanceAPIException as e:
        return f"An error occurred: {e}"
    
def cancel_all_futures_orders(client, symbol):
    try:
        # Cancel all open futures orders for the specified symbol
        result = client.futures_cancel_all_open_orders(symbol=symbol)
        return result
    except BinanceAPIException as e:
        return f"An error occurred: {e}"
    

'''------------------------------'''


def execute_trade(client, symbol, quantity, side, order_type, price=None, stop_price=None):
    try:
        # Convert side from 'long':+1/'short':-1 to 'BUY'/'SELL'
        if side == 1:
            trade_side = SIDE_BUY
        elif side == -1:
            trade_side = SIDE_SELL
        else:
            raise ValueError("Invalid side, choose 1 for 'long' or -1 for 'short'")
        
        # Set the order type and add price if necessary
        if order_type.lower() == 'market':
            trade_type = FUTURE_ORDER_TYPE_MARKET
            order_params = {
                'symbol': symbol,
                'side': trade_side,
                'type': trade_type,
                'quantity': quantity,
            }
        elif order_type.lower() == 'limit':
            if not price:
                raise ValueError("Price must be provided for limit orders")
            order_params = {
                'symbol': symbol,
                'side': trade_side,
                'type': FUTURE_ORDER_TYPE_LIMIT,
                'timeInForce': 'GTC',  # Good till cancelled
                'quantity': quantity,
                'price': price,
            }
        elif order_type.lower() == 'stop_market':
            if not stop_price:
                raise ValueError("Stop price must be provided for stop market orders")
            order_params = {
                'symbol': symbol,
                'side': trade_side,
                'type': FUTURE_ORDER_TYPE_STOP_MARKET,
                'quantity': quantity,
                'stopPrice': stop_price,
                'reduceOnly': True,
            }
        else:
            raise ValueError("Invalid order type, choose 'market', 'limit', or 'stop_market'")

        # Create order
        order = client.futures_create_order(**order_params)
        return order

    except BinanceAPIException as e:
        return f"An error occurred: {e}"



'''------------------------------'''


def start_trading(client,symbol, leverage=10,SL=10):
    current_time = datetime.now().strftime("%H:%M")

    try:
        market_condition = fetch_and_prepare_data(client)
        time.sleep(0.125)

        # Scale them for model
        temp = market_condition.reshape(-1, 19)
        temp_normalized = scaler.transform(temp)

        # Reshape back to 2D
        matrix_normalized = temp_normalized.reshape(-1,19*80)
        
        # prediction
        prediction = model.predict(matrix_normalized)[0]
        
        leverage = leverage
        
        current_position = get_current_position(client,symbol)
        time.sleep(0.125)

        position_amount = float(current_position['positionAmt'])

        # not trading
        if position_amount == 0:

            free_usdt = get_usdt_balance(client)
            time.sleep(0.125)
            current_price = get_current_price(client,symbol)
            time.sleep(0.125)
            amount = how_much_amount(balance = free_usdt, current_price = current_price, leverage=leverage)

            if current_price*amount <5:
                message = 'Not enough money'
                # print(current_time, message)
                time.sleep(5)

                return market_condition, [prediction,0], message, 0, free_usdt, position_amount, 0, 0, 0
                #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss
            
        
            if prediction == 0:
                message = 'prediction = 0'
                # print(current_time, message)
                time.sleep(5)
                
                return market_condition, [prediction,0], message, 0, free_usdt, position_amount, 0, 0, 0
                #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss

            
            # start trade
            elif prediction != 0: 
                message = 'start trading'
                print(current_time, message)
                
                # Determine direction and calculate Take Profit and Stop Loss
                direction = 1 if prediction>0 else -1
                if direction == 1:  # long
                    TakeProfit = round(current_price * 1.01, 1)
                    StopLoss = round(current_price * (1-SL/1000), 1)
                else:  # short
                    TakeProfit = round(current_price * 0.99, 1)
                    StopLoss = round(current_price * (1+SL/1000), 1)

                # Execute the trade
                start_trade = execute_trade(client, symbol, quantity=amount, side=direction, order_type='market')
                print(start_trade)
                time.sleep(0.5)

                current_position = get_current_position(client,symbol)
                print(current_position)
                time.sleep(0.125)
                entry_price = float(current_position['entryPrice'])
                if entry_price == 0 or type(entry_price) != float:
                    message = 'Error in fetching entry price'
                    print(message)
                    return market_condition, [prediction,0], message, 0, free_usdt, 0, 0, 0, 0
                    #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss

                # Set TP and SL
                SL_trade = execute_trade(client, symbol, quantity=amount, side=-direction, order_type='stop_market', stop_price=StopLoss)
                time.sleep(0.125)
                TP_trade = execute_trade(client, symbol, quantity=amount, side=-direction, order_type='limit', price=TakeProfit)
                time.sleep(5)

                return market_condition, [prediction,0], message, direction, free_usdt, amount*entry_price, entry_price, TakeProfit, StopLoss
                    #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss
        
        # we are trading, reset
        elif position_amount != 0:
            
            message = 'no reset'
            direction = 0
            amount = 0
            entry_price = 0
            # print(current_time, message)
            time.sleep(5)

            return market_condition, [prediction,0], message, direction, 0, amount*entry_price, 0, 0, 0
            #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss

    except BinanceAPIException as e:
        message = f"An error occurred: {e}"
        return np.zeros((80,19)), [-2,0], message, 0, 0, 0, 0, 0, 0
        #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss


'''------------------------------'''


def recovery(client, symbol, SL=10):
    try:
        # first, find out if we are trading.
        current_position = get_current_position(client, symbol)
        while type(current_position) == str:
            print(current_position)
            current_position = get_current_position(client, symbol)


        time.sleep(0.125)
        position_amount = float(current_position['positionAmt'])
        amount = np.abs(position_amount)

        # if trading
        if position_amount != 0:

            if np.abs(position_amount) <0.003:
                # if error, so 0.001 BTC left in trade, close all position
                close_positions_and_cancel_orders_for_symbol(client,symbol)
            # fetch open order to see TP, SL remains
            open_orders = get_current_futures_open_orders(client, symbol)
            time.sleep(0.125)

            # both remains
            if len(open_orders) == 2:
                message = 'no problem'
                return np.zeros((80,19)), np.zeros((2,)), message, 0,0,0,0,0,0
                #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss
            
            
            # only 1 order remains (TP or SL)
            if len(open_orders) == 1:
                entry_price = float(current_position['entryPrice'])
                current_price = get_current_price(client, symbol)
                time.sleep(0.125)

                # Determine order type and position type
                is_limit_order = (open_orders[0]['type'] == 'LIMIT')
                is_long_position = position_amount > 0
                is_short_position = position_amount < 0

                # Calculate StopLoss for long and short positions
                if is_limit_order:  # Only TP remains
                    message = 'only TP remains, recovery'
                    print(message)

                    StopLoss = round(entry_price * (1-SL/1000), 1) if is_long_position else round(entry_price * (1+SL/1000), 1)
                    direction = -1 if is_long_position else 1

                    SL_trade = execute_trade(client, symbol, quantity=amount, side=direction, order_type='stop_market', stop_price=StopLoss)
                    print(SL_trade)
                    time.sleep(0.125)

                    return np.zeros((80,19)), np.zeros((2,)), message, direction, 0,amount*entry_price, entry_price, 0, StopLoss
                    #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss

                else:  # Only SL remains
                    message = 'only SL remains, recovery'
                    print(message)

                    entry_price = float(current_position['entryPrice'])

                    TakeProfit = round(entry_price*1.01, 1) if is_long_position else round(entry_price*0.99, 1)
                    direction = -1 if is_long_position else 1
                    TP_trade = execute_trade(client, symbol, quantity=amount, side=direction, order_type='limit', price=TakeProfit)
                    print(TP_trade)
                    time.sleep(0.125)

                    return np.zeros((80,19)), np.zeros((2,)), message, direction, 0, amount*entry_price, entry_price, TakeProfit, 0
                    #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss

            # no open order remains
            elif len(open_orders) == 0:
                message = 'no open order remains, recovery'
                print(message)

                entry_price = float(current_position['entryPrice'])

                # Calculate Take Profit and Stop Loss based on position type
                if position_amount > 0:  # Long position
                    TakeProfit = round(entry_price*1.01, 1)
                    StopLoss = round(entry_price * (1-SL/1000), 1)
                    direction = -1  # To close a long position, we sell
                elif position_amount < 0:  # Short position
                    TakeProfit = round(entry_price*0.99, 1)
                    StopLoss = round(entry_price * (1+SL/1000), 1)
                    direction = 1  # To close a short position, we buy

                # Execute trades for TP and SL
                SL_trade = execute_trade(client, symbol, quantity=amount, side=direction, order_type='stop_market', stop_price=StopLoss)
                print(SL_trade)
                time.sleep(0.125)
                TP_trade = execute_trade(client, symbol, quantity=amount, side=direction, order_type='limit', price=TakeProfit)
                print(TP_trade)
                time.sleep(0.125)

                return np.zeros((80,19)), np.zeros((2,)), message, direction, 0, amount*entry_price, entry_price, TakeProfit, StopLoss
                #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss
            
            # more than 3 open order remains
            else:
                message = 'more than 3 open orders'
                print(message)
                return np.zeros((80,19)), np.zeros((2,)), message, 0, 0, 0, 0, 0, 0
                #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss
        
            
        # If not trading (no position open)
        if position_amount == 0:
            # fetch open order to see TP, SL remains
            open_orders = get_current_futures_open_orders(client, symbol)
            time.sleep(0.125)

            # no open order remains
            if len(open_orders) == 0:
                message = 'no problem'
                return np.zeros((80,19)), np.zeros((2,)), message, 0, 0, 0, 0, 0, 0
                #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss

            # if 1 open order remains
            elif len(open_orders) == 1:
                order_type = open_orders[0]['type']

                if order_type == 'LIMIT':
                    message = 'lost'
                elif order_type == 'STOP_MARKET':
                    message = 'win'
                print(message)
               
                cancel_all_orders = cancel_all_futures_orders(client, symbol)
                time.sleep(0.125)

                return np.zeros((80,19)), np.zeros((2,)), message, 0, 0, 0, 0, 0, 0
                #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss

            # if 2 open order remains
            elif len(open_orders) == 2:
                message = 'closed by hand'
                print(message)
                cancel_all_orders = cancel_all_futures_orders(client, symbol)
                time.sleep(0.125)

                return np.zeros((80,19)), np.zeros((2,)), message, 0, 0, 0, 0, 0, 0
                #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss
            

    except BinanceAPIException as e:
        if '2021' in str(e):
            close_positions_and_cancel_orders_for_symbol(client, symbol)
        message = f"An error occurred: {e}"
        return np.zeros((80,19)), np.zeros((2,)), message, 0, 0, 0, 0, 0, 0
        #return market_condition, prediction, message, direction, balance, balance on trade, entry price, take profit, stop loss




'''------------------------------'''



def show_current_situation(client, symbol):
    try:
        current_position = get_current_position(client, symbol)
        initialMargin = float(current_position['initialMargin'])
        unrealizedProfit = float(current_position['unrealizedProfit'])
        entryPrice = float(current_position['entryPrice'])
        positionAmt = float(current_position['positionAmt'])

        open_orders = get_current_futures_open_orders(client, symbol)
        if open_orders[0]['type'] == 'LIMIT':
            TakeProfit = float(open_orders[0]['price'])
            StopLoss = float(open_orders[1]['stopPrice'])
        if open_orders[1]['type'] == 'LIMIT':
            TakeProfit = float(open_orders[1]['price'])
            StopLoss = float(open_orders[0]['stopPrice'])

        PNL = unrealizedProfit
        if initialMargin == 0:
            ROE = None
        else:
            ROE = 100*unrealizedProfit/initialMargin
            ROE = round(ROE,2)

        print(PNL, ROE, entryPrice)
        print('limit order at', TakeProfit)
        print('stop-market order at', StopLoss)
        print(' ')
        
    except BinanceAPIException as e:
        return f"An error occurred: {e}"
    
def close_all_positions_and_cancel_orders(client):
    try:
        # Cancel all open orders across all symbols
        client.futures_cancel_all_open_orders()

        # Fetch current positions
        positions = client.futures_account()['positions']

        # Close each position
        for position in positions:
            symbol = position['symbol']
            amount = float(position['positionAmt'])

            if amount != 0:
                side = 'SELL' if amount > 0 else 'BUY'
                client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=abs(amount)
                )
    except BinanceAPIException as e:
        return f"An error occurred: {e}"
    
def close_positions_and_cancel_orders_for_symbol(client, symbol):
    try:
        # Cancel all open orders for the specified symbol
        client.futures_cancel_all_open_orders(symbol=symbol)

        # Fetch current position for the symbol
        position_info = next((position for position in client.futures_account()['positions'] if position['symbol'] == symbol), None)
        
        if position_info:
            amount = float(position_info['positionAmt'])
            if amount != 0:
                side = 'SELL' if amount > 0 else 'BUY'
                client.futures_create_order(
                    symbol=symbol,
                    side=side,
                    type='MARKET',
                    quantity=abs(amount)
                )
    except BinanceAPIException as e:
        return f"An error occurred: {e}"
    

'''------------------------------'''


def run():
    try:
        model = joblib.load('Models/RFC_model_80.pkl')
        print('Model Load Success')
    except Exception as e:
        print(f"Model loading error: {e}")
        log_data([f"Model loading error: {e}"], 'error_log')
        return  # Exit if the model cannot be loaded
    
    while True:
        try:
            server_time = get_binance_server_time_unix(client)
            
            # every 3 minutes, at 2min 55sec
            if (server_time - 1698796800000) % (3 * 60 * 1000) > (2 * 60 + 55) * 1000:
                output = start_trading(client, symbol="BTCUSDT")
                log_data(prepare_log_data(output), 'trade_log')
                
                
            # every 2 seconds, always
            else:
                output = recovery(client, symbol="BTCUSDT")
                if output[2] != 'no problem':
                    log_data(prepare_log_data(output), 'trade_log')
                if output[2] in ['win', 'lost', 'closed by hand']:
                    USDT = get_usdt_balance(client)
                    log_data([USDT], 'balance_log')
                time.sleep(2)

        except Exception as e:
            print(f"An error occurred: {e}")
            log_data([f"Error: {e}"], 'error_log')
            time.sleep(10)


#run()
if __name__ == "__main__":
    run()