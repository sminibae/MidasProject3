# imports
import numpy as np
import pandas as pd
import os
import csv
import datetime

def find_threshold_crossing(df, start_index, percent):
    # Ensure the start_index is valid
    if start_index >= len(df):
        return None, None

    # Getting the open price at the start index
    open_price = df.loc[start_index, 'Open']

    # Calculating the thresholds
    threshold_up = open_price * (1 + percent/100)
    threshold_down = open_price * (1 - percent/100)

    # Variables to store the number of candles it took to cross each threshold
    up_cross = None
    down_cross = None

    # Calculate the maximum number of candles for 7 days
    max_candles = 7 * 24 * 20

    # Iterate through the dataframe starting from the start index
    for index, row in df.loc[start_index:].iterrows():
        if index - start_index > max_candles:
            # Stop if 7 days have passed without crossing either threshold
            break

        if up_cross is None and row['High'] >= threshold_up:
            up_cross = index - start_index

        if down_cross is None and row['Low'] <= threshold_down:
            down_cross = index - start_index

    return up_cross, down_cross

# read csv
df = pd.read_csv('Data/historical_DOGEUSDT_3min_data.csv')

# save all result into csv
csv_path = "Data/up_down_cross_3min_data.csv"
if not os.path.exists(csv_path): # Write header
    index =['up_cross','down_cross']
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(index)
    restart_index = 0

else:
    restart_index = len(pd.read_csv(csv_path))
# Write the chunk to the CSV file
with open(csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    print(f'{restart_index}/{len(df)} restarting...')

    for i in range(restart_index, len(df)):
        # calculate up down cross
        up_cross, down_cross = find_threshold_crossing(df=df, start_index=i, percent=1)
        output = [up_cross, down_cross]
        writer.writerow(output)
        file.flush()

        if i % 1000 == 0:
            current_time = datetime.datetime.now().strftime("%H:%M")
            print(f'{i}/{len(df)} processing... Time: {current_time}')


    
