# imports
import numpy as np
import pandas as pd
import os
import csv
import datetime
from multiprocessing import Pool, cpu_count
from math import ceil

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


def process_chunk(csv_path, chunk_size=1000, overlap=7*24*20):
    """
    Splits a CSV file into smaller chunks, each with a specified size and overlap.
    """
    # Calculate total number of rows in the CSV
    total_rows = sum(1 for _ in open(csv_path)) - 1  # Exclude header
    num_chunks = ceil(total_rows / chunk_size)
    csv_list = []

    # Read and split the CSV
    for chunk_id in range(num_chunks):
        skip_rows = chunk_id * chunk_size
        nrows = chunk_size + min(overlap, total_rows - chunk_id * chunk_size)
        if chunk_id == num_chunks - 1:  # Ensure we read all rows in the last chunk
            nrows = total_rows - chunk_id * chunk_size
            
        df_chunk = pd.read_csv(csv_path, skiprows=range(1, skip_rows), nrows=nrows, header=0)
        
        # Save each chunk to a new CSV file
        output_path = f"Data/chunk_{chunk_id}.csv"
        csv_list.append(output_path)
        df_chunk.to_csv(output_path, index=False)
        print(f"Chunk {chunk_id} saved to {output_path}.")
    
    for index, chunk_path in enumerate(csv_list):
        df = pd.read_csv(chunk_path)
        output_path = f'Data/up_down_cross_3min_data+{index}.csv'

    if not os.path.exists(output_path): # Write header
        index =['up_cross','down_cross']
        with open(ouptut_path, mode='a', newline='') as file:
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
        


def process_chunk(chunk_path):
    """
    Reads a chunk CSV file, applies the find_threshold_crossing function, and saves or returns the result.
    """
    df = pd.read_csv(chunk_path)
    output_csv_path = 
    result = find_threshold_crossing(df, start_index=0, percent=1)

    return result

def multiprocessing_handler(csv_chunks):
    """
    Handles multiprocessing across the given CSV chunks.
    """
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_chunk, csv_chunks)
    return results

# read csv
csv_path = 'Data/historical_BTCUSDT_3min_data.csv'
csv_list = split_csv_into_chunks(csv_path, chunk_size=1000, overlap=7*24*20)
with Pool(processes=cpu_count()) as pool:
    results = pool.map(process_chunk, csv_list)

# save all result into csv
csv_path = "Data/up_down_cross_3min_data.csv"



    
