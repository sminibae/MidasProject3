# imports
import numpy as np
import pandas as pd
import os
import csv
import datetime
from multiprocessing import Pool, cpu_count

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


def save_chunks(df, chunk_size=1000, overlap=7*24*20):
    num_rows = len(df)
    chunks = [(start, min(start + chunk_size + overlap, num_rows)) for start in range(0, num_rows, chunk_size)]
    
    for i, (start, end) in enumerate(chunks):
        chunk_df = df.iloc[start:end]
        chunk_df.to_csv(f'Data/temp_{i}.csv', index=False)


def process_chunk(chunk_num, chunk_size=1000):
    df = pd.read_csv(f'Data/temp_{chunk_num}.csv')
    csv_path = f'Data/up_down_cross_temp_{chunk_num}.csv'
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        index = ['up_cross', 'down_cross']
        writer.writerow(index)
        file.flush()

        for i in range(chunk_size):
            up_cross, down_cross = find_threshold_crossing(df=df, start_index=i, percent=1)
            writer.writerow([up_cross, down_cross])
            file.flush()

            if i % 1000 == 0:
                current_time = datetime.datetime.now().strftime("%H:%M")
                print(f'chunk {chunk_num}, {i}/{len(df)} processing... Time: {current_time}')


def run_parallel_processing(num_chunks):
    with Pool(processes=cpu_count()-1) as pool:
        pool.map(process_chunk, range(num_chunks))


def concatenate_results(num_chunks, final_path='Data/up_down_cross_3min_data.csv'):
    with open(final_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['up_cross', 'down_cross'])  # Writing header
        for i in range(num_chunks):
            with open(f'Data/up_down_cross_temp_{i}.csv', 'r') as infile:
                reader = csv.reader(infile)
                next(reader)  # Skip header
                for row in reader:
                    writer.writerow(row)
                    outfile.flush()
  

def cleanup_temp_files(num_chunks):
    for i in range(num_chunks):
        os.remove(f'Data/temp_{i}.csv')
        os.remove(f'Data/up_down_cross_temp_{i}.csv')


if __name__ == '__main__':
    # read csv
    df = pd.read_csv('Data/historical_XRPUSDT_3min_data.csv')
    num_chunks = len(df)//1000+1
    # crop df into 1000+7*24*20 size chunks
    save_chunks(df=df)

    # parallel process the chunks
    run_parallel_processing(num_chunks=num_chunks)

    # concatenate results
    concatenate_results(num_chunks=num_chunks)

    # clean up temp files
    cleanup_temp_files(num_chunks=num_chunks)
    
    
