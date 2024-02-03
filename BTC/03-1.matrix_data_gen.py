# imports
import pandas as pd
import numpy as np
import multiprocessing
import os, gc


data = pd.read_csv('Data/df_all.csv')
data = data.drop(['Ignore','up_cross','down_cross','minutes','log_minutes','side'], axis=1)



# looking for not continuous points
non_continuous_index =[]

for i in range(len(data)-1):
    if data['Open time'][i+1] - data['Open time'][i] != 3*60*1000:
        non_continuous_index.append(i)



# Initialize an empty list to hold your chunks of data
data_chunks = []

# Set the start index for the first chunk
start_idx = 0

# Iterate through the non-continuous indices and split the data into chunks
for idx in non_continuous_index:
    # Create a chunk from start_idx to the non-continuous index
    chunk = data.iloc[start_idx:idx+1]
    # Append the chunk to your list of data chunks
    data_chunks.append(chunk.reset_index(drop=True))
    # Update start_idx for the next chunk
    start_idx = idx + 1

# Don't forget to grab the last chunk of data after the last non-continuous index
final_chunk = data.iloc[start_idx:]
data_chunks.append(final_chunk.reset_index(drop=True))



# Define the window sizes for the moving averages
windows = [5, 10, 20, 30, 60, 120, 240]
# Define the window size and standard deviation multiplier for the Bollinger Bands
BB_window_size = 90
BB_std_multiplier = 1

# Iterate through each chunk in data_chunks
for i, chunk in enumerate(data_chunks):
    # ADDING MA
    for window in windows:
        # Calculate the moving average
        moving_avg = chunk['Close'].rolling(window).mean()
        # Add the moving average as a new column to the chunk
        chunk[f'MA{window}'] = moving_avg

    # Calculate the moving average and standard deviation
    BB_moving_avg = chunk['Close'].rolling(BB_window_size).mean()
    BB_std_dev = chunk['Close'].rolling(BB_window_size).std()

    # ADDING BB
    # Calculate the Bollinger Bands
    BB_upper_band = BB_moving_avg + (BB_std_multiplier * BB_std_dev)
    BB_lower_band = BB_moving_avg - (BB_std_multiplier * BB_std_dev)

    # Add the Bollinger Bands and moving average as new columns to the chunk
    chunk[f'MA{BB_window_size}'] = BB_moving_avg
    chunk[f'Upper_Band{BB_window_size}'] = BB_upper_band
    chunk[f'Lower_Band{BB_window_size}'] = BB_lower_band

    # drop NaN values
    chunk = chunk.dropna().reset_index(drop=True)
    chunk = chunk.drop(columns = ['Open time', 'Close time',], axis=1)
    # Optionally, update the chunk in data_chunks (if you want to keep the changes)
    data_chunks[i] = chunk

# Now each chunk in data_chunks has new columns for the moving averages



def process_chunk(chunk, chunk_index):
    matrix_list = []
    answer_list = []

    if len(chunk) >= 80:
        for i in range(len(chunk) - 79):
            matrix = chunk.drop(columns=['plus_6', 'minus_6', 'zero_6'], axis=1).iloc[i:i+80].values
            matrix_list.append(matrix)
            answer = chunk.iloc[i+59][['plus_6', 'minus_6', 'zero_6']].tolist()
            answer_list.append(answer)

            if i%1000 == 0:
                print(f'for {chunk_index}th chunk, {i}/{len(chunk)} processing...')

    # Save the processed data
    np.save(f'Data/matrix_array_80_{chunk_index}.npy', np.array(matrix_list))
    np.save(f'Data/answer_array_80_{chunk_index}.npy', np.array(answer_list))

    del matrix_list, answer_list
    gc.collect()


# Create a list of tuples (chunk, index)
chunk_data = [(chunk, index) for index, chunk in enumerate(data_chunks)]

# Define a helper function for multiprocessing to unpack arguments
def process_chunk_wrapper(args):
    return process_chunk(*args)

if __name__ == '__main__':
    # Assuming data_chunks is your list of chunks
    chunk_data = [(chunk, index) for index, chunk in enumerate(data_chunks)]

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        pool.map(process_chunk_wrapper, chunk_data)
    

    # List of file names to load
    file_names_matrix = [f"Data/matrix_array_80_{i}.npy" for i in range(11)]  # Adjust range as needed
    file_names_answer = [f"Data/answer_array_80_{i}.npy" for i in range(11)]

    # Load each file and store in a list
    loaded_arrays_matrix = [np.load(file_name) for file_name in file_names_matrix]
    loaded_arrays_answer = [np.load(file_name) for file_name in file_names_answer]

    # Concatenate all arrays into a single array
    matrix_array_80 = np.concatenate(loaded_arrays_matrix, axis=0)
    answer_array_80 = np.concatenate(loaded_arrays_answer, axis=0)

    # save the concatenated array
    np.save('Data/matrix_array_80.npy', matrix_array_80)
    np.save('Data/answer_array_80.npy', answer_array_80)

    # delete _0 ~ _10 .npy files
    for file_name in file_names_matrix + file_names_answer:
        os.remove(file_name)
    