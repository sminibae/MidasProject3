# imports
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn

from binance.client import Client
from datetime import datetime, timezone

import time
import csv
import gc
import os

from multiprocessing import Pool, cpu_count

from sklearn.preprocessing import StandardScaler
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import math

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

# ----------------------------------------------------------------------------------------
# 01. Data Fetch

#  client
API_Key = np.load('ReadOnlykeys.npy')[0]
Secret_Key = np.load('ReadOnlykeys.npy')[1]
client = Client(api_key=API_Key, api_secret=Secret_Key)

# Convert ISO 8601 date strings to Unix timestamp (milliseconds)
def iso_to_unix(iso_str):
    dt = datetime.strptime(iso_str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)  # Convert to milliseconds

# get historical data

# Define the time intervals
start_time = iso_to_unix("2020-09-01T00:00:00Z")
end_time = iso_to_unix("2024-04-12T00:00:00Z")
# end_time = iso_to_unix("2023-10-01T00:00:00Z")


# Open a CSV file for writing
# save all result into csv
folder_path = 'Data'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)

csv_path = "Data/historical_BTCUSDT_3min_data.csv"
if not os.path.exists(csv_path): # Write header
    index =["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume", "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(index)

# Fetch the data in chunks
while start_time < end_time:
    time.sleep(0.125)    # 8 requests/second is maximum for users
    chunk = client.get_historical_klines(
        symbol="BTCUSDT",
        interval=Client.KLINE_INTERVAL_3MINUTE,
        start_str=start_time,
        end_str=end_time,
        limit=1000  # max limit per request
    )
    if not chunk:
        break  # Break the loop if no more data is returned

    # Write the chunk to the CSV file
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(chunk)
        file.flush()

    # Update the start_time to the timestamp of the last kline in the chunk
    start_time = chunk[-1][0] + 3 * 60 * 1000  # Add 3 minutes in milliseconds

    # Optional: Force garbage collection to free up memory
    del chunk
    gc.collect()

# takes around 5 minutes


# get until 2023 data

# Define the time intervals
start_time = iso_to_unix("2020-09-01T00:00:00Z")
end_time = iso_to_unix("2024-01-01T00:00:00Z")
# end_time = iso_to_unix("2023-10-01T00:00:00Z")


# Open a CSV file for writing
# save all result into csv
csv_path = "Data/until_2023_BTCUSDT_3min_data.csv"
if not os.path.exists(csv_path): # Write header
    index =["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume", "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(index)

# Fetch the data in chunks
while start_time < end_time:
    time.sleep(0.2)
    chunk = client.get_historical_klines(
        symbol="BTCUSDT",
        interval=Client.KLINE_INTERVAL_3MINUTE,
        start_str=start_time,
        end_str=end_time,
        limit=1000  # max limit per request
    )
    if not chunk:
        break  # Break the loop if no more data is returned

    # Write the chunk to the CSV file
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(chunk)
        file.flush()

    # Update the start_time to the timestamp of the last kline in the chunk
    start_time = chunk[-1][0] + 3 * 60 * 1000  # Add 3 minutes in milliseconds

    # Optional: Force garbage collection to free up memory
    del chunk
    gc.collect()

# takes around 1 minutes


# get 2024 data


# Define the time intervals
start_time = iso_to_unix("2024-01-01T00:00:00Z")
end_time = iso_to_unix("2024-04-12T00:00:00Z")
# end_time = iso_to_unix("2023-10-01T00:00:00Z")


# Open a CSV file for writing
# save all result into csv
csv_path = "Data/2024_BTCUSDT_3min_data.csv"
if not os.path.exists(csv_path): # Write header
    index =["Open time", "Open", "High", "Low", "Close", "Volume", "Close time", "Quote asset volume", "Number of trades", "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"]
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(index)

# Fetch the data in chunks
while start_time < end_time:
    time.sleep(0.125)
    chunk = client.get_historical_klines(
        symbol="BTCUSDT",
        interval=Client.KLINE_INTERVAL_3MINUTE,
        start_str=start_time,
        end_str=end_time,
        limit=1000  # max limit per request
    )
    if not chunk:
        break  # Break the loop if no more data is returned

    # Write the chunk to the CSV file
    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(chunk)
        file.flush()

    # Update the start_time to the timestamp of the last kline in the chunk
    start_time = chunk[-1][0] + 3 * 60 * 1000  # Add 15 minutes in milliseconds

    # Optional: Force garbage collection to free up memory
    del chunk
    gc.collect()

# takes around 1 minutes

# ----------------------------------------------------------------------------------------
# 02. threshold_data_gen

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
                current_time = datetime.now().strftime("%H:%M")
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

# read csv
df = pd.read_csv('Data/historical_BTCUSDT_3min_data.csv')
num_chunks = len(df)//1000+1

# crop df into 1000+7*24*20 size chunks
save_chunks(df=df)

# parallel process the chunks
print('start parallel processing')
run_parallel_processing(num_chunks=num_chunks)

# concatenate results
print('concatenate start')
concatenate_results(num_chunks=num_chunks)

# clean up temp files
print('cleaning up temp files...')
cleanup_temp_files(num_chunks=num_chunks)

# ----------------------------------------------------------------------------------------
# 02. Data Analysis

data = pd.read_csv('Data/historical_BTCUSDT_3min_data.csv')
threshold_data = pd.read_csv('Data/up_down_cross_3min_data.csv')
temp = threshold_data.fillna(3360)

# Create the 'minutes' column
temp['minutes'] = temp[['up_cross', 'down_cross']].min(axis=1)*3

# Create the 'side' column
temp['side'] = temp.apply(lambda row: 1 if row['up_cross'] < row['down_cross'] else (-1 if row['up_cross'] > row['down_cross'] else 0), axis=1)

# Creating a new DataFrame with just the 'minutes' and 'side' columns
one_percent_data = temp[['minutes', 'side']]

temp2 = one_percent_data.copy()
temp2['plus_6'] = ((temp2['minutes']<6*60)&(temp2['side']>0)).astype(int)
temp2['minus_6'] = ((temp2['minutes']<6*60)&(temp2['side']<0)).astype(int)
temp2['zero_6'] = ((temp2['minutes']>=6*60)|(temp2['side']==0)).astype(int)
data2 = temp2

df_all = pd.concat([data,threshold_data,data2],axis=1)

df_all.to_csv('Data/df_all.csv', index=False)


# ----------------------------------------------------------------------------------------
# 03. matrix_data_gen

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



def process_chunk(chunk, chunk_index, seq_length):
    matrix_list = []
    answer_list = []

    if len(chunk) >= seq_length:
        for i in range(len(chunk) - (seq_length-1)):
            matrix = chunk.drop(columns=['plus_6', 'minus_6', 'zero_6'], axis=1).iloc[i:i+seq_length].values
            matrix_list.append(matrix)
            answer = chunk.iloc[i+(seq_length-1)][['plus_6', 'minus_6', 'zero_6']].tolist()
            answer_list.append(answer)

            if i%1000 == 0:
                print(f'for seq length: {seq_length}, {chunk_index}th chunk, {i}/{len(chunk)} processing...')

    # Save the processed data
    np.save(f'Data/matrix_array_{seq_length}_{chunk_index}.npy', np.array(matrix_list))
    np.save(f'Data/answer_array_{seq_length}_{chunk_index}.npy', np.array(answer_list))

    del matrix_list, answer_list
    gc.collect()


# Create a list of tuples (chunk, index)
chunk_data = [(chunk, index) for index, chunk in enumerate(data_chunks)]

# Define a helper function for multiprocessing to unpack arguments
def process_chunk_wrapper(args):
    return process_chunk(*args)


# Assuming data_chunks is your list of chunks
seq_lengths = [20, 40, 60, 80]
# Modify chunk_data to include every combination of chunks, indices, and sequence lengths
chunk_data_with_seq_length = [(chunk, index, seq_length) for index, chunk in enumerate(data_chunks) for seq_length in seq_lengths]


with Pool(cpu_count()) as pool:
    pool.map(process_chunk_wrapper, chunk_data_with_seq_length)

# concatenate the npy files
for seq_length in seq_lengths:
    # List of file names to load
    file_names_matrix = [f"Data/matrix_array_{seq_length}_{i}.npy" for i in range(11)]  # Adjust range as needed
    file_names_answer = [f"Data/answer_array_{seq_length}_{i}.npy" for i in range(11)]

    # Load each file and store in a list
    loaded_arrays_matrix = [np.load(file_name) for file_name in file_names_matrix]
    loaded_arrays_answer = [np.load(file_name) for file_name in file_names_answer]

    # Concatenate all arrays into a single array
    matrix_array = np.concatenate(loaded_arrays_matrix, axis=0)
    answer_array = np.concatenate(loaded_arrays_answer, axis=0)

    # save the concatenated array
    np.save(f'Data/matrix_array_{seq_length}.npy', matrix_array)
    np.save(f'Data/answer_array_{seq_length}.npy', answer_array)

    # delete _0 ~ _10 .npy files
    for file_name in file_names_matrix + file_names_answer:
        os.remove(file_name)
    
    del file_name, file_names_matrix, file_names_answer, loaded_arrays_matrix, loaded_arrays_answer, matrix_array, answer_array
    gc.collect()

# normalize the matrix & save scaler
for seq_length in seq_lengths:
    matrix_array = np.load(f'Data/matrix_array_{seq_length}.npy')
    answer_array = np.load(f'Data/answer_array_{seq_length}.npy')

    # Assuming data is your 600k matrices concatenated into a single 3D numpy array of shape (600000, 20, 19)
    matrix_array_reshaped = matrix_array.reshape(-1, 19)  # Reshape to 2D for standardization
    scaler = StandardScaler()
    matrix_array_normalized = scaler.fit_transform(matrix_array_reshaped)

    # Reshape back to 3D
    matrix_array_normalized = matrix_array_normalized.reshape(-1, seq_length, 19)

    folder_path2 = 'Scalers'
    if not os.path.exists(folder_path2):
        os.mkdir(folder_path2)
    joblib.dump(scaler, f'Scalers/StandardScaler_{seq_length}.pkl')

    np.save(f'Data/matrix_array_{seq_length}_normalized.npy', matrix_array_normalized)

    del matrix_array, answer_array, matrix_array_reshaped, matrix_array_normalized
    gc.collect()

# ----------------------------------------------------------------------------------------
# 04. ML_modeling

def MLs(seq_length):

    matrix_array = np.load(f'Data/matrix_array_{seq_length}_normalized.npy')
    answer_array = np.load(f'Data/answer_array_{seq_length}.npy')
    print("Data load Done.")

    X = matrix_array
    y = answer_array

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=1, stratify=y)

    del X_test, y_test
    gc.collect()

    # Flatten the X_train data
    # This assumes X_train is a list of numpy arrays with a shape of (20, 19)
    X_train_flattened = np.array([x.flatten() for x in X_train])

    # answer = chunk.iloc[i+19][['plus_6', 'minus_6', 'zero_6']].tolist()
    # Assuming y_train is a list or array of arrays like [[1, 0, 0], [0, 1, 0], [0, 0, 1], ...]
    y_train_transformed = np.array([1 if np.argmax(y) == 0 else (0 if np.argmax(y) == 2 else -1) for y in y_train])
    # zero = 0, up = 1, down = -1

    print('data transform done')

    RFC = RandomForestClassifier(random_state=1, verbose=1, n_jobs=-1)

    RFC.fit(X_train_flattened, y_train_transformed)

    # Assuming your model is named RFC
    joblib.dump(RFC, f'Models/RFC_model_{seq_length}.pkl')

    del RFC
    gc.collect()

    print('saved RFC model')


    # Assuming y_train is a list or array of arrays like [[1, 0, 0], [0, 1, 0], [0, 0, 1], ...]
    y_train_transformed = np.array([1 if np.argmax(y) == 0 else (0 if np.argmax(y) == 2 else 2) for y in y_train])
    # answer = chunk.iloc[i+19][['plus_6', 'minus_6', 'zero_6']].tolist()
    # 1 = up , 2 = down, 0 = zero


    # Initialize the XGBoost classifier
    XGB = xgb.XGBClassifier(objective='multi:softprob', random_state=0)  # multi:softprob for multi-class classification

    # Fit the model
    XGB.fit(X_train_flattened, y_train_transformed)
    # Assuming your model is named RFC
    joblib.dump(XGB, f'Models/XGB_model_{seq_length}.pkl')

    del XGB
    gc.collect() 
    
    print('saved XGB model')


folder_path = 'Models'
if not os.path.exists(folder_path):
    os.mkdir(folder_path)


seq_lengths = [20,40,60,80]
for seq_length in seq_lengths:
    MLs(seq_length)
    print(f'Done for seq length : {seq_length}')


# ----------------------------------------------------------------------------------------
# 05. NN_modeling

# data
def data_loaders(seq_length):
    matrix_array = np.load(f'Data/matrix_array_{seq_length}_normalized.npy')
    answer_array = np.load(f'Data/answer_array_{seq_length}.npy')
    
    labels = torch.tensor(answer_array)
    indices = torch.argmax(labels, dim=1)
    mapped_labels = torch.tensor([1 if i == 0 else 2 if i == 1 else 0 for i in indices])
    # answer = ['plus_6', 'minus_6', 'zero_6']
    # 1 = up , 2 = down, 0 = zero

    X = matrix_array
    y = mapped_labels

    X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size=0.2, random_state=1, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1, stratify=y_temp)

    # Convert Numpy arrays to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)  # LSTM use float32
    X_valid = torch.tensor(X_valid, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)  # CrossEntropyLoss use long
    y_valid = torch.tensor(y_valid, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # TensorDatasets
    train_dataset = TensorDataset(X_train, y_train)
    valid_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)

    # DataLoaders
    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader

def training(model_name, model_instance, seq_length, train_loader, valid_loader):
    model = model_instance

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device : ', device)

    model.to(device)

    # Loss, Optimizer, Num of epochs, Early Stopping, ReduceLROnPlateau
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    num_epochs = 1000000

    best_val_loss = float('inf')
    patience = 5

    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3, min_lr=0.001)

    # initialize history
    history = {'train_loss' : [], 'val_loss' : [], 'train_accuracy' : [], 'val_accuracy' : []}

    # start fitting 
    print(f'start fitting {model_name}_{seq_length}')

    # for epochs
    for epoch in range(num_epochs):
        # training process
        model.train()

        running_loss = 0.0
        correct_count = 0
        total_count = 0

        # Wrap loaders with tqdm for a progress bar
        pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}')
        
        # for batches
        for i, (inputs, labels) in pbar_train:
            # send X,y batches to device RAM
            inputs, labels = inputs.to(device), labels.to(device)

            # initializer optimizer with zero gradient
            optimizer.zero_grad()

            # calculate with model
            outputs = model(inputs)
            
            # loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # step each steps for batch
            optimizer.step()
            
            # for accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_count += labels.size(0)
            correct_count += (predicted == labels).sum().item()

            # Update progress bar
            current_loss = running_loss / (i+1)
            current_accuracy = correct_count / total_count
            pbar_train.set_postfix({'loss' : current_loss, 'accuracy' : current_accuracy})
                
        # for model.train()
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct_count / total_count

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)


        # Validation step
        model.eval()

        val_running_loss = 0.0
        val_correct_count = 0
        val_total_count = 0

        # Wrap loader with tqdm for a progress bar
        pbar_eval = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f'Epoch {epoch+1}/{num_epochs}')

        # no gradient in validation step
        with torch.no_grad():
            for i, (inputs, labels) in pbar_eval:
                # X,y to device RAM
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)

                loss = criterion(outputs,labels)
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total_count += labels.size(0)
                val_correct_count += (predicted == labels).sum().item()

                # Update progress bar
                current_val_running_loss = val_running_loss / (i+1)
                current_val_accuracy = val_correct_count / val_total_count
                pbar_eval.set_postfix({'val_loss' : current_val_running_loss, 'val_accuracy' : current_val_accuracy})

        # for each model.eval()
        val_loss = val_running_loss / len(valid_loader)
        val_accuracy = val_correct_count / val_total_count

        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
    
        # for record in command prompt
        logs = f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}\n'
            
        print(logs)
        with open('NN_modeling_logs.txt','a') as f:
            f.write(logs)

        # Reduce LR on Plateau
        # Call scheduler step after completing the validation phase of the current epoch
        scheduler.step(val_loss)

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'Models/best_model_state_dict.pth')
            torch.save(model, 'Models/best_model.pth')
            patience = 5  # Reset patience when finding a better model

        else:
            patience -= 1
            if patience == 0:
                break  # break whole epoch iteration

    # for epoch iteration done.
    print(f'Training complete {model_name}')

    # Save model and history
    torch.save(model.state_dict(), f'Models/{model_name}_model_state_dict_{seq_length}.pth')
    torch.save(model, f'Models/{model_name}_model_{seq_length}.pth')
    print(f'Saved {model_name}_{seq_length} model')

    with open(f'Models/{model_name}_history_{seq_length}.json', 'w') as f:
        json.dump(history, f)
    print('Saved history.json')


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()

        # Multi-layer LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True, dropout=0.2)

        # Flatten layer 

        # Define Linear layers with Batch Normalization, GELU, and Dropout
        self.linear1 = nn.Linear(hidden_dim, 64)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.2)

        self.linear2 = nn.Linear(64, 32)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.2)

        self.linear3 = nn.Linear(32, 16)
        self.batch_norm3 = nn.BatchNorm1d(16)
        self.gelu3 = nn.GELU()

        # Output layer
        self.output_layer = nn.Linear(16, output_dim)

    def forward(self, x):
        # Pass input through LSTM layers
        # print("Output type1:", type(x))
        (lstm_out, _) = self.lstm(x)

        # Taking the output of the last time step
        # print("Output type2:", type(x))
        x = lstm_out[:, -1, :]

        # Pass through Linear layers
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        x = self.batch_norm3(x)
        x = self.gelu3(x)

        # Output layer
        x = self.output_layer(x)
        
        return x 

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # GRU Layer
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

        # Flatten layer 

        # Define Linear layers with Batch Normalization, GELU, and Dropout
        self.linear1 = nn.Linear(hidden_dim, 64)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.gelu1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.2)

        self.linear2 = nn.Linear(64, 32)
        self.batch_norm2 = nn.BatchNorm1d(32)
        self.gelu2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.2)

        self.linear3 = nn.Linear(32, 16)
        self.batch_norm3 = nn.BatchNorm1d(16)
        self.gelu3 = nn.GELU()

        # Output layer
        self.output_layer = nn.Linear(16, output_dim)

    def forward(self, x):
        # Pass input through GRU layers
        # print("Output type1:", type(x))
        gru_out, _ = self.gru(x)

        # Taking the output of the last time step
        # print("Output type2:", type(x))
        x = gru_out[:, -1, :]

        # Pass through Linear layers
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.dropout1(x)

        x = self.linear2(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        x = self.dropout2(x)

        x = self.linear3(x)
        x = self.batch_norm3(x)
        x = self.gelu3(x)

        # Output layer
        x = self.output_layer(x)
        
        return x 

class Conv1DModel(nn.Module):
    def __init__(self, num_features, output_dim, seq_length):
        super(Conv1DModel, self).__init__()

        # Conv1D Layers
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(128)  
        self.gelu1 = nn.GELU()
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(0.2)  # Dropout
        
        

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(64)  
        self.gelu2 = nn.GELU()
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout2 = nn.Dropout(0.2)  # Dropout
        

        self.conv3 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm1d(32)  
        self.gelu3 = nn.GELU()
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2, padding=0)
        self.dropout3 = nn.Dropout(0.2)  # Dropout
        

        # Flatten layer 
        self.seq_length_after_conv_and_pool =seq_length // 2 // 2 // 2 # Pooling 3 times with stride 2

        # Define Linear layers with Batch Normalization, GELU, and Dropout
        self.linear1 = nn.Linear(32 * self.seq_length_after_conv_and_pool, 64)
        self.batch_norm_lin1 = nn.BatchNorm1d(64)
        self.gelu_lin1 = nn.GELU()
        self.dropout_lin1 = nn.Dropout(0.2)

        self.linear2 = nn.Linear(64, 32)
        self.batch_norm_lin2 = nn.BatchNorm1d(32)
        self.gelu_lin2 = nn.GELU()
        self.dropout_lin2 = nn.Dropout(0.2)

        self.linear3 = nn.Linear(32, 16)
        self.batch_norm_lin3 = nn.BatchNorm1d(16)
        self.gelu_lin3 = nn.GELU()

        # Output layer
        self.output_layer = nn.Linear(16, output_dim)


    def forward(self, x):
        # Assuming x shape is (batch_size, seq_length, num_features)
        # Conv1d expects (batch_size, in_channels, seq_length), so transpose x
        x = x.transpose(1, 2)  # Now x shape: (batch_size, num_features, seq_length)

        # Apply Conv1D layers followed by pooling
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.gelu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.gelu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.gelu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(-1, 32 * self.seq_length_after_conv_and_pool)

        # Pass through Linear layers
        x = self.linear1(x)
        x = self.batch_norm_lin1(x)
        x = self.gelu_lin1(x)
        x = self.dropout_lin1(x)

        x = self.linear2(x)
        x = self.batch_norm_lin2(x)
        x = self.gelu_lin2(x)
        x = self.dropout_lin2(x)

        x = self.linear3(x)
        x = self.batch_norm_lin3(x)
        x = self.gelu_lin3(x)

        # Output layer
        x = self.output_layer(x)
        
        return x 
    
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length, num_classes, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=256, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.seq_length = seq_length

        # Input embedding layer
        self.embedding = nn.Linear(input_dim, d_model)

        # Positional Encoding (Not using nn.Embedding here to keep it simple)
        self.positional_encoding = PositionalEncoding(d_model, dropout, seq_length)

        # Transformer
        transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=num_encoder_layers)


        # Define Linear layers with Batch Normalization, GELU, and Dropout
        self.linear1 = nn.Linear(seq_length * d_model, 64)
        self.batch_norm_lin1 = nn.BatchNorm1d(64)
        self.gelu_lin1 = nn.GELU()
        self.dropout_lin1 = nn.Dropout(0.2)

        self.linear2 = nn.Linear(64, 32)
        self.batch_norm_lin2 = nn.BatchNorm1d(32)
        self.gelu_lin2 = nn.GELU()
        self.dropout_lin2 = nn.Dropout(0.2)

        self.linear3 = nn.Linear(32, 16)
        self.batch_norm_lin3 = nn.BatchNorm1d(16)
        self.gelu_lin3 = nn.GELU()

        # Output layer
        self.output_layer = nn.Linear(16, output_dim)


    def forward(self, src):
        # Assuming src shape is (batch_size, seq_length, input_dim)
        # Transformer expects (seq_length, batch_size, input_dim), so transpose src
        src = src.transpose(0, 1)

        # Embedding and positional encoding
        src = self.embedding(src)  # Now shape is (seq_length, batch_size, d_model)
        src = self.positional_encoding(src)

        # Transformer
        output = self.transformer(src)

        # For linear layers, we'll consider the output of all positions.
        # Reshape output to (batch_size, seq_length * d_model) before passing to linear layers.
        # Note: Adjusting this as per the expected input for linear layers.
        output = output.transpose(0, 1)  # Change back to (batch_size, seq_length, d_model)
        x = output.flatten(start_dim=1)

        # Pass through Linear layers
        x = self.linear1(x)
        x = self.batch_norm_lin1(x)
        x = self.gelu_lin1(x)
        x = self.dropout_lin1(x)

        x = self.linear2(x)
        x = self.batch_norm_lin2(x)
        x = self.gelu_lin2(x)
        x = self.dropout_lin2(x)

        x = self.linear3(x)
        x = self.batch_norm_lin3(x)
        x = self.gelu_lin3(x)

        # Output layer
        x = self.output_layer(x)
        
        return x 

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LinearModel(nn.Module):
    def __init__(self, seq_length, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        
        # Calculate the flattened input size
        self.flattened_size = seq_length * input_dim
        
        # Define Linear layers with Batch Normalization, GELU, and Dropout
        self.linear1 = nn.Linear(self.flattened_size, 128)
        self.batch_norm_lin1 = nn.BatchNorm1d(128)
        self.gelu_lin1 = nn.GELU()
        self.dropout_lin1 = nn.Dropout(0.2)

        self.linear2 = nn.Linear(128, 48)
        self.batch_norm_lin2 = nn.BatchNorm1d(48)
        self.gelu_lin2 = nn.GELU()
        self.dropout_lin2 = nn.Dropout(0.2)

        self.linear3 = nn.Linear(48, 16)
        self.batch_norm_lin3 = nn.BatchNorm1d(16)
        self.gelu_lin3 = nn.GELU()

        # Output layer
        self.output_layer = nn.Linear(16, output_dim)


    def forward(self, x):
        # Flatten the input
        x = x.view(-1, self.flattened_size)  # Reshape input to (batch_size, seq_length*input_dim)
        
        # Pass through Linear layers
        x = self.linear1(x)
        x = self.batch_norm_lin1(x)
        x = self.gelu_lin1(x)
        x = self.dropout_lin1(x)

        x = self.linear2(x)
        x = self.batch_norm_lin2(x)
        x = self.gelu_lin2(x)
        x = self.dropout_lin2(x)

        x = self.linear3(x)
        x = self.batch_norm_lin3(x)
        x = self.gelu_lin3(x)

        # Output layer
        x = self.output_layer(x)
        
        return x
    

print('Start')

seq_length = 40
# data loaders
train_loader, valid_loader, test_loader = data_loaders(seq_length)
print('DataLoader Set')
# models
models = {
    'Conv1D' : Conv1DModel(num_features=19, output_dim=3, seq_length=seq_length),
    'Transformer' :  TransformerModel(input_dim=19, output_dim=3, seq_length=seq_length, num_classes=3, \
                                    d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.1),
    "Linear" : LinearModel(seq_length=seq_length, input_dim=19, output_dim=3),
}

for model_name, model_instance in models.items():
    training(model_name, model_instance, seq_length, train_loader, valid_loader)

print('Done for 40')

seq_lengths = [60, 80]
for seq_length in seq_lengths:

    # data loaders
    train_loader, valid_loader, test_loader = data_loaders(seq_length)
    print('DataLoader Set')

    # models
    models = {
        'LSTM' : LSTMModel(input_dim=19, hidden_dim=128, output_dim=3),
        'GRU' : GRUModel(input_dim=19, hidden_dim=128, output_dim=3, num_layers=3),
        'Conv1D' : Conv1DModel(num_features=19, output_dim=3, seq_length=seq_length),
        'Transformer' :  TransformerModel(input_dim=19, output_dim=3, seq_length=seq_length, num_classes=3, \
                                        d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=256, dropout=0.1),
        "Linear" : LinearModel(seq_length=seq_length, input_dim=19, output_dim=3),
    }

    for model_name, model_instance in models.items():
        training(model_name, model_instance, seq_length, train_loader, valid_loader)
    

print('Done')



# ----------------------------------------------------------------------------------------
# 06. Predictions

