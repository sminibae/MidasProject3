# imports
import pandas as pd
import numpy as np

import math

import torch
import torch.nn as nn
import torch.nn.init as init

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

import json, gc

from sklearn.model_selection import train_test_split


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
    batch_size = 128
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


def init_weights(m):
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:  # Input-hidden weights
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:  # Hidden-hidden weights
                init.orthogonal_(param.data)  # Helps maintain stability
            elif 'bias' in name:  # Biases initialization
                # Biases could be set to zero or using a more sophisticated scheme:
                # Setting biases such that the forget gate has initially more influence
                param.data.fill_(0)
                # It's common practice to initialize biases for gates to ensure better initial performance
                # For GRUs, biases are usually split into two parts, each for input and recurrent
                # For a GRU, the bias is a single vector with `2*hidden_size` elements
                # It may be beneficial to initialize the parts corresponding to the reset and update gates to 1
                n = param.size(0)
                param[n//4:n//2].fill_(1)  # Update gate bias

    elif isinstance(m, nn.Conv1d):
        # Initialize Conv1D layers with He initialization
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

    elif isinstance(m, nn.TransformerEncoder) or isinstance(m, nn.TransformerEncoderLayer):
        # Initialize Transformer weights using a more suitable scheme
        # Usually, Transformer weights are initialized slightly differently to prevent early saturation
        # and help convergence. A common practice is using xavier_uniform with gain adjusted for non-linearity.
        for name, param in m.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param, gain=init.calculate_gain('relu'))
            elif 'bias' in name:
                init.constant_(param, 0)

    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

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

        # Apply the custom initializer to all layers
        self.apply(init_weights)

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

        # Apply the custom initializer to all layers
        self.apply(init_weights)

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

        # Apply the custom initializer to all layers
        self.apply(init_weights)


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


# main
if __name__ == '__main__':
    print('Start')

    seq_length = 20
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