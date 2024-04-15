# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.init as init

from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from tqdm import  tqdm

import os, json, gc, io, joblib
from contextlib import redirect_stdout

from sklearn.model_selection import train_test_split

# Raw data
matrix_array_20 = np.load('Data/matrix_array_20_normalized.npy')
answer_array_20 = np.load('Data/answer_array_20.npy')

labels = torch.tensor(answer_array_20)
indices = torch.argmax(labels, dim=1)
mapped_labels = torch.tensor([1 if i == 0 else 2 if i == 1 else 0 for i in indices])

X = matrix_array_20
y = mapped_labels

X_train, X_temp, y_train, y_temp = train_test_split(X,y, test_size=0.2, random_state=1, stratify=y)
X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1, stratify=y_temp)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)  # Use torch.long for labels if using CrossEntropyLoss
X_valid_tensor = torch.tensor(X_valid, dtype=torch.float32)
y_valid_tensor = torch.tensor(y_valid, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# Create DataLoaders
batch_size = 128  # You can adjust the batch size as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Now your DataLoaders are ready to be used in the training loop


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
                new_param = param.clone()
                new_param[n//4:n//2].fill_(1)  # Update gate bias
                param.data = new_param

    elif isinstance(m, nn.Conv1d):
        # Initialize Conv1D layers with He initialization
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.constant_(m.bias, 0)

    elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


# Custom LSTM model
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


# Initialize the model
model = LSTMModel(input_dim=19, hidden_dim=128, output_dim=3)

# Setting device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model.to(device)

# Loss Function
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters())

num_epochs = 10000000

# Early Stopping and Model Checkpoint can be manually implemented in the training loop
best_val_loss = float('inf')
patience = 20  # For early stopping

scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3, min_lr=0.001)



# initialize history
history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}


print('start fitting')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Wrap your loader with tqdm for a progress bar
    pbar_train = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for i, (inputs, labels) in pbar_train:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # for history
        # Calculate predictions for accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
         # Update progress bar
        current_loss = running_loss / (i + 1)
        current_accuracy = 100 * correct / total
        pbar_train.set_postfix({'loss' : current_loss, 'accuracy': current_accuracy})
        
    # Calculate average loss and accuracy over the epoch
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    
    history['train_loss'].append(train_loss)
    history['train_accuracy'].append(train_accuracy)
    
    # Validation step
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    # Wrap your loader with tqdm for a progress bar
    pbar_eval = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=f"Epoch {epoch+1}/{num_epochs}")
    
    with torch.no_grad():
        for i, (inputs, labels) in pbar_eval:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            # for history
            # Calculate predictions for accuracy
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            # Update progress bar
            current_val_loss = val_loss / (i + 1)
            current_val_accuracy = 100 * val_correct / val_total
            pbar_eval.set_postfix({'val_loss': current_val_loss, 'val_accuracy': current_val_accuracy})
            
    # Calculate average loss and accuracy over the validation set
    val_loss = val_loss / len(valid_loader)
    val_accuracy = 100 * val_correct / val_total
    
    # Append to history after each epoch
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)
    
    
    # for record in command prompt
    logs = f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}\n'
        
    print(logs)
    with open('logs.txt','a') as f:
        f.write(logs)

    
    
    # Early stopping and Model checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'Models/LSTM/best_model_state_dict.pth')
        torch.save(model, 'Models/LSTM/best_model.pth')
        patience = 20  # Reset patience since we found a better model
    else:
        patience -= 1
        if patience == 0:
            break
    
    # Garbage collection
    gc.collect()

    
print("Training complete")


with open('Models/LSTM/LSTM_history2.json', 'w') as f:
    json.dump(history, f)
print('Saved history.json')

# Save final model
torch.save(model.state_dict(), 'Models/LSTM/LSTM_model_state_dict_custominitialize.pth')
torch.save(model, 'Models/LSTM/LSTM_model_custominitialize.pth')
print('Saved LSTM model')


