# imports
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import joblib, gc, os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
import xgboost as xgb

# Get the current absolute path of the directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))



def MLs(seq_length):
    # Construct the full file paths
    matrix_array_path = os.path.join(BASE_DIR, f'Data/matrix_array_{seq_length}_normalized.npy')
    answer_array_path = os.path.join(BASE_DIR, f'Data/answer_array_{seq_length}.npy')

    # Before loading, check if the files exist to ensure the paths are correct
    if os.path.exists(matrix_array_path) and os.path.exists(answer_array_path):
        matrix_array = np.load(matrix_array_path)
        answer_array = np.load(answer_array_path)
        print("Files loaded successfully.")
    else:
        print("One or more file paths are incorrect or the files do not exist.")

    X = matrix_array
    y = answer_array

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=1, stratify=y)

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


    # ### Linear SVC 
    # Initialize the LinearSVC model
    linear_svc = LinearSVC(random_state=0)

    # Fit the model
    linear_svc.fit(X_train_flattened, y_train_transformed)

    joblib.dump(linear_svc, f'Models/Linear_SVC_model_{seq_length}.pkl')

    del linear_svc
    gc.collect()

    print('saved Linear SVC model')

    # ### Nu SVC

    # Initialize the NuSVC model
    # The nu parameter may need to be adjusted based on your dataset
    nu_svc = NuSVC(nu=0.5, random_state=0)

    # Fit the model
    nu_svc.fit(X_train_flattened, y_train_transformed)

    # Assuming your model is named RFC
    joblib.dump(nu_svc, f'Models/Nu_SVC_model_{seq_length}.pkl')

    del nu_svc
    gc.collect()

    print('saved NuSVC model')

    # ### SVC
    # Initialize the SVC model
    # You can change the kernel to 'linear', 'poly', 'rbf', 'sigmoid', etc.
    svc = SVC(kernel='rbf', random_state=0)

    # Fit the model
    svc.fit(X_train_flattened, y_train_transformed)

    # Assuming your model is named RFC
    joblib.dump(svc, f'Models/SVC_model_{seq_length}.pkl')

    del svc
    gc.collect()

    print('saved SVC model')



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

if __name__ == '__main__':
    seq_lengths = [20,40,60,80]
    for seq_length in seq_lengths:
        MLs(seq_length)
        print(f'Done for seq length : {seq_length}')