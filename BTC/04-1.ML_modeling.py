# imports
import numpy as np

import joblib, gc, os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb


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


    # # Assuming y_train is a list or array of arrays like [[1, 0, 0], [0, 1, 0], [0, 0, 1], ...]
    # y_train_transformed = np.array([1 if np.argmax(y) == 0 else (0 if np.argmax(y) == 2 else 2) for y in y_train])
    # # answer = chunk.iloc[i+19][['plus_6', 'minus_6', 'zero_6']].tolist()
    # # 1 = up , 2 = down, 0 = zero


    # # Initialize the XGBoost classifier
    # XGB = xgb.XGBClassifier(objective='multi:softprob', random_state=0)  # multi:softprob for multi-class classification

    # # Fit the model
    # XGB.fit(X_train_flattened, y_train_transformed)
    # # Assuming your model is named RFC
    # joblib.dump(XGB, f'Models/XGB_model_{seq_length}.pkl')

    # del XGB
    # gc.collect() 
    
    # print('saved XGB model')

if __name__ == '__main__':
    folder_path = 'Models'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


    seq_lengths = [20,40,60,80]
    for seq_length in seq_lengths:
        MLs(seq_length)
        print(f'Done for seq length : {seq_length}')