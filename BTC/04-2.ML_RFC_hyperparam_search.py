# imports
import numpy as np
import pandas as pd

import joblib, gc, os
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

class TimeLogging_RandomForestClassifier(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None):
        start_time = datetime.now()
        print(f"Starting training at {start_time.strftime('%H:%M')}")
        
        # Call the original 'fit' method to perform training
        result = super().fit(X, y, sample_weight=sample_weight)
        
        end_time = datetime.now()
        elapsed_time = end_time - start_time
        
        elapsed_hours = elapsed_time.seconds // 3600
        elapsed_minutes = (elapsed_time.seconds % 3600) // 60
        elapsed_seconds = (elapsed_time.seconds % 60)
        
        print(f"Finished training at {end_time.strftime('%H:%M')}")
        print(f"Elapsed time: {elapsed_hours}h {elapsed_minutes}m {elapsed_seconds}s")

        return result


if __name__ == '__main__':

    matrix_array = np.load(f'Data/matrix_array_20_normalized.npy')
    answer_array = np.load(f'Data/answer_array_20.npy')
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

    # Example parameters to search over
    param_grid = {
        'n_estimators': [100, 200, 300],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
        'bootstrap': [True, False]        # Method for sampling data points (with or without replacement)
    }

    # Initialize the classifier
    RFC = TimeLogging_RandomForestClassifier(random_state=1, verbose=1, n_jobs=-1)

    # Setup the grid search with F1 scoring
    grid_search = GridSearchCV(estimator=RFC, param_grid=param_grid, cv=3, scoring='f1_macro', n_jobs=1, verbose=2)


    # Assuming X_train_flattened and y_train_transformed are already defined
    grid_search.fit(X_train_flattened, y_train_transformed)

    # Assuming grid_search has completed
    results = pd.DataFrame(grid_search.cv_results_)
    results.to_csv('Data/grid_search_results.csv', index=False)

    top_5 = results_sorted.head(5)  # Adjust the number as needed

    # Directory to save models
    if not os.path.exists('Models'):
        os.makedirs('Models')

    # Iterate over the top 5 parameter settings
    for index, row in top_5.iterrows():
        # Setup the RandomForestClassifier with the best parameters
        model = RandomForestClassifier(
            n_estimators=row['param_n_estimators'],
            max_depth=row['param_max_depth'],
            min_samples_split=row['param_min_samples_split'],
            min_samples_leaf=row['param_min_samples_leaf'],
            bootstrap=row['param_bootstrap'],
            random_state=1, verbose=1, n_jobs=-1
        )

        # Train the model with the same training set
        model.fit(X_train_flattened, y_train_transformed)

        # Save each model
        model_path = f'Models/RFC_model_rank_{row["rank_test_score"]}.pkl'
        joblib.dump(model, model_path)
        print(f'Model saved to {model_path}')

    print('Top 5 models saved.')