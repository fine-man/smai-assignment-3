import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src import *
from src.classifiers import *

def load_wine_dataset(path):
    wine_df = pd.read_csv(path)
    X, y = wine_df.iloc[:, :-2].to_numpy(), wine_df.iloc[:, -2].to_numpy()
    y -= np.min(y) # making the range of classes to be between [0, 5]
    return X, y

def get_and_process_wine_dataset(path, random_state=42):
    # DATA Loading and Pre-Processing
    X, y = load_wine_dataset(path)

    # Train, Val, Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.33, random_state=42
    )

    # Standarizing the data
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test

def minibatch_generator(X, y, minibatch_size=100):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0],
                            minibatch_size):
        minibatch_idx = indices[start_idx: start_idx + minibatch_size]
        yield X[minibatch_idx], y[minibatch_idx]