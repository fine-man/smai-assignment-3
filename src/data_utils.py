import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer

from src import *
from src.classifiers import *

def minibatch_generator(X, y, minibatch_size=100):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0],
                            minibatch_size):
        minibatch_idx = indices[start_idx: start_idx + minibatch_size]
        yield X[minibatch_idx], y[minibatch_idx]

# Wine Dataset
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

# Advertisement Dataset
def preprocess_advertisement_dataset(data_df):
    """
    Pre-Process the advertisement.csv data
    
    Bool columns are converted to {0, 1}
    String columns are converted to Ordinal Encoding
    Other columns are left unchanged
    
    Input:
        data_df : a pandas dataframe containing the data without the labels
    
    Output:
        feature_values : a numpy array containing the features for each datapoint
    """
    
    data_df["married"] = data_df["married"].astype(int)
    
    # find all the boolean columns
    # find all the string columns
    
    feature_values = data_df.values
    column_transform = ColumnTransformer([
        ('onehot', OrdinalEncoder(), [1, 3, 6, 7, 9]),
        ('nothing', 'passthrough', [0, 2, 4, 5, 8]),
    ])
    
    feature_values = column_transform.fit_transform(feature_values).astype(float)
    
    return feature_values

def get_and_process_advertisement_dataset(path, random_state=42):
    advert_df = pd.read_csv(path)

    # Finding all the unique labels
    all_labels = advert_df["labels"].str.split(" ").values # (1000, )

    all_labels = np.concatenate(all_labels, axis=0) # (2758, )
    unique_labels, label_freq = np.unique(all_labels, return_counts=True)

    features = preprocess_advertisement_dataset(advert_df.iloc[:, :-1])

    # creating a multi label binarizer/encoder for class labels
    mlb = MultiLabelBinarizer()

    mlb.fit([unique_labels])

    # splitting the label string
    labels_np = advert_df["labels"].str.split(" ").values # (1000, ) each element is a list

    labels_powerset = mlb.transform(labels_np) # converts each list to a vector containing 1's and 0's

    # Splitting into Train/Val/Test
    # Test data = 20% of entire data, Val data = 10 % of entire data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels_powerset, test_size=0.3, random_state=random_state
    )

    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.50, random_state=random_state
    )

    # Standard Scaling of all data
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    print(f"Mean of training features:\n{scaler.mean_}\n")
    print(f"Variance of training features: {scaler.var_}")

    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Housing Dataset
def preprocess_housing_dataset(housing_df):
    for column_name in housing_df.columns:
        # Calculate the median of the correct column
        median_value = housing_df[column_name].median()

        # Fill NaN value in the current column with the median
        housing_df[column_name].fillna(median_value, inplace=True)
    
    return housing_df

def get_and_process_housing_dataset(path, random_state=42):
    # reading the housing data
    housing_df = pd.read_csv(path)

    # pre-processing the data
    housing_df = preprocess_housing_dataset(housing_df)

    # converting to numpy
    X = housing_df.iloc[:, :-1].to_numpy()
    y = housing_df.iloc[:, -1].to_numpy()
    
    # splitting into train, val, test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=random_state
    )

    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size=0.33, random_state=random_state
    )

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test
