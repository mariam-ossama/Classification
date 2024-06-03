import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def train_test_split(data, test_size=0.25, random_state=None):
    if isinstance(data, pd.DataFrame):
        data = data.values

    if random_state:
        np.random.seed(random_state)

    # Shuffle the indices
    indices = np.random.permutation(len(data))

    # Calculate the number of samples for training and testing
    n_train = int(len(data) * 0.75)
    n_test = len(data) - n_train

    if n_train == 0 or n_test == 0:
        raise ValueError("Insufficient data for train-test split. Please use a smaller percentage.")

    # Split the data
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]


    '''The target variable 'diabetes' is selected as the last column (data[:, -1]) and separated from the feature matrix.'''
    X_train, X_test = data[train_indices, :-1], data[test_indices, :-1]
    y_train, y_test = data[train_indices, -1].astype(int), data[test_indices, -1].astype(int)

    return X_train, X_test, y_train, y_test

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def fit(self, data):
        # Remove duplicated rows
        data = data.drop_duplicates().copy()  # Make a copy of the DataFrame

        # Encode categorical columns: gender and smoking_history
        columns_to_encode = ['gender', 'smoking_history']
        for column in columns_to_encode:
            self.label_encoders[column] = LabelEncoder()
            data[column] = self.label_encoders[column].fit_transform(data[column])

        # Save the columns to scale
        self.columns_to_scale = [col for col in data.columns if col != 'diabetes']

        # Fit the scaler on the data
        self.scaler.fit(data[self.columns_to_scale])

    def transform(self, data):
        # Apply encoding
        for column in self.label_encoders:
            data.loc[:, column] = self.label_encoders[column].transform(data[column])

        # Apply feature scaling
        data.loc[:, self.columns_to_scale] = self.scaler.transform(data[self.columns_to_scale])

        return data