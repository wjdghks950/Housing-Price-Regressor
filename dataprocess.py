import torch
import numpy as np
import pandas as pd
import xlrd # download xlrd
from torch.utils.data import Dataset

class HousePriceDataset(Dataset):
    def __init__(self, path, transform=None, label=False):
        self.data = pd.read_excel(path)
        self.label = label
        self.X, self.Y = self.preprocess()
        self.feature_len = self.X.shape[1]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y[idx])
    
    def get_feature_dim(self):
        return self.feature_len

    def preprocess(self):
        self.data.iloc[:,23] = pd.to_numeric(self.data.iloc[:,23], errors='coerce').fillna(0)
        numeric_features = self.data.dtypes[self.data.dtypes != 'object'].index # Numeric features only index
        normalized_data = self.data.copy()
        normalized_data[numeric_features] = normalized_data[numeric_features].apply(lambda x: (x-x.mean()) / x.std())
        normalized_data[numeric_features] = normalized_data[numeric_features].fillna(0)
        # Use numeric only
        normalized_data = normalized_data[numeric_features]
        num_data = self.data.shape[0]
        norm_X = np.array(normalized_data[:num_data].values, dtype=np.float32)
        if (self.label):
            labels = np.array(self.data[list(self.data.columns)[2]].values, dtype=np.float32).reshape(-1,1) # Test data has no label (regression problem)
        else:
            labels = np.zeros((self.data.shape[0], 1))
        return norm_X, labels


def load_data(path_train, path_test):
    train_data = pd.read_excel(path_train)
    test_data = pd.read_excel(path_test)
    print('[ Before Normalization ]')
    print(train_data.shape)
    print(test_data.shape)
    # Convert `parking_per` column to numeric dtype
    train_data.iloc[:,23] = pd.to_numeric(train_data.iloc[:,23], errors='coerce').fillna(0)
    test_data.iloc[:,23] = pd.to_numeric(test_data.iloc[:,23], errors='coerce').fillna(0)

    train_num_features = train_data.dtypes[train_data.dtypes != 'object'].index # Numeric features only index
    test_num_features = test_data.dtypes[test_data.dtypes != 'object'].index
    # Normalize numeric features
    norm_train_data = train_data.copy()
    norm_test_data = test_data.copy()
    norm_train_data[train_num_features] = norm_train_data[train_num_features].apply(lambda x: (x-x.mean()) / x.std())
    norm_train_data[train_num_features] = norm_train_data[train_num_features].fillna(0)
    norm_test_data[test_num_features] = norm_test_data[test_num_features].apply(lambda x: (x-x.mean()) / x.std())
    norm_test_data[test_num_features] = norm_test_data[test_num_features].fillna(0)
    # Set non-numeric values (discrete values) to one-hot values (0 or 1)
    # norm_train_data = pd.get_dummies(norm_train_data, dummy_na=True)
    # norm_test_data = pd.get_dummies(norm_train_data, dummy_na=True)
    # print('[ After Normalization ]')
    # print(norm_train_data.shape)
    # print(norm_test_data.shape)

    # Use numeric only
    norm_train_data = norm_train_data[train_num_features]
    norm_test_data = norm_test_data[test_num_features]
    print('[Numeric only(train)] : ', norm_train_data.shape)
    print('[Numeric only(test)] : ',norm_test_data.shape)

    n_train = train_data.shape[0]
    n_test = test_data.shape[0]
    train_features = np.array(norm_train_data[:n_train].values, dtype=np.float32)
    test_features = np.array(norm_test_data[:n_test].values, dtype=np.float32)
    train_labels = np.array(train_data[list(train_data.columns)[2]].values, dtype=np.float32).reshape(-1,1)

    # print(list(norm_train_data.columns))
    print('\n< Training labels >\n', train_labels)

    return train_features, train_labels, test_features