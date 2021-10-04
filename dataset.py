import numpy as np
import scipy.io as scio
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, path: str):
        self.raw_train_data = scio.loadmat(path + "/data_train.mat")['data_train']
        self.raw_train_label = scio.loadmat(path + "/label_train.mat")['label_train']

        self.train_data = np.empty(0)
        self.train_label = np.empty(0)
        self.validation_data = np.empty(0)
        self.validation_label = np.empty(0)

        self.split_train_data()

        self.test_data = scio.loadmat(path + "/data_test.mat")['data_test']

    def get_train_data(self) -> (np.ndarray, np.ndarray):
        return self.train_data, self.train_label.ravel()

    def get_validation_data(self) -> (np.ndarray, np.ndarray):
        return self.validation_data, self.validation_label.ravel()

    def get_test_data(self) -> np.ndarray:
        return self.test_data

    def split_train_data(self):
        self.train_data, self.validation_data, self.train_label, self.validation_label = train_test_split(
            self.raw_train_data, self.raw_train_label, test_size=0.30, random_state=123)
        return
