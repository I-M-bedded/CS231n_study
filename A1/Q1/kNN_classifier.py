import numpy as np
from cs231n import data_utils
from cs231n import vis_utils

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store the training data."""
        self.X_train = X
        self.y_train = y

    def predict_twoloops(self, X):
        """Predict the labels for the test data using two loops."""
        num_test = X.shape[0]
        y_pred = np.zeros(num_test, dtype=self.y_train.dtype)

        for i in range(num_test):
            distances = np.zeros(self.X_train.shape[0])
            for j in range(self.X_train.shape[0]):
                distances[j] = np.linalg.norm(X[i] - self.X_train[j])
            y_pred[i] = self._predict_label(distances)

        return y_pred