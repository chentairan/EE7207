from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


class SVM(object):

    def __init__(self, kernel: str):
        """ SVM network
        # Arguments
            kernel: the kernel used in SVM
        """
        self.model = svm.SVC(kernel=kernel, C=10, gamma=0.1)

    def fit(self, X, Y):
        """ Fits SVM
        # Arguments
            X: training samples
            Y: targets
        # Input shape
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        self.model.fit(X, Y)
        self.model.score(X, Y)

    def predict(self, X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        predictions = self.model.predict(X)
        return predictions
