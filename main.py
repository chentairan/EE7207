import scipy.io as scio
import numpy as np
from model.RBFN import RBFN
from model.SVM import SVM
from model.SOM import MiniSom
from sklearn.cluster import KMeans


def load_data():
    train_data = scio.loadmat("data/data_train.mat")['data_train']
    train_label = scio.loadmat("data/label_train.mat")['label_train']
    test_data = scio.loadmat("data/data_test.mat")['data_test']
    return train_data, train_label.ravel(), test_data


def eval(predict, label):
    RMSE = np.linalg.norm(label - predict) ** 2 / len(label)
    right = label == predict
    Accuracy = sum(right) / len(label)
    return RMSE, Accuracy

def test_rbf():
    train, label, test = load_data()
    rbf = RBFN(50, 1.0)
    rbf.fit(train, label)
    predict = rbf.predict(train)

    # Eval
    RMSE, Accuracy = eval(predict, label)
    print(f"RBF train RMSE: {RMSE}, Accuracy: {Accuracy}")
    return rbf.predict(test)


def test_rbf_with_kmeans():
    train, label, test = load_data()
    rbf = RBFN(50, 1.0, True)

    kms = KMeans(n_clusters=50)
    kms.fit(train)

    rbf.set_centers(kms.cluster_centers_)
    rbf.fit(train, label)
    predict = rbf.predict(train)

    # Eval
    RMSE, Accuracy = eval(predict, label)
    print(f"RBF with Kmeans train RMSE: {RMSE}, Accuracy: {Accuracy}")
    return rbf.predict(test)


def test_rbf_with_som():
    train, label, test = load_data()
    rbf = RBFN(49, 1.0, True)

    som = MiniSom(7, 7, len(train[0]), sigma=0.3, learning_rate=0.5)
    som.train(train, 1000)
    centers = som.get_weights().reshape(-1, len(train[0]))

    rbf.set_centers(centers)
    rbf.fit(train, label)
    predict = rbf.predict(train)

    # Eval
    RMSE, Accuracy = eval(predict, label)
    print(f"RBF with SOM train RMSE: {RMSE}, Accuracy: {Accuracy}")
    return rbf.predict(test)


def test_svm():
    train, label, test = load_data()
    svm = SVM(kernel='rbf')

    svm.fit(train, label)
    predict = svm.predict(train)

    # Eval
    RMSE, Accuracy = eval(predict, label)
    print(f"SVM train RMSE: {RMSE}, Accuracy: {Accuracy}")
    return svm.predict(test)


if __name__ == "__main__":
    rbf_test_label = test_rbf()
    rbf_kmeans_test_label = test_rbf_with_kmeans()
    test_rbf_with_som()
    svm_test_label = test_svm()
