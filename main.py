import numpy as np
from dataset import Dataset
from model.RBFN import RBFN
from model.SVM import SVM
from model.SOM import MiniSom
from sklearn.cluster import KMeans

data_loader = Dataset('data')


def eval_model(predict, label):
    RMSE = np.linalg.norm(label - predict) ** 2 / len(label)
    predict_label = np.sign(predict)
    right = predict_label == label
    Accuracy = sum(right) / len(label)
    return RMSE, Accuracy


def test_rbf():
    train, train_label = data_loader.get_train_data()
    val, val_label = data_loader.get_validation_data()
    test = data_loader.get_test_data()
    rbf = RBFN(90, 1.0)
    rbf.fit(train, train_label)
    train_predict = rbf.predict(train)
    val_predict = rbf.predict(val)

    # Eval
    print("\n--------- RBF ---------")
    RMSE, Accuracy = eval_model(train_predict, train_label)
    print(f"RBF train RMSE: {RMSE}, Accuracy: {Accuracy}")

    RMSE, Accuracy = eval_model(val_predict, val_label)
    print(f"RBF val RMSE: {RMSE}, Accuracy: {Accuracy}")
    return np.sign(rbf.predict(test))


def test_rbf_with_kmeans(cnt: int):
    train, train_label = data_loader.get_train_data()
    val, val_label = data_loader.get_validation_data()
    test = data_loader.get_test_data()
    rbf = RBFN(cnt, 1.0, True)

    kms = KMeans(n_clusters=cnt)
    kms.fit(train)

    rbf.set_centers(kms.cluster_centers_)
    rbf.fit(train, train_label)
    train_predict = rbf.predict(train)
    val_predict = rbf.predict(val)

    # Eval
    print("\n--------- RBF with Kmeans ---------")
    RMSE, Accuracy_train = eval_model(train_predict, train_label)
    print(f"RBF with Kmeans train RMSE: {RMSE}, Accuracy: {Accuracy_train}")

    RMSE, Accuracy_val = eval_model(val_predict, val_label)
    print(f"RBF with Kmeans val RMSE: {RMSE}, Accuracy: {Accuracy_val}")

    return Accuracy_train, Accuracy_val, np.sign(rbf.predict(test))


def test_rbf_with_som():
    train, train_label = data_loader.get_train_data()
    val, val_label = data_loader.get_validation_data()
    test = data_loader.get_test_data()
    rbf = RBFN(50, 1.0, True)

    som = MiniSom(5, 10, len(train[0]), sigma=0.3, learning_rate=0.1)
    som.train(train, 10000)
    centers = som.get_weights().reshape(-1, len(train[0]))

    rbf.set_centers(centers)
    rbf.fit(train, train_label)
    train_predict = rbf.predict(train)
    val_predict = rbf.predict(val)

    # Eval
    print("\n--------- RBF with SOM ---------")
    RMSE, Accuracy = eval_model(train_predict, train_label)
    print(f"RBF with SOM train RMSE: {RMSE}, Accuracy: {Accuracy}")

    RMSE, Accuracy = eval_model(val_predict, val_label)
    print(f"RBF with SOM val RMSE: {RMSE}, Accuracy: {Accuracy}")

    return np.sign(rbf.predict(test))


def test_svm():
    train, train_label = data_loader.get_train_data()
    val, val_label = data_loader.get_validation_data()
    test = data_loader.get_test_data()
    svm = SVM(kernel='rbf')

    svm.fit(train, train_label)
    train_predict = svm.predict(train)
    val_predict = svm.predict(val)

    # Eval
    print("\n--------- SVM ---------")
    RMSE, Accuracy = eval_model(train_predict, train_label)
    print(f"SVM train RMSE: {RMSE}, Accuracy: {Accuracy}")

    RMSE, Accuracy = eval_model(val_predict, val_label)
    print(f"SVM val RMSE: {RMSE}, Accuracy: {Accuracy}")

    return np.sign(svm.predict(test))

def svm_grid_search():
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split

    X_train, y_train = data_loader.get_train_data()
    X_val, y_val = data_loader.get_validation_data()
    # 把要调整的参数以及其候选值 列出来；
    param_grid = {"gamma": [0.001, 0.01, 0.1, 1, 10, 100],
                  "C": [0.001, 0.01, 0.1, 1, 10, 100]}
    print("Parameters:{}".format(param_grid))

    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)  # 实例化一个GridSearchCV类
    grid_search.fit(X_train, y_train)  # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
    print("Test set score:{:.2f}".format(grid_search.score(X_val, y_val)))
    print("Train set score:{:.2f}".format(grid_search.score(X_train, y_train)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))


if __name__ == "__main__":
    # rbf_test_label = test_rbf()
    # train_accs = []
    # val_accs = []
    # for i in range(1, 200):
    #     train_acc, val_acc = test_rbf_with_kmeans(i)
    #     train_accs.append(train_acc)
    #     val_accs.append(val_acc)
    # print(train_accs)
    # print(val_accs)
    # rbf_som_test_label = test_rbf_with_som()
    svm_test_label = test_svm()
    train_acc, val_acc, test = test_rbf_with_kmeans(81)
    print(svm_test_label)
    print(test)
    print(svm_test_label - test)
