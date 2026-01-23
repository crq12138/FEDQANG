from mnist import MNIST
from sklearn import svm, linear_model, neural_network
import pdb
import numpy as np
import matplotlib.pyplot as plt


def main():

    mndata = MNIST('./MNIST/raw/')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    n = len(images)
    d = len(images[0])
    t = len(images_test)

    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))

    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])

    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q])

    print("Training classifier.")

    clf = linear_model.SGDClassifier(loss='log', max_iter=1000, tol=0.01)
    clf.fit(Xtrain, ytrain)

    # Training error
    y_hat = clf.predict(Xtrain)
    train_error = np.mean(y_hat != ytrain)
    print("Training Err: " + str(train_error))

    y_hat_test = clf.predict(Xtest)
    test_error = np.mean(y_hat_test != ytest)
    print("Test Err: " + str(test_error))

    nn = neural_network.MLPClassifier()
    nn.fit(Xtrain, ytrain)

    # Training error
    y_hat = nn.predict(Xtrain)
    train_error = np.mean(y_hat != ytrain)
    print("Training Err: " + str(train_error))

    y_hat_test = nn.predict(Xtest)
    test_error = np.mean(y_hat_test != ytest)
    print("Test Err: " + str(test_error))

    pdb.set_trace()

def dirichlet_split_noniid(alpha=0.5, client_number=5):
    '''
    使用参数 alpha 的 Dirichlet 分布将训练数据划分为 client_number 个非 IID 子集，并保存每个子集的数据。
    
    参数:
        alpha (float): Dirichlet 分布的参数，控制分布的均匀性。较小的 alpha 值会导致更不均匀的分布。
        client_number (int): 客户端的数量。
    '''
    mndata = MNIST('./MNIST/raw/')
    images, labels = mndata.load_training()
    n = len(images)
    d = len(images[0])

    Xtrain = np.zeros((n, d))
    ytrain = np.asarray(labels)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])

    # 标准化每一列
    # print("Standardize columns")
    # Xtrain = Xtrain / 100.0

    # 使用 Dirichlet 分布进行数据划分
    client_idcs = dirichlet_split_noniid_split(ytrain, alpha=alpha, client_number=client_number)

    # 保存每个客户端的数据
    for client_idx, idcs in enumerate(client_idcs):
        client_X = Xtrain[idcs]
        client_y = ytrain[idcs]
        client_data = np.hstack((client_X, client_y[:, None]))
        np.save(f"mnist_noniid_{alpha}_client_{client_idx}", client_data)
        print(f"Client {client_idx} data saved with shape {client_data.shape}")

def dirichlet_split_noniid_split(train_labels, alpha=0.5, client_number=5):
    '''
    使用参数 alpha 的 Dirichlet 分布将数据索引划分为 client_number 个子集，实现非 IID 分布。
    
    参数:
        train_labels (np.ndarray): 训练数据的标签数组。
        alpha (float): Dirichlet 分布的参数，控制分布的均匀性。
        client_number (int): 客户端的数量。
    
    返回:
        client_idcs (List[np.ndarray]): 每个客户端对应的样本索引列表。
    '''
    n_clients = client_number
    n_classes = train_labels.max() + 1  # 总类别数
    
    # 为每个类别生成 Dirichlet 分布的比例
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    
    # 记录每个类别对应的样本下标
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    
    # 初始化每个客户端的样本索引列表
    client_idcs = [[] for _ in range(n_clients)]
    
    # 对每个类别进行分配
    for c, fracs in zip(class_idcs, label_distribution):
        np.random.shuffle(c)  # 打乱类别内的样本顺序
        proportions = (np.cumsum(fracs)[:-1] * len(c)).astype(int)
        split_idcs = np.split(c, proportions)
        for i, idcs in enumerate(split_idcs):
            client_idcs[i].extend(idcs)
    
    # 转换为 NumPy 数组
    client_idcs = [np.array(idcs) for idcs in client_idcs]
    
    return client_idcs



def slice_uniform():

    mndata = MNIST('./MNIST/raw/')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    n = len(images)
    d = len(images[0])
    t = len(images_test)

    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))

    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])

    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q])

    # standardize each column
    # print("Standardize columns")
    # Xtrain = Xtrain / 100.0
    # Xtrain, _, _ = standardize_cols(Xtrain)
    # Xtest, _, _ = standardize_cols(Xtest)

    for k in range(10):

        randIdx = np.random.permutation(n)[0:5000]

        class_slice = Xtrain[randIdx]
        data_slice = np.hstack((class_slice, ytrain[randIdx][:, None]))

        print("slice " + str(k) + " is shape " + str(data_slice.shape))
        np.save("mnist_unif" + str(k), data_slice)

    # pdb.set_trace()


def slice_for_tm():

    mndata = MNIST('./MNIST/raw/')

    images, labels = mndata.load_training()
    images_test, labels_test = mndata.load_testing()

    # data pre-proccessing
    n = len(images)
    t = len(images_test)
    d = len(images[0])
    print("training data size: ", n)
    print("test data size: ", t)
    print("images size: ", d)
    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))
    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    # training data loading
    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i])
    # test data loading
    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q])

    # standardize each column
    # print("Standardize columns")
    # Xtrain = Xtrain / 100.0

    for k in range(10):
        idx = np.where(ytrain == k)[0]
        class_slice = Xtrain[idx][:5000]  # ✅ 仅保留前5000个样本
        label_slice = ytrain[idx][:5000]  # ✅ 对应标签
        data_slice = np.hstack((class_slice, label_slice[:, None]))
        print("slice " + str(k) + " is shape " + str(data_slice.shape))
        np.save("mnist" + str(k), data_slice)

    # train_slice = np.hstack((Xtrain, np.reshape(ytrain, (len(ytrain), 1))))
    # np.save("mnist_train", train_slice)

    # test_slice = np.hstack((Xtest, np.reshape(ytest, (len(ytest), 1))))
    # np.save("mnist_test", test_slice)

    # pdb.set_trace()


def show_digit(image):

    plt.imshow(image, cmap='gray')
    plt.show()


def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma, mu, sigma


if __name__ == "__main__":

    slice_uniform()
    # slice_for_tm()
    # dirichlet_split_noniid(alpha=0.1, client_number=10)
