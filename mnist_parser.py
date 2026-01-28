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


def generate_exp2_data():
    """
    实验二数据生成：偏科天才 vs 冗余混子
    """
    import numpy as np
    from mnist import MNIST

    mndata = MNIST('./MNIST/raw/')
    images, labels = mndata.load_training()
    X = np.array(images)
    y = np.array(labels)

    # 分类数据容器
    class_data = {i: [] for i in range(10)}
    for img, lbl in zip(X, y):
        class_data[lbl].append(np.hstack((img, [lbl]))) # 图片+标签

    # 转为 array
    for i in range(10):
        class_data[i] = np.array(class_data[i])

    # 1. 生成 8 个普通节点 (Client 0-7): 拥有 0-7 类
    # 将 0-7 类的数据混合并均分给 8 个人
    common_data = np.vstack([class_data[i] for i in range(8)])
    np.random.shuffle(common_data)
    chunks = np.array_split(common_data, 8)
    
    for i in range(8):
        np.save(f"mnist_exp2_client_{i}.npy", chunks[i])
        print(f"Client {i} (Common 0-7) saved.")

    # 2. 生成 1 个偏科天才 (Client 8): 拥有 8-9 类
    genius_data = np.vstack([class_data[8], class_data[9]])
    np.random.shuffle(genius_data)
    # 为了公平，可以控制数据量和普通节点差不多，或者少一点也无所谓
    # 假设取 5000 个样本
    genius_data = genius_data[:5000]
    np.save(f"mnist_exp2_client_8.npy", genius_data)
    print(f"Client 8 (Genius 8-9) saved.")

    # 3. 生成 1 个冗余混子 (Client 9): 拥有 0-1 类 (重复知识)
    redundant_data = np.vstack([class_data[0], class_data[1]])
    np.random.shuffle(redundant_data)
    # 取新的数据，模拟它确实有数据，但是是重复的知识
    # 这里直接重用数据也没关系，因为我们看的是泛化贡献
    redundant_data = redundant_data[:5000]
    np.save(f"mnist_exp2_client_9.npy", redundant_data)
    print(f"Client 9 (Redundant 0-1) saved.")

if __name__ == "__main__":

    # slice_uniform()
    # slice_for_tm()
    # dirichlet_split_noniid(alpha=0.1, client_number=10)
    generate_exp2_data()
