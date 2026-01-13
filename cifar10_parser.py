import numpy as np
import pickle

# 加载CIFAR-10数据批次
def load_cifar10_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    return batch

def slice_uniform():
    # 加载CIFAR-10训练数据批次
    data_batches = []
    labels_batches = []

    for i in range(1, 6):
        batch = load_cifar10_batch(f'cifar-10-batches-py/data_batch_{i}')
        data_batches.append(batch[b'data'])
        labels_batches.append(batch[b'labels'])

    # 合并数据和标签
    Xtrain = np.concatenate(data_batches, axis=0)
    ytrain = np.concatenate(labels_batches, axis=0)

    # # 加载CIFAR-10测试数据批次
    # test_batch = load_cifar10_batch('cifar-10-batches-py/test_batch')
    # Xtest = test_batch[b'data']
    # ytest = test_batch[b'labels']

    # 标准化训练数据
    # print("Standardizing columns")
    # Xtrain = Xtrain / 255.0
    # Xtest = Xtest / 255.0

    # 每个客户端的数据大小
    client_data_size = 10000
    total_train_size = Xtrain.shape[0]

    for k in range(10):
        randIdx = np.random.permutation(total_train_size)[:client_data_size]  # 随机选择5000个样本

        class_slice = Xtrain[randIdx]
        data_slice = np.hstack((class_slice, ytrain[randIdx][:, None]))

        print(f"Slice {k} is shape {data_slice.shape}")
        np.save(f"cifar-10-batches-py/cifar10/cifar10_unif_10000_{k}", data_slice)

    # 其他处理逻辑和保存测试集数据
    print("Data slicing complete.")

def slice_for_cifar10():
    # 加载CIFAR-10训练数据批次
    data_batches = []
    labels_batches = []

    for i in range(1, 6):
        batch = load_cifar10_batch(f'cifar-10-batches-py/data_batch_{i}')
        data_batches.append(batch[b'data'])
        labels_batches.append(batch[b'labels'])
    
    # 合并数据和标签
    Xtrain = np.concatenate(data_batches, axis=0)
    ytrain = np.concatenate(labels_batches, axis=0)

    # 加载CIFAR-10测试数据批次
    test_batch = load_cifar10_batch('cifar-10-batches-py/test_batch')
    Xtest = test_batch[b'data']
    ytest = test_batch[b'labels']
    # print(Xtest)

    # 标准化训练数据
    print("Standardizing columns")
    # Xtrain = Xtrain / 255.0
    # Xtest = Xtest / 255.0
    print(Xtest)
    # 划分训练数据并保存
    for k in range(10):
        idx = np.where(ytrain == k)[0]
        class_slice = Xtrain[idx]
        data_slice = np.hstack((class_slice, ytrain[idx][:, None]))
        print(f"Slice {k} is shape {data_slice.shape}")
        np.save(f"cifar-10-batches-py/cifar10/cifar10_{k}", data_slice)

    # 保存训练数据和测试数据
    train_slice = np.hstack((Xtrain, np.reshape(ytrain, (len(ytrain), 1))))
    np.save("cifar-10-batches-py/cifar10/cifar10_train", train_slice)
    test_slice = np.hstack((Xtest, np.reshape(ytest, (len(ytest), 1))))
    np.save("cifar-10-batches-py/cifar10/cifar10_test", test_slice)

    print("Data slicing complete.")

# 调用函数
# slice_for_cifar10()
slice_uniform()