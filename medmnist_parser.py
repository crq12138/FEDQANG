import numpy as np
import os
import medmnist
from medmnist import INFO

def generate_medmnist_data(root_dir='./medmnist_data'):
    """
    下载 PathMNIST 数据并转换为项目通用的 .npy 格式。
    生成：
    1. medmnist_test.npy (测试集)
    2. medmnist_{client_id}.npy (模拟不同医院的私有数据)
    """
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    data_flag = 'pathmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    print(f"Downloading {data_flag}...")
    # 下载数据
    train_dataset = DataClass(split='train', transform=None, download=True, root=root_dir)
    test_dataset = DataClass(split='test', transform=None, download=True, root=root_dir)

    # 提取数据 (N, 28, 28, 3) 和 标签 (N, 1)
    train_images = train_dataset.imgs
    train_labels = train_dataset.labels
    test_images = test_dataset.imgs
    test_labels = test_dataset.labels

    # 展平图像数据: (N, 28*28*3)
    n_train, h, w, c = train_images.shape
    n_test = test_images.shape[0]
    
    flat_dim = h * w * c
    X_train = train_images.reshape(n_train, flat_dim)
    X_test = test_images.reshape(n_test, flat_dim)

    # 拼接数据和标签 [Features, Label] -> 保存为 .npy
    # 训练集切分 (模拟10个客户端)
    num_clients = 10
    samples_per_client = n_train // num_clients
    
    print(f"Splitting training data into {num_clients} clients (IID)...")
    
    # 打乱数据
    perm = np.random.permutation(n_train)
    X_train = X_train[perm]
    train_labels = train_labels[perm]

    for i in range(num_clients):
        start = i * samples_per_client
        end = (i + 1) * samples_per_client
        
        client_X = X_train[start:end]
        client_y = train_labels[start:end]
        
        # 拼接: (N, 2352) + (N, 1) -> (N, 2353)
        client_data = np.hstack((client_X, client_y))
        
        save_path = os.path.join(root_dir, f"pathmnist_{i}.npy")
        np.save(save_path, client_data)
        print(f"Saved {save_path}, shape: {client_data.shape}")

    # 保存测试集
    test_data = np.hstack((X_test, test_labels))
    save_path_test = os.path.join(root_dir, "pathmnist_test.npy")
    np.save(save_path_test, test_data)
    print(f"Saved Test Set: {save_path_test}, shape: {test_data.shape}")

if __name__ == "__main__":
    generate_medmnist_data()