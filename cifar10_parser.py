import numpy as np
import pickle
import os
import sys

def load_cifar10_batch(file_path):
    """加载单个 CIFAR-10 batch 文件"""
    with open(file_path, 'rb') as f:
        # Python 3 需要 encoding='bytes'
        batch = pickle.load(f, encoding='bytes')
    return batch

def load_raw_cifar10(root_dir):
    """
    加载所有原始 CIFAR-10 数据 (data_batch_1 ~ 5)
    返回: 
        X_train (50000, 3072)
        y_train (50000,)
        X_test  (10000, 3072)
        y_test  (10000,)
    """
    data_batches = []
    labels_batches = []

    # 加载训练集 batch 1-5
    for i in range(1, 6):
        file_path = os.path.join(root_dir, f'data_batch_{i}')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot find {file_path}. Please check directory.")
            
        batch = load_cifar10_batch(file_path)
        data_batches.append(batch[b'data'])
        labels_batches.append(batch[b'labels'])

    X_train = np.concatenate(data_batches, axis=0)
    y_train = np.concatenate(labels_batches, axis=0)

    # 加载测试集
    test_path = os.path.join(root_dir, 'test_batch')
    test_batch = load_cifar10_batch(test_path)
    X_test = test_batch[b'data']
    y_test = np.array(test_batch[b'labels'])

    print(f"Loaded CIFAR-10: Train {X_train.shape}, Test {X_test.shape}")
    print("Note: Pixel values kept in 0-255 range (no normalization applied here).")
    
    return X_train, y_train, X_test, y_test

def save_data(X, y, filepath):
    """辅助函数：拼接数据和标签并保存为 .npy"""
    # X: (N, 3072), y: (N,) -> (N, 3073)
    data = np.hstack((X, y[:, None]))
    np.save(filepath, data)
    print(f"Saved {filepath}, shape: {data.shape}")

def dirichlet_split_noniid_split(train_labels, alpha=0.1, client_number=10):
    '''Dirichlet 分布划分辅助函数'''
    train_labels = np.array(train_labels)
    n_classes = train_labels.max() + 1
    
    # 采样分布比例
    label_distribution = np.random.dirichlet([alpha] * client_number, n_classes)
    
    # 获取每个类别的索引
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    
    client_idcs = [[] for _ in range(client_number)]
    
    for c, fracs in zip(class_idcs, label_distribution):
        np.random.shuffle(c)
        # 计算切分点
        proportions = (np.cumsum(fracs)[:-1] * len(c)).astype(int)
        split_idcs = np.split(c, proportions)
        for i, idcs in enumerate(split_idcs):
            client_idcs[i].extend(idcs)
    
    client_idcs = [np.array(idcs) for idcs in client_idcs]
    return client_idcs

# ==========================================
# 实验一：有效性验证 (9 Non-IID + 1 IID)
# ==========================================
def generate_cifar10_exp1_data(input_dir, output_dir):
    print("\n>>> Generating Exp1 Data (9 Non-IID + 1 IID)...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X, y, X_test, y_test = load_raw_cifar10(input_dir)
    
    # 保存测试集
    save_data(X_test, y_test, os.path.join(output_dir, "cifar10_test.npy"))

    # 1. 预留 10% 数据给 IID Client (Client 9)
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    split_point = int(len(X) * 0.9)
    
    X_noniid = X[:split_point]
    y_noniid = y[:split_point]
    
    X_iid = X[split_point:]
    y_iid = y[split_point:]
    
    # 保存 Client 9 (IID)
    save_data(X_iid, y_iid, os.path.join(output_dir, "cifar10_exp1_client_9.npy"))
    
    # 2. 对 Client 0-8 进行 Dirichlet 划分 (alpha=0.1)
    client_idcs = dirichlet_split_noniid_split(y_noniid, alpha=0.1, client_number=9)
    
    for i, idcs in enumerate(client_idcs):
        client_X = X_noniid[idcs]
        client_y = y_noniid[idcs]
        save_data(client_X, client_y, os.path.join(output_dir, f"cifar10_exp1_client_{i}.npy"))

# ==========================================
# 实验二：寻找天才 (Genius Hunt)
# ==========================================
def generate_cifar10_exp2_data(input_dir, output_dir):
    print("\n>>> Generating Exp2 Data (Genius Hunt)...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X, y, X_test, y_test = load_raw_cifar10(input_dir)
    
    # 保存测试集
    save_data(X_test, y_test, os.path.join(output_dir, "cifar10_test.npy"))

    # 按类别整理数据
    class_data = {i: [] for i in range(10)}
    for img, lbl in zip(X, y):
        class_data[lbl].append(img)
    
    for i in range(10):
        class_data[i] = np.array(class_data[i])

    # --- 配置 ---
    # Client 0-7 (8个普通): 拥有类别 0-7 (缺失 8,9)
    # Client 8   (1个天才): 拥有类别 8,9 (Ship, Truck)
    # Client 9   (1个混子): 拥有类别 0,1 (Plane, Car)
    
    # 1. 生成 8 个普通节点
    common_classes = [0, 1, 2, 3, 4, 5, 6, 7]
    common_X = np.vstack([class_data[c] for c in common_classes])
    common_y = np.concatenate([[c] * len(class_data[c]) for c in common_classes])
    
    # 打乱
    perm = np.random.permutation(len(common_X))
    common_X = common_X[perm]
    common_y = common_y[perm]
    
    # 均分
    chunks_X = np.array_split(common_X, 8)
    chunks_y = np.array_split(common_y, 8)
    
    for i in range(8):
        save_data(chunks_X[i], chunks_y[i], os.path.join(output_dir, f"cifar10_exp2_client_{i}.npy"))

    # 2. 生成 1 个天才节点 (Client 8) - 独占 8, 9 类
    genius_classes = [8, 9]
    genius_X = np.vstack([class_data[c] for c in genius_classes])
    genius_y = np.concatenate([[c] * len(class_data[c]) for c in genius_classes])
    
    # 随机取 5000 个样本
    perm = np.random.permutation(len(genius_X))
    genius_X = genius_X[perm][:5000]
    genius_y = genius_y[perm][:5000]
    
    save_data(genius_X, genius_y, os.path.join(output_dir, "cifar10_exp2_client_8.npy"))

    # 3. 生成 1 个冗余混子 (Client 9) - 只懂 0, 1 类
    redundant_classes = [0, 1]
    redundant_X = np.vstack([class_data[c] for c in redundant_classes])
    redundant_y = np.concatenate([[c] * len(class_data[c]) for c in redundant_classes])
    
    # 随机取 5000 个样本
    perm = np.random.permutation(len(redundant_X))
    redundant_X = redundant_X[perm][:5000]
    redundant_y = redundant_y[perm][:5000]
    
    save_data(redundant_X, redundant_y, os.path.join(output_dir, "cifar10_exp2_client_9.npy"))


if __name__ == "__main__":
    # 路径配置
    # 输入目录：存放 data_batch_1 等文件的目录
    INPUT_DIR = 'cifar-10-batches-py' 
    # 输出目录：存放生成的 .npy 文件的目录
    OUTPUT_DIR = 'cifar-10-batches-py/cifar10'

    # --- 选择要运行的实验 ---
    
    # 运行实验一 (Non-IID + IID 对比)
    generate_cifar10_exp1_data(INPUT_DIR, OUTPUT_DIR)
    
    # 运行实验二 (寻找天才节点)
    # generate_cifar10_exp2_data(INPUT_DIR, OUTPUT_DIR)