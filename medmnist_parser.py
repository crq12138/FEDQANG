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

def dirichlet_split_noniid_split(train_labels, alpha=0.5, client_number=5):
    '''
    Dirichlet 分布划分辅助函数
    '''
    n_clients = client_number
    # PathMNIST 标签可能是 (N, 1)，需要 flatten
    train_labels = train_labels.flatten()
    n_classes = train_labels.max() + 1
    
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    
    client_idcs = [[] for _ in range(n_clients)]
    
    for c, fracs in zip(class_idcs, label_distribution):
        np.random.shuffle(c)
        proportions = (np.cumsum(fracs)[:-1] * len(c)).astype(int)
        split_idcs = np.split(c, proportions)
        for i, idcs in enumerate(split_idcs):
            client_idcs[i].extend(idcs)
    
    client_idcs = [np.array(idcs) for idcs in client_idcs]
    return client_idcs

def load_flat_medmnist(root_dir, split='train'):
    """
    辅助函数：加载并展平 MedMNIST 数据
    返回: X (N, 2352), y (N, 1)
    """
    data_flag = 'pathmnist'
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])
    
    dataset = DataClass(split=split, transform=None, download=True, root=root_dir)
    
    images = dataset.imgs
    labels = dataset.labels
    
    n, h, w, c = images.shape
    flat_dim = h * w * c
    X = images.reshape(n, flat_dim)
    
    return X, labels

def generate_medmnist_exp1_data(root_dir='./medmnist'):
    """
    【实验一：有效性验证】
    配置：
    - Client 0-8 (9个): 极度 Non-IID (alpha=0.1)
    - Client 9   (1个): IID (独立同分布)
    """
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    print("Generating Exp1 Data (9 Non-IID + 1 IID)...")
    
    # 加载数据
    X, y = load_flat_medmnist(root_dir, split='train')
    y = y.flatten() # 转换为 (N,)
    
    # 1. 预留 10% 数据给 IID Client (Client 9)
    perm = np.random.permutation(len(X))
    X = X[perm]
    y = y[perm]
    
    split_point = int(len(X) * 0.9)
    
    # Non-IID 池 (90%)
    X_noniid = X[:split_point]
    y_noniid = y[:split_point]
    
    # IID 池 (10%)
    X_iid = X[split_point:]
    y_iid = y[split_point:]
    
    # 2. 保存 Client 9 (IID)
    data_iid = np.hstack((X_iid, y_iid[:, None]))
    np.save(os.path.join(root_dir, "pathmnist_exp1_client_9.npy"), data_iid)
    print(f"Client 9 (IID) saved, shape {data_iid.shape}")
    
    # 3. 对 Client 0-8 进行 Dirichlet 划分
    client_idcs = dirichlet_split_noniid_split(y_noniid, alpha=0.1, client_number=9)
    
    for i, idcs in enumerate(client_idcs):
        client_X = X_noniid[idcs]
        client_y = y_noniid[idcs]
        data = np.hstack((client_X, client_y[:, None]))
        np.save(os.path.join(root_dir, f"pathmnist_exp1_client_{i}.npy"), data)
        print(f"Client {i} (Non-IID alpha=0.1) saved, shape {data.shape}")
        
    # 生成测试集 (如果还没生成过)
    X_test, y_test = load_flat_medmnist(root_dir, split='test')
    test_data = np.hstack((X_test, y_test))
    np.save(os.path.join(root_dir, "pathmnist_test.npy"), test_data)

def generate_medmnist_exp2_data(root_dir='./medmnist'):
    """
    【实验二：寻找天才】
    PathMNIST 共 9 类 (0-8)。
    配置：
    - Client 0-7 (8个普通): 拥有类别 0-6 (7类) -> 缺失 7, 8
    - Client 8   (1个天才): 拥有类别 7-8 (稀缺类)
    - Client 9   (1个混子): 拥有类别 0-1 (冗余类)
    """
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
        
    print("Generating Exp2 Data (Genius Hunt)...")
    
    X, y = load_flat_medmnist(root_dir, split='train')
    y = y.flatten()
    
    # 按类别整理数据
    class_data = {i: [] for i in range(9)}
    for img, lbl in zip(X, y):
        class_data[lbl].append(np.hstack((img, [lbl])))
        
    for i in range(9):
        class_data[i] = np.array(class_data[i])

    # 1. 生成 8 个普通节点 (Client 0-7): 瓜分 0-6 类
    # 将类别 0-6 的数据混合
    common_classes = [0, 1, 2, 3, 4, 5, 6]
    common_pool = np.vstack([class_data[i] for i in common_classes])
    np.random.shuffle(common_pool)
    
    # 均分给 8 个客户端
    chunks = np.array_split(common_pool, 8)
    for i in range(8):
        np.save(os.path.join(root_dir, f"pathmnist_exp2_client_{i}.npy"), chunks[i])
        print(f"Client {i} (Common 0-6) saved, shape {chunks[i].shape}")

    # 2. 生成 1 个天才节点 (Client 8): 独占 7-8 类
    genius_classes = [7, 8]
    genius_pool = np.vstack([class_data[i] for i in genius_classes])
    np.random.shuffle(genius_pool)
    # 为了实验公平，控制数据量大约为 3000-5000 (PathMNIST 训练集约 90k，平均每类 10k)
    # 两个类全拿大约 20k，我们取一部分，模拟它数据量不大但是很关键
    genius_data = genius_pool[:5000] 
    np.save(os.path.join(root_dir, "pathmnist_exp2_client_8.npy"), genius_data)
    print(f"Client 8 (Genius 7-8) saved, shape {genius_data.shape}")

    # 3. 生成 1 个冗余混子 (Client 9): 只懂 0-1 类
    redundant_classes = [0, 1]
    # 注意：这里我们重新采样 0-1 类的数据，或者复用。为了简单，直接取原始数据的前部分。
    # 现实中混子的数据往往是重复的或者大路货。
    redundant_pool = np.vstack([class_data[i] for i in redundant_classes])
    np.random.shuffle(redundant_pool)
    redundant_data = redundant_pool[:5000] # 数据量和天才差不多，控制变量
    np.save(os.path.join(root_dir, "pathmnist_exp2_client_9.npy"), redundant_data)
    print(f"Client 9 (Redundant 0-1) saved, shape {redundant_data.shape}")
    
    # 生成测试集
    X_test, y_test = load_flat_medmnist(root_dir, split='test')
    test_data = np.hstack((X_test, y_test))
    np.save(os.path.join(root_dir, "pathmnist_test.npy"), test_data)

if __name__ == "__main__":
    # generate_medmnist_data()
    # --- 1. 运行实验一数据生成 ---
    generate_medmnist_exp1_data()
    
    # --- 2. 运行实验二数据生成 ---
    # generate_medmnist_exp2_data()