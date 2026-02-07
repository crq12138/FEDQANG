import numpy as np
import os
import medmnist
from medmnist import INFO
from torchvision import datasets

# 配置
# 直接在这里切换数据集: 'pathmnist' / 'mnist' / 'cifar10'
DATASET_NAME = 'pathmnist'

DATASET_CONFIG = {
    'pathmnist': {
        'root_dir': './medmnist',
        'output_prefix': 'pathmnist_exp5_client',
        'test_file': 'pathmnist_test.npy',
    },
    'mnist': {
        'root_dir': './mnist',
        'output_prefix': 'mnist_exp5_client',
        'test_file': 'mnist_test.npy',
    },
    'cifar10': {
        'root_dir': './cifar-10-batches-py/cifar10',
        'output_prefix': 'cifar10_exp5_client',
        'test_file': 'cifar10_test.npy',
    }
}

if DATASET_NAME not in DATASET_CONFIG:
    supported = ', '.join(DATASET_CONFIG.keys())
    raise ValueError(f"Unsupported dataset '{DATASET_NAME}'. Supported: {supported}")

ROOT_DIR = DATASET_CONFIG[DATASET_NAME]['root_dir']
OUTPUT_PREFIX = DATASET_CONFIG[DATASET_NAME]['output_prefix']
TEST_FILE = DATASET_CONFIG[DATASET_NAME]['test_file']

def load_raw_data(root_dir, split='train'):
    if DATASET_NAME == 'pathmnist':
        info = INFO['pathmnist']
        DataClass = getattr(medmnist, info['python_class'])
        dataset = DataClass(split=split, transform=None, download=True, root=root_dir)
        images = dataset.imgs
        labels = dataset.labels
        n = images.shape[0]
        return images.reshape(n, -1), labels.flatten()

    if DATASET_NAME == 'mnist':
        dataset = datasets.MNIST(
            root=root_dir,
            train=(split == 'train'),
            download=True,
        )
        images = dataset.data.numpy()
        labels = np.array(dataset.targets)
        n = images.shape[0]
        return images.reshape(n, -1), labels.flatten()

    if DATASET_NAME == 'cifar10':
        dataset = datasets.CIFAR10(
            root=root_dir,
            train=(split == 'train'),
            download=True,
        )
        images = np.array(dataset.data)
        labels = np.array(dataset.targets)
        n = images.shape[0]
        return images.reshape(n, -1), labels.flatten()

    raise ValueError(f"Unsupported dataset '{DATASET_NAME}'")

def sample_data(X, y, classes, n_samples):
    indices = []
    for cls in classes:
        indices.extend(np.where(y == cls)[0])
    if not indices: return np.array([]), np.array([])
    # 允许重复采样以达到大数据量
    selected = np.random.choice(indices, n_samples, replace=True) 
    return X[selected], y[selected]

def sample_repetitive_data(X, y, classes, n_total_samples, n_unique_seeds=50):
    """
    【恶意采样】制造虚假的大数据量
    逻辑：只取极少量(n_unique_seeds)的图片，然后重复无数次填满 n_total_samples
    """
    # 1. 找到目标类的所有候选图片
    candidate_indices = []
    for cls in classes:
        candidate_indices.extend(np.where(y == cls)[0])
    
    if not candidate_indices: return np.array([]), np.array([])
    
    # 2. 只选极少量的"种子" (比如 50 张)
    # 如果候选不够50张，就全选
    n_seeds = min(len(candidate_indices), n_unique_seeds)
    seeds = np.random.choice(candidate_indices, n_seeds, replace=False)
    
    # 3. 疯狂重复这些种子来凑数 (比如重复 160 次)
    selected_indices = np.random.choice(seeds, n_total_samples, replace=True)
    
    return X[selected_indices], y[selected_indices]

def generate_exp5():
    if not os.path.exists(ROOT_DIR): os.makedirs(ROOT_DIR)
    X, y = load_raw_data(ROOT_DIR, 'train')
    all_cls = np.unique(y)

    print(f"Generating Exp5 split for {DATASET_NAME}...")

    # 1. 天才 (Client 0): IID, 数据少 (Trap: FedAvg会忽略它)
    print("Genius (Small Data)...")
    X0, y0 = sample_data(X, y, all_cls, 1000) 
    np.save(f"{ROOT_DIR}/{OUTPUT_PREFIX}_0.npy", np.hstack((X0, y0[:,None])))

    # --- 2. 混子 (Client 1-3): Class 0/1, 数据巨大, 但高度重复! ---
    print("Generating Client 1-3 (Free-riders: Huge Size, Biased, Highly Repetitive)...")
    for i in range(1, 4):
        # 使用新的恶意采样函数
        # 只有 50 张不一样的图，重复凑成 8000 张
        Xi, yi = sample_repetitive_data(X, y, [0, 1], n_total_samples=8000, n_unique_seeds=50)
        
        np.save(f"{ROOT_DIR}/{OUTPUT_PREFIX}_{i}.npy", np.hstack((Xi, yi[:,None])))
        print(f"  -> Client {i}: 8000 samples (generated from only 50 unique images)")

    # 3. 普通 (Client 4-9): 随机偏好, 数据中等
    print("Ordinary (Medium Data)...")
    for i in range(4, 10):
        # 随机3个类
        cls = np.random.choice(all_cls, 7, replace=False)
        Xi, yi = sample_data(X, y, cls, 3000)
        np.save(f"{ROOT_DIR}/{OUTPUT_PREFIX}_{i}.npy", np.hstack((Xi, yi[:,None])))

    # Test set
    Xt, yt = load_raw_data(ROOT_DIR, 'test')
    np.save(f"{ROOT_DIR}/{TEST_FILE}", np.hstack((Xt, yt[:,None])))
    print("Done.")

if __name__ == "__main__":
    generate_exp5()
