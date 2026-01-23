from mnist_dataset import MNISTDataset
# from lfw_dataset import LFWDataset
from cifar_dataset import CIFARDataset
# 引入新数据集
from medmnist_dataset import MedMNISTDataset 
import torch
import numpy as np
from torchvision import transforms # 移到顶层导入

def get_dataset(dataset):
    if dataset == "mnist":
        return MNISTDataset
    elif dataset == "cifar":
        return CIFARDataset
    elif dataset == "medmnist":  # 新增
        return MedMNISTDataset
    else: 
        print("Error: dataset " + dataset + " not defined")
    
def get_num_params(dataset):
    if dataset == "mnist":
        return 7850
    elif dataset == "cifar":
        return -1 
    elif dataset == "medmnist": # 新增
        return -1 # 取决于模型
    else:
        print("Error: dataset " + dataset + " not defined")

def get_num_features(dataset):
    if dataset == "mnist":
        return 784
    elif dataset == "cifar":
        return 32*32*3
    elif dataset == "medmnist": # 新增: PathMNIST 28x28x3
        return 28*28*3
    else:
        print("Error: dataset " + dataset + " not defined")
    
def get_num_classes(dataset):
    if dataset == "mnist":
        return 10
    elif dataset == "cifar":
        return 10
    elif dataset == "medmnist": # 新增: PathMNIST 是 9 类
        return 9
    else: 
        print("Error: dataset " + dataset + " not defined")

# 修改 get_proxy_dataloader 以支持 medmnist 并修复单通道/多通道 normalize 问题
def get_proxy_dataloader(dataset_name, root_dir, batch_size=32, sample_size=200):
    Dataset = get_dataset(dataset_name)
    
    test_filename = f"{dataset_name}_test" 
    if dataset_name == "cifar":
         test_filename = "cifar10_test" 
    elif dataset_name == "medmnist": # 新增
         test_filename = "pathmnist_test"

    try:
        # ==== 核心修复：根据数据集类型选择正确的 Normalize 参数 ====
        if dataset_name == "mnist":
            # MNIST 是单通道 (Grayscale)
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            # CIFAR / MedMNIST 是三通道 (RGB)
            transform = transforms.Compose([
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        # ========================================================
        
        full_test_set = Dataset(test_filename, root_dir, is_train=False, transform=transform)

        indices = np.random.choice(len(full_test_set), min(len(full_test_set), sample_size), replace=False)
        subset = torch.utils.data.Subset(full_test_set, indices)
        
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
        print(f"Proxy Consensus Dataset (Root Dataset) created with {len(subset)} samples.")
        return loader
    except Exception as e:
        print(f"Error creating proxy dataset: {e}")
        return None