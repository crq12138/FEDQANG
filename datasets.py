from mnist_dataset import MNISTDataset
# from lfw_dataset import LFWDataset
from cifar_dataset import CIFARDataset
# from credit_dataset import CreditDataset
import torch
import numpy as np

def get_dataset(dataset):
    if dataset == "mnist":
        return MNISTDataset
    elif dataset == "lfw":
        return LFWDataset
    elif dataset == "cifar":
        return CIFARDataset
    elif dataset == "creditcard":
        return CreditDataset
    else: 
        print("Error: dataset " + dataset + "not defined")
    
def get_num_params(dataset):
    if dataset == "mnist":
        return 7850
    elif dataset == "lfw":
        return 18254
    elif dataset == "cifar":
        return -1 # Find out how many.
    elif dataset == "creditcard":
        return 50
    else:
        print("Error: dataset " + dataset + "not defined")

def get_num_features(dataset):
    if dataset == "mnist":
        return 784
    elif dataset == "lfw":
        return 8742 #62 47 3
    elif dataset == "cifar":
        return 32*32*3
    elif dataset == "creditcard":
        return 24
    else:
        print("Error: dataset " + dataset + "not defined")
    
def get_num_classes(dataset):
    if dataset == "mnist":
        return 10
    elif dataset == "lfw":
        return 12
    elif dataset == "cifar":
        return 10
    elif dataset == "creditcard":
        return 2
    else: 
        print("Error: dataset " + dataset + "not defined")

def get_proxy_dataloader(dataset_name, root_dir, batch_size=32, sample_size=200):
    """
    获取代理共识数据集（Root Dataset）。
    从测试集中随机采样一个小的平衡子集。
    """
    Dataset = get_dataset(dataset_name)
    
    # 假设测试集文件名为 {dataset}_test，例如 mnist_test
    # 这里的 root_dir 需要根据实际情况传入，例如 "./mnist"
    test_filename = f"{dataset_name}_test" 
    if dataset_name == "cifar":
         # CIFAR 数据集通常不需要 .npy 后缀来加载整个测试集，这里假设 CIFAR10 的处理逻辑
         # 如果你的 CIFAR 数据集也是切分好的 npy，需要指定一个公共的测试集文件
         test_filename = "cifar10_test" 

    try:
        # 加载完整测试集 (is_train=False)
        full_test_set = Dataset(test_filename, root_dir, is_train=False, transform=None) # transform 在 Dataset 内部处理或者外部传入
        
        # 这里的 transform 需要和 Client 中一致，为了简单，我们在 Client 调用时传入 transform
        # 但 Dataset 类初始化时如果需要 transform，这里先传 None，由 Dataset 内部逻辑决定
        # 查看 Dataset 源码，transform 是在 __getitem__ 用的。
        # 我们需要在 Loader 层面或者 Dataset 初始化时处理。
        # 重新实例化带 transform 的 Dataset (需要引入 transforms)
        from torchvision import transforms
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        full_test_set = Dataset(test_filename, root_dir, is_train=False, transform=transform)

        # 随机采样 sample_size 个样本作为 Root Dataset
        indices = np.random.choice(len(full_test_set), min(len(full_test_set), sample_size), replace=False)
        subset = torch.utils.data.Subset(full_test_set, indices)
        
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=False)
        print(f"Proxy Consensus Dataset (Root Dataset) created with {len(subset)} samples.")
        return loader
    except Exception as e:
        print(f"Error creating proxy dataset: {e}")
        return None