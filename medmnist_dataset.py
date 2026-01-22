from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class MedMNISTDataset(Dataset):
    def __init__(self, filename, root_dir, transform=None, is_train=True, train_cut=1.0):
        self.filename = filename
        self.root_dir = root_dir
        self.transform = transform
        
        # 加载 .npy 文件
        file_path = os.path.join(root_dir, filename + ".npy")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
            
        data = np.load(file_path)
        n, d = data.shape
        self.n = n

        # 数据切分 (如果需要仅使用部分数据)
        cut = int(n * train_cut)
        self.cut = cut

        # 最后一列是 Label
        if is_train:
            self.X = data[0:cut, 0:d - 1]
            self.y = data[0:cut, -1]
        else:
            self.X = data[0:n, 0:d - 1] # 测试集通常全用
            self.y = data[0:n, -1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # PathMNIST 是 28x28x3 (RGB)
        # 从扁平化数据恢复形状
        img_data = self.X[idx].reshape(28, 28, 3).astype(np.uint8)
        sample = Image.fromarray(img_data)

        if self.transform:
            sample = self.transform(sample)
        
        # Label 需要是 Long 类型
        label = int(self.y[idx])
        
        return {'image': sample, 'label': label}
    
    def getData(self):
        return self.X, self.y