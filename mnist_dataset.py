from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class MNISTDataset(Dataset):
    def __init__(self, filename, root_dir, transform=None, is_train=True, train_cut=0.8):
        self.filename = filename
        self.root_dir = root_dir
        self.transform = transform
        data = np.load(os.path.join(root_dir, filename + ".npy"))
        n, d = data.shape
        self.n = n

        cut = int(n*train_cut)
        self.cut = cut
        # self.X = data[0:n-1, 0:d - 1]
        # self.y = data[0:n-1, -1]
        if is_train == True:
            self.X = data[0:cut-1, 0:d - 1]
            self.y = data[0:cut-1, -1]
        else:
            self.X = data[cut:n-1, 0:d - 1]
            self.y = data[cut:n-1, -1]


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # ==== 修改开始 ====
        # 1. 取出数据并 reshape
        img_data = np.reshape(self.X[idx], (28, 28))
        
        # 2. 关键修复：强制转换为 uint8 类型 (PIL 需要 0-255 的 uint8)
        img_data = img_data.astype(np.uint8)
        
        # 3. 再转为 Image 对象
        sample = Image.fromarray(img_data)
        # ==== 修改结束 ====

        if self.transform:
            sample = self.transform(sample)
        
        return {'image': sample, 'label': self.y[idx]}
    
    def getData(self):
        return self.X, self.y
    # 

# data = np.load("./mnist/mnist_noniid_0.1_client_1.npy")
# print(data.shape)
# print(data.__len__)
# dataset = MNISTDataset("mnist_noniid_0.1_client_5", "./mnist")