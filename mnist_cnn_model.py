import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import numpy as np
import pdb
import datasets
import pickle
import bc_enum
import p2p
from blockchain import Blockchain
import torch.nn.functional as F

from client import Client

epsilon = 0.04
sigama=1e-5


class MNISTCNNModel(nn.Module):
    def __init__(self, D_in=None, D_out=None):
        super(MNISTCNNModel, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        # 定义全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        # 记录输入和输出维度（与原方法保持一致）
        self.D_in = D_in
        self.D_out = D_out

    def forward(self, x):
        # x 的形状应为 [batch_size, 1, 28, 28]
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))        # 输出形状: [batch_size, 32, 28, 28]
        x = F.max_pool2d(x, 2)           # 输出形状: [batch_size, 32, 14, 14]
        x = F.relu(self.conv2(x))        # 输出形状: [batch_size, 64, 14, 14]
        x = F.max_pool2d(x, 2)           # 输出形状: [batch_size, 64, 7, 7]
        x = x.view(-1, 64 * 7 * 7)       # 展平
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    # 展平梯度的逆操作，用于将扁平化的梯度还原为模型参数的形状
    def reshape(self, flat_gradient):
        layers = []
        idx = 0

        # Conv1 权重和偏置
        conv1_weight_size = self.conv1.weight.numel()
        conv1_bias_size = self.conv1.bias.numel()
        conv1_weight = torch.from_numpy(
            flat_gradient[idx:idx + conv1_weight_size].reshape(self.conv1.weight.shape)
        ).float().to(self.conv1.weight.device)
        idx += conv1_weight_size
        conv1_bias = torch.from_numpy(
            flat_gradient[idx:idx + conv1_bias_size]
        ).float().to(self.conv1.bias.device)
        idx += conv1_bias_size
        layers.append(conv1_weight)
        layers.append(conv1_bias)

        # Conv2 权重和偏置
        conv2_weight_size = self.conv2.weight.numel()
        conv2_bias_size = self.conv2.bias.numel()
        conv2_weight = torch.from_numpy(
            flat_gradient[idx:idx + conv2_weight_size].reshape(self.conv2.weight.shape)
        ).float().to(self.conv2.weight.device)
        idx += conv2_weight_size
        conv2_bias = torch.from_numpy(
            flat_gradient[idx:idx + conv2_bias_size]
        ).float().to(self.conv2.bias.device)
        idx += conv2_bias_size
        layers.append(conv2_weight)
        layers.append(conv2_bias)

        # FC1 权重和偏置
        fc1_weight_size = self.fc1.weight.numel()
        fc1_bias_size = self.fc1.bias.numel()
        fc1_weight = torch.from_numpy(
            flat_gradient[idx:idx + fc1_weight_size].reshape(self.fc1.weight.shape)
        ).float().to(self.fc1.weight.device)
        idx += fc1_weight_size
        fc1_bias = torch.from_numpy(
            flat_gradient[idx:idx + fc1_bias_size]
        ).float().to(self.fc1.bias.device)
        idx += fc1_bias_size
        layers.append(fc1_weight)
        layers.append(fc1_bias)

        # FC2 权重和偏置
        fc2_weight_size = self.fc2.weight.numel()
        fc2_bias_size = self.fc2.bias.numel()
        fc2_weight = torch.from_numpy(
            flat_gradient[idx:idx + fc2_weight_size].reshape(self.fc2.weight.shape)
        ).float().to(self.fc2.weight.device)
        idx += fc2_weight_size
        fc2_bias = torch.from_numpy(
            flat_gradient[idx:idx + fc2_bias_size]
        ).float().to(self.fc2.bias.device)
        idx += fc2_bias_size
        layers.append(fc2_weight)
        layers.append(fc2_bias)

        return layers