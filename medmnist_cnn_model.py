import torch
import torch.nn as nn
import torch.nn.functional as F

class MedMNISTCNNModel(nn.Module):
    def __init__(self, D_in=None, D_out=9):
        super(MedMNISTCNNModel, self).__init__()
        # Input: 3 x 28 x 28
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Linear layer
        # 28x28 -> pool -> 14x14 -> pool -> 7x7 -> pool -> 3x3 (approx) or adapt pooling
        # MaxPool 2x2: 28->14, 14->7, 7->3
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, D_out)

    def forward(self, x):
        # [Batch, 3, 28, 28]
        x = x.view(-1, 3, 28, 28)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2) # -> 14x14
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2) # -> 7x7
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2) # -> 3x3
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def reshape(self, flat_gradient):
        layers = []
        idx = 0
        
        for name, param in self.named_parameters():
            # 获取当前参数的元素个数
            num_param = param.numel()
            # 从扁平梯度中切片并 reshape
            layer_param = torch.from_numpy(
                flat_gradient[idx:idx + num_param].reshape(param.shape)
            ).float().to(param.device)
            layers.append(layer_param)
            idx += num_param
            
        return layers