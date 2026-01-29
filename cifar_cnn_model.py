import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFARCNNModel(nn.Module):
    def __init__(self, D_in=None, D_out=10):
        super(CIFARCNNModel, self).__init__()
        # Input: 3 x 32 x 32
        
        # Conv1: 3 -> 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        # 关键修改：使用 GroupNorm (32个通道分8组)
        self.gn1 = nn.GroupNorm(8, 32)
        
        # Conv2: 32 -> 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 关键修改：使用 GroupNorm (64个通道分16组)
        self.gn2 = nn.GroupNorm(16, 64)
        
        # Conv3: 64 -> 64
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.gn3 = nn.GroupNorm(16, 64)

        # 展平后的维度: 64通道 * 4 * 4 (经过3次 max_pool 2x2, 32->16->8->4)
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, D_out)

    def forward(self, x):
        # x: [Batch, 3, 32, 32]
        
        # Layer 1
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.max_pool2d(x, 2) # -> 16x16
        
        # Layer 2
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.max_pool2d(x, 2) # -> 8x8
        
        # Layer 3
        x = F.relu(self.gn3(self.conv3(x)))
        x = F.max_pool2d(x, 2) # -> 4x4
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def reshape(self, flat_gradient):
        layers = []
        idx = 0
        for name, param in self.named_parameters():
            num_param = param.numel()
            layer_param = torch.from_numpy(
                flat_gradient[idx:idx + num_param].reshape(param.shape)
            ).float().to(param.device)
            layers.append(layer_param)
            idx += num_param
        return layers