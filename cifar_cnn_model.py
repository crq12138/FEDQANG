import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# class CIFARCNNModel(nn.Module):
#     def __init__(self, D_in, D_out):
#         super(CIFARCNNModel, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(32, 64, 3)
#         self.conv3 = nn.Conv2d(64, 64, 3)
#         self.fc1 = nn.Linear(64 * 4 * 4, 64)
#         self.fc2 = nn.Linear(64, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = F.relu(self.conv3(x))
#         x = x.view(-1, 64 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

#     # 展平梯度的逆操作，用于将扁平化的梯度还原为模型参数的形状
#     def reshape(self, flat_gradient):
#         layers = []
#         idx = 0

#         # Conv1 权重和偏置
#         conv1_weight_size = self.conv1.weight.numel()
#         conv1_bias_size = self.conv1.bias.numel()
#         conv1_weight = torch.from_numpy(
#             flat_gradient[idx:idx + conv1_weight_size].reshape(self.conv1.weight.shape)
#         ).float().to(self.conv1.weight.device)
#         idx += conv1_weight_size
#         conv1_bias = torch.from_numpy(
#             flat_gradient[idx:idx + conv1_bias_size]
#         ).float().to(self.conv1.bias.device)
#         idx += conv1_bias_size
#         layers.append(conv1_weight)
#         layers.append(conv1_bias)

#         # Conv2 权重和偏置
#         conv2_weight_size = self.conv2.weight.numel()
#         conv2_bias_size = self.conv2.bias.numel()
#         conv2_weight = torch.from_numpy(
#             flat_gradient[idx:idx + conv2_weight_size].reshape(self.conv2.weight.shape)
#         ).float().to(self.conv2.weight.device)
#         idx += conv2_weight_size
#         conv2_bias = torch.from_numpy(
#             flat_gradient[idx:idx + conv2_bias_size]
#         ).float().to(self.conv2.bias.device)
#         idx += conv2_bias_size
#         layers.append(conv2_weight)
#         layers.append(conv2_bias)

#         # Conv3 权重和偏置
#         conv3_weight_size = self.conv3.weight.numel()
#         conv3_bias_size = self.conv3.bias.numel()
#         conv3_weight = torch.from_numpy(
#             flat_gradient[idx:idx + conv3_weight_size].reshape(self.conv3.weight.shape)
#         ).float().to(self.conv3.weight.device)
#         idx += conv3_weight_size
#         conv3_bias = torch.from_numpy(
#             flat_gradient[idx:idx + conv3_bias_size]
#         ).float().to(self.conv3.bias.device)
#         idx += conv3_bias_size
#         layers.append(conv3_weight)
#         layers.append(conv3_bias)

#         # FC1 权重和偏置
#         fc1_weight_size = self.fc1.weight.numel()
#         fc1_bias_size = self.fc1.bias.numel()
#         fc1_weight = torch.from_numpy(
#             flat_gradient[idx:idx + fc1_weight_size].reshape(self.fc1.weight.shape)
#         ).float().to(self.fc1.weight.device)
#         idx += fc1_weight_size
#         fc1_bias = torch.from_numpy(
#             flat_gradient[idx:idx + fc1_bias_size]
#         ).float().to(self.fc1.bias.device)
#         idx += fc1_bias_size
#         layers.append(fc1_weight)
#         layers.append(fc1_bias)

#         # FC2 权重和偏置
#         fc2_weight_size = self.fc2.weight.numel()
#         fc2_bias_size = self.fc2.bias.numel()
#         fc2_weight = torch.from_numpy(
#             flat_gradient[idx:idx + fc2_weight_size].reshape(self.fc2.weight.shape)
#         ).float().to(self.fc2.weight.device)
#         idx += fc2_weight_size
#         fc2_bias = torch.from_numpy(
#             flat_gradient[idx:idx + fc2_bias_size]
#         ).float().to(self.fc2.bias.device)
#         idx += fc2_bias_size
#         layers.append(fc2_weight)
#         layers.append(fc2_bias)

#         return layers
class CIFARCNNModel(nn.Module):
    """Convolutional Neural Network (CNN2) with three 3×3 conv layers and one FC layer.

    Architecture:
        • 3 × [Conv(3×3, 128 ch) → ReLU → 2×2 MaxPool]
        • 1 × Fully‑Connected layer

    The constructor keeps the original `D_in` / `D_out` signature for drop‑in
    compatibility with earlier code while defaulting to CIFAR‑10‑style inputs.
    """

    def __init__(self, D_in: Optional[int] = None, D_out: Optional[int] = None):
        super().__init__()

        num_classes = 10 if D_out is None else D_out

        # Three convolutional layers (each maintains spatial dim via padding=1)
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # After 3× (conv → pool): 32→16→8→4 → feature map 4×4
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

        self.D_in = D_in
        self.D_out = D_out

    # ─────────────────────────────────── forward ────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 3, 32, 32)  # Ensure [B, 3, 32, 32]

        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

    # ───────────────────────────────── reshape grad ─────────────────────────────────
    def reshape(self, flat_gradient):
        """Reconstruct parameter‑shaped tensors from a flattened gradient vector."""
        if isinstance(flat_gradient, torch.Tensor):
            flat_gradient = flat_gradient.cpu().numpy()

        layers = []
        idx = 0

        def _unflatten(module):
            nonlocal idx
            w_size = module.weight.numel()
            b_size = module.bias.numel()

            w = torch.from_numpy(
                flat_gradient[idx : idx + w_size].reshape(module.weight.shape)
            ).float().to(module.weight.device)
            idx += w_size
            b = torch.from_numpy(
                flat_gradient[idx : idx + b_size]
            ).float().to(module.bias.device)
            idx += b_size
            layers.extend([w, b])

        _unflatten(self.conv1)
        _unflatten(self.conv2)
        _unflatten(self.conv3)
        _unflatten(self.fc)

        return layers


# ────────────────────────────────────────────────────────────────────────────────
# CIFAR10CNNModel: MNIST‑style 2‑layer CNN adapted for CIFAR‑10
# ────────────────────────────────────────────────────────────────────────────────

class CIFAR10CNNModel(nn.Module):
    """Convolutional Neural Network (CNN2) with three 3×3 conv layers and one FC layer.

    Architecture:
        • 3 × [Conv(3×3, 128 ch) → ReLU → 2×2 MaxPool]
        • 1 × Fully‑Connected layer

    The constructor keeps the original `D_in` / `D_out` signature for drop‑in
    compatibility with earlier code while defaulting to CIFAR‑10‑style inputs.
    """

    def __init__(self, D_in: Optional[int] = None, D_out: Optional[int] = None):
        super().__init__()

        num_classes = 10 if D_out is None else D_out

        # Three convolutional layers (each maintains spatial dim via padding=1)
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # After 3× (conv → pool): 32→16→8→4 → feature map 4×4
        self.fc = nn.Linear(128 * 4 * 4, num_classes)

        self.D_in = D_in
        self.D_out = D_out

    # ─────────────────────────────────── forward ────────────────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, 3, 32, 32)  # Ensure [B, 3, 32, 32]

        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)

        x = x.view(x.size(0), -1)  # Flatten
        return self.fc(x)

    # ───────────────────────────────── reshape grad ─────────────────────────────────
    def reshape(self, flat_gradient):
        """Reconstruct parameter‑shaped tensors from a flattened gradient vector."""
        if isinstance(flat_gradient, torch.Tensor):
            flat_gradient = flat_gradient.cpu().numpy()

        layers = []
        idx = 0

        def _unflatten(module):
            nonlocal idx
            w_size = module.weight.numel()
            b_size = module.bias.numel()

            w = torch.from_numpy(
                flat_gradient[idx : idx + w_size].reshape(module.weight.shape)
            ).float().to(module.weight.device)
            idx += w_size
            b = torch.from_numpy(
                flat_gradient[idx : idx + b_size]
            ).float().to(module.bias.device)
            idx += b_size
            layers.extend([w, b])

        _unflatten(self.conv1)
        _unflatten(self.conv2)
        _unflatten(self.conv3)
        _unflatten(self.fc)

        return layers