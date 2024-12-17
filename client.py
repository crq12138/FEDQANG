# import time
# import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import numpy as np
# import pdb
import datasets
import pickle
import bc_enum
# from . import p2p
# from .blockchain import Blockchain
import torch.nn.functional as F




class Client():
    def __init__(self, dataset, filename, batch_size, model,port,train_cut=.80, credit_score=100.0, quality_score=100.0, p2p_node=None):
        # initializes dataset
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(torch.cuda.is_available())
        # 将模型移动到设备
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.ipport=port
        Dataset = datasets.get_dataset(dataset)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        self.trainset = Dataset(filename, "./" + dataset, is_train=True, transform=transform)
        self.testset = Dataset("mnist_test", "./" + dataset, is_train=False, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=len(self.testset), shuffle=False)
        # self.attackset = Dataset("mnist_digit1", "../mnist_data/" + dataset, is_train=False, transform=transform)
        # self.attackloader = torch.utils.data.DataLoader(self.attackset, batch_size=len(self.testset), shuffle=False)
        self.model = model

        ### Tunables ###
        # self.criterion = nn.MultiLabelMarginLoss()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.75, weight_decay=0.001)  # mnist_cnn
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5, weight_decay=0.001) # mnist_softmax
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.001) # lfw_cnn
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.001) # lfw_softmax
        self.aggregatedGradients = []
        self.loss = 0.0

        ### 激励机制相关 ###
        self.credit_score = credit_score
        self.quality_score = quality_score
        self.p2p_node = p2p_node  # 引用共享的 Node 实例
        
        from core import blockchain_instance
        self.blockchain = blockchain_instance  # 引用全局 Blockchain 实例
        

    # TODO:: Get noise for diff priv
    def getGrad(self):
        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            inputs = data['image'].float().to(self.device)
            labels = data['label'].long().to(self.device)

            # for svm
            # padded_labels = torch.zeros(self.batch_size,2).long()
            # padded_labels.transpose(0,1)[labels] = 1
            # labels = padded_labels

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 100)
            self.loss = loss.item()

            # TODO: Find more efficient way to flatten params
            # get gradients into layers
            # layers = np.zeros(0)
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         layers = np.concatenate((layers, param.grad.numpy().flatten()), axis=None)
            layers = np.zeros(0)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    layers = np.concatenate((layers, param.grad.cpu().numpy().flatten()), axis=None)
            return layers

    # Called when an aggregator receives a new gradient
    def updateGrad(self, gradient):
        # Reshape into original tensor
        layers = self.model.reshape(gradient)
        layers = [layer.to(self.device) for layer in layers]
        self.aggregatedGradients.append(layers)

    # Step in the direction of provided gradient.
    # Used in BlockML when gradient is aggregated in Go
    def simpleStep(self, gradient):
        print("Simple step")
        layers = self.model.reshape(gradient)
    # 将梯度移动到设备上
        layers = [layer.to(self.device) for layer in layers]
    # 手动更新参数梯度
        layer = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = layers[layer]
                layer += 1
    # 执行优化步骤
        self.optimizer.step()

    # Called when sufficient gradients are aggregated to generate updated model
    def step(self):
        # Aggregate gradients together in place
        for i in range(1, len(self.aggregatedGradients)):
            gradients = self.aggregatedGradients[i]
            for g, gradient in enumerate(gradients):
                self.aggregatedGradients[0][g] += gradient

        # Average gradients
        for g, gradient in enumerate(self.aggregatedGradients[0]):
            gradient /= len(self.aggregatedGradients)

        # Manually updates parameter gradients
        layer = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.aggregatedGradients[0][layer]
                layer += 1

        # Step in direction of parameter gradients
        self.optimizer.step()
        self.aggregatedGradients = []

    # Called when the aggregator shares the updated model
    def updateModel(self, modelWeights):

        layers = self.model.reshape(modelWeights)
        layer = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = layers[layer].to(self.device)
                layer += 1

    # def getModelWeights(self):
    #     layers = np.zeros(0)
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             layers = np.concatenate((layers, param.data.numpy().flatten()), axis=None)
    #     return layers

    def getModelWeights(self):
        layers = np.zeros(0)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                layers = np.concatenate((layers, param.data.cpu().numpy().flatten()), axis=None)
        return layers

    def getLoss(self):
        return self.loss

    def getModel(self):
        return self.model

    def getTestErr(self):
        for i, data in enumerate(self.testloader, 0):
            # get the inputs
            inputs = data['image'].float().to(self.device)
            labels = data['label'].long().to(self.device)
            inputs, labels = Variable(inputs), Variable(labels)
            out = self.model(inputs)
            pred = np.argmax(out.detach().cpu().numpy(), axis=1)
        return 1 - accuracy_score(pred, labels.cpu())
    
    ### 激励机制相关 ###
    def update_credit_score(self, delta):
        self.credit_score += delta
        if self.credit_score < 0:
            self.credit_score = 0.0

    def update_quality_score(self, delta):
        self.quality_score += delta
        if self.quality_score < 0:
            self.quality_score = 0.0

    def get_Hash_stake(self):
        return self.credit_score - self.quality_score


    def send_grad_to_committee(self, grad):
        """将梯度发送给当前的委员会成员"""
        committee_members = self.blockchain.committee.copy()
        if self.is_committee_member():
            # 如果当前节点也是委员会成员，排除自己避免自发送
            committee_members = [m for m in committee_members if m != self.p2p_node.selfipport]
        
        if not committee_members:
            print("当前没有委员会成员或只有自己是委员会成员，无需发送梯度。")
            return
        
        # 序列化梯度
        grad_bytes = pickle.dumps(grad)
        # 发送梯度到委员会成员
        for member in committee_members:
            self.p2p_node.send_grad(grad_bytes, member)
            print(f"梯度已发送到委员会成员: {member}")



    def is_committee_member(self):
        """判断当前客户端是否为委员会成员"""
        # print(type(self.ipport))
        # print(type(self.blockchain.committee))
        return self.ipport in self.blockchain.committee
    





    

# class MNISTCNNModel(nn.Module):
#     def __init__(self, D_in=None, D_out=None):
#         super(MNISTCNNModel, self).__init__()
#         # 定义卷积层
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         # 定义全连接层
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 10)
#         # 记录输入和输出维度（与原方法保持一致）
#         self.D_in = D_in
#         self.D_out = D_out

#     def forward(self, x):
#         # x 的形状应为 [batch_size, 1, 28, 28]
#         x = x.view(-1, 1, 28, 28)
#         x = F.relu(self.conv1(x))        # 输出形状: [batch_size, 32, 28, 28]
#         x = F.max_pool2d(x, 2)           # 输出形状: [batch_size, 32, 14, 14]
#         x = F.relu(self.conv2(x))        # 输出形状: [batch_size, 64, 14, 14]
#         x = F.max_pool2d(x, 2)           # 输出形状: [batch_size, 64, 7, 7]
#         x = x.view(-1, 64 * 7 * 7)       # 展平
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


# class Client():
#     def __init__(self, dataset, filename, batch_size, model,port,train_cut=.80, credit_score=100.0, quality_score=100.0):
#         # initializes dataset
#         # 设置设备
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # print(torch.cuda.is_available())
#         # 将模型移动到设备
#         self.model = model.to(self.device)
#         self.batch_size = batch_size
#         self.port=port
#         Dataset = datasets.get_dataset(dataset)
#         transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
#         self.trainset = Dataset(filename, "./" + dataset, is_train=True, transform=transform)
#         self.testset = Dataset("mnist_test", "./" + dataset, is_train=False, transform=transform)
#         self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
#         self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=len(self.testset), shuffle=False)
#         # self.attackset = Dataset("mnist_digit1", "../mnist_data/" + dataset, is_train=False, transform=transform)
#         # self.attackloader = torch.utils.data.DataLoader(self.attackset, batch_size=len(self.testset), shuffle=False)
#         self.model = model

#         ### Tunables ###
#         # self.criterion = nn.MultiLabelMarginLoss()
#         self.criterion = nn.CrossEntropyLoss()
#         self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.75, weight_decay=0.001)  # mnist_cnn
#         # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5, weight_decay=0.001) # mnist_softmax
#         # self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.001) # lfw_cnn
#         # self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.001) # lfw_softmax
#         self.aggregatedGradients = []
#         self.loss = 0.0

#         ### 激励机制相关 ###
#         self.credit_score = credit_score
#         self.quality_score = quality_score

#     # TODO:: Get noise for diff priv
#     def getGrad(self):
#         for i, data in enumerate(self.trainloader, 0):
#             # get the inputs
#             inputs = data['image'].float().to(self.device)
#             labels = data['label'].long().to(self.device)

#             # for svm
#             # padded_labels = torch.zeros(self.batch_size,2).long()
#             # padded_labels.transpose(0,1)[labels] = 1
#             # labels = padded_labels

#             # zero the parameter gradients
#             self.optimizer.zero_grad()

#             # forward + backward + optimize
#             outputs = self.model(inputs)
#             loss = self.criterion(outputs, labels)
#             loss.backward()
#             nn.utils.clip_grad_norm_(self.model.parameters(), 100)
#             self.loss = loss.item()

#             # TODO: Find more efficient way to flatten params
#             # get gradients into layers
#             # layers = np.zeros(0)
#             # for name, param in self.model.named_parameters():
#             #     if param.requires_grad:
#             #         layers = np.concatenate((layers, param.grad.numpy().flatten()), axis=None)
#             layers = np.zeros(0)
#             for name, param in self.model.named_parameters():
#                 if param.requires_grad:
#                     layers = np.concatenate((layers, param.grad.cpu().numpy().flatten()), axis=None)
#             return layers

#     # Called when an aggregator receives a new gradient
#     def updateGrad(self, gradient):
#         # Reshape into original tensor
#         layers = self.model.reshape(gradient)
#         layers = [layer.to(self.device) for layer in layers]
#         self.aggregatedGradients.append(layers)

#     # Step in the direction of provided gradient.
#     # Used in BlockML when gradient is aggregated in Go
#     def simpleStep(self, gradient):
#         print("Simple step")
#         layers = self.model.reshape(gradient)
#     # 将梯度移动到设备上
#         layers = [layer.to(self.device) for layer in layers]
#     # 手动更新参数梯度
#         layer = 0
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 param.grad = layers[layer]
#                 layer += 1
#     # 执行优化步骤
#         self.optimizer.step()

#     # Called when sufficient gradients are aggregated to generate updated model
#     def step(self):
#         # Aggregate gradients together in place
#         for i in range(1, len(self.aggregatedGradients)):
#             gradients = self.aggregatedGradients[i]
#             for g, gradient in enumerate(gradients):
#                 self.aggregatedGradients[0][g] += gradient

#         # Average gradients
#         for g, gradient in enumerate(self.aggregatedGradients[0]):
#             gradient /= len(self.aggregatedGradients)

#         # Manually updates parameter gradients
#         layer = 0
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 param.grad = self.aggregatedGradients[0][layer]
#                 layer += 1

#         # Step in direction of parameter gradients
#         self.optimizer.step()
#         self.aggregatedGradients = []

#     # Called when the aggregator shares the updated model
#     def updateModel(self, modelWeights):

#         layers = self.model.reshape(modelWeights)
#         layer = 0
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 param.data = layers[layer].to(self.device)
#                 layer += 1

#     # def getModelWeights(self):
#     #     layers = np.zeros(0)
#     #     for name, param in self.model.named_parameters():
#     #         if param.requires_grad:
#     #             layers = np.concatenate((layers, param.data.numpy().flatten()), axis=None)
#     #     return layers

#     def getModelWeights(self):
#         layers = np.zeros(0)
#         for name, param in self.model.named_parameters():
#             if param.requires_grad:
#                 layers = np.concatenate((layers, param.data.cpu().numpy().flatten()), axis=None)
#         return layers

#     def getLoss(self):
#         return self.loss

#     def getModel(self):
#         return self.model

#     def getTestErr(self):
#         for i, data in enumerate(self.testloader, 0):
#             # get the inputs
#             inputs = data['image'].float().to(self.device)
#             labels = data['label'].long().to(self.device)
#             inputs, labels = Variable(inputs), Variable(labels)
#             out = self.model(inputs)
#             pred = np.argmax(out.detach().cpu().numpy(), axis=1)
#         return 1 - accuracy_score(pred, labels.cpu())
    
#     ### 激励机制相关 ###
#     def update_credit_score(self, delta):
#         self.credit_score += delta
#         if self.credit_score < 0:
#             self.credit_score = 0.0

#     def update_quality_score(self, delta):
#         self.quality_score += delta
#         if self.quality_score < 0:
#             self.quality_score = 0.0

#     def get_Hash_stake(self):
#         return self.credit_score - self.quality_score