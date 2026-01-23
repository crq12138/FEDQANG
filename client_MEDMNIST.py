import time
import torch
from copy import deepcopy
from torch.optim.lr_scheduler import StepLR
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
# import bc_enum
# from . import p2p
# from .blockchain import Blockchain
import torch.nn.functional as F


class Client():
    def __init__(self, dataset, filename, dir, batch_size, model,port,train_cut=.80, p2p_node=None):
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
        self.trainset = Dataset(filename, "./" + dir, is_train=True, transform=transform)
        self.testset = Dataset("pathmnist_test", "./" + dataset, is_train=False, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)
        self.models = []
        self.datasize = len(self.trainset)
        # self.attackset = Dataset("mnist_digit1", "../mnist_data/" + dataset, is_train=False, transform=transform)
        # self.attackloader = torch.utils.data.DataLoader(self.attackset, batch_size=len(self.testset), shuffle=False)
        # self.model = model

        ### Tunables ###
        # self.criterion = nn.MultiLabelMarginLoss()s
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)  # mnist_cnn
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.995)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.5, weight_decay=0.001) # mnist_softmax
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.001) # lfw_cnn
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.5, weight_decay=0.001) # lfw_softmax
        self.aggregatedGradients = []
        self.loss = 0.0

        ### 激励机制相关 ###
        self.p2p_node = p2p_node  # 引用共享的 Node 实例
        
        from core import blockchain_instance
        self.blockchain = blockchain_instance  # 引用全局 Blockchain 实例
        # self.blockchain = None
        
    def set_train_datasize(self, new_size):
        """
        动态修改当前训练数据的大小，每一轮联邦学习可以调用此函数调整数据量。

        :param new_size: 需要使用的训练数据大小（样本数）
        """
        new_size = int(new_size*0.8)
        if new_size <= 0:
             print("Warning: new_size <= 0, setting to 1")
             new_size = 1
        # 重新采样训练数据
        subset_indices = np.random.choice(len(self.trainset), new_size, replace=False)
        subset = torch.utils.data.Subset(self.trainset, subset_indices)
        # 重新构造 DataLoader
        self.trainloader = torch.utils.data.DataLoader(subset, batch_size=self.batch_size, shuffle=True)

        print(f"Training data size set to: {new_size}")
        self.datasize = new_size

    # TODO:: Get noise for diff priv
    def getGrad(self):
        self.model.train()
        
        # 1. 记录训练前的模型参数
        initial_params = {name: param.clone().detach().cpu() for name, param in self.model.named_parameters() if param.requires_grad}

        # 2. 进行多个 batch 的训练
        for i, data in enumerate(self.trainloader, 0):
            inputs = data['image'].float().to(self.device)
            labels = data['label'].long().to(self.device)
            
            # 清空梯度
            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # 反向传播
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 100)
            # self.loss = loss.item()
            # for name, param in self.model.named_parameters():
            #     if param.requires_grad:
            #         print(param.grad)
            # 更新参数
            self.optimizer.step()

        # 3. 记录训练后的模型参数
        updated_params = {name: param.clone().detach().cpu() for name, param in self.model.named_parameters() if param.requires_grad}

        # 4. 计算本地更新（训练后的参数 - 训练前的参数）
        local_update = np.zeros(0)
        for name in initial_params.keys():
            update = (updated_params[name] - initial_params[name]).numpy().flatten()
            local_update = np.concatenate((local_update, update), axis=None)
        # layers = self.model.reshape(local_update)
        # layers = [layer.to(self.device) for layer in layers]
        # 5. 恢复模型到训练前状态
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in initial_params:
                    param.copy_(initial_params[name].to(self.device))  # 复制回原始参数
        # print(local_update.shape)
        # time.sleep(3000)
        return local_update

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

    def step(self):
        self.model.train()
        # Aggregate gradients together in place
        for i in range(1, len(self.aggregatedGradients)):
            gradients = self.aggregatedGradients[i]
            for g, gradient in enumerate(gradients):
                self.aggregatedGradients[0][g] += gradient

        # Average gradients
        for g, gradient in enumerate(self.aggregatedGradients[0]):
            gradient /= len(self.aggregatedGradients)

        # 手动更新参数梯度
        layer = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.add_(self.aggregatedGradients[0][layer])  # 直接加上更新量
                    layer += 1
        # ==== 新增：学习率衰减 ====
        self.scheduler.step()
        # 清空累计梯度
        current_lr = self.optimizer.param_groups[0]['lr']
        print("Current learning rate:", current_lr)
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
    def TestLoss(self):
        total_loss = 0.0  # 用于累计总损失
        total_samples = 0  # 用于记录总样本数
        with torch.no_grad():  # 在测试过程中，不需要计算梯度
            for i, data in enumerate(self.testloader, 0):
                # 获取输入和标签
                inputs = data['image'].float().to(self.device)
                labels = data['label'].long().to(self.device)
                
                # 模型输出
                out = self.model(inputs)
                
                # 计算当前批次的损失
                loss = self.criterion(out, labels)
                
                # 累加损失
                total_loss += loss.item() * inputs.size(0)  # loss.item() 提取当前批次的损失值
                total_samples += inputs.size(0)  # 更新总样本数

        # 返回平均损失
        self.loss =  total_loss / total_samples

    def getModel(self):
        return self.model

    def getTestErr(self):
        all_preds = []  # 用于存储所有批次的预测结果
        all_labels = []  # 用于存储所有批次的标签
        with torch.no_grad():  # 在测试过程中，不需要计算梯度
            for i, data in enumerate(self.testloader, 0):
                # 获取输入和标签
                inputs = data['image'].float().to(self.device)
                labels = data['label'].long().to(self.device)
            
                # 将输入和标签封装为Variable（已不再需要，因为PyTorch 1.0 后可以直接使用Tensor）
                inputs, labels = Variable(inputs), Variable(labels)
            
                # 模型输出
                out = self.model(inputs)
            
                # 获取预测值
                pred = np.argmax(out.detach().cpu().numpy(), axis=1)
            
                # 将预测值和标签添加到列表中
                all_preds.extend(pred)  # 将当前批次的预测结果添加到all_preds
                all_labels.extend(labels.cpu().numpy())  # 将当前批次的标签添加到all_labels
    
        # 计算整个测试集的准确率
        return 1 - accuracy_score(all_labels, all_preds)
    
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


    def send_grad_to_committee(self, grad, cost):
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
            # self.p2p_node.send_grad(grad_bytes, member, int(self.trainset.n))
            self.p2p_node.send_grad(grad_bytes, member, self.datasize, cost)
            print(f"梯度、数据大小、本地训练成本已发送到委员会成员: {member}")



    def is_committee_member(self):
        """判断当前客户端是否为委员会成员"""
        # print(type(self.ipport))
        # print(type(self.blockchain.committee))
        return self.ipport in self.blockchain.committee
    
    def average_models(self):
        """
        对多个节点模型的参数进行平均，生成全局模型。

        :param models: List[torch.nn.Module]，包含多个节点的模型实例。
        :return: torch.nn.Module，全局平均模型。
        """
        # 确保至少有一个模型
        if not self.models:
            raise ValueError("模型列表为空，无法计算平均值。")
        print("开始进行模型平均")
    
        # 深拷贝第一个模型作为全局模型的初始结构
        global_model = deepcopy(self.models[0])
    
        # 初始化全局模型的参数为0
        for param in global_model.parameters():
            param.data.zero_()
    
        # 遍历所有模型，将它们的参数相加
        for model in self.models:
            for global_param, local_param in zip(global_model.parameters(), model.parameters()):
                global_param.data.add_(local_param.data)
    
        # 计算平均值
        num_models = len(self.models)
        for param in global_model.parameters():
            param.data.div_(num_models)
    
        return global_model

    # ==== 新增辅助函数：用于在 Proxy Dataset 上评估梯度 ====
    def evaluate_accuracy_on_loader(self, loader):
        """
        在指定的 Dataloader 上评估当前模型的 Accuracy。
        返回: accuracy (0.0 ~ 1.0)
        """
        correct = 0
        total = 0
        self.model.eval()
        with torch.no_grad():
            for i, data in enumerate(loader, 0):
                inputs = data['image'].float().to(self.device)
                labels = data['label'].long().to(self.device)
                outputs = self.model(inputs)
                
                # 获取预测类别
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        if total == 0: return 0.0
        return correct / total

    def apply_flat_update(self, flat_update):
        """将扁平化的更新向量应用到模型上 (Model += Update)"""
        layers = self.model.reshape(flat_update)
        layer_idx = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    # 注意：local_update = new - old
                    # 所以要得到 new，应该是 param.data + update
                    param.data.add_(layers[layer_idx].to(self.device))
                    layer_idx += 1
    
    def revert_flat_update(self, flat_update):
        """撤销扁平化的更新向量 (Model -= Update)"""
        layers = self.model.reshape(flat_update)
        layer_idx = 0
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.sub_(layers[layer_idx].to(self.device))
                    layer_idx += 1