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


class Client():
    def __init__(self, dataset, filename, batch_size, model,port,train_cut=.80):
        # initializes dataset
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(torch.cuda.is_available())
        # 将模型移动到设备
        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.port=port
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


def returnModel(D_in,D_out):
    # model = SoftmaxModel(D_in, D_out)
    model = MNISTCNNModel(D_in, D_out)
    return model
def gaussian_noise(grad):
    # 确保 grad 是 NumPy 数组
    if isinstance(grad, torch.Tensor):
        grad = grad.cpu().numpy()
    noise_grad = grad.copy()
    global epsilon, sigama
    sum1 = np.sum(noise_grad ** 2)
    a = math.sqrt(sum1)
    scale = (1e-6) * a * math.sqrt(2 * math.log(1.25 / sigama)) / epsilon
    noise = np.random.normal(0, scale, size=noise_grad.shape)
    noise_grad += noise
    return noise_grad
def BFT(grad):
    BFT_grad = grad
    for i in range(BFT_grad.size):
        BFT_grad[i]=10000000
    return BFT_grad
def Num_Bft(world_size,f):
    return int(world_size*f)
def run(f):
    iter_time = 1000
    D_in = datasets.get_num_features("mnist")
    D_out = datasets.get_num_classes("mnist")
    batch_size = 100
    train_cut = 0.8
    node_size=4
    num_bft=Num_Bft(node_size,f)
    model = returnModel(D_in, D_out)
    client=Client("mnist", "mnist_unif" + str(int(int(p2p.PORT)-50051)//2), batch_size, model, p2p.PORT, train_cut)
    node=p2p.Node()
    filename1 = "./log/cnn/unif/loss/" + "loss_" + str(node.PORT) + ".txt"
    filename2 = "./log/cnn/unif/error/" + "Test_error_" + str(node.PORT) + ".txt"
    filename3 = "./log/cnn/unif/time/" + "time_correspond" + str(node.PORT) + ".txt"
    # filename4 = "./log/cnn/time/" + "time_compute" + str(node.PORT) + ".txt"
    log_loss1 = open(filename1, "w")
    log_loss2 = open(filename2, "w")
    log_loss3 = open(filename3, "w")
    # log_loss4 = open(filename4, "w")
    blockchain = Blockchain()
    for iter in range(iter_time):
        # t1=time.time()
        grad_recv = list()
        t1 = time.time()
        if ((int(p2p.PORT)-50051)//2)<(node_size-num_bft):
            grad = client.getGrad()
        else:
            grad=BFT(client.getGrad())
        t2 = time.time()
        grad_recv.append(grad)
        print('grad_receive00000========',grad_recv)
        # grad1=gaussian_noise(grad)
        grad1 = grad
        a=pickle.dumps(grad1)
        t3 = time.time()
        node.broadcast(bc_enum.SERVICE * bc_enum.DESCOVERY + bc_enum.EXCHANGEGRAD, a)
        # print('grad_receive========',grad_recv)
        t4 = time.time()
        time.sleep(3)
        # for i in p2p.grad_list:
        #     if i not in grad_recv:
        #         grad_recv.append(pickle.loads(i.para))
        for i in p2p.grad_list:
            grad = pickle.loads(i.para)
            # 确保 grad 是 NumPy 数组
            if isinstance(grad, torch.Tensor):
                grad = grad.cpu().numpy()
            found = False
            for existing_grad in grad_recv:
                if np.array_equal(grad, existing_grad):
                    found = True
                    break
            if not found:
                grad_recv.append(grad)
        p2p.grad_list.clear()
        print('grad_receive11111========',grad_recv)
        grad_recv.sort(key=lambda x: sum(x))
        print('grad_receive22222========',grad_recv)
        krum_grad1 = krum(grad_recv, node_size - num_bft - 2, node_size - num_bft)
        b = pickle.dumps(krum_grad1)
        t5=time.time()
        # print("krum 的结果是：", b)
        blockchain.consensus_process(b)
        t6=time.time()
        # print("blockchain.lastBlock.krumgrad:", blockchain.lastBlock.krumgrad)
        if blockchain.lastBlock.krumgrad:
            krum_grad = pickle.loads(blockchain.lastBlock.krumgrad)
        else:
            print("Error: blockchain.lastBlock.krumgrad is None or empty")
    # 处理错误，例如跳过此轮或重新请求数据

        print('krumgrad==========',krum_grad)
        # Share updated model
        # TO DO add  krum and BC
        client.updateGrad(krum_grad)
        client.step()
        # t2=time.time()
        t_cor=t4-t3
        t_consen=t6-t5
        # t_com=t2-t1
        print('============== EPOCH=', iter, '==============')
        print('============== consensus_time=',t_consen, '==============')
        # print('============== t_all=', t_all, '==============')
        print('============== time_cor=', t_cor, '==============')
        # print('============== time_com=', t_com, '==============')
        print('============== LOSS = ', client.getLoss(), '==============')
        print('============== ERROR = ', client.getTestErr(), '==============')
        log_loss1.write('%d %3f\n' % (iter, client.getLoss()))
        log_loss2.write('%d %3f\n' % (iter, client.getTestErr()))
        log_loss1.flush()
        log_loss2.flush()
        log_loss3.write('%d %3f\n' % (iter, t_cor))
        # log_loss4.write('%d %3f\n' % (iter, t_com))
        log_loss3.flush()
        # log_loss4.flush()
        if blockchain.role=="leader":
            node.send_epoch()
        else:
            while True:
                if p2p.Epoch_overing:
                    p2p.Epoch_overing=False
                    break


def cul_score(vi, vecs, num):
    vec = [np.sum((x - vi) ** 2) for x in vecs]
    vec.sort()
    score = sum(vec[:num])
    return score
def krum(vecs, num, m):
    # print(vecs)
    # 确保所有 vecs 都是 NumPy 数组
    vecs = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in vecs]
    temp = (sorted(vecs, key=lambda x: cul_score(x, vecs, num + 1)))[0:m]
    return sum(temp) / m