import time
import math
import torch
# import torch.nn as nn
# import torch.optim as optim
import torch.utils.data
# from torch.autograd import Variable
# import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import numpy as np
# import pdb
import datasets
import pickle
import bc_enum
# import p2p
# from blockchain import Blockchain
# import torch.nn.functional as F
from mnist_cnn_model import MNISTCNNModel
from client_local import Client

epsilon = 0.04
sigama=1e-5


# print("11111111111111111111111111")

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

def calculate_model_size(model):
    """
    计算PyTorch模型的参数大小（以字节为单位）
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_size = total_params * 4  # 每个浮点数（float32）占4字节
    print(f"模型参数数量: {total_params}")
    print(f"模型大小: {total_size / 1024:.2f} KB")
    return total_size

# blockchain_instance = Blockchain()
def run(f):
    iter_time = 500
    D_in = datasets.get_num_features("mnist")
    D_out = datasets.get_num_classes("mnist")
    batch_size = 128
    train_cut = 0.8
    # node_size=10
    # # num_bft=Num_Bft(node_size,f)
    # num_bft = 0
    model = returnModel(D_in, D_out)
    print("length of the model is ", calculate_model_size(model))
    print(model)
    # node=p2p.Node()
    
    # from core import blockchain_instance
    client=Client("mnist", 
                  "mnist_unif0", 
                  batch_size, model, 
                  None, 
                  train_cut, 
                  credit_score=100,
                  quality_score=100,
                  p2p_node=None)
    # filename1 = "./log/cnn/unif/test/loss/" + "loss_" + str(node.PORT) + ".txt"
    # filename2 = "./log/cnn/unif/test/error/" + "Test_error_" + str(node.PORT) + ".txt"
    filename1 = "./log/cnn/CIFAR10/local/loss/" + "loss.txt"
    filename2 = "./log/cnn/CIFAR10/local/error/" + "Test_error.txt"
    # filename3 = "./log/cnn/unif/10/time/" + "time_correspond" + str(node.PORT) + ".txt"
    # filename4 = "./log/cnn/time/" + "time_compute" + str(node.PORT) + ".txt"
    log_loss1 = open(filename1, "w")
    log_loss2 = open(filename2, "w")
    # log_loss3 = open(filename3, "w")
    # log_loss4 = open(filename4, "w")
    # blockchain = blockchain_instance
    # models = []
    # non_committee = len(blockchain.nodes) - blockchain.committee_size
    
    
    for iter in range(iter_time):
        print("此节点不是委员会成员，开始进行梯度计算并发送梯度")
            # 非委员会成员：训练、获取梯度、发送给委员会成员
        grad = client.getGrad()
            # grad_noised = gaussian_noise(grad)
        grad_noised = grad
        print(grad_noised)
        # client.send_grad_to_committee(grad_noised)
        print(f"Epoch {iter}: 梯度已发送给委员会成员。")
            
        # 记录日志
        log_loss1.write(f"{iter} {client.getLoss()}\n")
        log_loss2.write(f"{iter} {client.getTestErr()}\n")
        # log_loss3.write(f"{iter} {0.0}\n")  # 这里的时间记录需要进一步完善
        log_loss1.flush()
        log_loss2.flush()
        # log_loss3.flush()
        
        
        
        client.updateGrad(grad_noised)
        client.step()
        print(f"Epoch {iter}: 模型已更新。")
        print('grad==========',grad_noised)


run(0)
