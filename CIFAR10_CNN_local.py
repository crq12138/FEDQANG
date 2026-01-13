import time
import math
import torch
# import torch.nn as nn
# import torch.optim as optim
import torch.utils.data
import random
# from torch.autograd import Variable
# import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import numpy as np
# import pdb
import datasets
import pickle
# import bc_enum
# import p2p
# from blockchain import Blockchain
# import torch.nn.functional as F
from cifar_cnn_model import CIFARCNNModel
from client_CIFAR_local import Client
epsilon = 0.04
sigama=1e-5


def print_model_parameters(model, num_values=5):
    """
    打印模型每个参数张量名、形状，以及前 num_values 个元素.
    """
    for name, param in model.named_parameters():
        # 打印参数名和形状
        print(f"Parameter name: {name}, shape: {param.shape}")
        
        # 打印该参数中前 num_values 个元素
        # .flatten() 用来将多维张量展开为一维方便显示
        # .cpu() 将参数拷贝到 CPU 上，避免 GPU 上打印不方便
        print(f"Values (first {num_values}): {param.flatten().cpu().data[:num_values]}")
        print("-" * 50)
def set_seed(seed: int):
    """
    设置 Python 内置、Numpy、PyTorch(CPU和GPU)的随机种子，以保证实验结果的可复现性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def returnModel(D_in, D_out, seed):
    # model = SoftmaxModel(D_in, D_out)
    # set_seed(seed)
    model = CIFARCNNModel(D_in, D_out)
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
    iter_time = 1000
    D_in = datasets.get_num_features("cifar")
    D_out = datasets.get_num_classes("cifar")
    batch_size = 128
    train_cut = 1.0
    seed = 42
    # node_size=10s
    # # num_bft=Num_Bft(node_size,f)
    # num_bft = 0
    model = returnModel(D_in, D_out, seed)
    print("length of the model is ", calculate_model_size(model))
    client=Client("cifar", 
                  "cifar10_train", 
                #   "mnist" + str(int(int(p2p.PORT)-50051)//2), 
                  batch_size, model, 
                  50051, 
                  train_cut, 
        
                  credit_score=100,
                  quality_score=100,
                  p2p_node=None)
    # filename1 = "./log/cnn/unif/test/loss/" + "loss_" + str(node.PORT) + ".txt"
    # filename2 = "./log/cnn/unif/test/error/" + "Test_error_" + str(node.PORT) + ".txt"
    filename3 = "./log/cnn/CIFAR10/local/quality_score/" + "Quality_score_update.txt"
    filename1 = "./log/cnn/CIFAR10/local/loss/" + "loss_update.txt"
    filename2 = "./log/cnn/CIFAR10/local/error/" + "Test_error_update.txt"
    # filename3 = "./log/cnn/non-iid/tm/quality_score/" + "Quality_score_" + str(node.PORT) + ".txt"
    # filename1 = "./log/cnn/non-iid/tm/loss/" + "loss_" + str(node.PORT) + ".txt"
    # filename2 = "./log/cnn/non-iid/tm/error/" + "Test_error_" + str(node.PORT) + ".txt"
    # filename3 = "./log/cnn/unif/10/time/" + "time_correspond" + str(node.PORT) + ".txt"
    # filename4 = "./log/cnn/time/" + "time_compute" + str(node.PORT) + ".txt"
    log_loss1 = open(filename1, "w")
    log_loss2 = open(filename2, "w")
    log_loss3 = open(filename3, "w")
    # log_loss3 = open(filename3, "w")
    # log_loss4 = open(filename4, "w")
    # blockchain = blockchain_instance
    # models = []
    # non_committee = len(blockchain.nodes) - blockchain.committee_size
    # node.broadcast(bc_enum.SERVICE * bc_enum.DESCOVERY + bc_enum.EXCHANGENODE, None)
    # for port in p2p.port_list:
    #     # print("check point 1!!!!!!!!")
    #     if port not in p2p.quality_score_dict:
    #         p2p.quality_score_dict[port] = 1.0

    
    
    for iter in range(iter_time):
        Loss = client.getLoss()
        print("此节点不是委员会成员，开始进行梯度计算并发送梯度")
        grad = client.getGrad()
        client.updateGrad(grad)
        client.step()
        log_loss1.write(f"{iter} {Loss}\n")
        log_loss2.write(f"{iter} {client.getTestErr()}\n")
        log_loss3.write(f"{iter} 100\n")
        # log_loss3.write(f"{iter} {0.0}\n")  # 这里的时间记录需要进一步完善
        log_loss1.flush()
        log_loss2.flush()
        log_loss3.flush()
        
        print(f"Epoch {iter}: 模型已更新。")
        print('krumgrad==========',grad)


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


def average(vecs, datasize_recv):
    """
    将所有向量按照对应的数据大小进行加权平均。
    
    :param vecs: 向量集合，可以是 NumPy 数组或 PyTorch 张量, 长度为 N
    :param datasize_recv: 每个参与方的数据大小列表，形如 [int, int, ...]，长度为 N
    :return: 加权平均后的向量
    """
    # 确保所有 vecs 都是 NumPy 数组
    vecs = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in vecs]
    
    # 将 datasize_recv 转为 NumPy 数组，便于后续运算
    # print(datasize_recv)

    datasize_recv1 = np.array(datasize_recv, dtype=np.float32)
    # datasize_recv1 = np.array(datasize_recv)
    
    # 加权求和
    weighted_sum = np.zeros_like(vecs[0], dtype=np.float32)
    for idx, v in enumerate(vecs):
        weighted_sum += datasize_recv1[idx] * v
    
    # 计算总的数据量
    total_size = np.sum(datasize_recv1)
    
    # 得到加权平均值
    avg_vec = weighted_sum / total_size
    
    return avg_vec


run(0)