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
import p2p
# from blockchain import Blockchain
# import torch.nn.functional as F
from mnist_cnn_model import MNISTCNNModel
from client import Client

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

# blockchain_instance = Blockchain()
def run(f):
    iter_time = 1500
    D_in = datasets.get_num_features("mnist")
    D_out = datasets.get_num_classes("mnist")
    batch_size = 100
    train_cut = 0.8
    # node_size=10
    # # num_bft=Num_Bft(node_size,f)
    # num_bft = 0
    model = returnModel(D_in, D_out)
    node=p2p.Node()
    
    from core import blockchain_instance
    client=Client("mnist", 
                  "mnist_unif" + str(int(int(p2p.PORT)-50051)//2), 
                  batch_size, model, 
                  p2p.SELF_IP_PORT, 
                  train_cut, 
                  credit_score=100,
                  quality_score=100,
                  p2p_node=node)
    filename1 = "./log/cnn/unif/10_committee/loss/" + "loss_" + str(node.PORT) + ".txt"
    filename2 = "./log/cnn/unif/10_committee/error/" + "Test_error_" + str(node.PORT) + ".txt"
    # filename3 = "./log/cnn/unif/10/time/" + "time_correspond" + str(node.PORT) + ".txt"
    # filename4 = "./log/cnn/time/" + "time_compute" + str(node.PORT) + ".txt"
    log_loss1 = open(filename1, "w")
    log_loss2 = open(filename2, "w")
    # log_loss3 = open(filename3, "w")
    # log_loss4 = open(filename4, "w")
    blockchain = blockchain_instance
    
    non_committee = len(blockchain.nodes) - blockchain.committee_size
    
    
    for iter in range(iter_time):
        # 每轮开始前选举委员会成员（已在 Blockchain 类中实现）
        
        # 获取当前节点是否为委员会成员
        node.broadcast(bc_enum.SERVICE * bc_enum.DESCOVERY + bc_enum.EXCHANGENODE, None)
        blockchain.elect_committee()  # 已在 add_block 中调用，不需要手动调用
        is_committee = client.is_committee_member()
        
        if not is_committee:
            print("此节点不是委员会成员，开始进行梯度计算并发送梯度")
            # 非委员会成员：训练、获取梯度、发送给委员会成员
            grad = client.getGrad()
            # grad_noised = gaussian_noise(grad)
            grad_noised = grad
            client.send_grad_to_committee(grad_noised)
            print(f"Epoch {iter}: 梯度已发送给委员会成员。")
            
            # # 记录日志
            # log_loss1.write(f"{iter} {client.getLoss()}\n")
            # log_loss2.write(f"{iter} {client.getTestErr()}\n")
            # log_loss1.flush()
            # log_loss2.flush()
            
            # # 等待 Epoch 完成信号
            # while not p2p.Epoch_overing:
            #     time.sleep(1)
            # p2p.Epoch_overing = False
            
        else:
            # 委员会成员：收集所有非委员会成员的梯度
            # 等待梯度收集完成
            time.sleep(5)
            print(f"Epoch {iter}: 作为委员会成员，开始收集梯度。")
            while(len(p2p.grad_list) != non_committee):
                time.sleep(1)
            
            grad_recv = []
            for grad_msg in p2p.grad_list:
                grad = pickle.loads(grad_msg.para)
                if isinstance(grad, torch.Tensor):
                    grad = grad.cpu().numpy()
                grad_recv.append(grad)
            p2p.grad_list.clear()
            print(f"Epoch {iter}: 委员会 {p2p.PORT} 收集到 {len(grad_recv)} 个梯度。")
            # print(grad_recv)
            # 使用 Krum 算法聚合梯度
            # krum_grad1 = krum(grad_recv, node_size - num_bft - 2, node_size - num_bft)
            krum_grad1 = average(grad_recv)
            print('grad_receive11111========',grad_recv)
            krum_grad_bytes = pickle.dumps(krum_grad1)

            
            # 进行区块链共识
            blockchain.consensus_process(krum_grad_bytes)
            print(f"Epoch {iter}: 区块链共识完成。")
            
            
        # 记录日志
        log_loss1.write(f"{iter} {client.getLoss()}\n")
        log_loss2.write(f"{iter} {client.getTestErr()}\n")
        # log_loss3.write(f"{iter} {0.0}\n")  # 这里的时间记录需要进一步完善
        log_loss1.flush()
        log_loss2.flush()
        # log_loss3.flush()
        
        blockchain.receive_new_block()
        
        # 更新客户端模型
        if blockchain.lastBlock.krumgrad:
            krum_grad = pickle.loads(blockchain.lastBlock.krumgrad)
            client.updateGrad(krum_grad)
            client.step()
            print(f"Epoch {iter}: 模型已更新。")
            print('krumgrad==========',krum_grad)
        else:
            print(f"Epoch {iter}: 错误 - 区块链的最后一个区块缺少 krumgrad。")
        # 轮次结束后的操作
        if blockchain.ipport == min(blockchain.committee, key=lambda node: int(node.split(":")[1])):
            node.send_epoch()
            # if blockchain.role == "leader":
            # node.send_epoch()
        else:
            print("进入等待轮次结束阶段")
            while True:
                if p2p.Epoch_overing:
                    p2p.Epoch_overing = False
                    break
                time.sleep(1)


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


def average(vecs):
    """
    将所有向量取平均值。
    :param vecs: 向量集合，可以是 NumPy 数组或 PyTorch 张量
    :return: 平均后的向量
    """
    # 确保所有 vecs 都是 NumPy 数组
    vecs = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in vecs]
    
    # 计算所有向量的平均值
    avg_vec = np.mean(vecs, axis=0)
    return avg_vec