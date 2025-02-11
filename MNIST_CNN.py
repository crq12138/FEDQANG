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
import bc_enum
import p2p
# from blockchain import Blockchain
# import torch.nn.functional as F
from mnist_cnn_model import MNISTCNNModel
from client import Client
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
    set_seed(seed)
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
    seed = 42
    # node_size=10
    # # num_bft=Num_Bft(node_size,f)
    # num_bft = 0
    model = returnModel(D_in, D_out, seed)
    print("length of the model is ", calculate_model_size(model))
    # print("===== Model A parameters =====")
    # print_model_parameters(model)
    # time.sleep(15)
    node=p2p.Node()
    
    from core import blockchain_instance
    import game_process
    if int(p2p.PORT) < 50061:
        client=Client("mnist", 
                #   "mnist_unif" + str(int(int(p2p.PORT)-50051)//2), 
                  "mnist_unif" + str(int(int(p2p.PORT)-50051)//2), 
                  batch_size, model, 
                  p2p.SELF_IP_PORT, 
                  train_cut, 
                  credit_score=100,
                  quality_score=100,
                  p2p_node=node)
    else:
        client=Client("mnist", 
                #   "mnist_unif" + str(int(int(p2p.PORT)-50051)//2), 
                  "mnist" + str(int(int(p2p.PORT)-50051)//2), 
                  batch_size, model, 
                  p2p.SELF_IP_PORT, 
                  train_cut, 
                  credit_score=100,
                  quality_score=100,
                  p2p_node=node)
    # client=Client("mnist", 
    #               "mnist_unif" + str(int(int(p2p.PORT)-50051)//2), 
    #             #   "mnist" + str(int(int(p2p.PORT)-50051)//2), 
    #               batch_size, model, 
    #               p2p.SELF_IP_PORT, 
    #               train_cut, 
        
    #               credit_score=100,
    #               quality_score=100,
    #               p2p_node=node)
    # filename1 = "./log/cnn/unif/test/loss/" + "loss_" + str(node.PORT) + ".txt"
    # filename2 = "./log/cnn/unif/test/error/" + "Test_error_" + str(node.PORT) + ".txt"
    filename3 = "./log/cnn/mixed/1_5n_5i/quality_score/" + "Quality_score_" + str(node.PORT) + ".txt"
    filename1 = "./log/cnn/mixed/1_5n_5i/loss/" + "loss_" + str(node.PORT) + ".txt"
    filename2 = "./log/cnn/mixed/1_5n_5i/error/" + "Test_error_" + str(node.PORT) + ".txt"
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
    blockchain = blockchain_instance
    # models = []
    non_committee = len(blockchain.nodes) - blockchain.committee_size
    node.broadcast(bc_enum.SERVICE * bc_enum.DESCOVERY + bc_enum.EXCHANGENODE, None)
    for port in p2p.port_list:
        # print("check point 1!!!!!!!!")
        if port not in p2p.quality_score_dict:
            p2p.quality_score_dict[port] = 1.0

    
    
    for iter in range(iter_time):
        # 每轮开始前选举委员会成员（已在 Blockchain 类中实现）
        # 获取当前节点是否为委员会成员
        node.broadcast(bc_enum.SERVICE * bc_enum.DESCOVERY + bc_enum.EXCHANGENODE, None)
        # blockchain.elect_committee()  # 已在 add_block 中调用，不需要手动调用
        is_committee = client.is_committee_member()
        # if(iter == 0):
        #     print("开始获取初始全局模型")
        #     client.send_models()
        
        Loss = client.getLoss()


        if not is_committee:
            print("此节点不是委员会成员，开始进行梯度计算并发送梯度")
            # 非委员会成员：训练、获取梯度、发送给委员会成员
            # print(p2p.quality_score_dict)
            # time.sleep(10)
            game_process.decentralized_game(client, Loss, iter)
            grad = client.getGrad()
            # grad_noised = gaussian_noise(grad)
            grad_noised = grad
            client.send_grad_to_committee(grad_noised)
            print(f"Epoch {iter}: 梯度已发送给委员会成员。")
            
        else:
            # 委员会成员：收集所有非委员会成员的梯度
            # 等待梯度收集完成
            time.sleep(5)
            print(f"Epoch {iter}: 作为委员会成员，开始收集梯度。")
            while(len(p2p.grad_list) != non_committee):
                time.sleep(2)
            
            grad_recv = []
            datasize_recv = p2p.datasize_list.copy()
            # print(datasize_recv)
            port_recv = p2p.port_list.copy()
            for grad_msg in p2p.grad_list:
                grad = pickle.loads(grad_msg)
                if isinstance(grad, torch.Tensor):
                    grad = grad.cpu().numpy()
                grad_recv.append(grad)
            p2p.grad_list.clear()
            p2p.datasize_list.clear()
            p2p.port_list.clear()
            print(f"Epoch {iter}: 委员会 {p2p.PORT} 收集到 {len(grad_recv)} 个梯度。")
            # print(grad_recv)
            # 使用 Krum 算法聚合梯度
            # krum_grad1 = krum(grad_recv, node_size - num_bft - 2, node_size - num_bft)
            krum_grad1 = average(grad_recv, datasize_recv)
            update_quality_scores(grad_recv, krum_grad1, port_recv)
            print('grad_receive11111========',grad_recv)
            krum_grad_bytes = pickle.dumps(krum_grad1)

            
            # 进行区块链共识
            blockchain.consensus_process(krum_grad_bytes, p2p.quality_score_dict)
            print(f"Epoch {iter}: 区块链共识完成。")
            
        
        print("新一轮质量分数列表为", p2p.quality_score_dict)    
        # 记录日志
        # Loss = client.getLoss()
        log_loss1.write(f"{iter} {Loss}\n")
        log_loss2.write(f"{iter} {client.getTestErr()}\n")
        log_loss3.write(f"{iter} {p2p.quality_score_dict[port]}\n")
        # log_loss3.write(f"{iter} {0.0}\n")  # 这里的时间记录需要进一步完善
        log_loss1.flush()
        log_loss2.flush()
        log_loss3.flush()
        
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

def update_quality_scores(grad_recv, w_avg, port_recv, D=0.01):
    """
    在获取全局梯度 w_avg 后，根据每个客户端的梯度与全局梯度的关系，更新其质量分数。
    
    参数：
    ----------
    grad_recv: list of np.array
        接收到的各客户端的梯度列表，每个元素为 NumPy 数组
    w_avg: np.array
        全局梯度向量（例如 average(grad_recv, datasize_recv) 的结果）
    port_recv: list
        存放各个梯度对应的客户端端口号，与 grad_recv 一一对应
    D: float, optional
        质量分数变化的缩放因子（或步长），默认为 0.1，可根据实验需要调整
    
    返回：
    ----------
    无（直接对 p2p.quality_score_dict 进行更新）
    """
    # 为数值稳定加一点偏置，防止除以 0
    epsilon = 1e-12

    # 计算全局梯度的范数
    norm_avg = np.linalg.norm(w_avg) + epsilon

    for i, w_i in enumerate(grad_recv):
        # 若仍是 torch.Tensor，则先转为 np.array
        if not isinstance(w_i, np.ndarray):
            w_i = w_i.cpu().numpy()

        # 计算 dot(w_i, w_avg)
        dot_val = np.sum(w_i * w_avg)

        # 计算局部梯度 w_i 的范数
        norm_i = np.linalg.norm(w_i) + epsilon

        # 计算余弦相似度
        cos_val = dot_val / (norm_i * norm_avg)

        # 计算本地梯度在全局梯度方向上的分量 |w_i|_avg
        w_i_avg = dot_val / norm_avg  # w_i 在 w_avg 方向上的投影长度

        # 计算公式中的比值 cos * (|w_i|_avg / |w_avg|)
        ratio = cos_val * (w_i_avg / norm_avg)

        # 计算本轮质量分数变化
        delta_score = D * ratio

        # 更新对应端口的质量分数，若此前没有该端口的记录则初始化
        port = port_recv[i]
        if port not in p2p.quality_score_dict:
            p2p.quality_score_dict[port] = 1.0 + delta_score
        p2p.quality_score_dict[port] += delta_score


    print("质量分数更新完成:", p2p.quality_score_dict)


