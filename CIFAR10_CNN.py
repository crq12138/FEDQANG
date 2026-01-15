import time
import math
import random
import pickle
from dataclasses import dataclass

import torch
import torch.utils.data
import numpy as np

import datasets
import bc_enum
import p2p
from cifar_cnn_model import CIFARCNNModel
from client_CIFAR import Client
from transfer import incentive
from CIFAR10_CNN_path import path

epsilon = 0.04
sigama = 1e-5


@dataclass(frozen=True)
class ExperimentConfig:
    iter_time: int = 100
    batch_size: int = 128
    train_cut: float = 1.0
    seed: int = 42
    wait_for_network_s: float = 5.0
    quality_score_init: float = 1.0

    # Experiment toggles
    use_game_process: bool = True
    send_initial_model: bool = False
    use_noise: bool = False
    zero_grad_when_small: bool = True

    # Dataset selection rule
    use_cifar_unif_threshold: int = 50052
    cifar_unif_prefix: str = "cifar10_unif_10000_"
    cifar_prefix: str = "cifar10_"
    dataset_dir: str = "cifar-10-batches-py/cifar10"


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
    model = CIFARCNNModel(D_in, D_out)
    return model

def calculate_model_size(model):
    """
    计算PyTorch模型的参数大小（以字节为单位）
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_size = total_params * 4  # 每个浮点数（float32）占4字节
    print(f"模型参数数量: {total_params}")
    print(f"模型大小: {total_size / 1024:.2f} KB")
    return total_size


def get_dataset_name(port, config: ExperimentConfig):
    if int(port) < config.use_cifar_unif_threshold:
        return f"{config.cifar_prefix}{int(int(port) - 50051) // 2}"
    return f"{config.cifar_unif_prefix}{int(int(port) - 50051) // 2}"


def log_paths(node_port):
    return {
        "loss": f"{path}loss/loss_{node_port}.txt",
        "error": f"{path}error/Test_error_{node_port}.txt",
        "quality": f"{path}quality_score/Quality_score_{node_port}.txt",
        "transfer": f"{path}pay_off/Transfer_{node_port}.txt",
    }


def gaussian_noise(grad):
    """
    可选的噪声注入接口：如需启用，请在此实现噪声逻辑。
    """
    raise NotImplementedError("gaussian_noise 未实现；请根据实验需求补充。")

new_error = 0.0
min_error = 1.0
min_count = 0
# blockchain_instance = Blockchain()
def run(f):
    global new_error, min_error, min_count
    config = ExperimentConfig()
    D_in = datasets.get_num_features("cifar")
    D_out = datasets.get_num_classes("cifar")
    model = returnModel(D_in, D_out, config.seed)
    print("length of the model is ", calculate_model_size(model))
    # print("===== Model A parameters =====")
    # print_model_parameters(model)
    # time.sleep(15)
    node = p2p.Node()
    
    from core import blockchain_instance
    import game_process
    dataset_name = get_dataset_name(p2p.PORT, config)
    client = Client(
        "cifar",
        dataset_name,
        config.dataset_dir,
        config.batch_size,
        model,
        p2p.SELF_IP_PORT,
        config.train_cut,
        p2p_node=node,
    )
    
    client.TestLoss()
    new_error = client.getTestErr()
    paths = log_paths(node.PORT)
    log_loss1 = open(paths["loss"], "w")
    log_loss2 = open(paths["error"], "w")
    log_loss3 = open(paths["quality"], "w")
    log_loss4 = open(paths["transfer"], "w")
    blockchain = blockchain_instance
    # models = []
    non_committee = len(blockchain.nodes) - blockchain.committee_size
    node.broadcast(bc_enum.SERVICE * bc_enum.DESCOVERY + bc_enum.EXCHANGENODE, None)
    node_list = node.get_nodes_list()
    port_list = [addr.split(':')[1] for addr in node_list]
    for port in port_list:
        if port not in p2p.quality_score_dict:
            p2p.quality_score_dict[port] = config.quality_score_init

    
    
    for iter in range(config.iter_time):
        # 每轮开始前选举委员会成员（已在 Blockchain 类中实现）
        # 获取当前节点是否为委员会成员
        node.broadcast(bc_enum.SERVICE * bc_enum.DESCOVERY + bc_enum.EXCHANGENODE, None)
        # blockchain.elect_committee()  # 已在 add_block 中调用，不需要手动调用
        is_committee = client.is_committee_member()
        if config.send_initial_model and iter == 0:
            print("开始获取初始全局模型")
            client.send_models()
        
        Loss = client.getLoss()


        if not is_committee:
            print("此节点不是委员会成员，开始进行梯度计算并发送梯度")
            # 非委员会成员：训练、获取梯度、发送给委员会成员
            cost = 0
            if config.use_game_process:
                train_data_size, cost, payoff = game_process.decentralized_game(client, Loss, iter)
                if config.zero_grad_when_small and train_data_size <= 128:
                    grad = torch.zeros(319242)
                    client.datasize = 0
                else:
                    client.set_train_datasize(train_data_size)
                    grad = client.getGrad()
            else:
                grad = client.getGrad()

            if config.use_noise:
                grad = gaussian_noise(grad)
            client.send_grad_to_committee(grad, cost)
            print(f"Epoch {iter}: 梯度已发送给委员会成员。")
            
        else:
            # 委员会成员：收集所有非委员会成员的梯度
            # 等待梯度收集完成
            time.sleep(config.wait_for_network_s)
            print(f"Epoch {iter}: 作为委员会成员，开始收集梯度。")
            while(len(p2p.grad_list) != non_committee):
                time.sleep(2)
            
            grad_recv = []
            cost_list = []
            cost_list = p2p.cost_list.copy()
            datasize_recv = p2p.datasize_list.copy()
            # print(datasize_recv)
            # time.sleep(3000)
            port_recv = p2p.port_list.copy()
            for grad_msg in p2p.grad_list:
                grad = pickle.loads(grad_msg)
                if isinstance(grad, torch.Tensor):
                    grad = grad.cpu().numpy()
                grad_recv.append(grad)
            p2p.grad_list.clear()
            p2p.datasize_list.clear()
            p2p.port_list.clear()
            p2p.cost_list.clear()
            print(f"Epoch {iter}: 委员会 {p2p.PORT} 收集到 {len(grad_recv)} 个梯度。")
            krum_grad1 = average(grad_recv, datasize_recv)
            # print(port_recv)
            p2p.transfer_dict = incentive(cost_list, port_recv, krum_grad1, grad_recv)
            for item in blockchain.committee:
                p2p.transfer_dict[item.split(':')[1]] = 0.0
            print(grad_recv)
            
            print(p2p.transfer_dict)
            print(datasize_recv)
            update_quality_scores(grad_recv, krum_grad1, port_recv)
            print('grad_receive11111========',grad_recv)
            krum_grad_bytes = pickle.dumps(krum_grad1)

            print("旧一轮质量分数列表为", p2p.quality_score_dict)
            # 进行区块链共识
            blockchain.consensus_process(krum_grad_bytes, p2p.quality_score_dict, p2p.transfer_dict)
            print(f"Epoch {iter}: 区块链共识完成。")
            
        blockchain.receive_new_block()
        # if iter == 2:
        # print("共识后的质量分数字典为", p2p.quality_score_dict)    
        # print("共识后的系统内部货币转移字典为", p2p.transfer_dict)
        client.TestLoss()
        new_error = client.getTestErr()
        if min_error > new_error:
            min_error = new_error.copy()
            min_count = 0
        else:
            min_count += 1

        log_loss1.write(f"{iter} {Loss}\n")
        log_loss2.write(f"{iter} {new_error}\n")
        log_loss3.write(f"{iter} {p2p.quality_score_dict[p2p.PORT]}\n")
        log_loss4.write(f"{iter} {p2p.transfer_dict[p2p.PORT]}\n")
        # log_loss3.write(f"{iter} {0.0}\n")  # 这里的时间记录需要进一步完善
        log_loss1.flush()
        log_loss2.flush()
        log_loss3.flush()
        log_loss4.flush()
        
        
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
        # if min_count == 40:
        #     break


# def cul_score(vi, vecs, num):
#     vec = [np.sum((x - vi) ** 2) for x in vecs]
#     vec.sort()
#     score = sum(vec[:num])
#     return score
# def krum(vecs, num, m):
#     # print(vecs)
#     # 确保所有 vecs 都是 NumPy 数组
#     vecs = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in vecs]
#     temp = (sorted(vecs, key=lambda x: cul_score(x, vecs, num + 1)))[0:m]
#     return sum(temp) / m


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
    datasize_recv1 = np.array(datasize_recv, dtype=np.float32)
    
    # 加权求和
    weighted_sum = np.zeros_like(vecs[0], dtype=np.float32)
    for idx, v in enumerate(vecs):
        # if datasize_recv[idx] != 0:
            # print(datasize_recv1[idx])
            # print(v)
            # print(datasize_recv)
            # print(vecs)
        weighted_sum += datasize_recv1[idx] * v
    
    # 计算总的数据量
    total_size = np.sum(datasize_recv1)
    
    # 得到加权平均值
    avg_vec = weighted_sum / total_size
    
    return avg_vec

def update_quality_scores(
        grad_recv,          # list[np.ndarray]  本轮各客户端梯度 g_i
        w_avg,              # np.ndarray        加权全局梯度 w_avg
        port_recv,          # list[int|str]     端口 / 客户端标识
        D=0.02,              # 可选整体缩放因子；如不需缩放设 1
        init_score=2.0):    # 新节点质量分数初始化值
    """
    ➤ 基于“将梯度投影到 w_avg 方向”更新 p2p.quality_score_dict  
      质量分数增量:  Δq_i = ||g_i|| · cos(g_i, w_avg) / w_distance

        • w_mean  : 所有 g_i 的简单平均梯度  
        • w_distance = ||w_mean||·cos(w_mean, w_avg) (即 w_mean 在 w_avg 方向上的投影长度)

    参数说明
    ----------
    grad_recv   : list[np.ndarray]     每个客户端上报的梯度
    w_avg       : np.ndarray           当前轮次的全局(加权)平均梯度
    port_recv   : list[int|str]        与 grad_recv 一一对应的节点标识
    D           : float                (可选) 将所有 Δq_i 统一乘以 D
    init_score  : float                新出现端口的初始质量分数
    """

    eps = 1e-12   # 数值稳定
    # ---- 1. 数据准备 ---------------------------------------------------------
    w_avg = w_avg if isinstance(w_avg, np.ndarray) else w_avg.cpu().numpy()
    norm_avg = np.linalg.norm(w_avg) + eps           # ||w_avg||
    u_avg = w_avg / norm_avg                         # w_avg 单位向量

    # 把所有梯度转成 np.ndarray 并堆叠
    grads = [g.cpu().numpy() if not isinstance(g, np.ndarray) else g
             for g in grad_recv]
    G = np.stack(grads, axis=0)

    # ---- 2. 计算 w_mean 及其在 w_avg 方向上的投影长度 --------------------------
    w_mean = G.mean(axis=0)                          # 简单平均梯度
    w_distance = np.dot(w_mean, u_avg)               # 投影长度 (可正可负)
    if abs(w_distance) < eps:
        # 若平均梯度与 w_avg 几乎正交，直接退出或设为极小非零值
        w_distance = eps

    # ---- 3. 计算每个客户端的投影长度，并更新质量分数 ---------------------------
    for g_i, port in zip(grads, port_recv):
        # g_i 在 w_avg 方向上的投影长度：||g_i||·cosθ = g_i·u_avg
        projection_len = np.dot(g_i, u_avg)

        # Δq_i = (projection_len / w_distance) * D
        delta_q = D * (projection_len / w_distance)

        # 更新 p2p.quality_score_dict
        if port not in p2p.quality_score_dict:
            p2p.quality_score_dict[port] = init_score
        p2p.quality_score_dict[port] += delta_q

    print("质量分数已更新:", p2p.quality_score_dict)

