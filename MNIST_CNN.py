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
from mnist_cnn_model import MNISTCNNModel
from client_MNIST import Client
from transfer import incentive
from MNIST_CNN_path import path

epsilon = 0.04
sigama = 1e-5


@dataclass(frozen=True)
class ExperimentConfig:
    iter_time: int = 50
    batch_size: int = 32
    train_cut: float = 1.0
    seed: int = 42
    wait_for_network_s: float = 5.0
    quality_score_init: float = 1.0

    # Experiment toggles
    use_game_process: bool = False
    send_initial_model: bool = False
    use_noise: bool = False
    zero_grad_when_small: bool = False

    # Dataset selection rule
    use_mnist_unif_threshold: int = 50052
    mnist_unif_prefix: str = "mnist_noniid_0.1_client_"
    mnist_prefix: str = "mnist_unif"


def print_model_parameters(model, num_values=5):
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, shape: {param.shape}")
        print(f"Values (first {num_values}): {param.flatten().cpu().data[:num_values]}")
        print("-" * 50)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def returnModel(D_in, D_out, seed):
    set_seed(seed)
    model = MNISTCNNModel(D_in, D_out)
    return model

def calculate_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_size = total_params * 4
    print(f"模型参数数量: {total_params}")
    print(f"模型大小: {total_size / 1024:.2f} KB")
    return total_size


def get_dataset_name(port, config: ExperimentConfig):
    if int(port) < config.use_mnist_unif_threshold:
        return f"{config.mnist_prefix}{int(50063 - int(port)) // 2}"
    return f"{config.mnist_unif_prefix}{int(int(port) - 50051) // 2}"


def log_paths(node_port):
    return {
        "loss": f"{path}loss/loss_{node_port}.txt",
        "error": f"{path}error/Test_error_{node_port}.txt",
        "quality": f"{path}quality_score/Quality_score_{node_port}.txt",
        "transfer": f"{path}pay_off/Transfer_{node_port}.txt",
    }


def gaussian_noise(grad):
    raise NotImplementedError("gaussian_noise 未实现；请根据实验需求补充。")

new_error = 0.0
min_error = 1.0
min_count = 0

def run(f):
    global new_error, min_error, min_count
    config = ExperimentConfig()
    D_in = datasets.get_num_features("mnist")
    D_out = datasets.get_num_classes("mnist")
    model = returnModel(D_in, D_out, config.seed)
    print("length of the model is ", calculate_model_size(model))
    node = p2p.Node()
    
    from core import blockchain_instance
    import game_process
    dataset_name = get_dataset_name(p2p.PORT, config)
    client = Client(
        "mnist",
        dataset_name,
        "mnist",
        config.batch_size,
        model,
        p2p.SELF_IP_PORT,
        config.train_cut,
        p2p_node=node,
    )
    
    # ==== 1. 加载 Proxy Consensus Dataset (Root Dataset) ====
    # 使用 ./mnist 作为 root_dir
    root_loader = datasets.get_proxy_dataloader("mnist", "./mnist", batch_size=config.batch_size, sample_size=500)
    
    client.TestLoss()
    new_error = client.getTestErr()
    paths = log_paths(node.PORT)
    log_loss1 = open(paths["loss"], "w")
    log_loss2 = open(paths["error"], "w")
    log_loss3 = open(paths["quality"], "w")
    log_loss4 = open(paths["transfer"], "w")
    blockchain = blockchain_instance

    non_committee = len(blockchain.nodes) - blockchain.committee_size
    node.broadcast(bc_enum.SERVICE * bc_enum.DESCOVERY + bc_enum.EXCHANGENODE, None)
    node_list = node.get_nodes_list()
    port_list = [addr.split(':')[1] for addr in node_list]
    for port in port_list:
        if port not in p2p.quality_score_dict:
            p2p.quality_score_dict[port] = config.quality_score_init

    
    for iter in range(config.iter_time):
        node.broadcast(bc_enum.SERVICE * bc_enum.DESCOVERY + bc_enum.EXCHANGENODE, None)
        is_committee = client.is_committee_member()
        if config.send_initial_model and iter == 0:
            print("开始获取初始全局模型")
            client.send_models()
        
        Loss = client.getLoss()

        if not is_committee:
            print("此节点不是委员会成员，开始进行梯度计算并发送梯度")
            cost = 0
            if config.use_game_process:
                train_data_size, cost, payoff = game_process.decentralized_game(client, Loss, iter)
                log_loss4.write(f"{iter} {payoff}\n")
                log_loss4.flush()
                if config.zero_grad_when_small and train_data_size <= 10:
                    grad = torch.zeros(1663370)
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
            p2p.transfer_dict = incentive(cost_list, port_recv, krum_grad1, grad_recv)
            for item in blockchain.committee:
                p2p.transfer_dict[item.split(':')[1]] = 0.0
            # print(grad_recv)

            print(p2p.transfer_dict)
            print(datasize_recv)
            
            # ==== 2. 调用新的质量评估函数 ====
            # 传入 client (包含 model) 和 root_loader
            update_quality_scores(grad_recv, port_recv, client, root_loader, blockchain)
            
            print('grad_receive11111========',grad_recv)
            krum_grad_bytes = pickle.dumps(krum_grad1)

            print("旧一轮质量分数列表为", p2p.quality_score_dict)
            blockchain.consensus_process(krum_grad_bytes, p2p.quality_score_dict, p2p.transfer_dict)
            print(f"Epoch {iter}: 区块链共识完成。")
            
        blockchain.receive_new_block()
        print("共识后的质量分数字典为", p2p.quality_score_dict)    
        print("共识后的系统内部货币转移字典为", p2p.transfer_dict)
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
        
        if blockchain.lastBlock.krumgrad:
            krum_grad = pickle.loads(blockchain.lastBlock.krumgrad)
            client.updateGrad(krum_grad)
            client.step()
            print(f"Epoch {iter}: 模型已更新。")
            print('krumgrad==========',krum_grad)
        else:
            print(f"Epoch {iter}: 错误 - 区块链的最后一个区块缺少 krumgrad。")
        if blockchain.ipport == min(blockchain.committee, key=lambda node: int(node.split(":")[1])):
            node.send_epoch()
        else:
            print("进入等待轮次结束阶段")
            while True:
                if p2p.Epoch_overing:
                    p2p.Epoch_overing = False
                    break
                time.sleep(1)


def average(vecs, datasize_recv):
    vecs = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in vecs]
    datasize_recv1 = np.array(datasize_recv, dtype=np.float32)
    weighted_sum = np.zeros_like(vecs[0], dtype=np.float32)
    for idx, v in enumerate(vecs):
        weighted_sum += datasize_recv1[idx] * v
    total_size = np.sum(datasize_recv1)
    avg_vec = weighted_sum / total_size
    return avg_vec


# ==== 3. 新的 update_quality_scores 函数 (Accuracy-based Relative Scaling) ====
def update_quality_scores(
        grad_recv,          # list[np.ndarray]  本轮各客户端梯度
        port_recv,          # list[int|str]     端口 / 客户端标识
        client,             # Client 对象
        root_loader,        # DataLoader        代理共识数据集
        blockchain,
        max_score_change=0.2, # 每一轮最大的分数变化量 (由您设定为 0.2)
        init_score=1.0):    
    
    if root_loader is None:
        print("Warning: Root Loader is None, skipping quality update.")
        return

    # 1. 计算基准精度 (Base Accuracy)
    base_acc = client.evaluate_accuracy_on_loader(root_loader)
    print(f"Base Accuracy on Proxy Dataset: {base_acc:.4f}")
    
    acc_gains = []

    # 2. 遍历每个梯度，计算精度增益 (Marginal Accuracy Gain)
    for g_i, port in zip(grad_recv, port_recv):
        # 确保格式为 numpy
        if isinstance(g_i, torch.Tensor):
            g_i = g_i.cpu().numpy()
            
        # A. 应用梯度 (模拟更新)
        client.apply_flat_update(g_i)
        
        # B. 计算新精度
        new_acc = client.evaluate_accuracy_on_loader(root_loader)
        
        # C. 撤销梯度 (恢复模型)
        client.revert_flat_update(g_i)
        
        # D. 计算增益
        marginal_gain = new_acc - base_acc
        acc_gains.append(marginal_gain)

# 3. 找到本轮最大的正向增益 (Max Positive Gain)
    # 过滤掉负值，只看正值。如果全是负值，则 max_gain = 0
    max_gain = max([g for g in acc_gains if g > 0]) if any(g > 0 for g in acc_gains) else 0.0
    
    total_delta = 0.0
    worker_count = 0

    for gain, port in zip(acc_gains, port_recv):
        # 核心逻辑：只奖励，不惩罚
        if gain > 0 and max_gain > 1e-6:
             # 相对缩放：表现最好的拿满 0.2，其他的按比例拿
             delta_q = (gain / max_gain) * max_score_change
        else:
             # 负增益或微小增益：不扣分，但也不加分
             # 这对理性节点已经是惩罚了（因为他们浪费了算力却没拿到分）
             delta_q = 0.0 
        
        # 更新分数
        if port not in p2p.quality_score_dict:
            p2p.quality_score_dict[port] = init_score
            
        p2p.quality_score_dict[port] += delta_q
        total_delta += delta_q
        worker_count += 1
        
        print(f"Node {port}: BaseAcc={base_acc:.4f}, NewAcc={base_acc+gain:.4f}, Gain={gain:.6f}, DeltaQ={delta_q:.4f}")

    avg_delta = total_delta / worker_count if worker_count > 0 else 0.0

    for member_str in blockchain.committee:
        port = member_str.split(':')[1]
        p2p.quality_score_dict[port] += avg_delta

    print("质量分数已更新 (含委员会补偿):", p2p.quality_score_dict)