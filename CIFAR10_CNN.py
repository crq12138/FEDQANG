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
import cost_compute
from transfer import incentive
from CIFAR10_CNN_path import path
import os

os.makedirs(path, exist_ok=True)
os.makedirs(path + "loss/", exist_ok=True)
os.makedirs(path + "error/", exist_ok=True)
os.makedirs(path + "quality_score/", exist_ok=True)
os.makedirs(path + "pay_off/", exist_ok=True)
os.makedirs(path + "cost/", exist_ok=True)
os.makedirs(path + "lambda/", exist_ok=True)
os.makedirs(path + "transfer/", exist_ok=True)

epsilon = 0.04
sigama = 1e-5


@dataclass(frozen=True)
class ExperimentConfig:
    iter_time: int = 100
    batch_size: int = 32
    train_cut: float = 1.0
    seed: int = 42
    wait_for_network_s: float = 5.0
    quality_score_init: float = 1.0

    # Experiment toggles
    use_game_process: bool = True
    lazy_game_once: bool = False
    send_initial_model: bool = False
    use_noise: bool = False
    zero_grad_when_small: bool = True
    use_random_strategy: bool = False

    # Dataset selection rule
    use_cifar_unif_threshold: int = 50052
    cifar_unif_prefix: str = "cifar10_exp5_client_"
    cifar_prefix: str = "cifar10_exp5_client_"
    dataset_dir: str = "cifar-10-batches-py/cifar10"


def print_model_parameters(model, num_values=5):
    """
    打印模型每个参数张量名、形状，以及前 num_values 个元素.
    """
    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, shape: {param.shape}")
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


def get_dataset_filename(port, config: ExperimentConfig):
    client_id = (int(port) - 50051) // 2
    return f"{config.cifar_prefix}{client_id}"


def log_paths(node_port):
    return {
        "loss": f"{path}loss/loss_{node_port}.txt",
        "error": f"{path}error/Test_error_{node_port}.txt",
        "quality": f"{path}quality_score/Quality_score_{node_port}.txt",
        "pay_off": f"{path}pay_off/pay_off{node_port}.txt",
        "cost": f"{path}cost/cost_{node_port}.txt",
        "lambda": f"{path}lambda/lambda_{node_port}.txt",
        "transfer": f"{path}transfer/Transfer_{node_port}.txt",
    }


def gaussian_noise(grad):
    """
    可选的噪声注入接口：如需启用，请在此实现噪声逻辑。
    """
    raise NotImplementedError("gaussian_noise 未实现；请根据实验需求补充。")


def calculate_lambda(cost_list, datasize_list, port_list, quality_score_dict):
    total_cost = float(np.sum(cost_list)) if cost_list else 0.0
    denominator = 0.0
    for port, datasize in zip(port_list, datasize_list):
        quality_score = quality_score_dict.get(port, 0.0)
        denominator += quality_score * datasize
    if denominator <= 0:
        return 0.0
    return total_cost / denominator


def read_lambda_from_log(target_port, iteration):
    if iteration < 0:
        return 0.0
    lambda_path = f"{path}lambda/lambda_{target_port}.txt"
    while True:
        try:
            with open(lambda_path, "r") as lambda_log:
                for line in reversed(lambda_log.readlines()):
                    parts = line.strip().split()
                    if len(parts) != 2:
                        continue
                    try:
                        iter_idx = int(parts[0])
                        if iter_idx == iteration:
                            return float(parts[1])
                    except ValueError:
                        continue
        except FileNotFoundError:
            pass
        time.sleep(0.5)

new_error = 0.0
min_error = 1.0
min_count = 0

def run(f):
    global new_error, min_error, min_count
    config = ExperimentConfig()
    D_in = datasets.get_num_features("cifar")
    D_out = datasets.get_num_classes("cifar")
    model = returnModel(D_in, D_out, config.seed)
    print("length of the model is ", calculate_model_size(model))
    
    node = p2p.Node()
    
    from core import blockchain_instance
    import game_process
    dataset_name = get_dataset_filename(p2p.PORT, config)
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
    
    # ==== 1. 加载 Proxy Consensus Dataset (Root Dataset) ====
    # 使用 config.dataset_dir 作为 root_dir, sample_size 可根据实验调整
    root_loader = datasets.get_proxy_dataloader("cifar", config.dataset_dir, batch_size=config.batch_size, sample_size=500)

    client.TestLoss()
    new_error = client.getTestErr()
    paths = log_paths(node.PORT)
    log_loss1 = open(paths["loss"], "w")
    log_loss2 = open(paths["error"], "w")
    log_loss3 = open(paths["quality"], "w")
    log_loss4 = open(paths["pay_off"], "w")
    log_loss5 = open(paths["cost"], "w")
    log_loss6 = open(paths["lambda"], "w")
    log_loss7 = open(paths["transfer"], "w")
    blockchain = blockchain_instance

    non_committee = len(blockchain.nodes) - blockchain.committee_size
    node.broadcast(bc_enum.SERVICE * bc_enum.DESCOVERY + bc_enum.EXCHANGENODE, None)
    node_list = node.get_nodes_list()
    port_list = [addr.split(':')[1] for addr in node_list]
    for port in port_list:
        if port not in p2p.quality_score_dict:
            p2p.quality_score_dict[port] = config.quality_score_init

    
    for epoch_idx in range(config.iter_time):
        cost_to_log = 0.0
        lambda_to_log = 0.0
        # 每轮开始前选举委员会成员（已在 Blockchain 类中实现）
        node.broadcast(bc_enum.SERVICE * bc_enum.DESCOVERY + bc_enum.EXCHANGENODE, None)
        is_committee = client.is_committee_member()
        if config.send_initial_model and epoch_idx == 0:
            print("开始获取初始全局模型")
            client.send_models()
        
        Loss = client.getLoss()

        if not is_committee:
            print("此节点不是委员会成员，开始进行梯度计算并发送梯度")
            cost = 0
            if config.use_game_process:
                train_data_size, cost, payoff = game_process.decentralized_game(
                    client,
                    Loss,
                    epoch_idx,
                    lazy_once=config.lazy_game_once,
                )
                log_loss4.write(f"{epoch_idx} {payoff}\n")
                log_loss4.flush()
                cost_to_log = cost
                if config.zero_grad_when_small and train_data_size <= 128:
                    grad = torch.zeros(319242)
                    client.datasize = 0
                else:
                    client.set_train_datasize(train_data_size)
                    grad = client.getGrad()
            elif config.use_random_strategy:
                max_data_len = client.trainset.n
                random_size = random.randint(0, max_data_len)
                client.set_train_datasize(random_size)
                grad = client.getGrad()
                cost = cost_compute.compute_cost(random_size)
                cost_to_log = cost
                print(f"Epoch {epoch_idx}: [Random Strategy] Data Size set to {random_size}/{max_data_len}")
            else:
                grad = client.getGrad()
                cost = cost_compute.compute_cost(client.datasize)
                cost_to_log = cost

            if config.use_noise:
                grad = gaussian_noise(grad)
            client.send_grad_to_committee(grad, cost)
            print(f"Epoch {epoch_idx}: 梯度已发送给委员会成员。")
            if blockchain.committee:
                committee_node = next(iter(blockchain.committee))
                committee_port = committee_node.split(":")[1]
                lambda_to_log = read_lambda_from_log(committee_port, epoch_idx)
            
        else:
            # 委员会成员：收集所有非委员会成员的梯度
            time.sleep(config.wait_for_network_s)
            print(f"Epoch {epoch_idx}: 作为委员会成员，开始收集梯度。")
            while(len(p2p.grad_list) != non_committee):
                time.sleep(2)
            
            grad_recv = []
            cost_list = []
            cost_list = p2p.cost_list.copy()
            datasize_recv = p2p.datasize_list.copy()
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
            print(f"Epoch {epoch_idx}: 委员会 {p2p.PORT} 收集到 {len(grad_recv)} 个梯度。")
            
            krum_grad1 = average(grad_recv, datasize_recv)
            lambda_to_log = calculate_lambda(cost_list, datasize_recv, port_recv, p2p.quality_score_dict)
            p2p.transfer_dict = incentive(cost_list, datasize_recv, port_recv, p2p.quality_score_dict, lambda_to_log)
            for item in blockchain.committee:
                p2p.transfer_dict[item.split(':')[1]] = 0.0
            print(grad_recv)
            
            print(p2p.transfer_dict)
            print(datasize_recv)
            
            # ==== 2. 调用新的质量评估函数 (Validation Gain) ====
            update_quality_scores(grad_recv, port_recv, client, root_loader, blockchain)
            
            print('grad_receive11111========',grad_recv)
            krum_grad_bytes = pickle.dumps(krum_grad1)

            print("旧一轮质量分数列表为", p2p.quality_score_dict)
            # 进行区块链共识
            blockchain.consensus_process(krum_grad_bytes, p2p.quality_score_dict, p2p.transfer_dict)
            print(f"Epoch {epoch_idx}: 区块链共识完成。")
            
        blockchain.receive_new_block()
        
        # print("共识后的质量分数字典为", p2p.quality_score_dict)    
        # print("共识后的系统内部货币转移字典为", p2p.transfer_dict)
        
        client.TestLoss()
        new_error = client.getTestErr()
        if min_error > new_error:
            min_error = new_error.copy() if hasattr(new_error, "copy") else new_error
            min_count = 0
        else:
            min_count += 1

        log_loss1.write(f"{epoch_idx} {Loss}\n")
        log_loss2.write(f"{epoch_idx} {new_error}\n")
        log_loss3.write(f"{epoch_idx} {p2p.quality_score_dict[p2p.PORT]}\n")
        log_loss5.write(f"{epoch_idx} {cost_to_log}\n")
        log_loss6.write(f"{epoch_idx} {lambda_to_log}\n")
        log_loss7.write(f"{epoch_idx} {p2p.transfer_dict[p2p.PORT]}\n")
        # log_loss3.write(f"{iter} {0.0}\n")  # 这里的时间记录需要进一步完善
        log_loss1.flush()
        log_loss2.flush()
        log_loss3.flush()
        log_loss5.flush()
        log_loss6.flush()
        log_loss7.flush()
        
        
        # 更新客户端模型
        if blockchain.lastBlock.krumgrad:
            krum_grad = pickle.loads(blockchain.lastBlock.krumgrad)
            client.updateGrad(krum_grad)
            client.step()
            print(f"Epoch {epoch_idx}: 模型已更新。")
            print('krumgrad==========',krum_grad)
        else:
            print(f"Epoch {epoch_idx}: 错误 - 区块链的最后一个区块缺少 krumgrad。")
        # 轮次结束后的操作
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
    """
    将所有向量按照对应的数据大小进行加权平均。
    """
    vecs = [x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in vecs]
    datasize_recv1 = np.array(datasize_recv, dtype=np.float32)
    weighted_sum = np.zeros_like(vecs[0], dtype=np.float32)
    for idx, v in enumerate(vecs):
        weighted_sum += datasize_recv1[idx] * v
    total_size = np.sum(datasize_recv1)
    avg_vec = weighted_sum / total_size
    return avg_vec


# ==== 修改后的 update_quality_scores (基于 Macro-F1) ====
def update_quality_scores(
        grad_recv,          
        port_recv,          
        client,             
        root_loader,        
        blockchain,
        max_score_change=0.2, 
        init_score=1.0):    
    
    if root_loader is None:
        return

    # 1. 计算基准 Macro-F1
    base_f1 = client.evaluate_f1_on_loader(root_loader)
    print(f"Base Macro-F1 on Proxy Dataset: {base_f1:.4f}")
    
    f1_gains = []

    # 2. 遍历梯度，计算 F1 增益
    for g_i, port in zip(grad_recv, port_recv):
        if isinstance(g_i, torch.Tensor):
            g_i = g_i.cpu().numpy()
            
        client.apply_flat_update(g_i)
        new_f1 = client.evaluate_f1_on_loader(root_loader) # 使用新函数
        client.revert_flat_update(g_i)
        
        # Gain = New - Old (正值越大约好)
        marginal_gain = new_f1 - base_f1
        f1_gains.append(marginal_gain)

    # 3. 找到本轮最大的正向增益
    # F1 的变化通常在 0.01 ~ 0.1 级别，比 Accuracy 敏感
    max_gain = max([g for g in f1_gains if g > 0]) if any(g > 0 for g in f1_gains) else 0.0
    
    print(f"Max F1 Gain this round: {max_gain:.6f}")

    total_delta = 0.0
    worker_count = 0

    for gain, port in zip(f1_gains, port_recv):
        # 阈值保护：F1 增益极小时忽略，防止除零或噪声放大
        if gain > 0 and max_gain > 1e-6:
             delta_q = (gain / max_gain) * max_score_change
        else:
             delta_q = 0.0 
        
        if port not in p2p.quality_score_dict:
            p2p.quality_score_dict[port] = init_score
            
        p2p.quality_score_dict[port] += delta_q
        total_delta += delta_q
        worker_count += 1
        
        print(f"Node {port}: Gain={gain:.6f}, DeltaQ={delta_q:.4f}")

    # 4. 委员会补偿
    avg_delta = total_delta / worker_count if worker_count > 0 else 0.0
    for member_str in blockchain.committee:
        port = member_str.split(':')[1]
        if port not in p2p.quality_score_dict:
            p2p.quality_score_dict[port] = init_score
        p2p.quality_score_dict[port] += avg_delta

    print("质量分数已更新 (Macro-F1):", p2p.quality_score_dict)
