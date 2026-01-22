# coding:utf-8
import threading
import time
import os
import socket
# import random
import re
import pickle
from concurrent import futures
import grpc
import blockchain
import grpc_pb2
import grpc_pb2_grpc
import bc_enum
import p2p
import client_MNIST
from core import blockchain_instance
import math
import numpy as np
from scipy.optimize import minimize_scalar
from MNIST_CNN_path import path as mnist_path
from CIFAR10_CNN_path import path as cifar10_path
from MedMNIST_CNN_path import path as medmnist_path


max_message_length = 100 * 1024 * 1024  # 设置为 100 MB，可根据需要调整
options = [
    ('grpc.max_send_message_length', max_message_length),
    ('grpc.max_receive_message_length', max_message_length),
]

# 假设我们加一个 global round_num (或在对象里管理), 表示第几轮博弈
round_num = 0
temp = 0.0
step_long = 0.4
# 用一个 dict 来存所有节点在各轮的决策: decisions[round_num][other_node_port] = datasize
decisions = dict()  # decisions = {0: {"50055": 1000, "50056": 400, ...}, 1: {...}, ...}
# decisions[0] = {}
convergence = dict()
data_contribution = dict()
port = str(p2p.PORT)
filename = medmnist_path + "datasize/traindata_" + port + ".txt"
log_data = open(filename, "w")
data_loop = 0
def decentralized_game(client, Loss, iter):
    global data_contribution, convergence
    global round_num, decisions, data_loop
    max_data_size = client.trainset.n
    # print("max_data_size: ", max_data_size)
    # time.sleep(20)
    quality_score_dict = p2p.quality_score_dict
    new_datasize = 0
    non_com_size = len(p2p.Node.get_nodes_list()) - blockchain_instance.committee_size
    if(iter == 0):
        datasize = int(client.trainset.cut/2)
    else:
        datasize = data_loop
    convergence_self = 0
    # iteration = 0
    if round_num not in decisions:
        decisions[round_num] = {}
    if port not in decisions[round_num]:
        decisions[round_num][port] = datasize
        # game_send(datasize, convergence_self, round_num)
        # 发送数据到其他参与方
        t = threading.Thread(target=game_send(datasize, convergence_self, round_num))
        t.start()
        t.join()
    while(len(decisions[round_num]) != non_com_size):
        time.sleep(2)
        print("博弈开始，目前收到的data_contribution是:", decisions[round_num])
    while True:
        convergence_self = 0
        round_num += 1
        print(f"这是第{round_num}次博弈")
        if round_num not in convergence:
            convergence[round_num] = 0
        if round_num not in decisions:
            decisions[round_num] = {}
        data_contribution = decisions[round_num-1].copy()
        print(data_contribution)
        new_datasize, new_cost, new_payoff = solve_optimal_data_contribution(data_contribution, quality_score_dict, port, None, Loss, 60.0, max_data_size)
        step = new_datasize - datasize
        new_datasize = int(step_long * step + datasize)
        # print("这一次的new_datasize大小为：", new_datasize)
        print("这一次的datasize大小为：", datasize)
        decisions[round_num][port] = new_datasize
        print("新一轮博弈的数据大小为：", new_datasize)
        if(abs(new_datasize) - abs(datasize) < 2):
            convergence_self = 1
            convergence[round_num] += 1
        datasize = new_datasize
        t = threading.Thread(target=game_send(datasize, convergence_self, round_num))
        t.start()
        t.join()
        while True:
            if len(decisions[round_num]) == non_com_size:
                break
            time.sleep(2)
            print("等待其他参与方的第", round_num, "轮博弈消息")
            print(len(decisions[round_num]))
        if(convergence[round_num] == non_com_size):
            print("博弈均衡之后的参与方数据为：")
            print(decisions[round_num])
            break
    temp = 0.0 + decisions[round_num][port]
    round_num = 0
    decisions.clear()
    convergence.clear()
    log_data.write(f"{iter} {datasize}\n")
    log_data.flush()
    data_loop = datasize
    return temp, new_cost, new_payoff
    
        
        
        

def game_send(trainsize, convergence_self, current_round):
    self_node = set()
    self_node.add(p2p.SELF_IP_PORT)
    nodes = set(p2p.Node.get_nodes_list()) - blockchain_instance.committee - self_node
    # print("Game send 阶段, 目标节点有", nodes)
    request = grpc_pb2.Gamemessage()
    request.round = current_round
    request.id = p2p.PORT
    request.msg = trainsize
    request.convergence = convergence_self
    for member in nodes:
        channel = grpc.insecure_channel(member, options=options)
        stub = grpc_pb2_grpc.GameStub(channel)
        # print(request)
        try:
            response = stub.Gamesend(request)
            # print(response.Result)
        except Exception as e:
            print("CONNECTION FAILED IN Gamesend PHASE!")
            print("Exception details:", e)
            break



def solve_optimal_data_contribution(
    all_data_contributions: dict,
    all_quality_scores: dict,
    participant_id: str,
    p_n_dict: dict = None,
    k: float = 0.1,
    T: float = 60.0,
    max_data_size: int = 0
):
    """
    求解单个参与方在本轮博弈中的最优数据贡献量 (键均为字符串).

    :param all_data_contributions:
        当前所有参与方贡献的数据大小, 例如 {"50055": 1000, "50056": 400, ...}
    :param all_quality_scores:
        当前所有参与方的质量分数 q, 例如 {"50055": 1.0, "50056": 1.05, ...}
    :param participant_id:
        当前参与方的标识(字符串形式), 例如 "50055"
    :param p_n_dict:
        (可选) 每个参与方对应的 p_n 值字典, 例如 {"50055": 7.9, "50056": 7.8, ...}
        若不提供, 则默认使用 8.0
    :param max_data_size_dict:
        (可选) 每个参与方可贡献的最大数据量, 例如 {"50055": 5000, "50056": 2000, ...}
        若不提供, 则默认使用 5000
    :param k:
        全局精确度损失参数, 默认为 0.1
    :param T:
        总时长, 默认为 60
    
    :return: 
        该参与方在本次博弈中应贡献的最优数据大小 (示例中向下取整为 int)
    """

    # 若未提供 p_n_dict, 则为所有参与方设定一个默认 p_n=8.0
    if p_n_dict is None:
        p_n_dict = {pid: 50 for pid in all_data_contributions.keys()}

    # 若未提供 max_data_size_dict, 则为所有参与方设定一个默认最大贡献量5000
    # if max_data_size_dict is None:
    #     max_data_size_dict = {pid: 5000 for pid in all_data_contributions.keys()}

    # ============== 以下为示例中的成本相关参数，可根据实际情况修改或动态输入 ==============
    cost_upload = 20/50 * 7.6 * 0.174 / 3600000   # 上传成本(示例)
    cost_download = 20/50 * 7.6 * 0.174 / 3600000 # 下载成本(示例)
    cost_investment = 0.22 * T / 3600            # 投资成本(示例)
    cost_energy_consumption = math.pow(10, -26) * 0.174 / 3600000  # 能耗(示例)
    time_upload = 0.16 / 42.06   # 上传时间(示例)
    time_download = 0.16 / 78.26 # 下载时间(示例)
    cpu_cycle_per_data_D = 0.00947555555
    local_train = 1
    # ===========================================================================

    # ------------------------------------------------------------------------
    # 1) 计算 G_k
    #    G_k = ((1 + sqrt(1 + 4*k*sum(q_i * x_i)))^2) / (4 * k^2 * sum(q_i * x_i))
    # ------------------------------------------------------------------------
    def compute_Gk(data_contributions, q_scores, k_value):
        # sum_m = Σ (q_i * x_i)
        # print(data_contributions)
        # print(q_scores)
        # time.sleep(10)
        sum_m = sum(q_scores[pid] * data_contributions[pid] for pid in data_contributions)
        if sum_m <= 0:
            # 防止分母为0或负; 可以视情况返回0或做其他处理
            return 0.0
        numerator = (1 + math.sqrt(1 + 4 * k_value * sum_m)) ** 2
        denominator = 4 * (k_value ** 2) * sum_m
        return numerator / denominator

    # ------------------------------------------------------------------------
    # 2) 计算 loss_decrease
    #    k' = 1 / sqrt( (q_n*x_n + sum_m_minus_n)*(G_k+1) ) + 1/(G_k+1)
    #    loss_decrease = k - k'
    # ------------------------------------------------------------------------
    def compute_loss_decrease(participant, x_n, data_contributions, q_scores, k_value):
        old_x = data_contributions[participant]
        data_contributions[participant] = x_n

        # 计算当前 G_k
        G_k_val = compute_Gk(data_contributions, q_scores, k_value)
        # 计算 k'
        q_n = q_scores[participant]
        # 除掉 participant 自身贡献的 sum_m_minus_n
        sum_m_minus_n = sum(
            q_scores[i] * data_contributions[i] 
            for i in data_contributions if i != participant
        )
        k_prime = 1.0 / math.sqrt((q_n * x_n + sum_m_minus_n) * (G_k_val + 1)) + 1.0 / (G_k_val + 1)
        loss_decrease_val = k_value - k_prime

        # 恢复
        data_contributions[participant] = old_x
        return G_k_val, k_prime, loss_decrease_val

    # ------------------------------------------------------------------------
    # 3) 效用函数 U = p_n * loss_decrease
    # ------------------------------------------------------------------------
    def utility_function(loss_decrease, p_n_value):
        return p_n_value * loss_decrease

    # ------------------------------------------------------------------------
    # 4) 成本函数
    # ------------------------------------------------------------------------
    def cost_function(x_n):
        # f = x_n * cpu_cycle_per_data_D * local_train / (T - (time_upload + time_download))
        processing_capacity_f = (x_n * cpu_cycle_per_data_D * local_train) / (T - time_upload - time_download)
        energy_term = cost_energy_consumption * math.pow(x_n * cpu_cycle_per_data_D * local_train, 3) \
                      / math.pow((T - time_upload - time_download), 2)
        return cost_upload + cost_download + cost_investment * processing_capacity_f + energy_term

    # ------------------------------------------------------------------------
    # 5) 定义目标函数： loss_function(x_n) = Cost_n - Utility_n
    # ------------------------------------------------------------------------
    def compute_loss_function(x_n, participant):
        # 计算损失下降量
        _, _, loss_dec = compute_loss_decrease(participant, x_n, 
                                               all_data_contributions,
                                               all_quality_scores,
                                               k)
        # 计算效用
        p_n_val = p_n_dict[participant] if participant in p_n_dict else 40
        util = utility_function(loss_dec, p_n_val)

        # 计算成本
        c_val = cost_function(x_n)

        return c_val - util

    # ------------------------------------------------------------------------
    # 6) 求解最优的 x_n (最小化 Cost-Utility, 即 maximize Utility-Cost)
    # ------------------------------------------------------------------------
    # participant_max_x = max_data_size_dict.get(participant_id, 5000)
    
    result = minimize_scalar(
        fun=lambda x: compute_loss_function(x, participant_id),
        bounds=(0, max_data_size),
        method='bounded'
    )

    optimal_x = result.x
    # 这里向下取整为 int，您也可视需求取 round 或保留 float
    optimal_x_floor = int(math.floor(optimal_x))
    final_loss = cost_function(optimal_x_floor)
    # final_payoff = compute_loss_function(optimal_x_floor, participant_id)
    # 计算效用 (Utility)
    p_n_val = p_n_dict.get(participant_id, 40)  # 获取该参与方的 p_n 值
    _, _, loss_dec = compute_loss_decrease(participant_id, optimal_x_floor,
                                        all_data_contributions,
                                        all_quality_scores, k)
    utility_val = utility_function(loss_dec, p_n_val)

    # 计算最终 pay_off (Utility - Cost)
    final_payoff = utility_val - final_loss

    return optimal_x_floor, final_loss, final_payoff




class Game(grpc_pb2_grpc.GameServicer):
    # exchange node list
    def Gamesend(self, request, context):
        global decisions, convergence
        current_round = request.round  # 发送端要写
        if current_round not in decisions:
            decisions[current_round] = {}
        decisions[current_round][request.id] = request.msg
        if current_round not in convergence:
            convergence[current_round] = 0
        convergence[current_round] += request.convergence
        # print('asdsadafasfasfasasfasf==========================')
        return grpc_pb2.TensorReceive(Result='Datasize Received Successfully from' + port)
