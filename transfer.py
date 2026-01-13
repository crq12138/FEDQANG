import time
import numpy as np
import torch

def incentive(cost_list, port_recv, krum_grad1, grad_recv):
    """
    根据本地更新在全局更新 krum_grad1 方向上的投影差值做线性变换，
    并通过双指针/顺序匹配来保证系统零和的货币转移方案。

    参数：
    --------
    cost_list : list of float
        本轮各参与方的训练成本（若为 0 则视为未实际训练）
    port_recv : list
        与 cost_list, grad_recv 对应的参与方端口号
    krum_grad1 : np.array 或 torch.Tensor
        聚合后的全局梯度，作为“方向”向量
    grad_recv : list of (np.array 或 torch.Tensor)
        各参与方的本地更新梯度

    返回：
    --------
    transfer_list : list of tuple
        形如 [(payer_port, receiver_port, transfer_amount), ...]
        表示从某个付款方节点到某个收款方节点的金额转移
    """
    print("===== Entering incentive function (projection-diff linear) =====")

    # (A) 先转换 krum_grad1 & grad_recv 为 np.array
    if isinstance(krum_grad1, torch.Tensor):
        krum_grad1 = krum_grad1.cpu().numpy()
    
    grad_nps = []
    for g in grad_recv:
        if isinstance(g, torch.Tensor):
            g = g.cpu().numpy()
        grad_nps.append(g)

    # (B) 判断 krum_grad1 是否近似全 0
    norm_krum = np.linalg.norm(krum_grad1)
    if norm_krum < 1e-12:
        print("krum_grad1 全局更新近似为 0，跳过货币转移。")
        return []

    # (C) 判断哪些梯度是全 0，用于识别明显的搭便车者
    def is_zero_grad(vec: np.ndarray) -> bool:
        return np.allclose(vec, 0.0, atol=1e-12)

    # (D) 计算非零梯度的平均梯度 average_grad
    non_zero_grads = [g for g in grad_nps if not is_zero_grad(g)]
    if len(non_zero_grads) == 0:
        print("所有梯度均为 0，无需转移。")
        return []
    
    sum_grad = np.zeros_like(grad_nps[0])
    for g in non_zero_grads:
        sum_grad += g
    average_grad = sum_grad / float(len(grad_nps))

    # (E) 计算 average_distance = average_grad 在 krum_grad1 方向上的投影
    def projection_length(vec):
        return np.dot(vec, krum_grad1) / norm_krum
    
    average_distance = projection_length(average_grad)

    # (F) 为每个客户端 i 计算 distance_i, delta_i
    #     distance_i = grad_i 在全局方向上的投影
    #     delta_i = distance_i - average_distance
    distance_list = []
    delta_list = []
    for g in grad_nps:
        d_i = projection_length(g)
        distance_list.append(d_i)
        delta_list.append(d_i - average_distance)

    # (G) 计算系统中真正进行训练的参与方的平均成本 (non-zero cost)
    #     若你想严格以 cost!=0 来界定“训练”，可直接看 cost_list
    non_zero_costs = [cost_list[i] for i, g in enumerate(grad_nps) if not is_zero_grad(g)]
    if len(non_zero_costs) == 0:
        # 说明所有人都没有报成本，都是 0
        print("本轮没有任何非零成本，视为所有人都没训练，跳过转移。")
        return []
    average_cost = sum(non_zero_costs) / len(non_zero_costs)
    print(cost_list)
    print(non_zero_costs)
    print(sum(non_zero_costs))
    print(len(non_zero_costs))
    print(average_cost)
    # time.sleep(3000)
    # (H) 若 average_distance 近似 0，则说明所有 (非零) grads 投影都差不多
    if abs(average_distance) < 1e-12:
        # 基本无差别，不需要做任何转移
        print("average_distance 近似 0，各节点在全局方向贡献差别极小，跳过转移。")
        return []

    # (I) 定义支付/收款额：|delta_i| 与 average_distance 做线性映射
    #     当 |delta_i| = average_distance => 金额 = average_cost
    #     即 cost_i = average_cost * (|delta_i| / average_distance)
    pay_or_receive_amount = []
    for i, dlt in enumerate(delta_list):
        cost_i = average_cost * (abs(dlt) / abs(average_distance))
        pay_or_receive_amount.append(cost_i)

    # (J') 不再做多笔转移匹配，而是直接计算每个节点的净额 net_transfer_dict

    net_transfer_dict = {}
    for i in range(len(grad_nps)):
        # 若 abs(delta_list[i]) < 1e-12，认为无实际支付或接收
        if abs(delta_list[i]) < 1e-12:
            net_transfer_dict[port_recv[i]] = 0.0
        else:
            # pay_or_receive_amount[i] 已根据 |delta_i| 映射到了 [0, +∞)
            # 若 delta_i < 0 => 该节点需要支付 => 记为负值
            # 若 delta_i > 0 => 该节点可以接收 => 记为正值
            sign = 1 if delta_list[i] > 0 else -1
            net_transfer_dict[port_recv[i]] = sign * pay_or_receive_amount[i]

    # (K') 直接返回 net_transfer_dict，或者转换成你需要的 list 形式
    # 例如返回 [(port, net_amount), ...]
    return net_transfer_dict

