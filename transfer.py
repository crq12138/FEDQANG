import numpy as np


def incentive(cost_list, datasize_list, port_recv, quality_score_dict, lambda_value=None):
    """
    按照系统内货币转移项计算每个参与方的净转移金额:
        transfer_n = lambda * (q_n * x_n - avg_{i!=n}(q_i * x_i))
    """
    if lambda_value is None:
        total_cost = float(np.sum(cost_list)) if cost_list else 0.0
        denominator = 0.0
        for port, datasize in zip(port_recv, datasize_list):
            quality_score = quality_score_dict.get(port, 0.0)
            denominator += quality_score * datasize
        lambda_value = total_cost / denominator if denominator > 0 else 0.0

    net_transfer_dict = {}
    for idx, port in enumerate(port_recv):
        q_n = quality_score_dict.get(port, 0.0)
        x_n = datasize_list[idx]
        other_values = [
            quality_score_dict.get(other_port, 0.0) * datasize_list[j]
            for j, other_port in enumerate(port_recv)
            if other_port != port
        ]
        avg_other = sum(other_values) / len(other_values) if other_values else 0.0
        net_transfer_dict[port] = lambda_value * (q_n * x_n - avg_other)

    return net_transfer_dict
