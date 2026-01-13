#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_transfer_summary.py
------------------------
该脚本用于读取 ./log/cnn/mixed/test/transfer 目录下的客户端转移文件，文件名形如：
    Transfer_{client_id}.txt
文件内容格式类似：
    iteration  net_money
    0          0.8914457228614308
    1          0.8204102051025512
    2          0.8999499749874937
    ...
我们会将其中每一行的 net_money 累加，得到该客户端在所有轮次的总转移金额（正为净收益，负为净支出）。

最后用柱状图展示各客户端的总转移金额。横轴为客户端 ID，纵轴为累加总额。
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取客户端 ID 列表
def read_client_ids(file_path):
    """
    从 ipport.txt 或类似文件读取客户端 ID 列表，
    假设其中每行格式为 'IP:Port'，我们只需要 Port 作为 client_id。
    """
    client_ids = []
    if not os.path.exists(file_path):
        print(f"客户端ID列表文件 {file_path} 不存在，返回空列表。")
        return client_ids

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(':')
                if len(parts) == 2:
                    client_ids.append(parts[1])
    return client_ids

# 2. 读取单个客户端的转移文件，并将第二列数值累加
def read_and_sum_transfer_file(file_path):
    """
    读取形如:
        iteration  net_money
        0          0.8914457228614308
        1          0.8204102051025512
    的日志文件，返回所有 net_money 的累加值。
    若文件不存在或为空，则返回 0.0。
    """
    total_net_money = 0.0
    if not os.path.exists(file_path):
        return total_net_money

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # parts[0] = iteration, parts[1] = net_money
            if len(parts) >= 2:
                net_money = float(parts[1])
                total_net_money += net_money
    return total_net_money

# 3. 绘制柱状图
def plot_transfer_bar(transfer_data, title, xlabel, ylabel, save_path=None):
    """
    transfer_data 是一个字典 {client_id: sum_of_transfer}
    画一张柱状图，横轴为 client_id，纵轴为 sum_of_transfer。
    """
    client_ids = list(transfer_data.keys())
    total_values = list(transfer_data.values())

    x = np.arange(len(client_ids))  # X 轴刻度位置
    plt.figure(figsize=(10, 6))

    plt.bar(x, total_values, color='blue', alpha=0.7)
    plt.xticks(x, client_ids, rotation=45)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y')

    # 可选：在柱状顶部显示数值
    for i, val in enumerate(total_values):
        # 如果 val>=0，数值标签放柱上方；否则放柱下方
        plt.text(x[i], val, f"{val:.3f}", ha='center',
                 va='bottom' if val >= 0 else 'top')

    plt.tight_layout()  # 避免标签挤在一起
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def main():
    # A) 读取客户端 ID 列表
    ipport_file = 'ipport.txt'
    client_ids = read_client_ids(ipport_file)
    if not client_ids:
        print("未读取到任何客户端ID，脚本终止。")
        return

    # B) 对每个客户端，读取转移日志并累加
    transfer_dir = './log/cnn/mixed/test/transfer'
    transfer_data = {}  # {client_id: total_net_money}

    for cid in client_ids:
        transfer_file = os.path.join(transfer_dir, f"Transfer_{cid}.txt")
        total_money = read_and_sum_transfer_file(transfer_file)
        transfer_data[cid] = total_money

    # C) 绘制柱状图，展示每个客户端的总转移金额
    plot_transfer_bar(
        transfer_data,
        title="Total Transfer for Each Client",
        xlabel="Client ID",
        ylabel="Sum of Transfers",
        save_path="./result/cnn/MNIST/mixed/test/transfer_bar.png"
    )

if __name__ == "__main__":
    main()
