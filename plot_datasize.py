#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
plot_datasize_summary.py
------------------------
该脚本用于读取 ./log/cnn/mixed/test/datasize 目录下的客户端数据大小日志文件，文件名形如：
    datasize_{client_id}.txt
内容格式示例：
    iteration  datasize
    0          100
    1          128
    2          64
    ...
我们会读取并对每个客户端所有轮次的 datasize 做平均，得到“该客户端在所有轮次中的平均数据大小”。
最后以柱状图形式展示。

需要:
1. ipport.txt 中存放客户端列表，如：
   127.0.0.1:50051
   127.0.0.1:50052
   ...
2. datasize_{client_id}.txt 存放在 ./log/cnn/mixed/test/datasize/ 下
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def read_client_ids(file_path):
    """
    从 ipport.txt 或类似文件读取客户端 ID 列表。
    假设行格式为 'IP:Port'，只需要 Port 作为 client_id。
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

def read_and_avg_datasize_file(file_path):
    """
    读取形如:
        iteration  datasize
        0          100
        1          128
    的日志文件，返回其所有 datasize 的平均值。
    如果文件不存在或没有有效的数据，则返回 0.0。
    """
    sum_datasize = 0.0
    count = 0
    if not os.path.exists(file_path):
        print("here")
        return 0.0

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                # parts[0] = iteration, parts[1] = datasize
                data_val = float(parts[1])
                print(float(parts[1]))
                sum_datasize += data_val
                count += 1

    if count == 0:
        return 0.0
    return sum_datasize / count

def plot_datasize_bar(datasize_data, title, xlabel, ylabel, save_path=None):
    """
    datasize_data: dict {client_id: avg_datasize}
    绘制柱状图，横轴为 client_id，纵轴为该客户端的平均数据大小。
    """
    client_ids = list(datasize_data.keys())
    avg_values = list(datasize_data.values())

    x = np.arange(len(client_ids))  # X 轴刻度
    plt.figure(figsize=(10, 6))

    plt.bar(x, avg_values, color='blue', alpha=0.7)
    plt.xticks(x, client_ids, rotation=45)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y')

    # 可选：在柱子顶部显示数值
    for i, val in enumerate(avg_values):
        plt.text(x[i], val, f"{val:.2f}", ha='center',
                 va='bottom' if val>=0 else 'top')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def main():
    # 1) 读取客户端 ID
    ipport_file = 'ipport.txt'
    client_ids = read_client_ids(ipport_file)
    if not client_ids:
        print("未读取到任何客户端ID，脚本终止。")
        return

    # 2) 读取每个客户端的数据大小日志并计算“平均数据大小”
    # datasize_dir = './log/cnn/mixed/8_3n_7i_nochange_5000data/datasize'
    datasize_dir = './log/cnn/mixed/test/datasize'
    datasize_data = {}  # {client_id: avg_datasize}

    for cid in client_ids:
        datasize_file = os.path.join(datasize_dir, f"traindata_{cid}.txt")
        avg_size = read_and_avg_datasize_file(datasize_file)
        datasize_data[cid] = avg_size

    # 3) 绘制柱状图
    plot_datasize_bar(
        datasize_data,
        title="Average Data Size for Each Client",
        xlabel="Client ID",
        ylabel="Average Data Size",
        save_path="./result/cnn/MNIST/mixed/test/datasize_bar.png"
    )

if __name__ == "__main__":
    main()
