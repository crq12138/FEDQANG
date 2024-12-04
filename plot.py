# import matplotlib.pyplot as plt
# import numpy as np
# import os

# def read_log_file(file_path):
#     iterations = []
#     values = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             parts = line.split()
#             if len(parts) >= 2:
#                 iteration = int(parts[0])
#                 value = float(parts[1])
#                 iterations.append(iteration)
#                 values.append(value)
#     return iterations, values

# def plot_data(iterations, values, title, xlabel, ylabel, save_path=None):
#     plt.figure(figsize=(10, 6))
#     plt.plot(iterations, values, marker='o', linestyle='-')
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.grid(True)
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()

# # # 设置日志文件路径
# # loss_log_path = './log/krum/loss/loss_50051.txt'
# # error_log_path = './log/krum/error/Test_error_50051.txt'
# # time_log_path = './log/krum/time/time_correspond50051.txt'
# loss_log_path = './log/cnn/unif/loss/loss_50051.txt'
# error_log_path = './log/cnn/unif/error/Test_error_50051.txt'
# time_log_path = './log/cnn/unif/time/time_correspond50051.txt'



# # 读取损失日志文件并绘制图形
# loss_iterations, loss_values = read_log_file(loss_log_path)
# plot_data(
#     loss_iterations,
#     loss_values,
#     'Loss over Iterations',
#     'Iteration',
#     'Loss',
#     save_path='./result/cnn/loss_plot.png'  # 修改保存路径
# )

# # 读取误差日志文件并绘制图形
# error_iterations, error_values = read_log_file(error_log_path)
# plot_data(
#     error_iterations,
#     error_values,
#     'Error over Iterations',
#     'Iteration',
#     'Error',
#     save_path='./result/cnn/error_plot.png'  # 修改保存路径
# )

# # 读取时间日志文件并绘制图形
# time_iterations, time_values = read_log_file(time_log_path)
# plot_data(
#     time_iterations,
#     time_values,
#     'Time over Iterations',
#     'Iteration',
#     'Time (s)',
#     save_path='./result/cnn/time_plot.png'  # 修改保存路径
# )

import matplotlib.pyplot as plt
import numpy as np
import os

def read_client_ids(file_path):
    """
    读取客户端ID列表
    """
    client_ids = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                # 假设行的格式为 'IP:Port'，提取 Port 作为客户端ID
                parts = line.split(':')
                if len(parts) == 2:
                    client_id = parts[1]
                    client_ids.append(client_id)
    return client_ids

def read_log_file(file_path):
    """
    读取日志文件，返回迭代次数和对应的值
    """
    iterations = []
    values = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                iteration = int(parts[0])
                value = float(parts[1])
                iterations.append(iteration)
                values.append(value)
    return iterations, values

def plot_multiple_clients(data_dict, title, xlabel, ylabel, save_path=None):
    """
    绘制多个客户端的数据在同一张图上
    """
    plt.figure(figsize=(10, 6))
    for client_id, (iterations, values) in data_dict.items():
        plt.plot(iterations, values, linestyle='-', label=f'Client {client_id}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_average(iterations, mean_values, title, xlabel, ylabel, save_path=None):
    """
    绘制平均值曲线
    """
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, mean_values, linestyle='-', label='Average')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 读取客户端ID列表
client_ids = read_client_ids('ipport.txt')

# 初始化数据字典
loss_data = {}
error_data = {}
time_data = {}

for client_id in client_ids:
    # 设置日志文件路径
    loss_log_path = f'./log/cnn/unif/loss/loss_{client_id}.txt'
    error_log_path = f'./log/cnn/unif/error/Test_error_{client_id}.txt'
    time_log_path = f'./log/cnn/unif/time/time_correspond{client_id}.txt'
    
    # 检查文件是否存在
    if not os.path.exists(loss_log_path) or not os.path.exists(error_log_path) or not os.path.exists(time_log_path):
        print(f"客户端 {client_id} 的日志文件不存在，跳过该客户端。")
        continue
    
    # 读取日志文件
    loss_iterations, loss_values = read_log_file(loss_log_path)
    error_iterations, error_values = read_log_file(error_log_path)
    time_iterations, time_values = read_log_file(time_log_path)
    
    # 存储数据
    loss_data[client_id] = (loss_iterations, loss_values)
    error_data[client_id] = (error_iterations, error_values)
    time_data[client_id] = (time_iterations, time_values)

# 绘制多个客户端的 Loss 曲线
plot_multiple_clients(
    loss_data,
    'Loss over Iterations',
    'Iteration',
    'Loss',
    save_path='./result/cnn/unif/loss_plot.png'
)

# 绘制多个客户端的 Error 曲线
plot_multiple_clients(
    error_data,
    'Error over Iterations',
    'Iteration',
    'Error',
    save_path='./result/cnn/unif/error_plot.png'
)

# 绘制多个客户端的 Time 曲线
plot_multiple_clients(
    time_data,
    'Time over Iterations',
    'Iteration',
    'Time (s)',
    save_path='./result/cnn/unif/time_plot.png'
)

# 计算并绘制平均曲线
def compute_average(data_dict):
    """
    计算平均值
    """
    all_values = []
    for client_id, (iterations, values) in data_dict.items():
        all_values.append(np.array(values))
    # 转置矩阵以按列（同一迭代）计算平均值
    all_values = np.vstack(all_values)
    mean_values = np.mean(all_values, axis=0)
    # 假设所有客户端的迭代次数相同
    return iterations, mean_values

# 计算并绘制平均 Loss 曲线
iterations, mean_loss_values = compute_average(loss_data)
plot_average(
    iterations,
    mean_loss_values,
    'Average Loss over Iterations',
    'Iteration',
    'Loss',
    save_path='./result/cnn/unif/average_loss_plot.png'
)

# 计算并绘制平均 Error 曲线
iterations, mean_error_values = compute_average(error_data)
plot_average(
    iterations,
    mean_error_values,
    'Average Error over Iterations',
    'Iteration',
    'Error',
    save_path='./result/cnn/unif/average_error_plot.png'
)

# 计算并绘制平均 Time 曲线
iterations, mean_time_values = compute_average(time_data)
plot_average(
    iterations,
    mean_time_values,
    'Average Time over Iterations',
    'Iteration',
    'Time (s)',
    save_path='./result/cnn/unif/average_time_plot.png'
)
