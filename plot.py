

# import matplotlib.pyplot as plt
# import numpy as np
# import os

# def read_client_ids(file_path):
#     """
#     读取客户端ID列表
#     """
#     client_ids = []
#     with open(file_path, 'r') as f:
#         for line in f:
#             line = line.strip()
#             if line:
#                 # 假设行的格式为 'IP:Port'，提取 Port 作为客户端ID
#                 parts = line.split(':')
#                 if len(parts) == 2:
#                     client_id = parts[1]
#                     client_ids.append(client_id)
#     return client_ids

# def read_log_file(file_path):
#     """
#     读取日志文件，返回迭代次数和对应的值
#     """
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

# def plot_multiple_clients(data_dict, title, xlabel, ylabel, save_path=None):
#     """
#     绘制多个客户端的数据在同一张图上
#     """
#     plt.figure(figsize=(10, 6))
#     for client_id, (iterations, values) in data_dict.items():
#         plt.plot(iterations, values, linestyle='-', label=f'Client {client_id}')
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend()
#     plt.grid(True)
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()

# def plot_average(iterations, mean_values, title, xlabel, ylabel, save_path=None):
#     """
#     绘制平均值曲线
#     """
#     plt.figure(figsize=(10, 6))
#     plt.plot(iterations, mean_values, linestyle='-', label='Average')
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.legend()
#     plt.grid(True)
#     if save_path:
#         plt.savefig(save_path)
#     plt.show()

# # 读取客户端ID列表
# client_ids = read_client_ids('ipport.txt')

# # 初始化数据字典
# loss_data = {}
# error_data = {}
# accuracy_data = {}
# time_data = {}
# quality_data = {}
# datasize_data = {}
# dir = 'formal'

# for client_id in client_ids:
#     # 设置日志文件路径
#     # loss_log_path = f'./log/cnn/unif/test/loss/loss_{client_id}.txt'
#     # # # print(loss_log_path)
#     # error_log_path = f'./log/cnn/unif/test/error/Test_error_{client_id}.txt'
#     # time_log_path = f'./log/cnn/unif/test/time/time_correspond{client_id}.txt'
    
#     # loss_log_path = f'./log/cnn/unif/10_committee_samemodel/' + dir + f'/loss/loss_{client_id}.txt'
#     # error_log_path = f'./log/cnn/unif/10_committee_samemodel/' + dir + f'/error/Test_error_{client_id}.txt'
#     # quality_log_path = f'./log/cnn/unif/10_committee_samemodel/' + dir + f'/quality_score/Quality_score_{client_id}.txt'
#     # datasize_log_path = f'./log/cnn/unif/10_committee_samemodel/' + dir + f'/datasize/traindata_{client_id}.txt'
    
    
#     loss_log_path = f'./log/cnn/non-iid/' + dir + f'/loss/loss_{client_id}.txt'
#     error_log_path = f'./log/cnn/non-iid/' + dir + f'/error/Test_error_{client_id}.txt'
#     quality_log_path = f'./log/cnn/non-iid/' + dir + f'/quality_score/Quality_score_{client_id}.txt'
#     datasize_log_path = f'./log/cnn/non-iid/' + dir + f'/datasize/traindata_{client_id}.txt'
    
#     # 检查文件是否存在
#     if not os.path.exists(loss_log_path) or not os.path.exists(error_log_path) or not os.path.exists(quality_log_path) or not os.path.exists(datasize_log_path):
#         print(f"客户端 {client_id} 的日志文件不存在，跳过该客户端。")
#         continue
    
#     # 读取日志文件
#     loss_iterations, loss_values = read_log_file(loss_log_path)
#     error_iterations, error_values = read_log_file(error_log_path)
#     # time_iterations, time_values = read_log_file(time_log_path)
#     quality_iterations, quality_values = read_log_file(quality_log_path)
    
#     # 存储数据
#     loss_data[client_id] = (loss_iterations, loss_values)
#     error_data[client_id] = (error_iterations, error_values)
#     # for v in error_values:
#     accuracy_values = [1 - error for error in error_values]
#     accuracy_data[client_id] = (error_iterations, accuracy_values)
#     # time_data[client_id] = (time_iterations, time_values)
#     quality_data[client_id] = (quality_iterations, quality_values)

# # 绘制多个客户端的 Loss 曲线
# plot_multiple_clients(
#     loss_data,
#     'Loss over Iterations',
#     'Iteration',
#     'Loss',
#     # save_path='./result/cnn/test/loss_plot.png'
#     # save_path='./result/cnn/unif/10_committee_samemodel/' + dir + '/loss_plot.png'
#     save_path='./result/cnn/non-iid/' + dir + '/loss_plot.png'
# )

# # 绘制多个客户端的 Accuracy 曲线
# plot_multiple_clients(
#     accuracy_data,
#     'Accuracy over Iterations',
#     'Iteration',
#     'Accuracy',
#     # save_path='./result/cnn/test/accuracy_plot.png'
#     # save_path='./result/cnn/unif/10_committee_samemodel/' + dir + '/accuracy_plot.png'
#     save_path='./result/cnn/non-iid/' + dir + '/accuracy_plot.png'
# )

# # 绘制多个客户端的 Error 曲线
# plot_multiple_clients(
#     error_data,
#     'Error over Iterations',
#     'Iteration',
#     'Error',
#     # save_path='./result/cnn/test/error_plot.png'
#     # save_path='./result/cnn/unif/10_committee_samemodel/' + dir + '/error_plot.png'
#     save_path='./result/cnn/non-iid/' + dir + '/error_plot.png'
# )

# plot_multiple_clients(
#     quality_data,
#     'Quality score over Iterations',
#     'Iteration',
#     'Quality',
#     # save_path='./result/cnn/test/accuracy_plot.png'
#     # save_path='./result/cnn/unif/10_committee_samemodel/' + dir + '/quality_plot.png'
#     save_path='./result/cnn/non-iid/' + dir + '/quality_plot.png'
# )

# 绘制多个客户端的 Time 曲线
# plot_multiple_clients(
#     time_data,
#     'Time over Iterations',
#     'Iteration',
#     'Time (s)',
#     save_path='./result/cnn/test/error_plot.png'
#     # save_path='./result/cnn/unif/10_committee_samemodel/' + dir + '/error_plot.png'
#     # save_path='./result/cnn/unif/10_committee/error_plot.png'
# )



import matplotlib.pyplot as plt
import numpy as np
import os

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
                # iterations.append(iteration)
                # values.append(value)
                if iteration <= 1000:  # 限制迭代次数
                    iterations.append(iteration)
                    values.append(value)
    return iterations, values

def plot_multiple_directories(data_dict, title, xlabel, ylabel, save_path=None):
    """
    绘制多个目录的数据在同一张图上
    """
    plt.figure(figsize=(10, 6))
    for dir_name, (iterations, values) in data_dict.items():
        plt.plot(iterations, values, label=f'{dir_name}')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# 定义红框中的目录路径
# base_path = './log/cnn/unif/10_committee_samemodel'
# sub_dirs = ['1_bz8_0.001', '2_bz16_0.001', '3_bz32_0.001', '4_bz64_0.001', '5_bz128_0.001', '6_bz128_0.1', '7_bz128_0.01']
base_path = './log/cnn'
sub_dirs = ['non-iid/tm', 'unif/10_committee_samemodel/formal_seed_0wd_ld0.995', 'mixed/1_5n_5i']

# 初始化数据字典
loss_data = {}
error_data = {}
accuracy_data = {}

# 遍历每个子目录读取日志文件
for sub_dir in sub_dirs:
    # 设置日志文件路径
    loss_log_path = os.path.join(base_path, sub_dir, 'loss/loss_50051.txt')
    error_log_path = os.path.join(base_path, sub_dir, 'error/Test_error_50051.txt')
    
    # 检查文件是否存在
    if not os.path.exists(loss_log_path) or not os.path.exists(error_log_path):
        print(f"目录 {sub_dir} 的日志文件不存在，跳过该目录。")
        continue

    # 读取日志文件
    loss_iterations, loss_values = read_log_file(loss_log_path)
    error_iterations, error_values = read_log_file(error_log_path)
    accuracy_values = [1 - error for error in error_values]
    
    # 存储数据
    loss_data[sub_dir] = (loss_iterations, loss_values)
    error_data[sub_dir] = (error_iterations, error_values)
    accuracy_data[sub_dir] = (error_iterations, accuracy_values)

# 绘制 Loss 曲线
plot_multiple_directories(
    loss_data,
    'Loss over Iterations (Multiple Directories)',
    'Iteration',
    'Loss',
    save_path='./result/cnn/compare/?iid/loss_plot_multiple.png'
)

# 绘制 Accuracy 曲线
plot_multiple_directories(
    accuracy_data,
    'Accuracy over Iterations (Multiple Directories)',
    'Iteration',
    'Accuracy',
    # save_path='./result/cnn/unif/10_committee_samemodel/accuracy_plot_multiple.png'
    save_path='./result/cnn/compare/?iid/accuracy_plot_multiple.png'
)

# 绘制 Error 曲线
plot_multiple_directories(
    error_data,
    'Error over Iterations (Multiple Directories)',
    'Iteration',
    'Error',
    # save_path='./result/cnn/unif/10_committee_samemodel/error_plot_multiple.png'
    save_path='./result/cnn/compare/?iid/error_plot_multiple.png'
)


# # 计算并绘制平均曲线
# def compute_average(data_dict):
#     """
#     计算平均值
#     """
#     all_values = []
#     for client_id, (iterations, values) in data_dict.items():
#         all_values.append(np.array(values))
#     # 转置矩阵以按列（同一迭代）计算平均值
#     all_values = np.vstack(all_values)
#     mean_values = np.mean(all_values, axis=0)
#     # 假设所有客户端的迭代次数相同
#     return iterations, mean_values

# # 计算并绘制平均 Loss 曲线
# iterations, mean_loss_values = compute_average(loss_data)
# plot_average(
#     iterations,
#     mean_loss_values,
#     'Average Loss over Iterations',
#     'Iteration',
#     'Loss',
#     # save_path='./result/cnn/unif/10_committee_samemodel/average_loss_plot.png'
#     # save_path='./result/cnn/unif/10_committee_samemodel/7_bz128_0.01/average_loss_plot1.png'
#     save_path='./result/cnn/unif/10_committee_samemodel/7_bz128_0.01/average_loss_plot_local.png'
# )

# 计算并绘制平均 Error 曲线
# iterations, mean_error_values = compute_average(error_data)
# plot_average(
#     iterations,
#     mean_error_values,
#     'Average Error over Iterations',
#     'Iteration',
#     'Error',
#     # save_path='./result/cnn/unif/10_committee_samemodel/average_error_plot.png'
#     # save_path='./result/cnn/unif/10_committee_samemodel/7_bz128_0.01/average_error_plot1.png'
#     save_path='./result/cnn/test/average_error_plot.png'
# )

# # 计算并绘制平均 Time 曲线
# iterations, mean_time_values = compute_average(time_data)
# plot_average(
#     iterations,
#     mean_time_values,
#     'Average Time over Iterations',
#     'Iteration',
#     'Time (s)',
#     # save_path='./result/cnn/unif/10_committee_samemodel/average_time_plot.png'
#     # save_path='./result/cnn/unif/10_committee_samemodel/7_bz128_0.01/average_time_plot1.png'
#     save_path='./result/cnn/unif/10_committee_samemodel/7_bz128_0.01/average_time_plot_local.png'
# )
