import matplotlib.pyplot as plt
import numpy as np
import os

# 定义读取数据和绘制分布图的函数
# 定义读取数据和绘制分布图的函数
def draw_stacked_bar_chart(data_path, num_clients, num_classes):
    # 存储每个客户端的标签分布
    client_label_distributions = []

    # 遍历所有客户端数据文件
    for client_id in range(num_clients):
        # client_file = os.path.join(data_path, f"mnist_unif{client_id}.npy")
        client_file = os.path.join(data_path, f"mnist_noniid_0.1_client_{client_id}.npy")
        if os.path.exists(client_file):
            # 加载客户端数据
            client_data = np.load(client_file)
            labels = client_data[:, -1]  # 假设最后一列是标签
            label_counts = np.zeros(num_classes, dtype=int)
            
            # 统计每个标签的数量
            for label in labels:
                label_counts[int(label)] += 1
            
            client_label_distributions.append(label_counts)
        else:
            print(f"数据文件 {client_file} 不存在")

    # 转换为 NumPy 数组便于处理
    client_label_distributions = np.array(client_label_distributions)

    # 绘制叠加柱状图
    save_path = os.path.join(os.getcwd(), 'save/img')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 叠加柱状图
    plt.figure(figsize=(12, 8))
    x = np.arange(num_classes)  # 标签类别
    bottom = np.zeros(num_classes)  # 用于叠加的基线
    for client_id in range(num_clients):
        plt.bar(x, client_label_distributions[client_id], bottom=bottom,
                label=f"Client {client_id}")
        bottom += client_label_distributions[client_id]  # 更新基线

    plt.xlabel("Label")
    plt.ylabel("Number of Samples")
    plt.xticks(x, [f"Class {i}" for i in range(num_classes)])
    plt.legend()
    plt.title("Label Distribution Across Clients (Stacked)")
    plt.savefig(os.path.join(save_path, 'stacked_label_distribution_clients_0.1.png'))
    plt.show()

    # 按客户端绘制叠加柱状图
    plt.figure(figsize=(12, 8))
    x = np.arange(num_clients)  # 客户端编号
    bottom = np.zeros(num_clients)  # 用于叠加的基线
    for class_id in range(num_classes):
        plt.bar(x, client_label_distributions[:, class_id], bottom=bottom,
                label=f"Class {class_id}")
        bottom += client_label_distributions[:, class_id]  # 更新基线

    plt.xlabel("Client ID")
    plt.ylabel("Number of Samples")
    plt.xticks(x, [f"Client {i}" for i in range(num_clients)])
    plt.legend()
    plt.title("Client Distribution Across Labels (Stacked)")
    plt.savefig(os.path.join(save_path, 'stacked_client_distribution_labels_0.1.png'))
    plt.show()




# 参数设置
data_directory = "./mnist/"  # 替换为你的数据文件夹路径
num_clients = 10  # 客户端数量
num_classes = 10  # 标签类别数量（MNIST为10）

# 调用函数
draw_stacked_bar_chart(data_directory, num_clients, num_classes)

# def dirichlet_split_noniid(classes,train_labels, alpha=100.0, client_number=5):
#     '''
#     参数为 alpha 的 Dirichlet 分布将数据索引划分为 n_clients 个子集
#     '''
#     n_clients = 10
#     # 总类别数
#     n_classes = train_labels.max()+1#也可以自己手动设置
#     label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
#     # 记录每个类别对应的样本下标
#     # 返回二维数组
#     class_idcs = [np.argwhere(train_labels==y).flatten()
#            for y in range(n_classes)]

#     # 定义一个空列表作最后的返回值
#     client_idcs = [[] for _ in range(n_clients)]
#     # 记录N个client分别对应样本集合的索引
#     for c, fracs in zip(class_idcs, label_distribution):
#         # np.split按照比例将类别为k的样本划分为了N个子集
#         # for i, idcs 为遍历第i个client对应样本集合的索引
#         for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
#             client_idcs[i] += [idcs]
#     client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    
#     return client_idcs

# train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_cifar10_train)
# train_client_idcs = dirichlet_split_noniid(train_dataset.classes,np.array(train_dataset.targets),alpha=100.0,n_clients=5)
