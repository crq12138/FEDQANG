import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

def natural_sort_key(s):
    """用于文件名的自然排序 (例如: client_1, client_2, ..., client_10)"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def plot_distribution(data_dir, file_prefix, title, num_classes=10, save_path=None):
    """
    读取 .npy 文件并画出堆叠柱状图
    
    参数:
        data_dir: 数据文件夹路径 (例如 './mnist_data')
        file_prefix: 文件名前缀 (例如 'mnist_' 或 'cifar10_')
        title: 图表标题
        num_classes: 总类别数 (MNIST/CIFAR10为10, PathMNIST为9)
    """
    
    # 1. 寻找匹配的文件
    search_pattern = os.path.join(data_dir, f"{file_prefix}*.npy")
    files = glob.glob(search_pattern)
    pattern = re.compile(rf"^{file_prefix}\d+\.npy$")
    files = [f for f in files if pattern.match(os.path.basename(f))]
    files.sort(key=natural_sort_key) # 排序确保 Client 0, 1, 2... 顺序正确

    # 过滤掉测试集文件 (通常含有 'test' 字符串)
    files = [f for f in files if "test" not in f]

    if not files:
        print(f"Warning: No files found in {data_dir} with prefix '{file_prefix}'")
        return

    print(f"Found {len(files)} client files in {data_dir}...")

    # 2. 读取数据并统计
    client_ids = []
    distribution_data = [] # List of dicts: [{class0: count, class1: count...}, ...]

    for f_path in files:
        # 提取 Client ID (假设文件名结尾是数字，如 mnist_3.npy)
        filename = os.path.basename(f_path)
        try:
            # 尝试从文件名提取数字作为 ID
            c_id = re.findall(r'\d+', filename)[-1]
        except IndexError:
            c_id = filename
        
        client_ids.append(f"Client {c_id}")

        # 加载 NPY 数据
        try:
            data = np.load(f_path)
            # 假设最后一列是 Label (根据您的 parser 逻辑)
            labels = data[:, -1]
            
            # 统计每个类的数量
            unique, counts = np.unique(labels, return_counts=True)
            counts_dict = dict(zip(unique, counts))
            distribution_data.append(counts_dict)
            
        except Exception as e:
            print(f"Error reading {f_path}: {e}")

    # 3. 准备绘图数据
    num_clients = len(client_ids)
    #构建矩阵: rows=classes, cols=clients
    matrix = np.zeros((num_classes, num_clients))

    for col_idx, counts_dict in enumerate(distribution_data):
        for class_id, count in counts_dict.items():
            if int(class_id) < num_classes:
                matrix[int(class_id), col_idx] = count

    # 4. 绘图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 定义颜色映射 (Tab20 适合分类较多的情况)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i) for i in np.linspace(0, 1, num_classes)]

    bottom = np.zeros(num_clients)

    for class_id in range(num_classes):
        values = matrix[class_id, :]
        ax.bar(client_ids, values, bottom=bottom, label=f'Class {class_id}', color=colors[class_id], width=0.6)
        bottom += values

    # 设置图表样式
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.set_xlabel('Client ID', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 图例放在外侧，避免遮挡
    ax.legend(title="Classes", bbox_to_anchor=(1.0, 1.0), loc='upper left')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Chart saved to {save_path}")
    
    plt.show()

# ==========================================
# 主程序入口：配置你要查看的数据集
# ==========================================
if __name__ == "__main__":
    
    # 示例 1: 查看 MNIST 分布
    # 请确保您已经运行了 mnist_parser.py 并在 ./mnist_data 下生成了数据
    # if os.path.exists("./mnist"):
    #     plot_distribution(
    #         data_dir="./mnist", 
    #         file_prefix="mnist_exp5_client_",  # 或者是 "mnist_unif_" 取决于您 parser 的设置
    #         title="MNIST Client Data Distribution",
    #         num_classes=10,
    #         save_path="mnist_dist.png"
    #     )

    # # 示例 2: 查看 CIFAR-10 分布
    if os.path.exists("./cifar-10-batches-py/cifar10"):
        plot_distribution(
            data_dir="./cifar-10-batches-py/cifar10", 
            file_prefix="cifar10_exp5_client_", 
            title="CIFAR-10 Client Data Distribution", 
            num_classes=10,
            save_path="cifar10_dist.png"
        )
        
    # # 示例 3: 查看 MedMNIST (PathMNIST) 分布
    # if os.path.exists("./medmnist"):
    #     plot_distribution(
    #         data_dir="./medmnist", 
    #         file_prefix="pathmnist_exp5_client_", 
    #         title="PathMNIST (MedMNIST) Distribution", 
    #         num_classes=9, # PathMNIST 是9类
    #         save_path="medmnist_dist.png"
    #     )