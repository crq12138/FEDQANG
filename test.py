import numpy as np
import matplotlib.pyplot as plt
import os

def setup_academic_style():
    """复刻学术风格设置，确保输出的 EPS 图表符合顶会/顶刊标准"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.linewidth'] = 1.5

def generate_mock_data(dataset_type, grid_x):
    """
    根据数据集特性合成数据
    """
    iid_ratio = np.ones_like(grid_x, dtype=float)
    
    if dataset_type == 'mnist':
        # MNIST: 基线 0.28，噪声标准差 0.08
        alpha, beta, decay_rate = 0.28, 0.72, 12.0
        base_decay = alpha + beta * np.exp(-grid_x / decay_rate)
        noise = np.random.normal(0, 0.08, size=len(grid_x))
        non_iid_ratio = base_decay + noise
        
    elif dataset_type == 'cifar10':
        # CIFAR-10: 基线 0.20，噪声标准差 0.10
        alpha, beta, decay_rate = 0.20, 0.80, 6.0
        base_decay = alpha + beta * np.exp(-grid_x / decay_rate)
        noise = np.random.normal(0, 0.10, size=len(grid_x))
        non_iid_ratio = base_decay + noise
        
    non_iid_ratio[0] = 1.0 
    non_iid_ratio = np.clip(non_iid_ratio, 0.01, 1.0)
    
    return iid_ratio, non_iid_ratio

def save_log_data(dataset_name, grid_x, iid_ratio, non_iid_ratio, log_dir='mock_logs'):
    """
    将合成的数据保存为日志文件，方便后续审查或使用原始脚本重新解析
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # 假设 IID 客户端端口号为 10000，Non-IID 为 10001 (可根据您的实际实验设定修改)
    iid_filename = os.path.join(log_dir, f'{dataset_name}_traindata_10000.txt')
    non_iid_filename = os.path.join(log_dir, f'{dataset_name}_traindata_10001.txt')
    
    # 写入 IID 数据
    with open(iid_filename, 'w', encoding='utf-8') as f:
        for r, val in zip(grid_x, iid_ratio):
            # 格式：Round: [轮次], Value: [数值]。若需严格匹配您的 parse_log_file 正则，请告知。
            f.write(f"Round: {r}, Value: {val:.6f}\n")
            
    # 写入 Non-IID 数据
    with open(non_iid_filename, 'w', encoding='utf-8') as f:
        for r, val in zip(grid_x, non_iid_ratio):
            f.write(f"Round: {r}, Value: {val:.6f}\n")
            
    print(f"[{dataset_name.upper()}] 数据日志已保存至: {log_dir}/")

def plot_simulated_dataset(dataset_name, grid_x, iid_ratio, non_iid_ratio):
    """绘制并保存 PNG 和 EPS 图像"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(grid_x, iid_ratio, label='独立同分布参与方', 
            color='tab:green', linewidth=2.0, marker='o', markersize=6)
    
    ax.plot(grid_x, non_iid_ratio, label='非独立同分布参与方', 
            color='tab:orange', linewidth=2.0, linestyle='-', marker='x', markersize=6)
    
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='初始贡献标准线')
    
    ax.set_xlabel('训练轮次', fontsize=18)
    ax.set_ylabel('数据贡献比例', fontsize=18)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.05, 1.05)
    
    ax.tick_params(axis='both', labelsize=16, length=6, width=1.5)
    
    legend = ax.legend(fontsize=16, loc='upper right', bbox_to_anchor=(0.97, 0.95), edgecolor='black')
    legend.get_frame().set_linewidth(1.5)
    
    plt.tight_layout()
    
    # 保存 PNG 和 EPS
    png_filename = f'Data_Contribution_Ratio_{dataset_name.upper()}.png'
    eps_filename = f'Data_Contribution_Ratio_{dataset_name.upper()}.eps'
    
    plt.savefig(png_filename, dpi=300)
    # 使用 bbox_inches='tight' 确保 EPS 格式下坐标轴标签不会被裁剪
    plt.savefig(eps_filename, format='eps', bbox_inches='tight') 
    
    print(f"[{dataset_name.upper()}] 图表绘制完成，已保存为 EPS 和 PNG 格式。")
    plt.close()

if __name__ == '__main__':
    setup_academic_style()
    grid_x = np.arange(0, 101, 2)
    np.random.seed(42) 
    
    for dataset in ['mnist', 'cifar10']:
        iid_ratio, non_iid_ratio = generate_mock_data(dataset, grid_x)
        save_log_data(dataset, grid_x, iid_ratio, non_iid_ratio)
        plot_simulated_dataset(dataset, grid_x, iid_ratio, non_iid_ratio)