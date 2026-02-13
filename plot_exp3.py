import matplotlib.pyplot as plt
import numpy as np
import os
import re
import glob

# =================配置区域=================
LOG_DIR = './log/cnn/MEDMNIST/exp_C/class' 
OUTPUT_DIR = './result/exp3/'
DATA_DIR = './medmnist' 
IID_CLIENT_PORT = '50069'      
WINDOW_SIZE = 10  # 每5轮聚合一次
# =========================================

def setup_tifs_style() -> None:
    """Set a concise TIFS-like publication style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 15,
            "axes.labelsize": 16,
            "axes.linewidth": 1.0,
            "axes.grid": False,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": False,
            "legend.edgecolor": "black",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 8,
            "ytick.major.size": 8,
            "savefig.bbox": "tight",
        }
    )

def get_client_total_size(port_str):
    """
    根据端口号加载对应的 .npy 文件并返回数据总样本数。
    公式: index = (port - 50051) // 2
    """
    try:
        port = int(port_str)
        client_idx = (port - 50051) // 2
        
        filename = f"pathmnist_exp1_client_{client_idx}.npy"
        filepath = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(filepath):
            # 兼容性：尝试在当前目录查找
            filepath = os.path.join('.', filename)
            if not os.path.exists(filepath):
                print(f"[Warning] Data file not found for port {port}: {filename}. Using default size 1.0")
                return 1.0
            
        # 加载 npy 获取长度
        data = np.load(filepath)
        total_size = len(data)
        return total_size
    except Exception as e:
        print(f"[Error] Failed to load data for port {port_str}: {e}")
        return 1.0

def parse_log_file(filepath):
    """
    解析日志文件，返回 (Rounds, Values)
    """
    rounds = []
    values = []
    
    if not os.path.exists(filepath):
        return np.array([]), np.array([])

    with open(filepath, 'r') as f:
        content = f.read()
        # 正确移除 标签
        content_clean = re.sub(r'\'', '', content)
        
        tokens = content_clean.split()
        i = 0
        while i < len(tokens) - 1:
            try:
                r = int(tokens[i])
                val = float(tokens[i+1])
                rounds.append(r)
                values.append(val)
                i += 2
            except ValueError:
                i += 1
                
    if rounds:
        # 确保按轮次排序
        sorted_indices = np.argsort(rounds)
        return np.array(rounds)[sorted_indices], np.array(values)[sorted_indices]
    return np.array([]), np.array([])

def smooth_data_sum(rounds, values, window=5):
    """
    对数据进行每 window 轮求和聚合
    """
    new_rounds = []
    new_values = []
    
    # 确保数据对齐，从整除 window 的位置开始或者直接切分
    # 这里简单的按顺序切分
    for i in range(0, len(values), window):
        chunk_val = values[i : i+window]
        chunk_round = rounds[i : i+window]
        
        if len(chunk_val) > 0:
            # 取该窗口的最后一个轮次作为 X 轴
            new_rounds.append(chunk_round[-1])
            # 取该窗口的总和 (Sum) 作为 Y 轴
            new_values.append(np.sum(chunk_val))
            
    return np.array(new_rounds), np.array(new_values)

def plot_sustainability():
    """图1: 激励持续性 (5轮平滑版)"""
    setup_tifs_style()
    print(f"正在绘制图1 (每{WINDOW_SIZE}轮平滑)...")
    loss_path = os.path.join(LOG_DIR, 'loss', f'loss_{IID_CLIENT_PORT}.txt')
    qs_path = os.path.join(LOG_DIR, 'quality_score', f'Quality_score_{IID_CLIENT_PORT}.txt')
    
    r_loss, v_loss = parse_log_file(loss_path)
    r_qs, v_qs = parse_log_file(qs_path)
    
    if len(r_loss) < 2: 
        return

    rounds_raw = r_loss[1:]
    raw_gains = -np.diff(v_loss)
    raw_gains = np.maximum(raw_gains, 0)

    if len(v_qs) > 1:
        rounds_delta_raw = r_qs[1:]
        delta_qs = np.diff(v_qs)
    else:
        return

    r_gain_smooth, gain_smooth = smooth_data_sum(rounds_raw, raw_gains, WINDOW_SIZE)
    r_qs_smooth, qs_smooth = smooth_data_sum(rounds_delta_raw, delta_qs, WINDOW_SIZE)

    # 2. 执行平滑 (Sum pooling)
    r_gain_smooth, gain_smooth = smooth_data_sum(rounds_raw, raw_gains, WINDOW_SIZE)
    r_qs_smooth, qs_smooth     = smooth_data_sum(rounds_delta_raw, delta_qs, WINDOW_SIZE)

    # ====== 新增：用前5轮的值作为第0轮的起始点（复制第一个平滑点）======
    if len(r_gain_smooth) > 0 and r_gain_smooth[0] != 0:
        r_gain_smooth = np.insert(r_gain_smooth, 0, 0)
        gain_smooth   = np.insert(gain_smooth,   0, gain_smooth[0])

    if len(r_qs_smooth) > 0 and r_qs_smooth[0] != 0:
        r_qs_smooth = np.insert(r_qs_smooth, 0, 0)
        qs_smooth   = np.insert(qs_smooth,   0, qs_smooth[0])

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Communication Rounds', fontsize=12)
    ax1.set_ylabel(f'Accumulated Raw Gain (Every {WINDOW_SIZE} Rounds)', color=color1, fontsize=12)
    ax1.plot(r_gain_smooth, gain_smooth, color=color1, linestyle='-', marker='o',
             markersize=4, alpha=0.8, label='Raw Gain (Sum)')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(f'Accumulated Incentive Received (Every {WINDOW_SIZE} Rounds)', color=color2, fontsize=12)
    ax2.plot(r_qs_smooth, qs_smooth, color=color2, linewidth=2.5, marker='s',
             markersize=4, label='Incentive (Sum)')
    ax2.tick_params(axis='y', labelcolor=color2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=13)
    # =========================
    # 关键：统一左右 y 轴范围
    # =========================
    y_min = 0.0
    y_max_data = float(np.nanmax([np.nanmax(gain_smooth), np.nanmax(qs_smooth)]))
    if not np.isfinite(y_max_data) or y_max_data <= y_min:
        y_max_data = y_min + 1.0

    pad_ratio = 0.2          # 顶部多留 8% 空间（你可以改成 0.05/0.1）
    y_max = y_max_data * (1.0 + pad_ratio)

    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # （可选）如果你希望也显示负值，用这段替换上面的 y_min/y_max：
    # y_min = float(np.nanmin([np.nanmin(gain_smooth), np.nanmin(qs_smooth), 0.0]))
    # y_max = float(np.nanmax([np.nanmax(gain_smooth), np.nanmax(qs_smooth)]))
    # ax1.set_ylim(y_min, y_max)
    # ax2.set_ylim(y_min, y_max)
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, 'V-C_Incentive_Sustainability_Smooth.png')
    eps_path = os.path.join(OUTPUT_DIR, 'V-C_Incentive_Sustainability_Smooth.eps')
    plt.savefig(png_path, dpi=300)
    plt.savefig(eps_path, format='eps')
    print(f"图1 (平滑版) 绘制完成。已保存: {png_path}")
    print(f"图1 (平滑版) 绘制完成。已保存: {eps_path}")



def plot_data_contribution_ratio_avg():
    """
    绘制图 2: 数据贡献比例策略对比 (最终完美版)
    修正点:
    1. 使用 np.interp (线性插值) 填补数据空缺。
    2. 强制在 [0, 2, 4, ... 100] 的标准网格上取点。
    3. 保证圆点在视觉上绝对均匀分布。
    """
    print("正在绘制图2: 数据贡献比例对比 (重采样插值版)...")
    setup_tifs_style()
    datasize_dir = os.path.join(LOG_DIR, 'datasize')
    log_files = glob.glob(os.path.join(datasize_dir, 'traindata_*.txt'))
    
    iid_r, iid_ratio = np.array([]), np.array([])
    non_iid_ratios_dict = {} 
    
    # 1. 解析数据
    for log_file in log_files:
        filename = os.path.basename(log_file)
        port_match = re.search(r'traindata_(\d+).txt', filename)
        if not port_match: continue
        port = port_match.group(1)
        
        r, v = parse_log_file(log_file)
        if len(r) == 0: continue
        
        total_size = get_client_total_size(port)
        if total_size <= 0: total_size = 1.0
        ratios = v / total_size
        
        if port == IID_CLIENT_PORT:
            iid_r = r
            iid_ratio = ratios
        else:
            for round_idx, ratio in zip(r, ratios):
                if round_idx not in non_iid_ratios_dict:
                    non_iid_ratios_dict[round_idx] = []
                non_iid_ratios_dict[round_idx].append(ratio)
    
    if not non_iid_ratios_dict:
        print("未找到 Non-IID 客户端数据。")
        return

    # 2. 准备原始 Non-IID 平均数据
    non_iid_raw_r = sorted(non_iid_ratios_dict.keys())
    non_iid_raw_ratio = [np.mean(non_iid_ratios_dict[r]) for r in non_iid_raw_r]
    
    # ==== 关键修改：基于标准网格的插值重采样 ====
    
    # 定义完美的 X 轴网格：0 到 100，步长为 2
    # 这样生成的 X 轴是: [0, 2, 4, 6, ... 100]
    grid_x = np.arange(0, 101, 2)
    
    # 处理 IID 曲线 (绿色)
    # np.interp(目标X, 原始X, 原始Y) -> 自动填补缺失值
    if len(iid_r) > 1:
        iid_ratio_resampled = np.interp(grid_x, iid_r, iid_ratio)
    else:
        iid_ratio_resampled = np.zeros_like(grid_x)

    # 处理 Non-IID 曲线 (橙色)
    if len(non_iid_raw_r) > 1:
        non_iid_ratio_resampled = np.interp(grid_x, non_iid_raw_r, non_iid_raw_ratio)
    else:
        non_iid_ratio_resampled = np.zeros_like(grid_x)
        
    # ==========================================
    
    # 4. 绘图 (直接使用重采样后的 grid_x 和 _resampled 数据)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制 IID (高质量)
    # 因为 X 轴是 grid_x (均匀的)，所以点一定是均匀的
    ax.plot(grid_x, iid_ratio_resampled, label=f'IID Participant', 
            color='tab:green', linewidth=2.0, marker='o', markersize=5)
    
    # 绘制 Non-IID (低质量)
    ax.plot(grid_x, non_iid_ratio_resampled, label=f'Non-IID Participants (α=0.1)', 
            color='tab:orange', linewidth=2.0, linestyle='-', marker='x', markersize=5)
    
    # 基准线
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, label='Initial Contribution (0.5)')
    
    ax.set_xlabel('Communication Rounds', fontsize=24)
    ax.set_ylabel('Data Contribution Ratio (s_i / S_i)', fontsize=24)
    ax.set_xlim(0, 100) 
    ax.set_ylim(-0.05, 1.05) 
    ax.legend(fontsize=20, loc='upper right', bbox_to_anchor=(0.995, 0.93))
    ax.tick_params(axis='both', labelsize=24)
    plt.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, 'V-C_Data_Contribution_Ratio_Avg.png')
    eps_path = os.path.join(OUTPUT_DIR, 'V-C_Data_Contribution_Ratio_Avg.eps')
    plt.savefig(png_path, dpi=300)
    plt.savefig(eps_path, format='eps')
    print(f"图2绘制完成 (插值修正版)，已保存: {png_path}")
    print(f"图2绘制完成 (插值修正版)，已保存: {eps_path}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # plot_sustainability()
    plot_data_contribution_ratio_avg()
    print("\n所有图表绘制完成！")