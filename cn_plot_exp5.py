import os
import re
import glob
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# =============== Configuration ===============
# OUTPUT_DIR = "./result/exp5"
OUTPUT_DIR = "./result/chinese/exp5"
SAMPLE_STEP = 5

# 图例显示名称: 文件夹名称
STRATEGIES = {
    "Our Scheme": "class",
    "Greedy": "greedy",
    "Random": "random",
    "Pure": "pure",
    "Lazy": "lazy",  # 若不存在会自动跳过并给出提示
}

# 扩展配置：新增 loss_fig_name 和 y_lim 分离
DATASET_CONFIG = {
    "MNIST": {
        "base_dir": "./log/cnn/MNIST/exp_E",
        "acc_fig_name": "mnist_expE_acc_convergence_comparison",
        "loss_fig_name": "mnist_expE_loss_convergence_comparison",
        "acc_y_lim": (0.85, 1.005),
        "loss_y_lim": None,  # None 表示自动缩放，如有需要可修改为具体元组，如 (0.0, 2.0)
    },
    "MEDMNIST": {
        "base_dir": "./log/cnn/MEDMNIST/exp_E",
        "acc_fig_name": "medmnist_expE_acc_convergence_comparison",
        "loss_fig_name": "medmnist_expE_loss_convergence_comparison",
        "acc_y_lim": (0.0, 1.02),
        "loss_y_lim": None,
    },
    "CIFAR10": {
        "base_dir": "./log/cnn/CIFAR10/exp_E",
        "acc_fig_name": "cifar10_expE_acc_convergence_comparison",
        "loss_fig_name": "cifar10_expE_loss_convergence_comparison",
        "acc_y_lim": (0.08, 0.75),
        "loss_y_lim": None,
    },
}
# ====================================


def parse_log_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取单个日志文件 (error 或 loss)，返回 (rounds, values)。"""
    rounds, values = [], []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"'", "", line)
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                r = int(parts[0])
                v = float(parts[1])
            except ValueError:
                continue
            rounds.append(r)
            values.append(v)

    if not rounds:
        return np.array([]), np.array([])

    pairs = sorted(zip(rounds, values), key=lambda x: x[0])
    rounds, values = zip(*pairs)
    return np.array(rounds), np.array(values)


def aggregate_strategy_metric(strategy_dir: str, metric_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    聚合某个策略下多个客户端的日志。
    metric_type: "acc" 或 "loss"
    返回: (x_rounds_downsampled, mean_val, std_val)
    """
    if metric_type == "acc":
        log_files = sorted(glob.glob(os.path.join(strategy_dir, "error", "Test_error_*.txt")))
    elif metric_type == "loss":
        log_files = sorted(glob.glob(os.path.join(strategy_dir, "loss", "loss_*.txt")))
    else:
        raise ValueError(f"不支持的 metric_type: {metric_type}")

    if not log_files:
        return np.array([]), np.array([]), np.array([])

    client_curves: List[np.ndarray] = []
    all_min_round = None
    all_max_round = None
    parsed_data = []

    for file in log_files:
        rounds, values = parse_log_file(file)
        if len(rounds) == 0:
            continue

        # 如果是 acc 模式，从 error 换算为 acc
        if metric_type == "acc":
            values = 1.0 - values
            
        parsed_data.append((rounds, values))

        cur_min, cur_max = int(rounds.min()), int(rounds.max())
        all_min_round = cur_min if all_min_round is None else min(all_min_round, cur_min)
        all_max_round = cur_max if all_max_round is None else max(all_max_round, cur_max)

    if not parsed_data:
        return np.array([]), np.array([]), np.array([])

    # 原始全量轮次
    x_rounds_full = np.arange(all_min_round, all_max_round + 1)

    for rounds, values in parsed_data:
        y_interp = np.interp(x_rounds_full, rounds, values)
        client_curves.append(y_interp)

    curves = np.vstack(client_curves)
    
    # --- 修改核心逻辑：确保最后一个 epoch 不被下采样截断 ---
    total_len = len(x_rounds_full)
    # 生成按步长采样的索引
    sample_indices = list(range(0, total_len, SAMPLE_STEP))
    
    # 严谨校验：如果最后一个索引不等于原始数据的最大索引，则强制追加
    if sample_indices[-1] != total_len - 1:
        sample_indices.append(total_len - 1)
        
    x_rounds = x_rounds_full[sample_indices]
    mean_val = curves.mean(axis=0)[sample_indices]
    std_val = curves.std(axis=0)[sample_indices]

    return x_rounds, mean_val, std_val


def setup_tifs_style() -> None:
    """设置符合学术规范的绘图风格，移除虚线。"""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["SimHei", "Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False, 
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.linewidth": 1.0,
            "axes.grid": False,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": False,
            "legend.edgecolor": "black",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 12,
            "ytick.major.size": 12,
            "savefig.bbox": "tight",
        }
    )


def plot_metric(dataset: str, metric_type: str) -> None:
    """
    绘制指定数据集的评估指标曲线。
    metric_type: "acc" 或 "loss"
    """
    config = DATASET_CONFIG[dataset]
    base_dir = config["base_dir"]
    
    if metric_type == "acc":
        fig_name = config["acc_fig_name"]
        y_lim = config.get("acc_y_lim", None)
        y_label = "准确率"
        legend_loc = "lower right"
    else:
        fig_name = config["loss_fig_name"]
        y_lim = config.get("loss_y_lim", None)
        y_label = "损失"
        legend_loc = "upper right" # Loss 下降，图例放右上角避免遮挡

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_tifs_style()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    # 统一 linestyle 为实线 "-"
    line_styles = {
        "Our Scheme": dict(color="#d62728", linestyle="-", marker="o"),
        "Greedy":     dict(color="#ff7f0e", linestyle="-", marker="s"),
        "Random":     dict(color="#2ca02c", linestyle="-", marker="^"),
        "Pure":       dict(color="#1f77b4", linestyle="-", marker="D"),
        "Lazy":       dict(color="#9467bd", linestyle="-", marker="x"),
    }

    plotted = 0
    for label, folder in STRATEGIES.items():
        strategy_dir = os.path.join(base_dir, folder)
        if not os.path.isdir(strategy_dir):
            if plotted == 0: # 只在开始时报一次警，避免输出太多干扰信息
                print(f"[Warning] 策略目录不存在或不完整: {strategy_dir}")
            continue

        rounds, mean_val, std_val = aggregate_strategy_metric(strategy_dir, metric_type)
        if len(rounds) == 0:
            continue

        style = line_styles.get(label, {"linestyle": "-"})
        marker_step = max(1, len(rounds) // 10)
        marker_idx = list(range(0, len(rounds), marker_step))
        if marker_idx[-1] != len(rounds) - 1:
            marker_idx.append(len(rounds) - 1)

        ax.plot(
            rounds,
            mean_val,
            label=label,
            linewidth=1.8, 
            markersize=5.0,
            markevery=marker_idx,
            **style,
        )
        ax.fill_between(rounds, mean_val - std_val, mean_val + std_val,
                        color=style.get("color", "gray"), alpha=0.12, linewidth=0)
        plotted += 1

    if plotted == 0:
        print(f"[Error] 未在 {base_dir} 中找到任何有效的 {metric_type} 曲线数据。")
        plt.close(fig)
        return

    ax.set_xlabel("训练轮次")
    ax.set_ylabel(y_label)
    
    if y_lim is not None:
        ax.set_ylim(*y_lim)
        
    ax.legend(loc=legend_loc, fontsize=12)

    png_path = os.path.join(OUTPUT_DIR, f"{fig_name}.png")
    eps_path = os.path.join(OUTPUT_DIR, f"{fig_name}.eps")
    fig.savefig(png_path, dpi=300)
    fig.savefig(eps_path, format="eps")
    plt.close(fig)

    print(f"[Done] [{dataset} - {metric_type.upper()}] 图形已保存至:")
    print(f"       -> {png_path}")
    print(f"       -> {eps_path}")


if __name__ == "__main__":
    # 批量遍历处理所有配置中的数据集
    print(f"下采样步长设定为: {SAMPLE_STEP} 轮")
    print("-" * 50)
    
    for dataset_name in DATASET_CONFIG.keys():
        print(f"正在处理数据集: {dataset_name} ...")
        plot_metric(dataset_name, metric_type="acc")
        plot_metric(dataset_name, metric_type="loss")
        print("-" * 50)
    
    print("所有数据集处理完毕！")