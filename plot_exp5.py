import os
import re
import glob
import argparse
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# =============== 配置 ===============
DATASET = "MNIST"  # 可选: MNIST / MEDMNIST / CIFAR10
OUTPUT_DIR = "./result/exp5"

# 图例显示名称: 文件夹名称
STRATEGIES = {
    "FEDQANG (Ours)": "class",
    "Greedy": "greedy",
    "Random": "random",
    "Pure": "pure",
    "Lazy": "lazy",  # 若不存在会自动跳过并给出提示
}

DATASET_CONFIG = {
    "MNIST": {
        "base_dir": "./log/cnn/MNIST/exp_E",
        "fig_name": "mnist_expE_acc_convergence_comparison",
        "y_lim": (0.85, 1.005),
    },
    "MEDMNIST": {
        "base_dir": "./log/cnn/MEDMNIST/exp_E",
        "fig_name": "medmnist_expE_acc_convergence_comparison",
        "y_lim": (0.0, 1.02),
    },
    "CIFAR10": {
        "base_dir": "./log/cnn/CIFAR10/exp_E",
        "fig_name": "cifar10_expE_acc_convergence_comparison",
        "y_lim": (0.08, 0.75),
    },
}
# ====================================


def parse_error_log(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """读取单个 error 日志文件，返回 (rounds, errors)。"""
    rounds, errors = [], []

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
                e = float(parts[1])
            except ValueError:
                continue
            rounds.append(r)
            errors.append(e)

    if not rounds:
        return np.array([]), np.array([])

    pairs = sorted(zip(rounds, errors), key=lambda x: x[0])
    rounds, errors = zip(*pairs)
    return np.array(rounds), np.array(errors)


def aggregate_strategy_accuracy(strategy_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    聚合某个策略下多个客户端日志：
    - 将 error 转换为 acc = 1 - error
    - 插值到统一轮次，并进行每两轮一次的下采样
    返回: (x_rounds_downsampled, mean_acc, std_acc)
    """
    error_files = sorted(glob.glob(os.path.join(strategy_dir, "error", "Test_error_*.txt")))
    if not error_files:
        return np.array([]), np.array([]), np.array([])

    client_curves: List[np.ndarray] = []
    all_min_round = None
    all_max_round = None

    parsed_data = []
    for file in error_files:
        rounds, errors = parse_error_log(file)
        if len(rounds) == 0:
            continue

        acc = 1.0 - errors
        parsed_data.append((rounds, acc))

        cur_min, cur_max = int(rounds.min()), int(rounds.max())
        all_min_round = cur_min if all_min_round is None else min(all_min_round, cur_min)
        all_max_round = cur_max if all_max_round is None else max(all_max_round, cur_max)

    if not parsed_data:
        return np.array([]), np.array([]), np.array([])

    # 原始全量轮次
    x_rounds_full = np.arange(all_min_round, all_max_round + 1)

    for rounds, acc in parsed_data:
        y_interp = np.interp(x_rounds_full, rounds, acc)
        client_curves.append(y_interp)

    curves = np.vstack(client_curves)
    
    # --- 修改点 1: 下采样处理 (每两轮取一个点) ---
    x_rounds = x_rounds_full[::2]
    mean_acc = curves.mean(axis=0)[::2]
    std_acc = curves.std(axis=0)[::2]

    return x_rounds, mean_acc, std_acc


def setup_tifs_style() -> None:
    """设置符合学术规范的绘图风格，移除虚线。"""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.linewidth": 1.0,
        "axes.grid": True,
        # --- 修改点 2: 网格线改为实线 ---
        "grid.linestyle": "-", 
        "grid.alpha": 0.15,
        "grid.linewidth": 0.5,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.fancybox": False,
        "savefig.bbox": "tight",
    })


def plot_comparison(dataset: str) -> None:
    config = DATASET_CONFIG[dataset]
    base_dir = config["base_dir"]
    fig_name = config["fig_name"]
    y_lim = config.get("y_lim", (0.0, 1.02))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_tifs_style()

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    # --- 修改点 3: 统一 linestyle 为实线 "-" ---
    line_styles = {
        "FEDQANG (Ours)": dict(color="#d62728", linestyle="-", marker="o"),
        "Greedy":         dict(color="#ff7f0e", linestyle="-", marker="s"),
        "Random":         dict(color="#2ca02c", linestyle="-", marker="^"),
        "Pure":           dict(color="#1f77b4", linestyle="-", marker="D"),
        "Lazy":           dict(color="#9467bd", linestyle="-", marker="x"),
    }

    plotted = 0
    for label, folder in STRATEGIES.items():
        strategy_dir = os.path.join(base_dir, folder)
        if not os.path.isdir(strategy_dir):
            print(f"[Warning] 策略目录不存在: {strategy_dir}")
            continue

        rounds, mean_acc, std_acc = aggregate_strategy_accuracy(strategy_dir)
        if len(rounds) == 0:
            continue

        style = line_styles.get(label, {"linestyle": "-"})
        marker_step = max(1, len(rounds) // 10)
        marker_idx = list(range(0, len(rounds), marker_step))
        if marker_idx[-1] != len(rounds) - 1:
            marker_idx.append(len(rounds) - 1)

        ax.plot(
            rounds,
            mean_acc,
            label=label,
            linewidth=1.8, # 稍微调细一点以增加学术精致感
            markersize=5.0,
            # 调整打点频率，避免下采样后 Marker 依然过密
            markevery=marker_idx,
            **style,
        )
        ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                        color=style.get("color", "gray"), alpha=0.12, linewidth=0)
        plotted += 1

    # (后续保存逻辑保持不变...)
    ax.set_xlabel("Communication Rounds", fontsize=12, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=12, fontweight="bold")
    ax.set_ylim(*y_lim)
    ax.legend(loc="lower right", fontsize=10)
    
    # 存图
    fig.savefig(os.path.join(OUTPUT_DIR, f"{fig_name}.pdf"))
    print(f"[Done] {dataset} 数据已按 2-round 间隔采样并绘制完成。")

if __name__ == "__main__":
    plot_comparison(DATASET)
