#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author : crq
Updated: 2025-10-13
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    # 基础字号
    "font.family": 'Times New Roman',            # 全局基础字体
    "font.size": 16,            # 全局基础字号
    "axes.titlesize": 24,       # 坐标系标题（ax.set_title）
    "axes.labelsize": 24,       # 轴标签（ax.set_xlabel / set_ylabel）
    "xtick.labelsize": 19,      # x轴刻度
    "ytick.labelsize": 15,      # y轴刻度
    "legend.fontsize": 15,      # 图例字号
    "legend.title_fontsize": 15,# 图例标题字号
    "figure.titlesize": 24,     # figure 级标题（若使用）

    # 输出清晰度
    "figure.dpi": 120,
    "savefig.dpi": 300,

    # EPS/PDF 字体类型：用 TrueType (Type 42)，便于后期编辑
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # 额外视觉细节（可选）
    "axes.titlepad": 8,         # 标题与图之间的间距
    "axes.labelpad": 6,         # 轴标签与轴的间距
})

# ========== ① 路径与实验配置 ==========
path = "log/cnn/MNIST/compare/2_low_quality/our_scheme/"
datasize_dir, transfer_dir, quality_dir = "datasize", "pay_off", "quality_score"

datasize_glob = "traindata_*.txt"
transfer_glob = "Transfer_*.txt"
quality_glob  = "Quality_score_*.txt"

save_path = "./result/cnn/MNIST/2_low_quality/"
os.makedirs(save_path, exist_ok=True)

# ---- 可调参数 ----
max_contrib   = 5000          # 定义“最大数据贡献值”
highlight_n   = 2              # 用于 Transfer 高亮 / 理论最优剔除数
n_low_quality = highlight_n    # 理论最优中，质量最低的 n 个 client 贡献 0
sigma_frac    = 0.15           # No‑Incentive 正态分布的 σ = sigma_frac * max_contrib
rng_seed      = 44             # 固定随机种子保证复现；改 None 获得每次不同结果
DELIMITER     = None           # 如果日志用逗号分隔则改 ","  

# ========== ② 工具函数 ==========

def read_two_col_logs(folder: str, pattern: str, value_col: int = 1):
    """读取 <index> <value> 或多列 txt，返回 {pid: np.ndarray(values)}"""
    data = {}
    for fp in sorted(glob.glob(os.path.join(folder, pattern))):
        pid = os.path.basename(fp).split("_")[-1].split(".")[0]
        arr = np.loadtxt(fp, delimiter=DELIMITER, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] <= value_col:
            raise ValueError(f"{fp} 第 {value_col+1} 列不存在")
        arr = arr[arr[:, 0].argsort()]
        data[pid] = arr[:, value_col].ravel()
    if not data:
        raise FileNotFoundError(f"{folder}/{pattern} 未找到日志")
    return data

# ========== ③ 读取日志 ==========

datasize_logs = read_two_col_logs(os.path.join(path, datasize_dir), datasize_glob)
transfer_logs = read_two_col_logs(os.path.join(path, transfer_dir), transfer_glob)
quality_logs  = read_two_col_logs(os.path.join(path, quality_dir), quality_glob)

# ========== ④ 参与方列表与颜色 ==========
participants = sorted(datasize_logs.keys())
client_labels = [f"client{i+1}" for i in range(len(participants))]

colors_clients = ['#8B0000' if i < highlight_n else 'C0'
                  for i in range(len(participants))]  # 暗红 / 默认蓝，用于 Transfer

# ========== ⑤ 计算 Datasize 四方案 ==========
# 方案 1: No‑Incentive —— 正态分布 N(max/2, (sigma_frac*max)^2) 并截断到 [0, max]
rng = np.random.default_rng(rng_seed)
scheme1_raw = rng.normal(loc=max_contrib/2.0,
                         scale=sigma_frac * max_contrib,
                         size=len(participants))
scheme1 = np.clip(scheme1_raw, 0, max_contrib).tolist()

# 方案 2: Theoretical optimum —— 找出平均质量最低的 n_low_quality 个 client
avg_quality = [(pid, quality_logs[pid].mean()) for pid in participants]
low_quality_pids = {pid for pid, _ in sorted(avg_quality, key=lambda t: t[1])[:n_low_quality]}

scheme2 = [0 if pid in low_quality_pids else max_contrib for pid in participants]

# 方案 3: Our scheme —— 平均数据贡献值
scheme3 = [datasize_logs[pid].mean() for pid in participants]

# 方案 4: Tang et al.
scheme4 = [max_contrib] * len(participants)

schemes       = [scheme1, scheme2, scheme3, scheme4]
scheme_names  = ["No‑Incentive", "Theoretical Opt", "Our Scheme", "Tang et al."]
bar_colors    = ["#AAAAAA", "#00B0F0", "#FF3333", "#2ca02c"]

# ========== ⑥ Datasize 4‑Scheme Grouped Bar 图 ==========
fig, ax = plt.subplots(figsize=(12, 6))
width = 0.18
x = np.arange(len(participants))
for i, (vals, col) in enumerate(zip(schemes, bar_colors)):
    ax.bar(x + (i - 1.5) * width, vals, width,
           label=scheme_names[i],
           color=col,
           edgecolor='black', linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(client_labels)
ax.set_xlabel("Participant")
ax.set_ylabel("Average number of samples")
ax.set_title("Datasize comparison under different incentive schemes (" + str(highlight_n) + " low-quality clients)")
ax.grid(alpha=0.4, linestyle='--', linewidth=1.0)
ax.legend()
fig.tight_layout()

# —— 保存 PNG + EPS ——
for ext in ("png", "eps"):
    fig.savefig(os.path.join(save_path, f"datasize_scheme_bar.{ext}"),
                dpi=300 if ext == "png" else None,
                format=ext)

plt.show()

# ========== ⑦ Transfer 柱状图 ==========
transfer_totals = [transfer_logs[pid].sum() for pid in participants]

plt.figure(figsize=(10, 6))
plt.bar(client_labels, transfer_totals,
        color=colors_clients,
        edgecolor='black', linewidth=0.8)
plt.xlabel("Participant")
plt.ylabel("Total monetary transfer")
plt.title("Total transfer per participant")
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.4, linestyle='--', linewidth=1.0)
plt.tight_layout()

# —— 保存 PNG + EPS ——
for ext in ("png", "eps"):
    plt.savefig(os.path.join(save_path, f"transfer_bar.{ext}"),
                dpi=300 if ext == "png" else None,
                format=ext)

plt.show()

# ========== ⑧ Quality score 折线图 ==========
plt.figure(figsize=(10, 6))
for pid, vals in quality_logs.items():
    plt.plot(range(1, len(vals) + 1), vals, label=pid, linewidth=1.5)
plt.xlabel("Training round")
plt.ylabel("Quality score")
plt.title("Quality score over rounds")
plt.grid(alpha=0.4, linestyle='--', linewidth=1.0)
plt.legend(title="Participant", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()

# —— 保存 PNG + EPS ——
for ext in ("png", "eps"):
    plt.savefig(os.path.join(save_path, f"quality_line.{ext}"),
                dpi=300 if ext == "png" else None,
                format=ext)

plt.show()
