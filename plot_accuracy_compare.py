#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot training *loss*, *error* and *accuracy* curves for several
federated‑learning schemes (optimal, our scheme, no‑incentive,
Tang et al.).

Key features
------------
* **Config block** – one place to change paths, scheme labels / colours,
  max iterations, *sampling step*, output dir, etc.
* **Sampling** – by default we keep *every 3rd round* (step = 3) so the
  figure is less cluttered; set ``SAMPLE_STEP = 1`` to plot every point.
* **Reusable helpers** – `read_two_col_log`, `collect_metric`, `plot_metric`.
* **Robustness** – warns and skips missing files, creates output directory
  automatically, fixed RNG seed for colour palette.

Author : <your‑name>
Updated: 2025‑06‑23
"""

from __future__ import annotations
import os
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    # 基础字号
    "font.family": 'Times New Roman',            # 全局基础字体
    "font.size": 16,            # 全局基础字号
    "axes.titlesize": 24,       # 坐标系标题（ax.set_title）
    "axes.labelsize": 24,       # 轴标签（ax.set_xlabel / set_ylabel）
    "xtick.labelsize": 15,      # x轴刻度
    "ytick.labelsize": 15,      # y轴刻度
    "legend.fontsize": 15,      # 图例字号
    "legend.title_fontsize": 15,# 图例标题字号
    "figure.titlesize": 24,     # figure 级标题（若使用）

    # 版面与清晰度
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",     # 自动紧致边距，避免文字被裁切

    # EPS/PDF 使用 TrueType (Type 42)
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # 细节观感（可选）
    "axes.titlepad": 8,          # 标题与图的间距
    "axes.labelpad": 6,          # 坐标轴标签与轴的间距
})
# ═════════════════════════════ CONFIG ════════════════════════════════
CONFIG = {
    # 根路径，下面按 scheme_name → sub‑dir 组织
    "base_log_path": Path("./log/cnn/CIFAR10/compare/1_low_quality"),
    # 方案名称映射到子目录名（可增删）
    "schemes": {
        "Theoretical Opt": "optimal",
        "Our Scheme"     : "our_scheme",
        "No-Incentive"   : "without_incentive",
        "Tang et al."    : "Tang",
    },
    # 颜色 (matplotlib 兼容)
    "colours": [
        "#00B0F0",  # 天蓝   – Theoretical Opt
        "#FF3333",  # 鲜红   – Our Scheme
        "#AAAAAA",  # 灰色   – No‑Incentive
        "#2ca02c",  # 绿色   – Tang et al.
    ],
    "max_iter"    : 1000,   # 仅读取 [0, max_iter]
    "sample_step" : 2,      # ← 每 *sample_step* 轮保留 1 个点
    "output_dir"  : Path("./result/cnn/CIFAR10/1_low_quality"),
    "file_names"  : {
        "loss"  : "loss/loss_50051.txt",
        "error" : "error/Test_error_50051.txt",
    }
}
# ═════════════════════════════════════════════════════════════════════

CONFIG["output_dir"].mkdir(parents=True, exist_ok=True)

# ──────────────────────────── Helpers ────────────────────────────────

def read_two_col_log(path: Path, *, max_iter: int | None = None,
                     sample_step: int = 1) -> tuple[list[int], list[float]]:
    """Return (iterations, values) lists, optionally truncated & subsampled."""
    iterations, values = [], []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            it, val = int(parts[0]), float(parts[1])
            if max_iter is not None and it > max_iter:
                continue
            iterations.append(it)
            values.append(val)
    # subsample
    if sample_step > 1:
        iterations = iterations[::sample_step]
        values      = values[::sample_step]
    return iterations, values


def collect_metric(metric: str, *, max_iter: int, sample_step: int):
    """Return dict {scheme_label: (iterations, values)} for given metric."""
    data = {}
    for (label, sub_dir), colour in zip(CONFIG["schemes"].items(), CONFIG["colours"]):
        file_rel = CONFIG["file_names"][metric]
        log_path = CONFIG["base_log_path"] / sub_dir / file_rel
        try:
            iters, vals = read_two_col_log(log_path, max_iter=max_iter,
                                           sample_step=sample_step)
        except FileNotFoundError:
            warnings.warn(f"Missing {metric} log for scheme '{label}' -> skip")
            continue
        if metric == "error":
            # convert to accuracy on the fly
            vals = [1 - v for v in vals]
        data[label] = (iters, vals)
    return data


def _set_grid_alpha(ax, alpha: float):
    """Helper: set alpha for both x- & y-grid lines."""
    for line in ax.get_xgridlines() + ax.get_ygridlines():
        line.set_alpha(alpha)

def plot_metric(data: dict[str, tuple[list[int], list[float]]], *,
                title: str, ylabel: str, outfile_prefix: str):
    fig, ax = plt.subplots(figsize=(10, 6))

    # ---------- 曲线 ----------
    for (label, (iters, vals)), colour in zip(data.items(), CONFIG["colours"]):
        ax.plot(iters, vals, label=label, linewidth=2, color=colour)

    # ---------- 样式 ----------
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.4, linestyle="--", linewidth=0.8)   # 半透明网格
    ax.legend()

    out_dir = CONFIG["output_dir"]
    png_path = out_dir / f"{outfile_prefix}.png"
    eps_path = out_dir / f"{outfile_prefix}.eps"

    # ---------- 保存 PNG（原样半透明） ----------
    fig.savefig(png_path, dpi=300)

    # ---------- 保存 EPS（先去掉透明度） ----------
    _set_grid_alpha(ax, 1.0)          # 网格改成不透明
    fig.savefig(eps_path, format="eps")
    _set_grid_alpha(ax, 0.4)          # 恢复透明度

    plt.show()


# ──────────────────────────── Main ───────────────────────────────────
if __name__ == "__main__":
    MAX_ITER   = CONFIG["max_iter"]
    SAMPLE_STEP = CONFIG["sample_step"]

    # Loss
    loss_data = collect_metric("loss", max_iter=MAX_ITER, sample_step=SAMPLE_STEP)
    plot_metric(loss_data, title="Loss over Iterations (sampled)",
                ylabel="Loss", outfile_prefix="loss_compare")

    # Accuracy (derived from error log)
    acc_data = collect_metric("error", max_iter=MAX_ITER, sample_step=SAMPLE_STEP)
    plot_metric(acc_data, title="Accuracy over Iterations (sampled)",
                ylabel="Accuracy", outfile_prefix="accuracy_compare")
