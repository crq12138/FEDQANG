import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AVG_CSV_PATH = "./game_log/non_coop_batch/non_coop_avg_total_rounds.csv"
DETAIL_CSV_PATH = "./game_log/non_coop_batch/non_coop_total_rounds_by_scenario.csv"
OUTPUT_DIR = "./result/chinese/exp6"
OUTPUT_NAME = "non_coop_total_rounds_barline_cn"


def setup_ieee_style() -> None:
    """设置 IEEE 顶刊风格：简洁、清晰、可黑白打印。"""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["SimHei", "Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.linewidth": 1.0,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": False,
            "legend.edgecolor": "black",
            "axes.grid": False,
            "savefig.bbox": "tight",
        }
    )


def load_and_merge_data() -> pd.DataFrame:
    if not os.path.exists(AVG_CSV_PATH):
        raise FileNotFoundError(f"未找到均值文件: {AVG_CSV_PATH}")
    if not os.path.exists(DETAIL_CSV_PATH):
        raise FileNotFoundError(f"未找到明细文件: {DETAIL_CSV_PATH}")

    avg_df = pd.read_csv(AVG_CSV_PATH)
    detail_df = pd.read_csv(DETAIL_CSV_PATH)

    required_avg_cols = {"participant_count", "avg_total_rounds"}
    required_detail_cols = {"participant_count", "total_rounds"}

    if not required_avg_cols.issubset(set(avg_df.columns)):
        raise ValueError(f"均值文件缺少必要列: {required_avg_cols}")
    if not required_detail_cols.issubset(set(detail_df.columns)):
        raise ValueError(f"明细文件缺少必要列: {required_detail_cols}")

    interval_df = (
        detail_df.groupby("participant_count", as_index=False)["total_rounds"]
        .agg(min_total_rounds="min", max_total_rounds="max")
        .sort_values("participant_count")
    )

    merged = (
        avg_df.merge(interval_df, on="participant_count", how="inner")
        .sort_values("participant_count")
        .reset_index(drop=True)
    )
    return merged


def plot_exp6() -> None:
    setup_ieee_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    merged = load_and_merge_data()

    x = merged["participant_count"].to_numpy(dtype=int)
    y = merged["avg_total_rounds"].to_numpy(dtype=float)
    y_min = merged["min_total_rounds"].to_numpy(dtype=float)
    y_max = merged["max_total_rounds"].to_numpy(dtype=float)

    yerr_lower = y - y_min
    yerr_upper = y_max - y

    fig, ax = plt.subplots(figsize=(7.4, 4.9))

    bars = ax.bar(
        x,
        y,
        width=2.8,
        color="#9ecae1",
        edgecolor="black",
        linewidth=1.0,
        label="平均总轮次",
        zorder=2,
    )

    ax.errorbar(
        x,
        y,
        yerr=np.vstack([yerr_lower, yerr_upper]),
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        capthick=1.2,
        zorder=4,
        label="场景区间（最小~最大）",
    )

    ax.plot(
        x,
        y,
        color="#d62728",
        linewidth=1.8,
        marker="o",
        markersize=5,
        label="均值趋势",
        zorder=5,
    )

    for bar, val in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{int(round(val))}",
            ha="center",
            va="bottom",
            fontsize=11,
            zorder=6,
        )

    ax.set_xlabel("参与方数量")
    ax.set_ylabel("总收敛轮次")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=1)

    y_max_plot = float(np.max(y_max))
    ax.set_ylim(0, y_max_plot * 1.15)

    # 避免图例重复（errorbar + line + bar）
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", fontsize=11)

    plt.tight_layout()

    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.png")
    eps_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.eps")
    pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.pdf")

    fig.savefig(png_path, dpi=300)
    fig.savefig(eps_path, format="eps")
    fig.savefig(pdf_path, format="pdf")
    plt.close(fig)

    print(f"[Done] 图已保存: {png_path}")
    print(f"[Done] 图已保存: {eps_path}")
    print(f"[Done] 图已保存: {pdf_path}")


if __name__ == "__main__":
    plot_exp6()
