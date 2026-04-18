import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


LOG_DIR = "./log/cnn/MEDMNIST/exp_A/class/quality_score"
# OUTPUT_DIR = "./result/exp1"
OUTPUT_DIR = "./result/exp_v2/exp1"
OUTPUT_NAME = "medmnist_expA_quality_score_comparison"
CLIENT_PORTS = [str(p) for p in range(50051, 50070)]
BEST_CLIENT = "50051"


def setup_tifs_style() -> None:
    """Set a concise TIFS-like publication style with Chinese support."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            # 将刚刚安装的 SimHei 放在列表首位
            "font.serif": ["SimHei", "Times New Roman", "Times", "DejaVu Serif"],
            "axes.unicode_minus": False,  # 确保负号正常显示
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


def parse_quality_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse one quality score file into sorted round/value arrays."""
    rounds, scores = [], []
    if not os.path.exists(filepath):
        return np.array([]), np.array([])

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().replace("'", "").split()
            if len(parts) < 2:
                continue
            try:
                # Log rounds are 0-based, but round 0 is actually the 1st round.
                # Shift by +1 so plotted rounds align with true communication rounds.
                round_id = int(parts[0])
                score = float(parts[1])
            except ValueError:
                continue
            rounds.append(round_id)
            scores.append(score)

    if not rounds:
        return np.array([]), np.array([])

    order = np.argsort(rounds)
    return np.array(rounds)[order], np.array(scores)[order]


def load_all_clients() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for port in CLIENT_PORTS:
        file_path = os.path.join(LOG_DIR, f"Quality_score_{port}.txt")
        rounds, scores = parse_quality_file(file_path)
        if len(rounds) == 0:
            print(f"[Warning] Missing or empty file: {file_path}")
            continue
        curves[port] = (rounds, scores)
    return curves


def plot_exp1() -> None:
    setup_tifs_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    curves = load_all_clients()
    if not curves:
        print("[Error] No valid quality score data found.")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    # Plot the IID participant curve with a dedicated marker shape.
    if BEST_CLIENT in curves:
        iid_rounds, iid_scores = curves[BEST_CLIENT]
        ax.plot(
            iid_rounds,
            iid_scores,
            color="#d62728",
            linewidth=2.3,
            marker="o",
            markersize=4.8,
            markevery=5,
            label="独立同分布参与方",
            zorder=6,
        )

    # Aggregate Non-IID participants using lower/mean/upper statistics.
    non_iid_rounds_list = [rounds for port, (rounds, _) in curves.items() if port != BEST_CLIENT]
    non_iid_rounds = (
        sorted(set(np.concatenate(non_iid_rounds_list).tolist())) if non_iid_rounds_list else []
    )

    if non_iid_rounds:
        non_iid_matrix = []
        for port, (rounds, scores) in curves.items():
            if port == BEST_CLIENT:
                continue
            round_to_score = dict(zip(rounds.tolist(), scores.tolist()))
            aligned_scores = [round_to_score.get(r, np.nan) for r in non_iid_rounds]
            non_iid_matrix.append(aligned_scores)

        non_iid_array = np.array(non_iid_matrix, dtype=float)
        non_iid_mean = np.nanmean(non_iid_array, axis=0)
        non_iid_lower = np.nanmin(non_iid_array, axis=0)
        non_iid_upper = np.nanmax(non_iid_array, axis=0)
        non_iid_rounds_array = np.array(non_iid_rounds)

        ax.fill_between(
            non_iid_rounds_array,
            non_iid_lower,
            non_iid_upper,
            color="#1f77b4",
            alpha=0.15,
            label="非独立同分布范围",
            zorder=2,
        )
        ax.plot(
            non_iid_rounds_array,
            non_iid_mean,
            color="#1f77b4",
            linewidth=2.0,
            marker="s",
            markersize=4.2,
            markevery=5,
            label="非独立同分布均值",
            zorder=4,
        )
        ax.plot(
            non_iid_rounds_array,
            non_iid_lower,
            color="#1f77b4",
            linewidth=1.3,
            linestyle="--",
            alpha=0.9,
            label="非独立同分布下界",
            zorder=3,
        )
        ax.plot(
            non_iid_rounds_array,
            non_iid_upper,
            color="#1f77b4",
            linewidth=1.3,
            linestyle=":",
            alpha=0.95,
            label="非独立同分布上界",
            zorder=3,
        )

    ax.set_xlabel("训练轮次")
    ax.set_ylabel("质量分数")
    ax.legend(loc="upper left", fontsize=12)

    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.png")
    eps_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.eps")
    pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(eps_path, format="eps")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[Done] Saved figure to: {png_path}")
    print(f"[Done] Saved figure to: {eps_path}")


if __name__ == "__main__":
    plot_exp1()
