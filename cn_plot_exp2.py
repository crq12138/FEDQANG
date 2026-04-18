import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


LOG_DIR = "./log/cnn/MEDMNIST/exp_B/class/quality_score"
# OUTPUT_DIR = "./result/exp2"
OUTPUT_DIR = "./result/exp_v2/exp2"
OUTPUT_NAME = "medmnist_expB_quality_score_comparison"
CLIENT_PORTS = [str(p) for p in range(50051, 50070)]
WORST_CLIENT = "50069"
SPECIAL_CLIENT = "50067"


def setup_tifs_style() -> None:
    """Set a concise TIFS-like publication style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
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

def apply_comfortable_ylim(ax: plt.Axes, y_series: np.ndarray, lower_pad_ratio: float = 0.05, upper_pad_ratio: float = 0.4) -> None:
    """Add vertical padding so legends/markers are less likely to occlude curves."""
    valid = [np.asarray(arr, dtype=float) for arr in y_series if arr is not None and len(arr) > 0]
    if not valid:
        return
    y_min = min(float(np.nanmin(arr)) for arr in valid)
    y_max = max(float(np.nanmax(arr)) for arr in valid)
    span = max(y_max - y_min, 1e-6)
    ax.set_ylim(y_min - span * lower_pad_ratio, y_max + span * upper_pad_ratio)


def plot_exp2() -> None:
    setup_tifs_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    curves = load_all_clients()
    if not curves:
        print("[Error] No valid quality score data found.")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    # Plot highlighted participants with dedicated marker shapes.
    if WORST_CLIENT in curves:
        worst_rounds, worst_scores = curves[WORST_CLIENT]
        ax.plot(
            worst_rounds,
            worst_scores,
            color="#1f77b4",
            linewidth=2.3,
            marker="o",
            markersize=4.8,
            markevery=5,
            label="冗余参与方",
            zorder=6,
        )

    if SPECIAL_CLIENT in curves:
        special_rounds, special_scores = curves[SPECIAL_CLIENT]
        ax.plot(
            special_rounds,
            special_scores,
            color="#ff7f0e",
            linewidth=2.3,
            marker="D",
            markersize=4.8,
            markevery=5,
            label="关键参与方",
            zorder=7,
        )

    # Aggregate ordinary participants using lower/mean/upper statistics.
    ordinary_rounds_list = [
        rounds
        for port, (rounds, _) in curves.items()
        if port not in {WORST_CLIENT, SPECIAL_CLIENT}
    ]
    ordinary_rounds = (
        sorted(set(np.concatenate(ordinary_rounds_list).tolist())) if ordinary_rounds_list else []
    )

    if ordinary_rounds:
        ordinary_matrix = []
        for port, (rounds, scores) in curves.items():
            if port in {WORST_CLIENT, SPECIAL_CLIENT}:
                continue
            round_to_score = dict(zip(rounds.tolist(), scores.tolist()))
            aligned_scores = [round_to_score.get(r, np.nan) for r in ordinary_rounds]
            ordinary_matrix.append(aligned_scores)

        ordinary_array = np.array(ordinary_matrix, dtype=float)
        ordinary_mean = np.nanmean(ordinary_array, axis=0)
        ordinary_lower = np.nanmin(ordinary_array, axis=0)
        ordinary_upper = np.nanmax(ordinary_array, axis=0)
        ordinary_rounds_array = np.array(ordinary_rounds)

        ax.fill_between(
            ordinary_rounds_array,
            ordinary_lower,
            ordinary_upper,
            color="#7f7f7f",
            alpha=0.18,
            label="普通参与方范围",
            zorder=2,
        )
        ax.plot(
            ordinary_rounds_array,
            ordinary_mean,
            color="#7f7f7f",
            linewidth=2.0,
            marker="s",
            markersize=4.2,
            markevery=5,
            label="普通参与方均值",
            zorder=4,
        )
        ax.plot(
            ordinary_rounds_array,
            ordinary_lower,
            color="#7f7f7f",
            linewidth=1.3,
            linestyle="--",
            alpha=0.9,
            label="普通参与方下界",
            zorder=3,
        )
        ax.plot(
            ordinary_rounds_array,
            ordinary_upper,
            color="#7f7f7f",
            linewidth=1.3,
            linestyle=":",
            alpha=0.95,
            label="普通参与方上界",
            zorder=3,
        )

    y_for_limits = []
    if WORST_CLIENT in curves:
        y_for_limits.append(curves[WORST_CLIENT][1])
    if SPECIAL_CLIENT in curves:
        y_for_limits.append(curves[SPECIAL_CLIENT][1])
    if ordinary_rounds:
        y_for_limits.extend([ordinary_mean, ordinary_lower, ordinary_upper])
    apply_comfortable_ylim(ax, y_for_limits)

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
    plot_exp2()
