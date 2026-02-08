import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


LOG_DIR = "./log/cnn/MEDMNIST/exp_A/class/quality_score"
OUTPUT_DIR = "./result/exp1"
OUTPUT_NAME = "medmnist_expA_quality_score_comparison"
CLIENT_PORTS = [str(p) for p in range(50051, 50070)]
BEST_CLIENT = "50069"


def setup_tifs_style() -> None:
    """Set a concise TIFS-like publication style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.linewidth": 1.0,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.5,
            "grid.linestyle": "-",
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.fancybox": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
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

    for port in CLIENT_PORTS:
        if port not in curves:
            continue
        rounds, scores = curves[port]

        if port == BEST_CLIENT:
            ax.plot(
                rounds,
                scores,
                color="#d62728",
                linewidth=2.3,
                label=f"Client {port} (Best)",
                zorder=5,
            )
        else:
            ax.plot(
                rounds,
                scores,
                color="#1f77b4",
                linewidth=1.2,
                alpha=0.5,
                zorder=2,
            )

    ax.set_xlabel("Communication Rounds")
    ax.set_ylabel("Quality Score")
    ax.set_title("MEDMNIST Exp-A: Quality Score Dynamics")
    ax.legend(loc="best", fontsize=10)

    png_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.png")
    pdf_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"[Done] Saved figure to: {png_path}")
    print(f"[Done] Saved figure to: {pdf_path}")


if __name__ == "__main__":
    plot_exp1()
