#!/usr/bin/env python3
# coding: utf-8
"""
批量运行非合作博弈实验（5个场景 x 5个参与方规模），汇总并画图。

默认会执行如下组合：
- 场景：quality_data/non_coop_quality_scenarios/1..5
- 参与方数量：5,10,15,20,25

每次组合调用：
python simulate_non_coop_game.py --scenario-profile legacy --participant-counts "xx" --total-datasize 500000

并记录：
- 每个场景配置下的总博弈轮数（total_rounds）
- 同一参与方数量下，5个场景配置的平均博弈轮数

输出：
- 明细 CSV
- 均值 CSV
- 柱状图（均值）+ 5次重复波动线
"""

import argparse
import csv
import subprocess
import sys
import shlex
from pathlib import Path
from statistics import mean
from typing import Dict, List



DEFAULT_SCENARIO_IDS = [1, 2, 3, 4, 5]
DEFAULT_PARTICIPANT_COUNTS = [5, 10, 15, 20, 25]


def run_command(cmd: List[str]) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"命令执行失败（exit={completed.returncode}）：{' '.join(cmd)}")


def read_single_row_csv(csv_path: Path) -> Dict[str, str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到文件：{csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"文件为空或缺少数据行：{csv_path}")
    if len(rows) != 1:
        raise RuntimeError(f"预期只有1条记录，实际有 {len(rows)} 条：{csv_path}")
    return rows[0]


def save_detailed_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scenario_id", "participant_count", "total_rounds", "iter_count", "all_converged"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["participant_count", "avg_total_rounds"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_bar_with_fluctuation_lines(
    out_path: Path,
    participant_counts: List[int],
    per_count_rounds: Dict[int, List[int]],
    avg_rounds: Dict[int, float],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("当前环境未安装 matplotlib，无法画图。请先安装后重试。") from exc

    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = list(range(len(participant_counts)))
    bar_values = [avg_rounds[c] for c in participant_counts]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(x, bar_values, color="#4C72B0", alpha=0.8, width=0.6, label="5次配置平均轮数")

    # 5次重复的波动线：每条线对应一个场景配置（1..5）
    repeat_num = len(next(iter(per_count_rounds.values())))
    for rep_idx in range(repeat_num):
        y_vals = [per_count_rounds[c][rep_idx] for c in participant_counts]
        plt.plot(
            x,
            y_vals,
            marker="o",
            linewidth=1.5,
            markersize=4,
            alpha=0.75,
            label=f"配置{rep_idx + 1}"
        )

    for i, bar in enumerate(bars):
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, h, f"{int(round(h))}", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, [str(c) for c in participant_counts])
    plt.xlabel("参与方个数")
    plt.ylabel("博弈次数（总轮数）")
    plt.title("非合作博弈：不同参与方规模的总轮数（5次配置重复）")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def main() -> None:
    parser = argparse.ArgumentParser(description="批量运行非合作博弈并汇总画图")
    parser.add_argument(
        "--scenario-root",
        default="quality_data/non_coop_quality_scenarios",
        help="场景根目录（其下应有1..5等子目录）",
    )
    parser.add_argument(
        "--scenario-ids",
        default="1,2,3,4,5",
        help="场景编号列表，逗号分隔",
    )
    parser.add_argument(
        "--participant-counts",
        default="5,10,15,20,25",
        help="参与方数量列表，逗号分隔",
    )
    parser.add_argument("--total-datasize", type=int, default=500000, help="传给 simulate 脚本的总数据量")
    parser.add_argument("--scenario-profile", default="legacy", choices=["legacy", "similar_distribution"])
    parser.add_argument("--output-dir", default="game_log/non_coop_batch", help="汇总输出目录")
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="python 可执行文件路径（默认使用当前解释器）",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="仅输出CSV，不画图（例如环境缺少matplotlib时）",
    )
    parser.add_argument(
        "--simulate-extra-args",
        default="",
        help="额外透传给 simulate_non_coop_game.py 的参数字符串，例如 '--max-rounds 100'",
    )
    args = parser.parse_args()

    scenario_root = Path(args.scenario_root)
    scenario_ids = [int(x.strip()) for x in args.scenario_ids.split(",") if x.strip()]
    participant_counts = [int(x.strip()) for x in args.participant_counts.split(",") if x.strip()]
    output_dir = Path(args.output_dir)

    detailed_rows: List[Dict[str, object]] = []
    per_count_rounds: Dict[int, List[int]] = {c: [] for c in participant_counts}

    for scenario_id in scenario_ids:
        scenario_dir = scenario_root / str(scenario_id)
        if not scenario_dir.exists():
            raise FileNotFoundError(f"场景目录不存在：{scenario_dir}")

        for participant_count in participant_counts:
            run_log_dir = output_dir / "runs" / f"scenario_{scenario_id}" / f"participants_{participant_count}"
            run_log_dir.mkdir(parents=True, exist_ok=True)

            extra_args = shlex.split(args.simulate_extra_args) if args.simulate_extra_args else []

            cmd = [
                args.python_bin,
                "simulate_non_coop_game.py",
                "--scenario-profile",
                args.scenario_profile,
                "--mock-base-dir",
                str(scenario_dir),
                "--participant-counts",
                str(participant_count),
                "--total-datasize",
                str(args.total_datasize),
                "--log-dir",
                str(run_log_dir),
            ] + extra_args
            run_command(cmd)

            run_summary_path = run_log_dir / f"non_coop_run_summary_participants_{participant_count}.csv"
            summary_row = read_single_row_csv(run_summary_path)
            total_rounds = int(summary_row["total_rounds"])
            iter_count = int(summary_row["iter_count"])
            all_converged = int(summary_row["all_converged"])

            detailed_rows.append(
                {
                    "scenario_id": scenario_id,
                    "participant_count": participant_count,
                    "total_rounds": total_rounds,
                    "iter_count": iter_count,
                    "all_converged": all_converged,
                }
            )
            per_count_rounds[participant_count].append(total_rounds)

            print(
                f"[OK] scenario={scenario_id}, participants={participant_count}, "
                f"total_rounds={total_rounds}, iter_count={iter_count}, all_converged={all_converged}",
                flush=True,
            )

    avg_rounds = {c: mean(per_count_rounds[c]) for c in participant_counts}

    summary_rows = [
        {"participant_count": c, "avg_total_rounds": f"{avg_rounds[c]:.6f}"}
        for c in participant_counts
    ]

    detailed_csv = output_dir / "non_coop_total_rounds_by_scenario.csv"
    summary_csv = output_dir / "non_coop_avg_total_rounds.csv"
    fig_png = output_dir / "non_coop_total_rounds_bar_with_fluctuation.png"

    save_detailed_csv(detailed_csv, detailed_rows)
    save_summary_csv(summary_csv, summary_rows)
    if args.skip_plot:
        print("[SKIP] 已按参数跳过画图，仅输出CSV。")
    else:
        plot_bar_with_fluctuation_lines(
            out_path=fig_png,
            participant_counts=participant_counts,
            per_count_rounds=per_count_rounds,
            avg_rounds=avg_rounds,
        )

    print("\n=== 完成 ===")
    print(f"明细文件: {detailed_csv}")
    print(f"平均文件: {summary_csv}")
    if args.skip_plot:
        print("图像文件: 已跳过")
    else:
        print(f"图像文件: {fig_png}")


if __name__ == "__main__":
    main()
