# coding:utf-8
"""
单独运行非合作博弈（去中心化数据贡献博弈）仿真实验脚本。

- 从 quality_score 目录读取参与方质量分数；
- 运行一次完整博弈直到纳什均衡（或达到最大轮次）；
- 记录每次博弈耗时与轮次到 ./game_log，便于后续画图。
"""

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class GameResult:
    game_id: int
    iter_idx: int
    converged: bool
    rounds: int
    duration_sec: float
    player_count: int
    quality_scores: Dict[str, float]
    final_datasizes: Dict[str, int]
    final_payoffs: Dict[str, float]


# ======================== 从 game_process.py 复制并整理的核心算法 ========================
def bounded_golden_section_search(func, low: float, high: float, tol: float = 1e-4, max_iter: int = 200):
    """无 scipy 环境下的 bounded 一维最小化。"""
    gr = (math.sqrt(5) - 1) / 2
    c = high - gr * (high - low)
    d = low + gr * (high - low)
    fc = func(c)
    fd = func(d)

    for _ in range(max_iter):
        if abs(high - low) < tol:
            break
        if fc < fd:
            high, d, fd = d, c, fc
            c = high - gr * (high - low)
            fc = func(c)
        else:
            low, c, fc = c, d, fd
            d = low + gr * (high - low)
            fd = func(d)
    x = (low + high) / 2
    return x


def solve_optimal_data_contribution(
    all_data_contributions: Dict[str, int],
    all_quality_scores: Dict[str, float],
    participant_id: str,
    p_n_dict: Dict[str, float] = None,
    k: float = 0.1,
    T: float = 60.0,
    max_data_size: int = 5000,
    lambda_prev: float = 0.0,
) -> Tuple[int, float, float]:
    """求解单个参与方在本轮博弈中的最优数据贡献量。"""
    if p_n_dict is None:
        p_n_dict = {pid: 3.0 for pid in all_data_contributions.keys()}

    cost_upload = 20 / 50 * 7.6 * 0.174 / 3600000
    cost_download = 20 / 50 * 7.6 * 0.174 / 3600000
    cost_investment = 0.22 * T / 3600
    cost_energy_consumption = math.pow(10, -26) * 0.174 / 3600000
    time_upload = 0.16 / 42.06
    time_download = 0.16 / 78.26
    cpu_cycle_per_data_D = 0.00947555555
    local_train = 1

    def compute_Gk(data_contributions, q_scores, k_value):
        sum_m = sum(q_scores[pid] * data_contributions[pid] for pid in data_contributions)
        if sum_m <= 0:
            return 0.0
        numerator = (1 + math.sqrt(1 + 4 * k_value * sum_m)) ** 2
        denominator = 4 * (k_value**2) * sum_m
        return numerator / denominator

    def compute_loss_decrease(participant, x_n, data_contributions, q_scores, k_value):
        old_x = data_contributions[participant]
        data_contributions[participant] = x_n

        G_k_val = compute_Gk(data_contributions, q_scores, k_value)
        q_n = q_scores[participant]
        sum_m_minus_n = sum(
            q_scores[i] * data_contributions[i] for i in data_contributions if i != participant
        )
        k_prime = 1.0 / math.sqrt((q_n * x_n + sum_m_minus_n) * (G_k_val + 1)) + 1.0 / (
            G_k_val + 1
        )
        loss_decrease_val = k_value - k_prime

        data_contributions[participant] = old_x
        return G_k_val, k_prime, loss_decrease_val

    def utility_function(loss_decrease, p_n_value):
        return p_n_value * loss_decrease

    def cost_function(x_n):
        processing_capacity_f = (x_n * cpu_cycle_per_data_D * local_train) / (
            T - time_upload - time_download
        )
        energy_term = (
            cost_energy_consumption
            * math.pow(x_n * cpu_cycle_per_data_D * local_train, 3)
            / math.pow((T - time_upload - time_download), 2)
        )
        return (
            cost_upload
            + cost_download
            + cost_investment * processing_capacity_f
            + energy_term
        )

    def transfer_term(participant, x_n, data_contributions, q_scores, lambda_value):
        if lambda_value == 0:
            return 0.0
        old_x = data_contributions[participant]
        data_contributions[participant] = x_n
        other_values = [
            q_scores[i] * data_contributions[i]
            for i in data_contributions
            if i != participant
        ]
        avg_other = sum(other_values) / len(other_values) if other_values else 0.0
        q_n = q_scores.get(participant, 0.0)
        data_contributions[participant] = old_x
        return lambda_value * (q_n * x_n - avg_other)

    def compute_loss_function(x_n, participant):
        _, _, loss_dec = compute_loss_decrease(
            participant,
            x_n,
            all_data_contributions,
            all_quality_scores,
            k,
        )
        p_n_val = p_n_dict[participant] if participant in p_n_dict else 40
        util = utility_function(loss_dec, p_n_val)
        c_val = cost_function(x_n)
        transfer_val = transfer_term(
            participant,
            x_n,
            all_data_contributions,
            all_quality_scores,
            lambda_prev,
        )
        return c_val - util - transfer_val

    optimal_x = bounded_golden_section_search(
        func=lambda x: compute_loss_function(x, participant_id),
        low=0.0,
        high=float(max_data_size),
    )
    optimal_x_floor = int(math.floor(optimal_x))
    final_loss = cost_function(optimal_x_floor)
    p_n_val = p_n_dict.get(participant_id, 40)
    _, _, loss_dec = compute_loss_decrease(
        participant_id,
        optimal_x_floor,
        all_data_contributions,
        all_quality_scores,
        k,
    )
    utility_val = utility_function(loss_dec, p_n_val)
    transfer_val = transfer_term(
        participant_id,
        optimal_x_floor,
        all_data_contributions,
        all_quality_scores,
        lambda_prev,
    )
    final_payoff = utility_val + transfer_val - final_loss

    return optimal_x_floor, final_loss, final_payoff


# ======================== 实验数据读取与记录 ========================
def parse_quality_file(file_path: Path) -> Dict[int, float]:
    values = {}
    with file_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            iter_idx = int(parts[0])
            score = float(parts[1])
            values[iter_idx] = score
    return values


def load_quality_scores(quality_dir: Path) -> Dict[str, Dict[int, float]]:
    result = {}
    for file_path in sorted(quality_dir.glob("Quality_score_*.txt")):
        player_id = file_path.stem.split("_")[-1]
        result[player_id] = parse_quality_file(file_path)
    if not result:
        raise FileNotFoundError(f"未在 {quality_dir} 找到 Quality_score_*.txt 文件")
    return result


def collect_iteration_indices(all_scores: Dict[str, Dict[int, float]]) -> List[int]:
    common = None
    for per_player in all_scores.values():
        keys = set(per_player.keys())
        common = keys if common is None else (common & keys)
    return sorted(common) if common else []


def run_non_cooperative_game(
    quality_scores: Dict[str, float],
    max_data_size: int,
    step_long: float,
    convergence_tol: int,
    max_rounds: int,
    k_loss: float,
) -> Tuple[bool, int, float, Dict[str, int], Dict[str, float]]:
    players = sorted(quality_scores.keys())
    datasizes = {pid: max_data_size // 2 for pid in players}
    payoffs = {pid: 0.0 for pid in players}

    start = time.perf_counter()
    converged = False
    rounds = 0

    while rounds < max_rounds:
        rounds += 1
        new_datasizes = {}
        converged_count = 0

        for pid in players:
            optimal_x, _, payoff = solve_optimal_data_contribution(
                all_data_contributions=datasizes,
                all_quality_scores=quality_scores,
                participant_id=pid,
                p_n_dict=None,
                k=k_loss,
                T=60.0,
                max_data_size=max_data_size,
                lambda_prev=0.0,
            )
            step = optimal_x - datasizes[pid]
            smoothed = int(step_long * step + datasizes[pid])
            new_datasizes[pid] = smoothed
            payoffs[pid] = payoff
            if abs(smoothed - datasizes[pid]) < convergence_tol:
                converged_count += 1

        datasizes = new_datasizes
        if converged_count == len(players):
            converged = True
            break

    duration = time.perf_counter() - start
    return converged, rounds, duration, datasizes, payoffs


def ensure_game_log_dir(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)


def append_csv_log(log_file: Path, row: dict) -> None:
    write_header = not log_file.exists()
    with log_file.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "game_id",
                "iter_idx",
                "converged",
                "rounds",
                "duration_sec",
                "player_count",
                "quality_scores_json",
                "final_datasizes_json",
                "final_payoffs_json",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_run_summary(log_file: Path, row: dict) -> None:
    write_header = not log_file.exists()
    with log_file.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run_timestamp_utc",
                "quality_dir",
                "iter_count",
                "total_rounds",
                "total_duration_sec",
                "avg_rounds_per_game",
                "avg_duration_sec_per_game",
                "all_converged",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="非合作博弈仿真实验")
    parser.add_argument(
        "--quality-dir",
        default="log/cnn/CIFAR10/exp_E/class/quality_score",
        help="质量分数文件目录",
    )
    parser.add_argument("--log-dir", default="./game_log", help="实验日志输出目录")
    parser.add_argument("--max-data-size", type=int, default=5000, help="单参与方最大数据量")
    parser.add_argument("--step-long", type=float, default=0.2, help="迭代平滑系数")
    parser.add_argument("--convergence-tol", type=int, default=2, help="收敛阈值")
    parser.add_argument("--max-rounds", type=int, default=1000000, help="最大博弈轮次")
    parser.add_argument("--k-loss", type=float, default=0.1, help="损失参数 k")
    parser.add_argument(
        "--iter-idx",
        type=int,
        default=None,
        help="仅运行指定训练迭代（默认运行所有可用迭代）",
    )
    args = parser.parse_args()

    quality_dir = Path(args.quality_dir)
    log_dir = Path(args.log_dir)
    ensure_game_log_dir(log_dir)

    all_scores = load_quality_scores(quality_dir)
    available_iters = collect_iteration_indices(all_scores)
    if not available_iters:
        raise RuntimeError("各参与方质量分数文件没有共同迭代编号，无法开展博弈")

    if args.iter_idx is None:
        target_iters = available_iters
    else:
        if args.iter_idx not in available_iters:
            raise ValueError(
                f"指定 iter_idx={args.iter_idx} 不在共同迭代集合中，可选范围: "
                f"{available_iters[0]}..{available_iters[-1]}"
            )
        target_iters = [args.iter_idx]

    csv_log_file = log_dir / "non_coop_game_metrics.csv"
    jsonl_log_file = log_dir / "non_coop_game_metrics.jsonl"
    run_summary_csv = log_dir / "non_coop_run_summary.csv"
    run_summary_jsonl = log_dir / "non_coop_run_summary.jsonl"

    total_rounds = 0
    total_duration = 0.0
    converged_games = 0

    for game_id, iter_idx in enumerate(target_iters, start=1):
        quality_scores = {
            pid: per_player_scores[iter_idx] for pid, per_player_scores in all_scores.items()
        }
        converged, rounds, duration, final_datasizes, final_payoffs = run_non_cooperative_game(
            quality_scores=quality_scores,
            max_data_size=args.max_data_size,
            step_long=args.step_long,
            convergence_tol=args.convergence_tol,
            max_rounds=args.max_rounds,
            k_loss=args.k_loss,
        )

        record = {
            "game_id": game_id,
            "iter_idx": iter_idx,
            "converged": int(converged),
            "rounds": rounds,
            "duration_sec": f"{duration:.6f}",
            "player_count": len(quality_scores),
            "quality_scores_json": json.dumps(quality_scores, ensure_ascii=False, sort_keys=True),
            "final_datasizes_json": json.dumps(final_datasizes, ensure_ascii=False, sort_keys=True),
            "final_payoffs_json": json.dumps(final_payoffs, ensure_ascii=False, sort_keys=True),
        }
        append_csv_log(csv_log_file, record)

        with jsonl_log_file.open("a", encoding="utf-8") as jf:
            jf.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(
            f"[game_id={game_id}] iter={iter_idx}, converged={converged}, "
            f"rounds={rounds}, duration={duration:.6f}s"
        )
        total_rounds += rounds
        total_duration += duration
        if converged:
            converged_games += 1

    run_summary_record = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "quality_dir": str(quality_dir),
        "iter_count": len(target_iters),
        "total_rounds": total_rounds,
        "total_duration_sec": f"{total_duration:.6f}",
        "avg_rounds_per_game": f"{(total_rounds / len(target_iters)):.6f}",
        "avg_duration_sec_per_game": f"{(total_duration / len(target_iters)):.6f}",
        "all_converged": int(converged_games == len(target_iters)),
    }
    append_run_summary(run_summary_csv, run_summary_record)
    with run_summary_jsonl.open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_summary_record, ensure_ascii=False) + "\n")

    print(
        "实验完成，总博弈轮次="
        f"{total_rounds}，总时间={total_duration:.6f}s；日志文件："
        f"{csv_log_file}、{jsonl_log_file}、{run_summary_csv}、{run_summary_jsonl}"
    )


if __name__ == "__main__":
    main()