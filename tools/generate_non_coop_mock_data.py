#!/usr/bin/env python3
# coding: utf-8
"""生成非合作博弈 mock 质量分数场景数据。"""

import json
import math
from pathlib import Path

BASE_DIR = Path("mock_data/non_coop_quality_scenarios")
ROUNDS = 60
TOTAL_DATASIZE = 50000

SCENARIOS = {
    "mock_5_clients": 5,
    "mock_15_clients": 15,
    "mock_20_clients": 20,
    "mock_25_clients": 25,
}


def build_caps(count: int):
    base = TOTAL_DATASIZE // count
    remainder = TOTAL_DATASIZE % count
    caps = {}
    for i in range(1, count + 1):
        pid = f"C{i:02d}"
        caps[pid] = base + (1 if i <= remainder else 0)
    return caps


def quality_value(idx: int, round_idx: int, count: int) -> float:
    # 场景化设计：客户端数量越多，整体均值略降，波动略减。
    base = 0.58 + 0.22 * (idx / max(1, count - 1))
    trend = 0.08 * (1 - math.exp(-round_idx / (12 + count // 2)))
    seasonal = 0.03 * math.sin(0.22 * round_idx + idx * 0.45)
    crowd_penalty = 0.01 * (count - 10) / 15
    q = base + trend + seasonal - crowd_penalty
    return max(0.35, min(0.98, q))


def write_scenario(name: str, count: int):
    scenario_dir = BASE_DIR / name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    caps = build_caps(count)
    with (scenario_dir / "datasize_budget.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": name,
                "description": f"{count}客户端mock场景，总数据量固定50000",
                "total_datasize": TOTAL_DATASIZE,
                "caps": caps,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    for idx, pid in enumerate(caps.keys(), start=1):
        fp = scenario_dir / f"Quality_score_{pid}.txt"
        with fp.open("w", encoding="utf-8") as f:
            for r in range(1, ROUNDS + 1):
                q = quality_value(idx, r, count)
                f.write(f"{r} {q:.6f}\n")


def main():
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for name, count in SCENARIOS.items():
        write_scenario(name, count)
    print(f"generated scenarios in {BASE_DIR}")


if __name__ == "__main__":
    main()
