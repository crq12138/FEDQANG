import argparse
import os
from typing import Iterable, Optional


def _extract_numeric_value(line: str) -> Optional[float]:
    """
    Extract the last numeric value from a line.
    Supports lines like: "0 0.123" or "0.123".
    """
    parts = line.strip().split()
    for token in reversed(parts):
        try:
            return float(token)
        except ValueError:
            continue
    return None


def _sum_values(lines: Iterable[str]) -> float:
    total = 0.0
    for line in lines:
        value = _extract_numeric_value(line)
        if value is not None:
            total += value
    return total


def _first_last_values(lines: Iterable[str]) -> tuple[Optional[float], Optional[float]]:
    values = [
        value
        for line in lines
        if (value := _extract_numeric_value(line)) is not None
    ]
    if not values:
        return None, None
    return values[0], values[-1]


def _resolve_file(base_dir: str, candidates: Iterable[str]) -> str:
    for rel_path in candidates:
        path = os.path.join(base_dir, rel_path)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        f"Unable to find expected file in {base_dir}. Tried: {', '.join(candidates)}"
    )


def compute_pay_off(participant_id: str, base_dir: str) -> float:
    cost_path = _resolve_file(
        base_dir,
        [
            os.path.join("cost", f"cost_{participant_id}.txt"),
            f"cost_{participant_id}.txt",
        ],
    )
    transfer_path = _resolve_file(
        base_dir,
        [
            os.path.join("transfer", f"Transfer_{participant_id}.txt"),
            os.path.join("pay_off", f"Transfer_{participant_id}.txt"),
            f"Transfer_{participant_id}.txt",
        ],
    )
    loss_path = _resolve_file(
        base_dir,
        [
            os.path.join("loss", f"loss_{participant_id}.txt"),
            f"loss_{participant_id}.txt",
        ],
    )
    pay_off_path = _resolve_file(
        base_dir,
        [
            os.path.join("pay_off", f"pay_off{participant_id}.txt"),
            os.path.join("pay_off", f"pay_off_{participant_id}.txt"),
            f"pay_off{participant_id}.txt",
            f"pay_off_{participant_id}.txt",
        ],
    )

    with open(cost_path, "r", encoding="utf-8") as cost_file:
        total_cost = _sum_values(cost_file)

    with open(transfer_path, "r", encoding="utf-8") as transfer_file:
        total_transfer = _sum_values(transfer_file)

    with open(loss_path, "r", encoding="utf-8") as loss_file:
        first_loss, last_loss = _first_last_values(loss_file)

    if first_loss is None or last_loss is None:
        raise ValueError(f"Loss file {loss_path} has no numeric values.")

    loss_decrease = first_loss - last_loss
    pay_off = loss_decrease + total_transfer - total_cost

    with open(pay_off_path, "w", encoding="utf-8") as pay_off_file:
        pay_off_file.write(f"{pay_off}\n")

    return pay_off


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute participant pay_off from logs.")
    parser.add_argument("participant", help="Participant identifier, e.g. 50051")
    parser.add_argument("folder", help="Base folder containing cost/transfer/loss logs")
    args = parser.parse_args()

    pay_off = compute_pay_off(args.participant, args.folder)
    print(f"pay_off for participant {args.participant}: {pay_off}")


if __name__ == "__main__":
    main()
