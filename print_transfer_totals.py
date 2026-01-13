#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""print_transfer_totals.py
---------------------------------
Compute **total monetary transfer per client** and print the results.
This script has **no plotting code** – it only reads the pay‑off logs and
outputs a tidy table. Adjust the `path` (root experiment folder) or the
file pattern if your directory layout differs.

Usage
~~~~~
$ python print_transfer_totals.py

Author : <your‑name>
Updated: 2025‑06‑24
"""

import os
import glob
import numpy as np

# ========== 1. Experiment paths (EDIT ME if needed) ==========
path           = "log/cnn/MNIST/compare/2_low_quality/our_scheme/"
transfer_dir   = "pay_off"
transfer_glob  = "Transfer_*.txt"   # pattern for pay‑off logs
DELIMITER      = None               # use "," if files are CSV

# ========== 2. Helper to read <round index> <value> logs ==========

def read_two_col_logs(folder: str, pattern: str, value_col: int = 1):
    """Return dict {pid: np.ndarray of the chosen value column}."""
    data = {}
    for fp in sorted(glob.glob(os.path.join(folder, pattern))):
        pid = os.path.basename(fp).split("_")[-1].split(".")[0]
        arr = np.loadtxt(fp, delimiter=DELIMITER, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.shape[1] <= value_col:
            raise ValueError(f"{fp} has no column {value_col}")
        data[pid] = arr[:, value_col].ravel()
    if not data:
        raise FileNotFoundError(f"No files match {folder}/{pattern}")
    return data

# ========== 3. Main ==========

def main():
    transfer_logs = read_two_col_logs(os.path.join(path, transfer_dir),
                                      transfer_glob)

    participants    = sorted(transfer_logs.keys())
    transfer_totals = [transfer_logs[pid].sum() for pid in participants]

    # ---- Print LaTeX‑friendly single line ----
    line = " & ".join(f"{v:.4f}" for v in transfer_totals)
    sum = 0
    for t in transfer_totals:
        sum += t 
    print(sum)
    print(line)

if __name__ == "__main__":
    main()
