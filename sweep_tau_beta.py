"""
sweep_tau_beta.py
-----------------
Grid sweep over NCE temperature (tau / nce_t) and critic loss weight (beta / -b)
for Contrastive Representation Distillation on CIFAR-100.

Usage:
    python sweep_tau_beta.py          # runs full 5x5 grid
    python sweep_tau_beta.py --dry_run  # just prints the commands

Results are saved to sweep_results/results.csv and a heatmap PNG.

Designed to run on Sharanga or any single-GPU machine.
Submit via:  sbatch sweep_tau_beta.slurm
"""

import subprocess
import itertools
import os
import sys
import csv
import re
import argparse
import json
from datetime import datetime

# ─── Grid definition ─────────────────────────────────────────────────────────
# Paper default: tau=0.07, beta=0.8
# We sweep around those defaults; adjust if you want a coarser/finer grid.
TAU_VALUES  = [0.03, 0.07, 0.14, 0.30, 0.60]   # nce_t
BETA_VALUES = [0.2,  0.4,  0.8,  1.2,  1.6]    # -b (weight of L_CRD)

# ─── Fixed training settings (fast proxy run) ────────────────────────────────
TEACHER_PATH = "./save/models/resnet32x4_vanilla/ckpt_epoch_240.pth"
STUDENT      = "resnet8x4"
DISTILL      = "crd"
EPOCHS       = 30           # full run is 240; 30 gives a reliable signal
BATCH_SIZE   = 64
NUM_WORKERS  = 4
# -r: CE loss weight, -a: KD loss weight (0 = pure CRD, no KD mixing)
R_WEIGHT     = 1.0
A_WEIGHT     = 0.0
SUBSET_FRAC  = 1.0          # 1.0 = full dataset (we cut compute via epochs)
                             # set to 0.3 if you need even faster runs

OUTPUT_DIR   = "sweep_results"
RESULTS_CSV  = os.path.join(OUTPUT_DIR, "results.csv")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dry_run", action="store_true",
                   help="Print commands without running them")
    p.add_argument("--tau_values", nargs="+", type=float, default=TAU_VALUES)
    p.add_argument("--beta_values", nargs="+", type=float, default=BETA_VALUES)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--subset_frac", type=float, default=SUBSET_FRAC)
    p.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    p.add_argument("--teacher_path", type=str, default=TEACHER_PATH)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args()


def build_command(tau, beta, trial_id, args):
    """Build the train_student.py command for one (tau, beta) combination."""
    cmd = [
        sys.executable, "train_student.py",
        "--path_t",      args.teacher_path,
        "--distill",     DISTILL,
        "--model_s",     STUDENT,
        "-r",            str(R_WEIGHT),
        "-a",            str(A_WEIGHT),
        "-b",            str(beta),
        "--nce_t",       str(tau),
        "--epochs",      str(args.epochs),
        "--batch_size",  str(BATCH_SIZE),
        "--num_workers", str(args.num_workers),
        "--trial",       str(trial_id),
        "--subset_frac", str(args.subset_frac),
        "--gpu",         str(args.gpu),
        "--print_freq",  "50",   # less verbose output
    ]
    return cmd


def extract_accuracy(log_text):
    """
    Parse the best test accuracy from train_student.py stdout.
    The script prints lines like:
      * Acc@1 75.48, Acc@5 93.21
    or in some versions:
      best accuracy: 75.48
    """
    # Try '* Acc@1 XX.XX' pattern (standard RepDistiller output)
    matches = re.findall(r"\*\s+Acc@1\s+([\d.]+)", log_text)
    if matches:
        return float(matches[-1])
    # Fallback: 'best accuracy: XX.XX'
    matches = re.findall(r"best accuracy[:\s]+([\d.]+)", log_text, re.IGNORECASE)
    if matches:
        return float(matches[-1])
    # Fallback: last 'test_acc XX.XX' line
    matches = re.findall(r"test_acc\s+([\d.]+)", log_text)
    if matches:
        return float(matches[-1])
    return None


def run_sweep(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    grid = list(itertools.product(args.tau_values, args.beta_values))
    total = len(grid)
    print(f"\n{'='*60}")
    print(f"  CRD tau-beta sweep: {len(args.tau_values)} taus × {len(args.beta_values)} betas = {total} runs")
    print(f"  Epochs per run : {args.epochs}")
    print(f"  Subset fraction: {args.subset_frac}")
    print(f"  Output dir     : {OUTPUT_DIR}")
    print(f"{'='*60}\n")

    # Write CSV header
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["trial_id", "tau", "beta", "best_acc", "status", "timestamp"])

    for idx, (tau, beta) in enumerate(grid):
        trial_id = idx + 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cmd = build_command(tau, beta, trial_id, args)

        print(f"\n[{idx+1}/{total}] tau={tau:.4f}  beta={beta:.2f}  trial={trial_id}")
        print("  CMD:", " ".join(cmd))

        if args.dry_run:
            continue

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2-hour hard timeout per run
            )
            log = result.stdout + result.stderr
            acc = extract_accuracy(log)
            status = "ok" if result.returncode == 0 else f"exit{result.returncode}"

            # Save per-run log
            log_path = os.path.join(OUTPUT_DIR, f"log_tau{tau}_beta{beta}.txt")
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(log)

            print(f"  → best_acc={acc}  status={status}")

        except subprocess.TimeoutExpired:
            acc, status = None, "timeout"
            print(f"  → TIMEOUT after 2h")
        except Exception as e:
            acc, status = None, f"error:{e}"
            print(f"  → ERROR: {e}")

        # Append to CSV immediately (so partial results survive if job dies)
        with open(RESULTS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trial_id, tau, beta, acc, status, timestamp])

    print(f"\n{'='*60}")
    print(f"  Sweep complete. Results saved to {RESULTS_CSV}")
    print(f"{'='*60}\n")

    if not args.dry_run:
        plot_heatmap()


def plot_heatmap():
    """Generate a tau vs beta accuracy heatmap from the CSV."""
    try:
        import pandas as pd
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for HPC
        import matplotlib.pyplot as plt

        df = pd.read_csv(RESULTS_CSV)
        df = df[df["status"] == "ok"].copy()
        df["best_acc"] = pd.to_numeric(df["best_acc"], errors="coerce")

        pivot = df.pivot_table(index="tau", columns="beta", values="best_acc", aggfunc="mean")

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(pivot.values, aspect="auto", cmap="viridis",
                       vmin=pivot.values.min(), vmax=pivot.values.max())
        plt.colorbar(im, ax=ax, label="Best Test Accuracy (%)")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.4f}" for v in pivot.index])
        ax.set_xlabel("Beta (β, CRD loss weight)")
        ax.set_ylabel("Tau (τ, NCE temperature)")
        ax.set_title("CRD Hyperparameter Sweep: τ vs β\n(resnet32x4 → resnet8x4, CIFAR-100)")

        # Annotate each cell
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color="white" if val < pivot.values.mean() else "black",
                            fontsize=9)

        # Highlight best cell
        best_idx = np.unravel_index(np.nanargmax(pivot.values), pivot.values.shape)
        ax.add_patch(plt.Rectangle(
            (best_idx[1] - 0.5, best_idx[0] - 0.5), 1, 1,
            fill=False, edgecolor="red", lw=3, label="Best"
        ))
        ax.legend(loc="upper right")

        heatmap_path = os.path.join(OUTPUT_DIR, "heatmap_tau_beta.png")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150)
        print(f"  Heatmap saved to {heatmap_path}")

        # Also print text table
        print("\n  Accuracy table (rows=tau, cols=beta):")
        print(pivot.to_string(float_format="%.2f"))

    except ImportError as e:
        print(f"  [warn] Could not generate heatmap (missing {e}). "
              f"Results are still in {RESULTS_CSV}.")


if __name__ == "__main__":
    args = parse_args()
    run_sweep(args)
