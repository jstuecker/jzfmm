import argparse
import csv
import json
from pathlib import Path

import numpy as np

from run_scaling_benchmark import (
    PARTICLE_COUNTS, BEST_INITIAL_SEEDS, output_files, summarize, validate_config,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--accuracy", type=int, choices=range(3), default=1)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--target-seeds", type=int, nargs="+", default=list(range(10)), choices=range(10))
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    logs = args.output_directory or directory / "logs" / (
        "scaling_benchmark" if args.accuracy == 1 else f"scaling_benchmark_accuracy{args.accuracy}"
    )
    rows = []

    for N in PARTICLE_COUNTS:
        results = []
        for target_seed in args.target_seeds:
            npz_file, config_file = output_files(logs, target_seed, N)
            if not npz_file.exists():
                raise FileNotFoundError(npz_file)
            validate_config(config_file, target_seed, BEST_INITIAL_SEEDS[target_seed], N, args.accuracy)
            results.append(summarize(npz_file))

        evaluations = np.asarray([result["evaluations"] for result in results])
        total = np.asarray([result["total_run_seconds"] for result in results])
        initialization = np.asarray([result["initialization_seconds"] for result in results])
        rows.append({
            "N": N,
            "mean_evaluations": float(evaluations.mean()),
            "std_evaluations": float(evaluations.std(ddof=1)),
            "mean_total_seconds": float(total.mean()),
            "std_total_seconds": float(total.std(ddof=1)),
            "mean_seconds_per_evaluation": float(np.mean(total / evaluations)),
            "std_seconds_per_evaluation": float(np.std(total / evaluations, ddof=1)),
            "mean_initialization_fraction": float(np.mean(initialization / total)),
        })

    tex = [r"\begin{tabular}{rrrr}", r"\hline",
           r"$N$ & Evaluations & Total time [min] & Time/eval. [s] \\", r"\hline"]
    for row in rows:
        tex.append(
            f"{row['N']:,} & "
            f"${row['mean_evaluations']:.0f} \\pm {row['std_evaluations']:.0f}$ & "
            f"${row['mean_total_seconds']/60:.1f} \\pm {row['std_total_seconds']/60:.1f}$ & "
            f"${row['mean_seconds_per_evaluation']:.2f} \\pm {row['std_seconds_per_evaluation']:.2f}$ " + r"\\"
        )
    tex += [r"\hline", r"\end{tabular}"]
    (logs / "summary.tex").write_text("\n".join(tex) + "\n")
    (logs / "summary_targets.json").write_text(json.dumps(args.target_seeds) + "\n")
    fields = tuple(rows[0])
    with (logs / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (logs / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Saved {logs / 'summary.csv'}")
    print(f"Saved {logs / 'summary.json'}")


if __name__ == "__main__":
    main()
