import csv
import json
from pathlib import Path

import numpy as np

from run_scaling_benchmark import PARTICLE_COUNTS, output_files, summarize


def main():
    directory = Path(__file__).resolve().parent
    logs = directory / "logs" / "scaling_benchmark"
    rows = []

    for N in PARTICLE_COUNTS:
        results = []
        for target_seed in range(10):
            npz_file, _ = output_files(logs, target_seed, N)
            if not npz_file.exists():
                raise FileNotFoundError(npz_file)
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
