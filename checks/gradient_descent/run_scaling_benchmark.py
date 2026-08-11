import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


PARTICLE_COUNTS = (20_000, 100_000, 1_000_000, 10_000_000)
BEST_INITIAL_SEEDS = {
    0: 100,
    1: 156,
    2: 139,
    3: 180,
    4: 130,
    5: 105,
    6: 167,
    7: 126,
    8: 123,
    9: 110,
}


def output_files(logs, target_seed, N):
    stem = f"target_{target_seed}_N_{N}"
    return logs / f"{stem}.npz", logs / f"{stem}.json"


def summarize(npz_file):
    with np.load(npz_file) as result:
        history = result["history"]
        best_index = int(np.argmin(history["loss"]))
        initialization = float(result["initialization_seconds"])
        optimization = float(result["optimization_seconds"])
        total = float(result["total_run_seconds"])
        evaluations = int(history["total_evaluations"][-1])
        return {
            "best_loss": float(history["loss"][best_index]),
            "optimal_loss": float(result["optimal_loss"]),
            "best_step": int(history["step"][best_index]),
            "final_step": int(history["step"][-1]),
            "evaluations": evaluations,
            "initialization_seconds": initialization,
            "optimization_seconds": optimization,
            "total_run_seconds": total,
            "seconds_per_evaluation": total / evaluations,
            "initialization_fraction": initialization / total,
        }


def validate_config(config_file, target_seed, initial_seed, N):
    config = json.loads(config_file.read_text())
    expected = {
        "N": N,
        "Nhaloes": 1,
        "mode": "sim",
        "integration_time_gyr": 4.0,
        "integration_steps": 40,
        "target_parameter_seed": target_seed,
        "initial_parameter_seed": initial_seed,
        "fix_mass": True,
        "early_stop": True,
    }
    mismatch = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatch:
        raise RuntimeError(f"Configuration mismatch in {config_file}: {mismatch}")


def print_summary(target_seed, N, result, existing=False):
    status = "Existing" if existing else "Completed"
    print(
        f"=== {status}: target {target_seed}, N={N:,}, "
        f"loss {result['best_loss']:.2e} / {result['optimal_loss']:.2e}, "
        f"evaluations {result['evaluations']}, total {result['total_run_seconds'] / 60:.1f} min, "
        f"setup {100 * result['initialization_fraction']:.1f}%, "
        f"{result['seconds_per_evaluation']:.3f} s/eval ===",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-seed", type=int, choices=range(10), required=True)
    args = parser.parse_args()

    directory = Path(__file__).resolve().parent
    logs = directory / "logs" / "scaling_benchmark"
    logs.mkdir(parents=True, exist_ok=True)
    initial_seed = BEST_INITIAL_SEEDS[args.target_seed]
    results = {}

    for N in PARTICLE_COUNTS:
        npz_file, config_file = output_files(logs, args.target_seed, N)
        if npz_file.exists() != config_file.exists():
            raise RuntimeError(f"Incomplete output pair: {npz_file}, {config_file}")
        if npz_file.exists():
            validate_config(config_file, args.target_seed, initial_seed, N)
            results[str(N)] = summarize(npz_file)
            print_summary(args.target_seed, N, results[str(N)], existing=True)
            (logs / f"target_{args.target_seed}_timing.json").write_text(json.dumps(results, indent=2) + "\n")
            continue

        command = [
            sys.executable,
            "cluster.py",
            f"--N={N}",
            "--Nhaloes=1",
            "--integration_time_gyr=4",
            "--integration_steps=40",
            "--accurate=1",
            "--sim_softening=4",
            "--loss_softening=4",
            "--steps=2000",
            "--patience=50",
            "--early-stop",
            "--fix_mass",
            f"--target_parameter_seed={args.target_seed}",
            f"--initial_parameter_seed={initial_seed}",
            f"--output_directory={logs}",
            f"--output_prefix=target_{args.target_seed}_N",
            f"--output_id={N}",
        ]
        print(f"\nStarting target {args.target_seed}, N={N:,}", flush=True)
        subprocess.run(command, cwd=directory, check=True)
        validate_config(config_file, args.target_seed, initial_seed, N)
        results[str(N)] = summarize(npz_file)
        print_summary(args.target_seed, N, results[str(N)])
        (logs / f"target_{args.target_seed}_timing.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
