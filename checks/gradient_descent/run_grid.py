import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np


NHALOES = (1,)
TIMES_GYR = (4,)
INITIAL_PARAMETER_SEEDS = (42,) #tuple(range(100,120))
TARGET_PARAMETER_SEEDS = tuple(range(10,14))


def print_summary(number, total, time_gyr, target_seed, initial_seed, nhaloes, result):
    target_number = TARGET_PARAMETER_SEEDS.index(target_seed) + 1
    initial_number = INITIAL_PARAMETER_SEEDS.index(initial_seed) + 1
    print(
        f"=== Run {number}/{total}: {nhaloes} haloes, {time_gyr} Gyr, "
        f"target seed {target_number}/{len(TARGET_PARAMETER_SEEDS)}, "
        f"initial seed {initial_number}/{len(INITIAL_PARAMETER_SEEDS)}, "
        f"best: {result['best_loss']:.2e} / {result['optimal_loss']:.2e}, "
        f"file: {result['output']} ===",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Stop launching new runs after this many hours",
    )
    args = parser.parse_args()

    directory = Path(__file__).resolve().parent
    state_file = directory / "logs/run_grid_completed.json"
    if state_file.exists():
        completed = json.loads(state_file.read_text())
    else:
        completed = {}

    jobs = product(
        TIMES_GYR,
        TARGET_PARAMETER_SEEDS,
        INITIAL_PARAMETER_SEEDS,
        NHALOES,
    )
    jobs = list(jobs)
    start = time.monotonic()

    for number, (time_gyr, target_seed, initial_seed, nhaloes) in enumerate(
        jobs, start=1
    ):
        key = f"target={target_seed},initial={initial_seed},haloes={nhaloes},time={time_gyr}"
        if key in completed:
            print_summary(
                number,
                len(jobs),
                time_gyr,
                target_seed,
                initial_seed,
                nhaloes,
                completed[key],
            )
            continue
        if args.hours is not None and time.monotonic() - start >= args.hours * 3600:
            print("Time budget reached; remaining runs can be resumed later.")
            break

        command = [
            sys.executable,
            "cluster.py",
            f"--Nhaloes={nhaloes}",
            f"--integration_time_gyr={time_gyr}",
            f"--integration_steps={10 * time_gyr}",
            "--N=10000",
            "--sim_softening=10.0",
            "--patience=20",
            "--restarts=2",
            "--accurate=1",
            f"--target_parameter_seed={target_seed}",
            f"--initial_parameter_seed={initial_seed}",
        ]
        print(f"\nRun {number}/{len(jobs)}: {key}", flush=True)
        previous_outputs = set((directory / "logs").glob("sim_*.npz"))
        subprocess.run(command, cwd=directory, check=True)

        new_outputs = set((directory / "logs").glob("sim_*.npz")) - previous_outputs
        if len(new_outputs) != 1:
            raise RuntimeError(f"Expected one new output file, found {len(new_outputs)}")
        output_file = new_outputs.pop()
        with np.load(output_file) as result:
            completed[key] = {
                "best_loss": float(np.min(result["history"]["loss"])),
                "optimal_loss": float(result["optimal_loss"]),
                "output": output_file.name,
            }
        state_file.write_text(json.dumps(completed, indent=2) + "\n")
        print_summary(
            number,
            len(jobs),
            time_gyr,
            target_seed,
            initial_seed,
            nhaloes,
            completed[key],
        )


if __name__ == "__main__":
    main()
