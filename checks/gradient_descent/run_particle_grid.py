import argparse
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np


N = 10_000
TIMES_GYR = (2.0,)
ACCURACY = 0
LOG10_MASS_MIN = 9.0
LOG10_MASS_MAX = 10.0
SIM_SOFTENING = 2.0
LOSS_SOFTENING = 5.0
TARGET_PARAMETER_SEEDS = tuple(range(10))
INITIAL_PARAMETER_SEED = 42


def mass_weighted_com(parameters):
    position = np.asarray(parameters["position"])
    velocity = np.asarray(parameters["velocity"])
    mass = np.broadcast_to(
        10.0 ** np.asarray(parameters["log10_mass"]),
        (len(position),),
    )
    return (
        np.average(position, axis=0, weights=mass),
        np.average(velocity, axis=0, weights=mass),
    )


def summarize(output_file):
    with np.load(output_file) as result:
        best_position, best_velocity = mass_weighted_com(
            result["best_parameters"]
        )
        target_position, target_velocity = mass_weighted_com(
            result["target_parameters"]
        )
        return {
            "best_loss": float(result["best_loss"]),
            "optimal_loss": float(result["optimal_loss"]),
            "com_error_kpc": float(
                1_000.0 * np.linalg.norm(best_position - target_position)
            ),
            "com_velocity_error_kms": float(
                1_000.0 * np.linalg.norm(best_velocity - target_velocity)
            ),
            "output": output_file.name,
        }


def result_files(logs):
    return {
        path
        for path in logs.glob("particle_sim_*.npz")
        if not path.name.endswith("_snapshots.npz")
    }


def print_summary(number, total, time_gyr, target_seed, result):
    target_number = TARGET_PARAMETER_SEEDS.index(target_seed) + 1
    print(
        f"=== Run {number}/{total}: {time_gyr:g} Gyr, "
        f"target seed {target_number}/{len(TARGET_PARAMETER_SEEDS)} "
        f"({target_seed}), "
        f"N={N}, logM={LOG10_MASS_MIN:g}–{LOG10_MASS_MAX:g}, "
        f"accuracy={ACCURACY}, real space, "
        f"best: {result['best_loss']:.2e} / "
        f"{result['optimal_loss']:.2e}, "
        f"IC COM: {result['com_error_kpc']:.1f} kpc, "
        f"IC velocity: {result['com_velocity_error_kms']:.1f} km/s, "
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
    logs = directory / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state_file = logs / "particle_run_grid_low_mass_completed.json"
    if state_file.exists():
        completed = json.loads(state_file.read_text())
    else:
        completed = {}

    jobs = list(product(TIMES_GYR, TARGET_PARAMETER_SEEDS))
    start = time.monotonic()

    for number, (time_gyr, target_seed) in enumerate(jobs, start=1):
        integration_steps = round(10 * time_gyr)
        key = (
            f"N={N},time={time_gyr},steps={integration_steps},"
            f"target={target_seed},initial={INITIAL_PARAMETER_SEED},"
            f"accuracy={ACCURACY},redshift_space=False,"
            f"log10_mass={LOG10_MASS_MIN}-{LOG10_MASS_MAX},"
            f"sim_softening={SIM_SOFTENING},"
            f"loss_softening={LOSS_SOFTENING}"
        )
        if key in completed:
            print_summary(
                number,
                len(jobs),
                time_gyr,
                target_seed,
                completed[key],
            )
            continue
        if (
            args.hours is not None
            and time.monotonic() - start >= args.hours * 3_600
        ):
            print("Time budget reached; remaining runs can be resumed later.")
            break

        command = [
            sys.executable,
            "particle_cluster.py",
            f"--N={N}",
            f"--integration_time_gyr={time_gyr}",
            f"--integration_steps={integration_steps}",
            f"--accurate={ACCURACY}",
            f"--log10_mass_min={LOG10_MASS_MIN}",
            f"--log10_mass_max={LOG10_MASS_MAX}",
            f"--sim_softening={SIM_SOFTENING}",
            f"--loss_softening={LOSS_SOFTENING}",
            "--adam",
            "--steps=2000",
            "--patience=50",
            "--target_weight=1",
            "--mass_weight=10",
            "--hernquist_weight=1",
            f"--target_parameter_seed={target_seed}",
            f"--initial_parameter_seed={INITIAL_PARAMETER_SEED}",
        ]
        print(f"\nRun {number}/{len(jobs)}: {key}", flush=True)
        previous_outputs = result_files(logs)
        subprocess.run(command, cwd=directory, check=True)

        new_outputs = result_files(logs) - previous_outputs
        if len(new_outputs) != 1:
            raise RuntimeError(
                f"Expected one new output file, found {len(new_outputs)}"
            )
        output_file = new_outputs.pop()
        completed[key] = summarize(output_file)
        state_file.write_text(json.dumps(completed, indent=2) + "\n")
        print_summary(
            number,
            len(jobs),
            time_gyr,
            target_seed,
            completed[key],
        )


if __name__ == "__main__":
    main()
