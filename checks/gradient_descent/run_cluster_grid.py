import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


NHALOES = 1
INTEGRATION_TIME_GYR = 4.0
INTEGRATION_STEPS = 40
CASES = ((20_000, 1),)
LOG10_MASS_MIN = 9.5
LOG10_MASS_MAX = 11.5
TARGET_LOG10_MASS_MIN = 10.0
TARGET_LOG10_MASS_MAX = 11.0
SIM_SOFTENING = 4.0
LOSS_SOFTENING = 4.0
TARGET_PARAMETER_SEEDS = (0, 1, 2, 3, 5, 7, 8)
INITIAL_PARAMETER_SEEDS = tuple(range(17, 31))


def log10_mass(parameters):
    sigmoid = 1.0 / (1.0 + np.exp(-parameters["exp_m"]))
    return LOG10_MASS_MIN + (
        LOG10_MASS_MAX - LOG10_MASS_MIN
    ) * sigmoid


def mass_weighted_com(parameters):
    mass = 10.0 ** log10_mass(parameters)
    return (
        np.average(parameters["position"], axis=0, weights=mass),
        np.average(parameters["velocity"], axis=0, weights=mass),
    )


def summarize(output_file):
    with np.load(output_file) as result:
        history = result["history"]
        best_index = int(np.argmin(history["loss"]))
        best = history["parameters"][best_index]
        target = result["target_parameters"]
        best_position, best_velocity = mass_weighted_com(best)
        target_position, target_velocity = mass_weighted_com(target)
        return {
            "best_loss": float(history["loss"][best_index]),
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
    return set(logs.glob("sim_*.npz"))


def print_summary(
    number,
    total,
    N,
    accuracy,
    target_seed,
    initial_seed,
    result,
):
    print(
        f"=== Run {number}/{total}: "
        f"{NHALOES} halo, {INTEGRATION_TIME_GYR:g} Gyr, "
        f"N={N}, accuracy={accuracy}, target seed {target_seed}, "
        f"initial seed {initial_seed}, "
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
    state_file = (
        logs
        / "cluster_run_grid_massive_host_4gyr_early_stop.json"
    )
    completed = (
        json.loads(state_file.read_text()) if state_file.exists() else {}
    )
    start = time.monotonic()

    jobs = [
        (N, accuracy, target_seed, initial_seed)
        for N, accuracy in CASES
        for initial_seed in INITIAL_PARAMETER_SEEDS
        for target_seed in TARGET_PARAMETER_SEEDS
    ]
    for number, (N, accuracy, target_seed, initial_seed) in enumerate(
        jobs,
        start=1,
    ):
        key = (
            f"N={N},Nhaloes={NHALOES},time={INTEGRATION_TIME_GYR},"
            f"steps={INTEGRATION_STEPS},target={target_seed},"
            f"initial={initial_seed},accuracy={accuracy},"
            f"log10_mass={LOG10_MASS_MIN}-{LOG10_MASS_MAX},"
            f"target_log10_mass="
            f"{TARGET_LOG10_MASS_MIN}-{TARGET_LOG10_MASS_MAX},"
            f"sim_softening={SIM_SOFTENING},"
            f"loss_softening={LOSS_SOFTENING}"
        )
        if key in completed:
            print_summary(
                number,
                len(jobs),
                N,
                accuracy,
                target_seed,
                initial_seed,
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
            "cluster.py",
            f"--N={N}",
            f"--Nhaloes={NHALOES}",
            f"--integration_time_gyr={INTEGRATION_TIME_GYR}",
            f"--integration_steps={INTEGRATION_STEPS}",
            f"--accurate={accuracy}",
            f"--log10_mass_min={LOG10_MASS_MIN}",
            f"--log10_mass_max={LOG10_MASS_MAX}",
            f"--target_log10_mass_min={TARGET_LOG10_MASS_MIN}",
            f"--target_log10_mass_max={TARGET_LOG10_MASS_MAX}",
            f"--sim_softening={SIM_SOFTENING}",
            f"--loss_softening={LOSS_SOFTENING}",
            "--steps=2000",
            "--patience=50",
            "--early-stop",
            f"--target_parameter_seed={target_seed}",
            f"--initial_parameter_seed={initial_seed}",
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
            N,
            accuracy,
            target_seed,
            initial_seed,
            completed[key],
        )


if __name__ == "__main__":
    main()
