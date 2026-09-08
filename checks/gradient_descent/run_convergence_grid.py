import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


N = 20_000
NHALOES = 1
ACCURACY = 1
SIM_SOFTENING = 4.0
LOSS_SOFTENING = 4.0
MAX_STEPS = 2_000
PATIENCE = 50
TARGET_SEEDS = tuple(range(10))
INITIAL_SEEDS = tuple(range(100, 200))
TIMES_GYR = tuple(range(5))
OUTPUT_ID_WIDTH = 4


def make_jobs():
    jobs = []
    run_id = 1
    for initial_seed in INITIAL_SEEDS:
        for time_gyr in TIMES_GYR:
            for target_seed in TARGET_SEEDS:
                jobs.append({
                    "run_id": run_id,
                    "time_gyr": time_gyr,
                    "integration_steps": 10 * time_gyr,
                    "target_seed": target_seed,
                    "initial_seed": initial_seed,
                })
                run_id += 1
    return jobs


def manifest_values(jobs, accuracy=ACCURACY):
    return {
        "description": (
            "One-halo convergence grid ordered by initialization round"
        ),
        "N": N,
        "Nhaloes": NHALOES,
        "accuracy": accuracy,
        "sim_softening_kpc": SIM_SOFTENING,
        "loss_softening_kpc": LOSS_SOFTENING,
        "max_steps": MAX_STEPS,
        "patience": PATIENCE,
        "fix_mass": True,
        "early_stop": True,
        "target_seeds": TARGET_SEEDS,
        "initial_seeds": INITIAL_SEEDS,
        "times_gyr": TIMES_GYR,
        "jobs": jobs,
    }


def write_or_check_manifest(filename, values):
    if filename.exists():
        existing = json.loads(filename.read_text())
        if existing != json.loads(json.dumps(values)):
            raise RuntimeError(
                f"Existing manifest does not match this grid: {filename}"
            )
        return
    temporary = filename.with_name(
        f".{filename.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(json.dumps(values, indent=2) + "\n")
    os.replace(temporary, filename)


def output_paths(logs, run_id):
    stem = f"run_{run_id:0{OUTPUT_ID_WIDTH}d}"
    return logs / f"{stem}.npz", logs / f"{stem}.json"


def validate_config(config_file, job, accuracy=ACCURACY):
    config = json.loads(config_file.read_text())
    expected = {
        "N": N,
        "Nhaloes": NHALOES,
        "mode": "ic" if job["time_gyr"] == 0 else "sim",
        "integration_time_gyr": job["time_gyr"],
        "integration_steps": job["integration_steps"],
        "target_parameter_seed": job["target_seed"],
        "initial_parameter_seed": job["initial_seed"],
        "fix_mass": True,
        "loss_softening": LOSS_SOFTENING,
        "max_steps": MAX_STEPS,
        "patience": PATIENCE,
        "early_stop": True,
        "output_prefix": "run",
        "output_id": job["run_id"],
        "output_id_width": OUTPUT_ID_WIDTH,
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Configuration mismatch for {config_file}: {mismatches}"
        )
    force = config["sim_config"]["force"]
    force_expected = {
        "p": (5, 6, 7)[accuracy],
        "opening theta": (0.8, 0.7, 0.6)[accuracy],
        "softening": SIM_SOFTENING,
    }
    force_actual = {
        "p": force["p"],
        "opening theta": force["opening"]["theta"],
        "softening": force["kernel"]["softening"],
    }
    if force_actual != force_expected:
        raise RuntimeError(
            f"Force configuration mismatch for {config_file}: "
            f"found {force_actual}, expected {force_expected}"
        )


def mass_weighted_com(parameters, config):
    exp_m = parameters["exp_m"]
    sigmoid = 1.0 / (1.0 + np.exp(-exp_m))
    log10_mass = config["log10_mass_min"] + (
        config["log10_mass_max"] - config["log10_mass_min"]
    ) * sigmoid
    mass = 10**log10_mass
    position = np.average(parameters["position"], axis=0, weights=mass)
    velocity = np.average(parameters["velocity"], axis=0, weights=mass)
    return position, velocity


def summarize(npz_file, config_file):
    config = json.loads(config_file.read_text())
    with np.load(npz_file) as result:
        history = result["history"]
        best_index = int(np.argmin(history["loss"]))
        best = history["parameters"][best_index]
        target = result["target_parameters"]
        best_position, best_velocity = mass_weighted_com(best, config)
        target_position, target_velocity = mass_weighted_com(target, config)
        return {
            "best_loss": float(history["loss"][best_index]),
            "optimal_loss": float(result["optimal_loss"]),
            "position_error_kpc": float(
                1_000 * np.linalg.norm(best_position - target_position)
            ),
            "velocity_error_kms": float(
                1_000 * np.linalg.norm(best_velocity - target_velocity)
            ),
        }


def print_summary(job, total, result, skipped=False):
    action = "Existing" if skipped else "Completed"
    print(
        f"=== {action} run {job['run_id']}/{total}: "
        f"{job['time_gyr']} Gyr, target {job['target_seed']}, "
        f"initial {job['initial_seed']}, "
        f"best {result['best_loss']:.2e} / "
        f"{result['optimal_loss']:.2e}, "
        f"IC position {result['position_error_kpc']:.1f} kpc, "
        f"IC velocity {result['velocity_error_kms']:.1f} km/s ===",
        flush=True,
    )


def command_for_job(directory, logs, job, accuracy=ACCURACY):
    command = [
        sys.executable,
        "cluster.py",
        f"--N={N}",
        f"--Nhaloes={NHALOES}",
        f"--integration_time_gyr={job['time_gyr']}",
        f"--integration_steps={job['integration_steps']}",
        f"--accurate={accuracy}",
        f"--sim_softening={SIM_SOFTENING}",
        f"--loss_softening={LOSS_SOFTENING}",
        f"--steps={MAX_STEPS}",
        f"--patience={PATIENCE}",
        "--early-stop",
        "--fix_mass",
        f"--target_parameter_seed={job['target_seed']}",
        f"--initial_parameter_seed={job['initial_seed']}",
        f"--output_directory={logs}",
        "--output_prefix=run",
        f"--output_id={job['run_id']}",
        f"--output_id_width={OUTPUT_ID_WIDTH}",
    ]
    if job["time_gyr"] == 0:
        command.append("--ic")
    return command


def run_job(directory, logs, job, accuracy=ACCURACY):
    command = command_for_job(directory, logs, job, accuracy)
    process = subprocess.Popen(
        command,
        cwd=directory,
        start_new_session=True,
    )
    stop_after_run = False
    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        stop_after_run = True
        print(
            "\nInterrupt received; asking the active run to save its "
            "partial history.",
            flush=True,
        )
        process.send_signal(signal.SIGINT)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return stop_after_run


def preserve_partial_output(npz_file, config_file):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    for filename in (npz_file, config_file):
        if filename.exists():
            partial = filename.with_name(
                f"{filename.stem}_partial_{timestamp}{filename.suffix}"
            )
            filename.rename(partial)
            print(f"Saved partial output as {partial}", flush=True)


def main():
    jobs = make_jobs()
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-run", type=int, default=1)
    parser.add_argument("--end-run", type=int, default=len(jobs))
    parser.add_argument("--accuracy", type=int, choices=range(3), default=ACCURACY)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    if not 1 <= args.start_run <= len(jobs):
        parser.error(f"--start-run must be in [1, {len(jobs)}]")
    if not args.start_run <= args.end_run <= len(jobs):
        parser.error(
            f"--end-run must be in [{args.start_run}, {len(jobs)}]"
        )

    directory = Path(__file__).resolve().parent
    logs = args.output_directory or directory / "logs" / (
        "convergence_grid" if args.accuracy == 1 else f"convergence_grid_accuracy{args.accuracy}"
    )
    logs = logs.resolve()
    logs.mkdir(parents=True, exist_ok=True)
    write_or_check_manifest(
        logs / "manifest.json",
        manifest_values(jobs, args.accuracy),
    )

    start = time.monotonic()
    selected_jobs = jobs[args.start_run - 1:args.end_run]
    for job in selected_jobs:
        npz_file, config_file = output_paths(logs, job["run_id"])
        if npz_file.exists() != config_file.exists():
            raise RuntimeError(
                f"Incomplete output pair for run {job['run_id']}: "
                f"{npz_file}, {config_file}"
            )
        if npz_file.exists():
            validate_config(config_file, job, args.accuracy)
            print_summary(
                job,
                len(jobs),
                summarize(npz_file, config_file),
                skipped=True,
            )
            continue

        print(
            f"\nStarting run {job['run_id']}/{len(jobs)}: "
            f"{job['time_gyr']} Gyr, target {job['target_seed']}, "
            f"initial {job['initial_seed']}",
            flush=True,
        )
        stop_after_run = run_job(directory, logs, job, args.accuracy)
        if stop_after_run:
            preserve_partial_output(npz_file, config_file)
            break
        validate_config(config_file, job, args.accuracy)
        print_summary(
            job,
            len(jobs),
            summarize(npz_file, config_file),
        )

    elapsed = time.monotonic() - start
    print(f"Grid invocation finished after {elapsed / 3_600:.2f} hours")


if __name__ == "__main__":
    main()
