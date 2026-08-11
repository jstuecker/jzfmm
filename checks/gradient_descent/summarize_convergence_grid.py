import csv
import json
from pathlib import Path

import numpy as np

import run_convergence_grid as grid


SUMMARY_DTYPE = np.dtype([
    ("run_id", np.int32),
    ("time_gyr", np.float64),
    ("target_seed", np.int32),
    ("initial_seed", np.int32),
    ("history_length", np.int32),
    ("best_history_index", np.int32),
    ("best_step", np.int32),
    ("final_step", np.int32),
    ("best_loss", np.float64),
    ("final_loss", np.float64),
    ("optimal_loss", np.float64),
    ("loss_ratio", np.float64),
    ("best_gradient_norm", np.float64),
    ("total_evaluations", np.int32),
    ("elapsed_seconds", np.float64),
    ("best_is_final", np.bool_),
    ("nonfinite_loss_count", np.int32),
    ("target_log10_mass", np.float64),
    ("best_log10_mass", np.float64),
    ("target_concentration", np.float64),
    ("best_concentration", np.float64),
    ("target_x_mpc", np.float64),
    ("target_y_mpc", np.float64),
    ("target_z_mpc", np.float64),
    ("best_x_mpc", np.float64),
    ("best_y_mpc", np.float64),
    ("best_z_mpc", np.float64),
    ("position_dx_kpc", np.float64),
    ("position_dy_kpc", np.float64),
    ("position_dz_kpc", np.float64),
    ("position_error_kpc", np.float64),
    ("target_vx_kms", np.float64),
    ("target_vy_kms", np.float64),
    ("target_vz_kms", np.float64),
    ("best_vx_kms", np.float64),
    ("best_vy_kms", np.float64),
    ("best_vz_kms", np.float64),
    ("velocity_dvx_kms", np.float64),
    ("velocity_dvy_kms", np.float64),
    ("velocity_dvz_kms", np.float64),
    ("velocity_error_kms", np.float64),
])


def mass_and_concentration(parameters, config):
    exp_m = float(parameters["exp_m"][0])
    fraction = 1.0 / (1.0 + np.exp(-exp_m))
    log10_mass = config["log10_mass_min"] + (
        config["log10_mass_max"] - config["log10_mass_min"]
    ) * fraction
    concentration = 10 ** float(parameters["log10_concentration"][0])
    return log10_mass, concentration


def summarize_run(npz_file, config_file, job):
    config = json.loads(config_file.read_text())
    grid.validate_config(config_file, job)
    with np.load(npz_file) as result:
        history = result["history"]
        if not len(history):
            raise RuntimeError(f"Empty history in {npz_file}")
        finite = np.isfinite(history["loss"])
        if not finite.any():
            raise RuntimeError(f"No finite loss in {npz_file}")
        finite_indices = np.flatnonzero(finite)
        best_index = int(
            finite_indices[np.argmin(history["loss"][finite])]
        )
        best = history["parameters"][best_index]
        target = result["target_parameters"]
        optimal_loss = float(result["optimal_loss"])

        target_position, target_velocity = grid.mass_weighted_com(
            target, config
        )
        best_position, best_velocity = grid.mass_weighted_com(best, config)
        position_delta = 1_000 * (best_position - target_position)
        velocity_delta = 1_000 * (best_velocity - target_velocity)
        target_log10_mass, target_concentration = mass_and_concentration(
            target, config
        )
        best_log10_mass, best_concentration = mass_and_concentration(
            best, config
        )

        values = (
            job["run_id"],
            job["time_gyr"],
            job["target_seed"],
            job["initial_seed"],
            len(history),
            best_index,
            int(history["step"][best_index]),
            int(history["step"][-1]),
            float(history["loss"][best_index]),
            float(history["loss"][-1]),
            optimal_loss,
            float(history["loss"][best_index] / optimal_loss),
            float(history["gradient_norm"][best_index]),
            int(history["total_evaluations"][-1]),
            float(history["elapsed_seconds"][-1]),
            best_index == len(history) - 1,
            int((~finite).sum()),
            target_log10_mass,
            best_log10_mass,
            target_concentration,
            best_concentration,
            *target_position,
            *best_position,
            *position_delta,
            float(np.linalg.norm(position_delta)),
            *(1_000 * target_velocity),
            *(1_000 * best_velocity),
            *velocity_delta,
            float(np.linalg.norm(velocity_delta)),
        )
    return values, target.copy()


def write_csv(filename, summary):
    with filename.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(summary.dtype.names)
        for row in summary:
            writer.writerow(row[name] for name in summary.dtype.names)


def validate_groups(summary, targets):
    counts = {}
    for time_gyr in grid.TIMES_GYR:
        for target_seed in grid.TARGET_SEEDS:
            count = int(np.sum(
                (summary["time_gyr"] == time_gyr)
                & (summary["target_seed"] == target_seed)
            ))
            counts[f"{time_gyr}Gyr_target{target_seed}"] = count
            if count != len(grid.INITIAL_SEEDS):
                raise RuntimeError(
                    f"Expected {len(grid.INITIAL_SEEDS)} runs for "
                    f"{time_gyr} Gyr, target {target_seed}; found {count}"
                )

    for target_seed in grid.TARGET_SEEDS:
        selected = [
            targets[index]
            for index, row in enumerate(summary)
            if row["target_seed"] == target_seed
        ]
        reference = selected[0]
        if not all(np.array_equal(reference, target) for target in selected[1:]):
            raise RuntimeError(
                f"Target parameters differ for target seed {target_seed}"
            )

    optimal_spread = {}
    for time_gyr in grid.TIMES_GYR:
        for target_seed in grid.TARGET_SEEDS:
            selected = summary["optimal_loss"][
                (summary["time_gyr"] == time_gyr)
                & (summary["target_seed"] == target_seed)
            ]
            relative_spread = (
                (selected.max() - selected.min()) / np.median(selected)
            )
            optimal_spread[f"{time_gyr}Gyr_target{target_seed}"] = float(
                relative_spread
            )
    return counts, optimal_spread


def main():
    directory = Path(__file__).resolve().parent
    logs = directory / "logs" / "convergence_grid"
    analysis = logs / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)

    jobs = grid.make_jobs()
    manifest = json.loads((logs / "manifest.json").read_text())
    expected_manifest = json.loads(json.dumps(grid.manifest_values(jobs)))
    if manifest != expected_manifest:
        raise RuntimeError("Grid manifest does not match the expected setup")

    summary = np.empty(len(jobs), dtype=SUMMARY_DTYPE)
    targets = []
    for index, job in enumerate(jobs):
        npz_file, config_file = grid.output_paths(logs, job["run_id"])
        if not npz_file.exists() or not config_file.exists():
            raise FileNotFoundError(
                f"Missing output for run {job['run_id']}: "
                f"{npz_file}, {config_file}"
            )
        values, target = summarize_run(npz_file, config_file, job)
        summary[index] = values
        targets.append(target)
        if (index + 1) % 500 == 0:
            print(f"Read {index + 1}/{len(jobs)} runs", flush=True)

    counts, optimal_spread = validate_groups(summary, targets)
    np.savez(analysis / "summary.npz", summary=summary)
    write_csv(analysis / "summary.csv", summary)

    validation = {
        "number_of_runs": len(summary),
        "number_of_nonfinite_history_values": int(
            summary["nonfinite_loss_count"].sum()
        ),
        "runs_with_nonfinite_history_values": int(np.sum(
            summary["nonfinite_loss_count"] > 0
        )),
        "group_counts": counts,
        "maximum_relative_optimal_loss_spread": max(
            optimal_spread.values()
        ),
        "relative_optimal_loss_spread": optimal_spread,
        "total_gpu_hours": float(
            summary["elapsed_seconds"].sum() / 3_600
        ),
    }
    (analysis / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )

    print(f"Saved {analysis / 'summary.npz'}")
    print(f"Saved {analysis / 'summary.csv'}")
    print(f"Saved {analysis / 'validation.json'}")
    print(
        f"Total recorded GPU time: {validation['total_gpu_hours']:.1f} h"
    )


if __name__ == "__main__":
    main()
