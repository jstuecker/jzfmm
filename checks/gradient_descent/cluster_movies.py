import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

import cluster as cl


DEFAULT_SUCCESS_RUN = 93
DEFAULT_FAILURE_RUN = 103
DEFAULT_FPS = 10
DEFAULT_MOVIE_DIRECTORY = Path("movies")


def step_spaced_history_indices(history, best_index, nstates):
    steps = np.asarray(history["step"][:best_index + 1])
    target_steps = np.linspace(steps[0], steps[-1], nstates)
    indices = [
        int(np.argmin(np.abs(steps - target_step)))
        for target_step in target_steps
    ]
    return np.unique([0, *indices, best_index])


def load_run(run):
    with np.load(f"logs/sim_{run}.npz") as result:
        target = result["target_parameters"].copy()
        history = result["history"].copy()
    config = cl.Config.from_json(f"logs/sim_{run}.json")
    config.sim_config.logging.level = 0
    return target, history, config


def particle_outputs(parameters, config, seed, outputs):
    return [
        (float(np.asarray(time)), np.asarray(particles.pos))
        for time, particles in cl.get_particles(
            parameters,
            config,
            seed=seed,
            outputs=outputs,
        )
    ]


def parameter_position(parameters):
    return np.asarray(parameters["position"])[0] * 1_000


def parameter_velocity(parameters):
    return np.asarray(parameters["velocity"])[0]


def virial_radius(config):
    return cl.RvirOfMvir(
        config.sim_config.external_potential.mass
    ) * 1_000


def orbit_extent(trajectory, rvir):
    apocenter = max(
        np.linalg.norm(np.mean(positions, axis=0))
        for _, positions in trajectory
    )
    return 1.5 * apocenter / rvir


def prepare_axes(title, extent):
    fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
    ax.scatter(0, 0, marker="*", s=55, color="black", zorder=5)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$x/r_{\rm vir}$")
    ax.set_ylabel(r"$y/r_{\rm vir}$")
    ax.set_title(title)
    return fig, ax


def save_animation(animation, filename, fps, dpi):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {filename}", flush=True)
    animation.save(
        filename,
        writer=FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=5_000,
            extra_args=["-pix_fmt", "yuv420p"],
        ),
        dpi=dpi,
    )
    plt.close(animation._fig)
    print(f"Saved {filename}", flush=True)


def save_frames(fig, update, frames, names, movie_filename, dpi):
    frame_directory = Path(movie_filename).with_suffix("")
    frame_directory = frame_directory.with_name(
        f"{frame_directory.name}_frames"
    )
    frame_directory.mkdir(parents=True, exist_ok=True)
    for frame, name in zip(frames, names):
        update(int(frame))
        frame_path = frame_directory / name
        fig.savefig(frame_path, dpi=dpi)
        print(f"Saved {frame_path}", flush=True)


def true_evolution_movie(
    run,
    filename,
    fps=DEFAULT_FPS,
    dpi=160,
    still_frames=5,
):
    target, _, config = load_run(run)
    print(f"Caching true evolution for sim_{run}", flush=True)
    trajectory = particle_outputs(
        target,
        config,
        config.target_particle_seed,
        outputs=config.integration_steps + 1,
    )
    rvir = virial_radius(config)
    extent = orbit_extent(trajectory, rvir)

    fig, ax = prepare_axes("True evolution", extent)
    scatter = ax.scatter(
        [], [], marker=".", s=2, alpha=0.045, color="tab:blue", zorder=2
    )
    time_text = ax.text(
        0.03, 0.96, "", transform=ax.transAxes, va="top", fontsize=12
    )

    def update(frame):
        time_gyr, positions = trajectory[frame]
        scatter.set_offsets(positions[:, :2] / rvir)
        time_text.set_text(f"{time_gyr:.1f} Gyr")
        return scatter, time_text

    selected_frames = np.unique(np.linspace(
        0,
        len(trajectory) - 1,
        min(still_frames, len(trajectory)),
        dtype=int,
    ))
    save_frames(
        fig,
        update,
        selected_frames,
        [
            f"frame_{number:02d}_t_{trajectory[frame][0]:.1f}Gyr.png"
            for number, frame in enumerate(selected_frames)
        ],
        filename,
        dpi,
    )

    endpoint_hold_frames = round(0.5 * fps)
    movie_frames = (
        [0] * endpoint_hold_frames
        + list(range(len(trajectory)))
        + [len(trajectory) - 1] * endpoint_hold_frames
    )
    animation = FuncAnimation(
        fig,
        update,
        frames=movie_frames,
        interval=1_000 / fps,
        blit=True,
    )
    save_animation(animation, filename, fps, dpi)


def optimization_movie(
    run,
    filename,
    nstates=15,
    replay_outputs=None,
    replay_seconds=4.0,
    final_hold_seconds=2.5,
    endpoint_hold_seconds=0.5,
    fps=DEFAULT_FPS,
    dpi=160,
    still_frames=5,
):
    target, history, config = load_run(run)
    if replay_outputs is None:
        replay_outputs = config.integration_steps + 1
    best_index = int(np.argmin(history["loss"]))
    state_indices = step_spaced_history_indices(
        history,
        best_index,
        nstates,
    )

    print(f"Caching target for sim_{run}", flush=True)
    target_trajectory = particle_outputs(
        target,
        config,
        config.target_particle_seed,
        outputs=replay_outputs,
    )
    target_final = target_trajectory[-1][1]
    rvir = virial_radius(config)
    extent = orbit_extent(target_trajectory, rvir)

    trajectories = []
    for number, index in enumerate(state_indices, start=1):
        print(
            f"Caching optimizer state {number}/{len(state_indices)} "
            f"(step {history['step'][index]})",
            flush=True,
        )
        trajectories.append(particle_outputs(
            history["parameters"][index],
            config,
            config.model_particle_seed,
            outputs=replay_outputs,
        ))

    fig, ax = prepare_axes("Optimization and evolution", extent)
    ax.scatter(
        target_final[:, 0] / rvir,
        target_final[:, 1] / rvir,
        marker=".",
        s=2,
        alpha=0.022,
        color="tab:blue",
        zorder=2,
    )
    true_initial = parameter_position(target)
    true_initial_xy = true_initial[:2] / rvir
    ax.scatter(
        true_initial_xy[0],
        true_initial_xy[1],
        marker="X",
        s=65,
        facecolor="white",
        edgecolor="black",
        linewidth=0.9,
        zorder=6,
    )
    true_velocity_xy = 0.1 * parameter_velocity(target)[:2]
    ax.add_patch(FancyArrowPatch(
        true_initial_xy,
        true_initial_xy + true_velocity_xy,
        arrowstyle="-|>",
        mutation_scale=13,
        color="black",
        lw=1.3,
        zorder=7,
    ))
    inferred = ax.scatter(
        [], [], marker=".", s=2, alpha=0.045, color="tab:red", zorder=3
    )
    initial_com_history = (
        np.asarray(history["parameters"]["position"][:, 0, :2])
        * 1_000
        / rvir
    )
    initial_com_path, = ax.plot(
        [], [], color="0.25", lw=1.0, alpha=0.7, zorder=5
    )
    inferred_initial = ax.scatter(
        [], [], marker="o", s=48, facecolor="none",
        edgecolor="tab:red", linewidth=1.2, zorder=6
    )
    inferred_velocity = FancyArrowPatch(
        (0, 0),
        (0, 0),
        arrowstyle="-|>",
        mutation_scale=13,
        color="tab:red",
        lw=1.3,
        zorder=7,
    )
    ax.add_patch(inferred_velocity)
    status = ax.text(
        0.03,
        0.97,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        zorder=10,
    )
    ax.legend(
        handles=[
            Line2D([], [], marker="o", ls="none", color="tab:blue",
                   label="Target final state"),
            Line2D([], [], marker="o", ls="none", color="tab:red",
                   label="Inferred evolution"),
            Line2D([], [], marker="X", ls="none", markerfacecolor="white",
                   markeredgecolor="black", color="black",
                   label="True initial COM"),
            Line2D([], [], marker="o", ls="none", markerfacecolor="none",
                   markeredgecolor="tab:red", color="tab:red",
                   label="Inferred initial COM"),
            Line2D([], [], color="0.25", lw=1.0,
                   label="Initial COM path"),
            Line2D([], [], color="black", lw=1.3, marker=">",
                   markevery=[1], label="True initial velocity"),
            Line2D([], [], color="tab:red", lw=1.3, marker=">",
                   markevery=[1], label="Inferred initial velocity"),
        ],
        loc="lower left",
        ncol=2,
        fontsize=9,
        columnspacing=0.8,
        handletextpad=0.4,
        framealpha=0.9,
    )

    frames_per_replay = max(
        replay_outputs,
        round(replay_seconds * fps),
    )
    endpoint_hold_frames = max(1, round(endpoint_hold_seconds * fps))
    frame_schedule = []
    state_final_frames = []
    for state_number in range(len(state_indices)):
        frame_schedule.extend(
            [(state_number, 0)] * endpoint_hold_frames
        )
        frame_schedule.extend(
            (state_number, round(
                frame
                * (replay_outputs - 1)
                / max(frames_per_replay - 1, 1)
            ))
            for frame in range(frames_per_replay)
        )
        frame_schedule.extend(
            [(state_number, replay_outputs - 1)] * endpoint_hold_frames
        )
        state_final_frames.append(len(frame_schedule) - 1)
    replay_frames = len(frame_schedule)
    frame_schedule.extend(
        [(len(state_indices) - 1, replay_outputs - 1)]
        * round(final_hold_seconds * fps)
    )

    def update(frame):
        state_number, output_number = frame_schedule[frame]
        holding = frame >= replay_frames

        history_index = state_indices[state_number]
        time_gyr, positions = trajectories[state_number][output_number]
        inferred.set_offsets(positions[:, :2] / rvir)
        initial_position = parameter_position(
            history["parameters"][history_index]
        )
        inferred_initial.set_offsets(initial_position[None, :2] / rvir)
        inferred_initial_xy = initial_position[:2] / rvir
        inferred_velocity_xy = 0.1 * parameter_velocity(
            history["parameters"][history_index]
        )[:2]
        inferred_velocity.set_positions(
            inferred_initial_xy,
            inferred_initial_xy + inferred_velocity_xy,
        )
        initial_com_path.set_data(
            initial_com_history[:history_index + 1, 0],
            initial_com_history[:history_index + 1, 1],
        )
        label = "Best fit\n" if holding else ""
        status.set_text(
            f"{label}step {history['step'][history_index]}\n"
            f"loss = {history['loss'][history_index]:.2e}\n"
            f"t = {time_gyr:.1f} Gyr"
        )
        return (
            inferred,
            inferred_initial,
            inferred_velocity,
            initial_com_path,
            status,
        )

    selected_states = np.unique(np.linspace(
        0,
        len(state_indices) - 1,
        min(still_frames, len(state_indices)),
        dtype=int,
    ))
    selected_frames = [
        (
            replay_frames
            if state_number == len(state_indices) - 1
            else state_final_frames[state_number]
        )
        for state_number in selected_states
    ]
    save_frames(
        fig,
        update,
        selected_frames,
        [
            f"frame_{number:02d}_step_"
            f"{history['step'][state_indices[state_number]]}.png"
            for number, state_number in enumerate(selected_states)
        ],
        filename,
        dpi,
    )

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_schedule),
        interval=1_000 / fps,
        blit=True,
    )
    save_animation(animation, filename, fps, dpi)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--movie",
        choices=("all", "true", "success", "failure"),
        default="all",
    )
    parser.add_argument("--success-run", type=int, default=DEFAULT_SUCCESS_RUN)
    parser.add_argument("--failure-run", type=int, default=DEFAULT_FAILURE_RUN)
    parser.add_argument("--states", type=int, default=15)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument(
        "--still-frames",
        type=int,
        default=5,
        help="Number of representative PNG frames saved per movie",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_MOVIE_DIRECTORY,
    )
    args = parser.parse_args()
    if args.still_frames < 1:
        parser.error("--still-frames must be positive")

    if args.movie in ("all", "true"):
        true_evolution_movie(
            args.success_run,
            args.output_directory / f"sim_{args.success_run}_true.mp4",
            fps=args.fps,
            dpi=args.dpi,
            still_frames=args.still_frames,
        )
    if args.movie in ("all", "success"):
        optimization_movie(
            args.success_run,
            args.output_directory / f"sim_{args.success_run}_optimization.mp4",
            nstates=args.states,
            fps=args.fps,
            dpi=args.dpi,
            still_frames=args.still_frames,
        )
    if args.movie in ("all", "failure"):
        optimization_movie(
            args.failure_run,
            args.output_directory / f"sim_{args.failure_run}_optimization.mp4",
            nstates=args.states,
            fps=args.fps,
            dpi=args.dpi,
            still_frames=args.still_frames,
        )


if __name__ == "__main__":
    main()
