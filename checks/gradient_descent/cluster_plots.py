import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

import cluster as cl


MNRAS_PAGE_WIDTH = 7.1  # inches


def square_limits(particles, padding=0.03):
    xy = np.concatenate([
        np.asarray(part.pos[:, :2])
        for part in particles
    ])
    lower = xy.min(axis=0)
    upper = xy.max(axis=0)
    center = 0.5 * (lower + upper)
    size = np.max(upper - lower) * (1 + 2 * padding)
    return (
        center[0] - size / 2,
        center[0] + size / 2,
        center[1] - size / 2,
        center[1] + size / 2,
    )


def log_spaced_history_indices(history, best_index, nstates):
    loss = np.asarray(history["loss"][:best_index + 1])
    best_so_far = np.minimum.accumulate(loss)
    levels = np.geomspace(best_so_far[0], best_so_far[-1], nstates)

    indices = []
    for level in levels:
        reached = np.flatnonzero(best_so_far <= level)
        indices.append(reached[0] if len(reached) else best_index)
    return np.unique([0, *indices, best_index])


def parameter_com(parameters, config):
    exp_m = np.asarray(parameters["exp_m"])
    position = np.asarray(parameters["position"])
    mass_fraction = 1 / (1 + np.exp(-exp_m))
    log10_mass = (
        config.log10_mass_min
        + (config.log10_mass_max - config.log10_mass_min)
        * mass_fraction
    )
    mass = 10**log10_mass
    return np.sum(
        position * mass[..., None],
        axis=-2,
    ) / np.sum(mass, axis=-1)[..., None]


def add_host_markers(ax, rvir):
    for fraction, linestyle in (
        (0.25, ":"),
        (0.50, "--"),
        (0.75, "-."),
        (1.00, "-"),
    ):
        ax.add_patch(plt.Circle(
            (0, 0),
            fraction * rvir,
            fill=False,
            color="black",
            lw=0.7,
            ls=linestyle,
            zorder=1,
        ))
    ax.scatter(
        0,
        0,
        marker="*",
        s=32,
        color="black",
        zorder=3,
    )


def plot_runs(sims, noutputs=5, nstates=10):
    sims = np.atleast_1d(sims)
    nrows = len(sims)
    figure_height = 1.7 * nrows + 1.0
    time_legend_handles = None

    with plt.rc_context({
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }):
        fig, axs = plt.subplots(
            nrows,
            4,
            figsize=(MNRAS_PAGE_WIDTH, figure_height),
            squeeze=False,
        )
        fig.subplots_adjust(
            left=0.09,
            right=0.995,
            bottom=0.80 / figure_height,
            top=1 - 0.28 / figure_height,
            wspace=0.08,
            hspace=0.03,
        )

        for row, sim in enumerate(sims):
            result = np.load(f"logs/sim_{sim}.npz")
            config = cl.Config.from_json(f"logs/sim_{sim}.json")
            config.sim_config.logging.level = 0
            history = result["history"]
            best_index = np.argmin(history["loss"])
            best = history["parameters"][best_index]

            target_outputs = list(cl.get_particles(
                result["target_parameters"],
                config,
                seed=config.target_particle_seed,
                outputs=noutputs,
            ))
            inferred_outputs = list(cl.get_particles(
                best,
                config,
                seed=config.model_particle_seed,
                outputs=noutputs,
            ))
            target_particles = [part for _, part in target_outputs]
            inferred_particles = [part for _, part in inferred_outputs]
            target_final = target_particles[-1]
            inferred_final = inferred_particles[-1]

            # Final target and inferred distributions.
            axs[row, 0].scatter(
                target_final.pos[:, 0],
                target_final.pos[:, 1],
                marker=".",
                s=1,
                alpha=0.035,
                color="tab:blue",
                zorder=2,
            )
            axs[row, 0].scatter(
                inferred_final.pos[:, 0],
                inferred_final.pos[:, 1],
                marker=".",
                s=1,
                alpha=0.035,
                color="tab:red",
                zorder=2,
            )
            final_limits = square_limits([target_final, inferred_final])
            axs[row, 0].set_xlim(*final_limits[:2])
            axs[row, 0].set_ylim(*final_limits[2:])

            # Target and inferred time evolution.
            colors = plt.cm.viridis(np.linspace(0, 1, noutputs))
            for color, (time_gyr, particles) in zip(colors, target_outputs):
                axs[row, 1].scatter(
                    particles.pos[:, 0],
                    particles.pos[:, 1],
                    marker=".",
                    s=1,
                    alpha=0.025,
                    color=color,
                    zorder=2,
                )
            for color, (_, particles) in zip(colors, inferred_outputs):
                axs[row, 2].scatter(
                    particles.pos[:, 0],
                    particles.pos[:, 1],
                    marker=".",
                    s=1,
                    alpha=0.025,
                    color=color,
                    zorder=2,
                )

            shared_limits = square_limits(
                target_particles + inferred_particles
            )

            # Optimization path and selected final distributions.
            state_indices = log_spaced_history_indices(
                history,
                best_index,
                nstates,
            )
            state_colors = plt.cm.plasma(
                np.linspace(0, 1, len(state_indices))
            )
            initial_com_history = (
                parameter_com(history["parameters"], config) * 1_000
            )
            axs[row, 3].plot(
                initial_com_history[:best_index + 1, 0],
                initial_com_history[:best_index + 1, 1],
                color="black",
                lw=0.7,
                alpha=0.75,
                zorder=4,
            )

            for color, history_index in zip(state_colors, state_indices):
                outputs = list(cl.get_particles(
                    history["parameters"][history_index],
                    config,
                    seed=config.model_particle_seed,
                    outputs=2,
                ))
                final = outputs[-1][1]
                axs[row, 3].scatter(
                    final.pos[:, 0],
                    final.pos[:, 1],
                    marker=".",
                    s=1,
                    alpha=0.025,
                    color=color,
                    zorder=2,
                )
                initial_com = initial_com_history[history_index]
                axs[row, 3].scatter(
                    initial_com[0],
                    initial_com[1],
                    marker="o",
                    s=24,
                    color=color,
                    edgecolor="black",
                    linewidth=0.35,
                    zorder=5,
                    label=f"{history['step'][history_index]}",
                )

            optimization_legend = axs[row, 3].legend(
                title="step",
                loc="best",
                ncol=4,
                fontsize=8,
                title_fontsize=8,
                markerscale=0.85,
                handlelength=0.8,
                handletextpad=0.2,
                columnspacing=0.5,
                labelspacing=0.25,
                borderpad=0.3,
                framealpha=0.85,
            )
            optimization_legend.set_zorder(100)

            host_mass = config.sim_config.external_potential.mass
            rvir = cl.RvirOfMvir(host_mass) * 1_000
            for column in range(4):
                add_host_markers(axs[row, column], rvir)

            for column in range(1, 4):
                axs[row, column].set_xlim(*shared_limits[:2])
                axs[row, column].set_ylim(*shared_limits[2:])

            if time_legend_handles is None:
                time_legend_handles = [
                    Line2D(
                        [],
                        [],
                        marker="o",
                        ls="none",
                        color=color,
                        markerfacecolor=color,
                        markeredgecolor=color,
                        alpha=1,
                        label=f"{float(np.asarray(time_gyr)):.1g}Gyr",
                    )
                    for color, (time_gyr, _) in zip(
                        colors,
                        target_outputs,
                    )
                ]

            for column, ax in enumerate(axs[row]):
                ax.set_aspect("equal")
                ax.set_xticks([])
                ax.set_yticks([])
                if row == nrows - 1:
                    ax.set_xlabel("x [kpc]")
                if column == 0:
                    ax.set_ylabel("y [kpc]")

        titles = (
            "Final distributions",
            "True evolution",
            "Inferred evolution",
            "Optimization path",
        )
        for ax, title in zip(axs[0], titles):
            ax.set_title(title)

        distribution_handles = [
            Line2D(
                [], [], marker="o", ls="none", color="tab:blue",
                markerfacecolor="tab:blue", alpha=1, label="Target",
            ),
            Line2D(
                [], [], marker="o", ls="none", color="tab:red",
                markerfacecolor="tab:red", alpha=1, label="Inferred",
            ),
        ]
        host_handles = [
            Line2D(
                [], [], color="black", lw=0.8,
                label=r"$r_{\rm vir}$",
            ),
            Line2D(
                [], [], color="black", lw=0.8, ls="-.",
                label=r"$3r_{\rm vir}/4$",
            ),
            Line2D(
                [], [], color="black", lw=0.8, ls="--",
                label=r"$r_{\rm vir}/2$",
            ),
            Line2D(
                [], [], color="black", lw=0.8, ls=":",
                label=r"$r_{\rm vir}/4$",
            ),
            Line2D(
                [], [], marker="*", ls="none", color="black",
                markersize=7, label="Host centre",
            ),
        ]

        legend_handles = (
            distribution_handles
            + time_legend_handles
            + host_handles
        )
        legend_columns = (len(legend_handles) + 1) // 2
        top_row = legend_handles[:legend_columns]
        bottom_row = legend_handles[legend_columns:]
        legend_handles = []
        for index, handle in enumerate(top_row):
            legend_handles.append(handle)
            if index < len(bottom_row):
                legend_handles.append(bottom_row[index])

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01 / figure_height),
            ncol=legend_columns,
            frameon=False,
            columnspacing=0.7,
            handletextpad=0.3,
        )

    return fig, axs
