from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jzfmm-matplotlib")

import aegis
import jzfmm
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs" / "hernquist_gradients"

ts = (0., 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)#, 8.0, 9.0, 10.0)
# ts = (0., 1.0, 2.0, 3.0, 4.0)

FIELDNAMES = (
    "N",
    "mode",
    "precision_mode",
    "force_mode",
    "loss_mode",
    "ntc",
    "softening",
    "par1",
    "par2",
    "par3",
    "target_dloss",
    "actual_dloss",
)
KEY_FIELDNAMES = FIELDNAMES[:-2]

PRECISION_MODES = ("float", "int", "double")
FORCE_MODES = ("fmm", "direct")
LOSS_MODES = ("com_y", "radius", "radius2")
LATTICE_LIMIT = 50.0
DEFAULT_OPENING_ANGLE = jzfmm.OpeningByAngle().theta


def _key_value(value: int | float | str) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def results_file() -> Path:
    return LOG_DIR / "loss_experiment.csv"


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row.get(name, "") for name in KEY_FIELDNAMES)


def _experiment_key(
    N: int,
    perturbation: str,
    precision_mode: str,
    force_mode: str,
    p_order: int,
    loss_mode: str,
    ntc: float,
    softening: float,
    par1: float,
    opening_angle: float,
) -> tuple[str, ...]:
    return tuple(
        _key_value(value)
        for value in (
            N, perturbation, precision_mode, force_mode, loss_mode, ntc, softening,
            par1, p_order, opening_angle,
        )
    )


def load_cached_result(
    N: int,
    perturbation: str,
    precision_mode: str,
    force_mode: str,
    p_order: int,
    loss_mode: str,
    ntc: float,
    softening: float,
    par1: float,
    opening_angle: float,
):
    path = results_file()
    if not path.exists():
        return None

    key = _experiment_key(
        N, perturbation, precision_mode, force_mode, p_order, loss_mode, ntc, softening, par1, opening_angle,
    )
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if _row_key(row) == key:
                print(f"Loading cached loss experiment from {path}: {key}")
                return float(row["target_dloss"]), float(row["actual_dloss"])

    return None


def save_result(
    N: int,
    perturbation: str,
    precision_mode: str,
    force_mode: str,
    p_order: int,
    loss_mode: str,
    ntc: float,
    softening: float,
    par1: float,
    opening_angle: float,
    target_dloss: float,
    actual_dloss: float,
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = results_file()
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "N": _key_value(N),
                "mode": perturbation,
                "precision_mode": precision_mode,
                "force_mode": force_mode,
                "loss_mode": loss_mode,
                "ntc": _key_value(ntc),
                "softening": _key_value(softening),
                "par1": _key_value(par1),
                "par2": _key_value(p_order),
                "par3": _key_value(opening_angle),
                "target_dloss": f"{target_dloss:.16e}",
                "actual_dloss": f"{actual_dloss:.16e}",
            }
        )


@lru_cache
def sample_ics(N: int, precision_mode: str):
    dtype = np.float64 if precision_mode == "double" else np.float32
    prof = aegis.profiles.HernquistProfile(a=1.0, M=1.0)
    np.random.seed(43)
    pos, vel, mass = prof.sample_particles(N, "pos_vel_m", ramax=20.0)
    return np.asarray(pos, dtype=dtype), np.asarray(vel, dtype=dtype), np.asarray(mass, dtype=dtype)


def sim_config(
    precision_mode: str,
    force_mode: str,
    softening: float,
    p_order: int,
    opening_angle: float,
):
    if force_mode not in FORCE_MODES:
        raise ValueError(f"Unknown force mode {force_mode!r}. Expected 'fmm' or 'direct'.")

    cfg = jzfmm.SimConfig()
    cfg.force.kernel.softening = softening
    cfg.force.p = p_order
    cfg.force.opening = jzfmm.OpeningByAngle(theta=opening_angle)
    if force_mode == "direct":
        cfg.force = jzfmm.DirectSummationConfig(kernel=jzfmm.PlummerKernel(softening=softening))
    if precision_mode == "int":
        intmax = np.iinfo(np.int32).max
        prof = aegis.profiles.HernquistProfile(a=1.0, M=1.0)
        cfg.integrator = jzfmm.DKDLatticeConfig(
            dx=LATTICE_LIMIT / intmax,
            dv=50.0 * prof.vcirc(1.0) / intmax,
            int_dtype=jnp.int32,
        )
    cfg.force.alloc_fac_ilist = cfg.force.alloc_fac_ilist * (0.8/opening_angle)**3.
    return cfg


def scalar_loss(part, loss_mode: str):
    mass = jnp.broadcast_to(part.mass, part.pos.shape[:-1])
    mtot = jnp.sum(mass)
    if loss_mode == "com_y":
        return jnp.sum(mass * part.pos[:, 1]) / mtot
    if loss_mode == "radius":
        return jnp.sum(mass * jnp.linalg.norm(part.pos, axis=-1)) / mtot
    if loss_mode == "radius2":
        return jnp.sum(mass * jnp.sum(part.pos**2, axis=-1)) / mtot
    raise ValueError(f"Unknown loss mode {loss_mode!r}. Expected 'com_y', 'radius', or 'radius2'.")


def simulate_loss(
    part0,
    precision_mode: str,
    force_mode: str,
    p_order: int,
    opening_angle: float,
    loss_mode: str,
    ntc: float,
    softening: float,
):
    cfg = sim_config(precision_mode, force_mode, softening, p_order, opening_angle)
    if ntc != 0.0:
        prof = aegis.profiles.HernquistProfile(a=1.0, M=1.0)
        ts_here = jnp.linspace(0.0, ntc * prof.tcirc(1.0), int(ntc * 100) + 1, dtype=part0.pos.dtype)
        part0 = jzfmm.time_integration.simulate(part0, ts=ts_here, cfg=cfg)
    return scalar_loss(part0, loss_mode)


@lru_cache
def gradient_result(
    N: int,
    precision_mode: str,
    force_mode: str,
    p_order: int,
    opening_angle: float,
    loss_mode: str,
    ntc: float,
    softening: float,
):
    double = precision_mode == "double"
    with jax.enable_x64(double):
        pos, vel, mass = sample_ics(N, precision_mode)
        part0 = jzfmm.data.Particles(
            pos=jnp.asarray(pos),
            vel=jnp.asarray(vel),
            mass=jnp.asarray(mass),
        )

        loss1, gpart0 = jax.jit(jax.value_and_grad(
            lambda part: simulate_loss(
                part, precision_mode, force_mode, p_order, opening_angle, loss_mode, ntc, softening,
            )
        ))(part0)

        return (
            float(jax.block_until_ready(loss1)),
            pos,
            vel,
            mass,
            np.asarray(jax.block_until_ready(gpart0.pos)),
            np.asarray(jax.block_until_ready(gpart0.vel)),
            np.asarray(jax.block_until_ready(gpart0.mass)),
        )


def loss_experiment(
    N: int = int(1e4),
    perturbation: str = "pos",
    loss_mode: str = "com_y",
    ntc: float = 2.0,
    alpha_frac: float = 1e-4,
    mass_fac: float = 1e-4,
    dscale_fac: float = 1e-4,
    softening: float = 0.1,
    mode: str = "float",
    force_mode: str = "fmm",
    p_order: int = 4,
    opening_angle: float = DEFAULT_OPENING_ANGLE,
) -> tuple[float, float]:
    if mode not in PRECISION_MODES:
        raise ValueError(f"Unknown precision mode {mode!r}. Expected 'float', 'int', or 'double'.")
    if force_mode not in FORCE_MODES:
        raise ValueError(f"Unknown force mode {force_mode!r}. Expected 'fmm' or 'direct'.")
    if loss_mode not in LOSS_MODES:
        raise ValueError(f"Unknown loss mode {loss_mode!r}. Expected 'com_y', 'radius', or 'radius2'.")

    if perturbation in ("pos", "com"):
        par1 = alpha_frac
    elif perturbation == "mass":
        par1 = mass_fac
    elif perturbation == "scale":
        par1 = dscale_fac
    else:
        raise ValueError(f"Unknown perturbation {perturbation!r}. Expected 'pos', 'mass', 'com', or 'scale'.")

    cached = load_cached_result(
        N, perturbation, mode, force_mode, p_order, loss_mode, ntc, softening, par1, opening_angle,
    )
    if cached is not None:
        return cached

    print(
        "Running loss experiment "
        f"N={N:g}, perturbation={perturbation}, precision={mode}, force={force_mode}, p={p_order}, "
        f"theta={opening_angle:g}, loss_mode={loss_mode}, "
        f"ntc={ntc:g}, softening={softening:g}, par1={par1:g}"
    )

    double = mode == "double"
    dtype = np.float64 if double else np.float32
    with jax.enable_x64(double):
        a = 1.0
        alpha = alpha_frac * a
        loss1, pos, vel, mass, gpos, gvel, gmass = gradient_result(
            N, mode, force_mode, p_order, opening_angle, loss_mode, ntc, softening,
        )
        part0 = jzfmm.data.Particles(
            pos=jnp.asarray(pos, dtype=dtype),
            vel=jnp.asarray(vel, dtype=dtype),
            mass=jnp.asarray(mass, dtype=dtype),
        )
        gpos = jnp.asarray(gpos, dtype=dtype)
        gvel = jnp.asarray(gvel, dtype=dtype)
        gmass = jnp.asarray(gmass, dtype=dtype)

        if perturbation == "pos":
            assert loss_mode in LOSS_MODES
            dpos = alpha * gpos / (jnp.linalg.norm(gpos, axis=-1, keepdims=True) + 1e-30)
            target_dloss = jnp.sum(gpos * dpos)
            part2 = replace(part0, pos=part0.pos + dpos)
        elif perturbation == "mass":
            dmass = mass_fac * part0.mass * jnp.sign(gmass)
            target_dloss = jnp.sum(gmass * dmass)
            part2 = replace(part0, mass=part0.mass + dmass)
        elif perturbation == "com":
            assert loss_mode == "com_y", "The com perturbation is only intended for loss_mode='com_y'."
            gpos_com = jnp.sum(gpos, axis=0)
            dpos = alpha * gpos_com / (jnp.linalg.norm(gpos_com) + 1e-30)
            target_dloss = jnp.sum(gpos_com * dpos)
            part2 = replace(part0, pos=part0.pos + dpos)
        elif perturbation == "scale":
            assert loss_mode in ("radius", "radius2"), (
                "The scale perturbation is only intended for loss_mode='radius' or 'radius2'."
            )
            scale_fac = 1.0 + dscale_fac
            dpos = (scale_fac - 1.0) * part0.pos
            dvel = (scale_fac**-0.5 - 1.0) * part0.vel * 0.
            target_dloss = jnp.sum(gpos * dpos) + jnp.sum(gvel * dvel)
            part2 = replace(part0, pos=part0.pos + dpos, vel=part0.vel + dvel)

        loss2 = jax.jit(
            lambda part: simulate_loss(part, mode, force_mode, p_order, opening_angle, loss_mode, ntc, softening)
        )(part2)
        target_dloss = float(jax.block_until_ready(target_dloss))
        actual_dloss = float(jax.block_until_ready(loss2 - loss1))

    save_result(
        N, perturbation, mode, force_mode, p_order, loss_mode, ntc, softening, par1, opening_angle,
        target_dloss, actual_dloss,
    )
    return target_dloss, actual_dloss


def plot_loss_experiment(
    N: int = int(1e4),
    loss_mode: str = "com_y",
    alpha_frac: float = 1e-4,
    mass_fac: float = 1e-4,
    dscale_fac: float = 1e-4,
    softening: float = 0.1,
    mode: str = "float",
    force_mode: str = "fmm",
    p_order: int = 4,
    opening_angle: float = DEFAULT_OPENING_ANGLE,
):
    if mode not in PRECISION_MODES:
        raise ValueError(f"Unknown precision mode {mode!r}. Expected 'float', 'int', or 'double'.")
    if force_mode not in FORCE_MODES:
        raise ValueError(f"Unknown force mode {force_mode!r}. Expected 'fmm' or 'direct'.")
    if loss_mode not in LOSS_MODES:
        raise ValueError(f"Unknown loss mode {loss_mode!r}. Expected 'com_y', 'radius', or 'radius2'.")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    perturbations = ("pos", "mass", "com") if loss_mode == "com_y" else ("pos", "mass", "scale")
    for perturbation in perturbations:
        res = [
            loss_experiment(
                N=N,
                perturbation=perturbation,
                loss_mode=loss_mode,
                ntc=t,
                alpha_frac=alpha_frac,
                mass_fac=mass_fac,
                dscale_fac=dscale_fac,
                softening=softening,
                mode=mode,
                force_mode=force_mode,
                p_order=p_order,
                opening_angle=opening_angle,
            )
            for t in ts
        ]
        target_dloss, actual_dloss = np.transpose(res)
        ax.plot(ts, actual_dloss / target_dloss, marker="o", label=perturbation)

    ax.axhline(1.0, linestyle="dashed", color="black")
    ax.axhline(0.0, linestyle="dashed", color="black")
    ax.set_xlabel(r"$t / t_c$")
    ax.set_ylabel(r"$\Delta L_\mathrm{measured} / \Delta L_\mathrm{target}$")
    ax.set_title(
        fr"N={N:g}, loss={loss_mode}, force={force_mode}, p={p_order}, $\theta$={opening_angle:g}, "
        fr"$\alpha/a$={alpha_frac:g}, "
        fr"mass_fac={mass_fac:g}, dscale_fac={dscale_fac:g}, softening={softening:g}"
    )
    ax.legend()
    ax.set_ylim(-0.2, 2.2)
    fig.tight_layout()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / (
        f"loss_experiment_N{N:g}_{loss_mode}_{force_mode}_p{p_order}_theta{_key_value(opening_angle)}"
        f"_alpha{_key_value(alpha_frac)}"
        f"_mass{_key_value(mass_fac)}_scale{_key_value(dscale_fac)}"
        f"_soft{_key_value(softening)}_{mode}.pdf"
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Wrote {path}")
    plt.close(fig)


def plot_case(
    ax,
    loss_mode: str,
    N: int,
    p_order: int,
    opening_angle: float,
    color,
    label: str,
    alpha_frac: float = 1e-4,
    mass_fac: float = 1e-4,
    dscale_fac: float = 1e-4,
    softening: float = 0.1,
    mode: str = "int",
    force_mode: str = "fmm",
):
    for perturbation, linestyle in (("pos", "-"), ("mass", "--")):
        res = [
            loss_experiment(
                N=N,
                perturbation=perturbation,
                loss_mode=loss_mode,
                ntc=t,
                alpha_frac=alpha_frac,
                mass_fac=mass_fac,
                dscale_fac=dscale_fac,
                softening=softening,
                mode=mode,
                force_mode=force_mode,
                p_order=p_order,
                opening_angle=opening_angle,
            )
            for t in ts
        ]
        target_dloss, actual_dloss = np.transpose(res)
        ax.plot(
            ts,
            actual_dloss / target_dloss,
            marker="o",
            linestyle=linestyle,
            color=color,
            alpha=0.6,
            label=label if perturbation == "pos" else None,
        )


def plot_experiments(
    alpha_frac: float = 1e-4,
    mass_fac: float = 1e-4,
    dscale_fac: float = 1e-4,
    softening: float = 0.1,
    mode: str = "int",
    force_mode: str = "fmm",
):
    scenarios = (
        (int(1e4), 5, 0.8, r"$N=10^4$, default acc."),
        # (int(1e5), 5, 0.8, r"$N=10^5$, $p=5$, $\theta=0.8$"),
        # (int(1e4), 5, 0.4, r"$N=10^4$, $p=5$, $\theta=0.4$"),
        # (int(1e5), 5, 0.4, r"$N=10^5$, $p=5$, $\theta=0.4$"),
        (int(1e4), 7, 0.4, r"$N=10^4$, high acc."),
        # (int(1e4), 7, 0.8, r"$N=10^4$, $p=7$, $\theta=0.8$"),
        # (int(1e5), 7, 0.8, r"$N=10^5$, $p=7$, $\theta=0.8$"),
        (int(1e5), 7, 0.4, r"$N=10^5$, high acc."),
        # (int(4e5), 7, 0.4, r"$N=4 \times 10^5$, high acc."),
        (int(1e6), 7, 0.4, r"$N=10^6$, high acc."),
    )

    fig, axes = plt.subplots(2, 1, figsize=(5.5, 5), sharex=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for ax, loss_mode, title in (
        (axes[0], "com_y", r"$L = \langle x \rangle$"),
        (axes[1], "radius", r"$L = \langle r \rangle$"),
    ):
        for i, (N, p_order, opening_angle, label) in enumerate(scenarios):
            plot_case(
                ax,
                loss_mode=loss_mode,
                N=N,
                p_order=p_order,
                opening_angle=opening_angle,
                color=colors[i],
                label=label,
                alpha_frac=alpha_frac,
                mass_fac=mass_fac,
                dscale_fac=dscale_fac,
                softening=softening,
                mode=mode,
                force_mode=force_mode,
            )
        ax.axhline(1.0, linestyle=":", color="black", linewidth=1.0)
        # ax.axhline(0.0, linestyle=":", color="black", linewidth=1.0)
        ax.set_ylabel(r"$\Delta L_\mathrm{actual} / \Delta L_\mathrm{grad}$")
        ax.set_title(title)
        ax.set_ylim(0., 1.5)
        ax.set_xlim(0., 8.)

    axes[0].plot([], color="black", ls="solid", label=r"pos displacement")
    axes[0].plot([], color="black",ls="dashed", label=r"mass displacement")

    axes[0].legend(ncol=2, loc="lower center")
    # axes[1].legend(loc="upper center")
    axes[1].set_xlabel(r"$t / t_c$")

    fig.tight_layout()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(LOG_DIR / "loss_experiment.pdf", bbox_inches="tight")
    print(f"Wrote {LOG_DIR / "loss_experiment.pdf"}")
    plt.close(fig)


if __name__ == "__main__":
    # plot_loss_experiment(N=int(1e4), loss_mode="com_y", alpha_frac=1e-3)
    # plot_loss_experiment(N=int(1e4), loss_mode="com_y", alpha_frac=1e-3, mode="int")
    # plot_loss_experiment(N=int(1e4), loss_mode="radius", alpha_frac=1e-3, dscale_fac=1e-5)
    # plot_loss_experiment(N=int(1e4), loss_mode="radius", alpha_frac=1e-3, force_mode="direct", dscale_fac=1e-5)
    # plot_loss_experiment(N=int(1e4), loss_mode="radius", alpha_frac=1e-3, mode="int", dscale_fac=1e-5)
    # plot_loss_experiment(N=int(1e5), loss_mode="com_y", alpha_frac=1e-3)
    # plot_loss_experiment(N=int(1e5), loss_mode="radius", alpha_frac=1e-3, mode="int", dscale_fac=1e-4)
    # plot_loss_experiment(N=int(1e5), loss_mode="radius", alpha_frac=1e-3, mode="int", dscale_fac=1e-2)
    # plot_loss_experiment(N=int(1e4), loss_mode="radius", alpha_frac=1e-3, mode="int", dscale_fac=1e-3, force_mode="direct")
    # plot_loss_experiment(N=int(1e4), loss_mode="radius", alpha_frac=1e-3, mode="int", dscale_fac=1e-4, p_order=3)
    # plot_loss_experiment(N=int(1e4), loss_mode="radius", alpha_frac=1e-3, mode="int", dscale_fac=1e-4, p_order=5)
    # plot_loss_experiment(N=int(1e4), loss_mode="radius", alpha_frac=1e-3, mode="int", dscale_fac=1e-4, p_order=7)
    # plot_loss_experiment(N=int(1e4), loss_mode="radius", alpha_frac=1e-3, mode="int", dscale_fac=1e-4, p_order=5, opening_angle=0.4)
    # plot_loss_experiment(N=int(1e4), loss_mode="radius", alpha_frac=1e-4, mode="int", dscale_fac=1e-4, p_order=5, opening_angle=0.4)
    # plot_loss_experiment(N=int(1e4), loss_mode="radius", alpha_frac=1e-4, mode="int", dscale_fac=1e-5, p_order=5, opening_angle=0.4)
    # plot_loss_experiment(N=int(1e4), loss_mode="radius", alpha_frac=1e-4, mode="int", dscale_fac=1e-5, p_order=7, opening_angle=0.4)
    # plot_loss_experiment(N=int(1e4), loss_mode="radius", alpha_frac=1e-4, mode="int", dscale_fac=1e-5, p_order=7)

    # plot_loss_experiment(N=int(1e5), loss_mode="radius", alpha_frac=1e-4, mode="int", dscale_fac=1e-5, p_order=7, opening_angle=0.4)

    # plot_loss_experiment(N=int(1e5), loss_mode="radius", alpha_frac=1e-3, mode="int", dscale_fac=1e-4, p_order=3)
    # plot_loss_experiment(N=int(1e5), loss_mode="radius", alpha_frac=1e-3, mode="int", dscale_fac=1e-4, p_order=5)
    # plot_loss_experiment(N=int(1e5), loss_mode="radius", alpha_frac=1e-3, mode="int", dscale_fac=1e-4, p_order=7)

    # plot_loss_experiment(N=int(1e5), loss_mode="radius", alpha_frac=1e-3, mode="int", dscale_fac=1e-6)
    # plot_loss_experiment(N=int(1e6), loss_mode="radius", alpha_frac=1e-3, mode="int", dscale_fac=1e-4)

    plot_experiments()
