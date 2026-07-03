from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import csv
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fmdj-matplotlib")

import aegis
import fmdj
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs" / "hernquist_gradients"

ts = (0., 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0)

FIELDNAMES = (
    "N",
    "mode",
    "ntc",
    "ltfrac",
    "softening",
    "disp",
    "target_dloss",
    "actual_dloss",
)


def _key_value(value: int | float | str) -> str:
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def precision_suffix(double: bool) -> str:
    return "_double" if double else ""


def results_file(double: bool) -> Path:
    return LOG_DIR / f"loss_experiment{precision_suffix(double)}.csv"


def _row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[name] for name in FIELDNAMES[:6])


def _experiment_key(N: int, mode: str, ntc: float, ltfrac: float, softening: float, disp: float) -> tuple[str, ...]:
    return tuple(_key_value(value) for value in (N, mode, ntc, ltfrac, softening, disp))


def load_cached_result(N: int, mode: str, ntc: float, ltfrac: float, softening: float, disp: float, double: bool):
    path = results_file(double)
    if not path.exists():
        return None

    key = _experiment_key(N, mode, ntc, ltfrac, softening, disp)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if _row_key(row) == key:
                print(f"Loading cached loss experiment from {path}: {key}")
                return float(row["target_dloss"]), float(row["actual_dloss"])

    return None


def save_result(
    N: int,
    mode: str,
    ntc: float,
    ltfrac: float,
    softening: float,
    disp: float,
    target_dloss: float,
    actual_dloss: float,
    double: bool,
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = results_file(double)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "N": _key_value(N),
                "mode": mode,
                "ntc": _key_value(ntc),
                "ltfrac": _key_value(ltfrac),
                "softening": _key_value(softening),
                "disp": _key_value(disp),
                "target_dloss": f"{target_dloss:.16e}",
                "actual_dloss": f"{actual_dloss:.16e}",
            }
        )


def loss(part, part0):
    cfg_fmm = fmdj.config.FMMConfig()
    cfg_fmm.kernel = fmdj.config.SoftenedDistanceKernel(1e-1)
    cfg_fmm.p = 4
    return fmdj.loss.maximum_mean_discrepancy(part, part0, cfg_fmm=cfg_fmm)


def loss_experiment(
    N: int = int(1e4),
    mode: str = "pos",
    ntc: float = 2.0,
    ltfrac: float = 1e-2,
    softening: float = 0.1,
    disp: float = 0.1,
    double: bool = False,
) -> tuple[float, float]:
    cached = load_cached_result(N, mode, ntc, ltfrac, softening, disp, double)
    if cached is not None:
        return cached

    print(
        "Running loss experiment "
        f"N={N:g}, mode={mode}, ntc={ntc:g}, ltfrac={ltfrac:g}, softening={softening:g}, disp={disp:g}"
    )
    dtype = np.float64 if double else np.float32
    with jax.enable_x64(double):
        prof = aegis.profiles.HernquistProfile(a=1.0, M=1.0)

        np.random.seed(42)
        pos, vel, mass = prof.sample_particles(N, "pos_vel_m", ramax=1e3)
        part_ref = fmdj.data.Particles(
            pos=np.asarray(pos, dtype=dtype),
            vel=np.asarray(vel, dtype=dtype),
            mass=np.asarray(mass, dtype=dtype),
        )

        np.random.seed(43)
        pos, vel, mass = prof.sample_particles(N, "pos_vel_m", ramax=1e3)
        part0 = fmdj.data.Particles(
            pos=np.asarray(pos, dtype=dtype),
            vel=np.asarray(vel, dtype=dtype),
            mass=np.asarray(mass, dtype=dtype),
        )
        part0 = fmdj.time_integration.clean_particles(part0)

        def mysim_loss(part0):
            cfg = fmdj.SimConfig()
            cfg.centered = False
            cfg.force.kernel.softening = softening
            if double:
                cfg.force.p = 4
            if ntc == 0.0:
                part = part0
            else:
                part = fmdj.time_integration.simulate(part0, tend=ntc*prof.tcirc(1.0), nsteps=int(ntc * 100), cfg=cfg)

            ex = part_ref.pos / jnp.linalg.norm(part_ref.pos, axis=-1, keepdims=True)
            return loss(part.pos, jax.lax.stop_gradient(part_ref.pos + disp * ex))

        mysim_loss.val_grad = jax.jit(jax.value_and_grad(mysim_loss))
        mysim_loss.jit = jax.jit(mysim_loss)

        loss1, gpart0 = mysim_loss.val_grad(part0)
        target_dloss = ltfrac * loss1

        if mode == "pos":
            gamp = jnp.sum(gpart0.pos**2)
            part2 = replace(part0, pos=part0.pos + target_dloss * gpart0.pos / gamp)
        elif mode == "mass":
            gamp = jnp.sum(gpart0.mass**2)
            part2 = replace(part0, mass=part0.mass + target_dloss * gpart0.mass / gamp)
        else:
            raise ValueError(f"Unknown mode {mode!r}. Expected 'pos' or 'mass'.")

        loss2 = mysim_loss.jit(part2)
        target_dloss = float(jax.block_until_ready(target_dloss))
        actual_dloss = float(jax.block_until_ready(loss2 - loss1))

    save_result(N, mode, ntc, ltfrac, softening, disp, target_dloss, actual_dloss, double)
    return target_dloss, actual_dloss


def plot_loss_experiment(
    N: int = int(1e4),
    ltfrac: float = 1e-2,
    softening: float = 0.1,
    disp: float = 0.5,
    double: bool = False,
    include_double: bool = True,
):
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    precision_cases = [(False, "-")]
    if include_double:
        precision_cases.append((True, "--"))
    elif double:
        precision_cases = [(True, "-")]

    for use_double, linestyle in precision_cases:
        precision_label = "double" if use_double else "float"
        for mode in ("pos", "mass"):
            times = ts[1:] if mode == "mass" else ts
            res = [
                loss_experiment(
                    N=N,
                    mode=mode,
                    ntc=t,
                    ltfrac=ltfrac,
                    softening=softening,
                    disp=disp,
                    double=use_double,
                )
                for t in times
            ]
            target_dloss, actual_dloss = np.transpose(res)
            ax.plot(
                times,
                actual_dloss / target_dloss,
                marker="o",
                linestyle=linestyle,
                label=f"{mode} ({precision_label})",
            )

    ax.axhline(1.0, linestyle="dashed", color="black")
    ax.axhline(0.0, linestyle="dashed", color="black")
    ax.set_xlabel(r"$t / t_c$")
    ax.set_ylabel(r"$\Delta L_\mathrm{measured} / \Delta L_\mathrm{target}$")
    ax.set_title(fr"N={N:g}, ltfrac={ltfrac:g}, softening={softening:g}, disp={disp:g}")
    ax.legend()
    ax.set_ylim(-0.2,1.2)
    fig.tight_layout()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    precision_tag = "_float_double" if include_double else precision_suffix(double)
    path = LOG_DIR / (
        f"loss_experiment_N{N:g}_ltfrac{_key_value(ltfrac)}_soft{_key_value(softening)}"
        f"_disp{_key_value(disp)}{precision_tag}.pdf"
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    plot_loss_experiment(N=int(1e4), ltfrac=0.01)
    plot_loss_experiment(N=int(1e5), ltfrac=0.01)
    plot_loss_experiment(N=int(1e4), ltfrac=0.001)
    # plot_loss_experiment(N=int(1e5), ltfrac=0.01)
