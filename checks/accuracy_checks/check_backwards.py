from __future__ import annotations

from pathlib import Path
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
LOG_DIR = HERE / "logs" / "backwards"

ntc_values = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def _float_key(value: float) -> str:
    return f"{value:.12g}".replace("-", "m").replace(".", "p").replace("+", "")


def setup_name(N: int, soft: float, dsteps: int, integrator: str = "float") -> str:
    return f"N{N:g}_soft{_float_key(soft)}_dsteps{dsteps:g}_{integrator}"


def precision_suffix(double: bool) -> str:
    return "_double" if double else ""


def cache_path(N: int, soft: float, ntc: float, dsteps: int, double: bool = False, integrator: str = "float") -> Path:
    return LOG_DIR / f"{setup_name(N, soft, dsteps, integrator)}{precision_suffix(double)}" / f"eps_ntc{_float_key(ntc)}_seed42.npz"


def get_err_estim(
    N: int = int(1e4),
    soft: float = 1e-1,
    ntc: float = 1.0,
    dsteps: int = 100,
    double: bool = False,
    integrator: str = "float",
) -> jax.Array:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(N=N, soft=soft, ntc=ntc, dsteps=dsteps, double=double, integrator=integrator)
    path.parent.mkdir(parents=True, exist_ok=True)
    nsteps = int(dsteps * ntc)
    dtype = np.float64 if double else np.float32

    if path.exists():
        print(f"Loading cached eps from {path}")
        with np.load(path) as data:
            with jax.enable_x64(double):
                return jnp.asarray(data["eps"])

    print(f"Running backward integration check and writing {path}")
    with jax.enable_x64(double):
        prof = aegis.profiles.HernquistProfile(a=1.0, M=1.0)
        np.random.seed(42)
        pos, vel, mass = prof.sample_particles(N, "pos_vel_m")
        part0 = fmdj.data.Particles(
            pos=np.asarray(pos, dtype=dtype),
            vel=np.asarray(vel, dtype=dtype),
            mass=np.asarray(mass, dtype=dtype),
        )

        cfg = fmdj.SimConfig()
        cfg.force.kernel.softening = soft
        if double:
            cfg.force.p = 4
        if integrator == "int32":
            cfg.integrator = fmdj.DKDLatticeConfig(dx=1e-5, dv=1e-5, int_dtype=jnp.int32)
        elif integrator == "int64":
            cfg.integrator = fmdj.DKDLatticeConfig(dx=1e-5, dv=1e-5, int_dtype=jnp.int64)
        elif integrator == "kdk":
            cfg.integrator = fmdj.KDKConfig()
        elif integrator != "float":
            raise ValueError(f"Unknown integrator {integrator!r}. Expected 'float', 'kdk', 'int32', or 'int64'.")

        tend = prof.tcirc(1.0) * ntc
        ts = jnp.linspace(0.0, tend, nsteps + 1, dtype=part0.pos.dtype)
        part = fmdj.time_integration.simulate.jit(part0, ts=ts, cfg=cfg)
        part0b = fmdj.time_integration.simulate.jit(part, ts=ts[::-1], cfg=cfg)

        eps = (
            2.0
            * jnp.linalg.norm(part0.pos - part0b.pos, axis=-1)
            / (jnp.linalg.norm(part0.pos, axis=-1) + jnp.linalg.norm(part0b.pos, axis=-1))
        )
        eps = jax.block_until_ready(eps)

    np.savez_compressed(path, eps=np.asarray(eps))
    return eps


def plot_hist(
    N: int = int(1e4),
    soft: float = 1e-1,
    dsteps: int = 100,
    title: str | None = None,
    double: bool = False,
    integrator: str = "float",
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    if double:
        bins = np.linspace(-18, 0, 101)
    else:
        bins = np.linspace(-9, 0, 101)

    for ntc in ntc_values:
        eps = get_err_estim(N=N, soft=soft, ntc=ntc, dsteps=dsteps, double=double, integrator=integrator)
        ax.hist(
            np.log10(np.asarray(eps) + 1e-19),
            bins=bins,
            density=True,
            alpha=0.5,
            label=fr"{ntc:g} $t_c$",
        )

    ax.set_xlabel(r"log10($\epsilon_x$)")
    ax.set_ylabel(r"d$N$/dlog10($\epsilon_x$)")
    ax.legend(loc="upper right")
    ax.set_title(title if title is not None else fr"N={N:g}, soft={soft:g}, dt={1 / dsteps:g}")
    fig.tight_layout()

    path = LOG_DIR / f"backwards_{setup_name(N, soft, dsteps, integrator)}{precision_suffix(double)}.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Wrote {path}")
    plt.close(fig)

def plot_line(
    ax,
    N: int = int(1e4),
    soft: float = 1e-1,
    dsteps: int = 100,
    title: str | None = None,
    double: bool = False,
    integrator: str = "float",
) -> None:
    perc = []
    for ntc in ntc_values:
        eps = get_err_estim(N=N, soft=soft, ntc=ntc, dsteps=dsteps, double=double, integrator=integrator)
        perc.append(np.percentile(eps, 90))
    label = title if title is not None else fr"N={N:g}, soft={soft:g}, dt={1 / dsteps:g}"
    ax.plot(ntc_values, perc, label=label, marker="o")

    ax.set_xlabel(r"$t / t_c$")
    ax.set_ylabel(r"$\epsilon_x$ (90th percentile)")

if __name__ == "__main__":
    plot_hist(N=int(1e4), soft=0.05, dsteps=100, title=r"N=$10^4$, soft=0.05, dt=0.01")
    plot_hist(N=int(1e4), soft=1e-1, dsteps=100, title=r"N=$10^4$, soft=0.1, dt=0.01")
    plot_hist(N=int(1e4), soft=1e-1, dsteps=100, title=r"N=$10^4$, soft=0.1, dt=0.01, DKD int32", double=False, integrator="int32")
    plot_hist(N=int(1e4), soft=1e-1, dsteps=100, title=r"N=$10^4$, soft=0.1, dt=0.01, DKD int64", double=False, integrator="int64")
    plot_hist(N=int(1e4), soft=1e-1, dsteps=100, title=r"N=$10^4$, soft=0.1, dt=0.01, double", double=True)
    plot_hist(N=int(1e4), soft=1e-1, dsteps=200, title=r"N=$10^4$, soft=0.1, dt=0.005")
    plot_hist(N=int(1e5), soft=1e-1, dsteps=100, title=r"N=$10^5$, soft=0.1, dt=0.01")
    plot_hist(N=int(1e5), soft=1e-2, dsteps=100, title=r"N=$10^5$, soft=0.01, dt=0.01")
    plot_hist(N=int(1e5), soft=5e-2, dsteps=100, title=r"N=$10^5$, soft=0.05, dt=0.01")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot_line(ax, N=int(1e4), soft=1e-1, dsteps=100, title=r"N=$10^4$, soft=0.1")
    # plot_line(ax, N=int(1e4), soft=1e-1, dsteps=10, title=r"N=$10^4$, soft=0.1, dt=0.1")
    # plot_line(ax, N=int(1e4), soft=1e-1, dsteps=200, title=r"N=$10^4$, soft=0.1, dt=0.005")
    plot_line(ax, N=int(1e4), soft=1e-1, dsteps=100, title=r"N=$10^4$, soft=0.1, dt=0.01, double", double=True)
    plot_line(ax, N=int(1e5), soft=1e-1, dsteps=100, title=r"N=$10^5$, soft=0.1")
    plot_line(ax, N=int(1e6), soft=1e-1, dsteps=100, title=r"N=$10^6$, soft=0.1")
    plot_line(ax, N=int(1e5), soft=5e-2, dsteps=100, title=r"N=$10^5$, soft=0.05")
    # plot_line(ax, N=int(1e5), soft=1e-2, dsteps=100, title=r"N=$10^5$, soft=0.01, dt=0.01")
    plot_line(ax, N=int(1e6), soft=5e-2, dsteps=100, title=r"N=$10^6$, soft=0.01")
    plot_line(ax, N=int(1e4), soft=1e-1, dsteps=100, title=r"N=$10^4$, soft=0.1, dt=0.01, DKD int32", double=False, integrator="int32")
    plot_line(ax, N=int(1e4), soft=1e-1, dsteps=100, title=r"N=$10^4$, soft=0.1, dt=0.01, DKD int64", double=False, integrator="int64")
    ax.loglog()
    ax.legend(loc="upper left")
    plt.axhline(1., linestyle="dashed", color="black")
    fig.tight_layout()
    fig.savefig(LOG_DIR / f"backwards_vs_time.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)
