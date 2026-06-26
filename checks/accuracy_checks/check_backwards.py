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


def setup_name(N: int, soft: float, dsteps: int) -> str:
    return f"N{N:g}_soft{_float_key(soft)}_dsteps{dsteps:g}"


def cache_path(N: int, soft: float, ntc: float, dsteps: int) -> Path:
    return LOG_DIR / setup_name(N, soft, dsteps) / f"eps_ntc{_float_key(ntc)}_seed42.npz"


def get_err_estim(N: int = int(1e4), soft: float = 1e-1, ntc: float = 1.0, dsteps: int = 100) -> jax.Array:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = cache_path(N=N, soft=soft, ntc=ntc, dsteps=dsteps)
    path.parent.mkdir(parents=True, exist_ok=True)
    nsteps = int(dsteps * ntc)

    if path.exists():
        print(f"Loading cached eps from {path}")
        with np.load(path) as data:
            return jnp.asarray(data["eps"])

    print(f"Running backward integration check and writing {path}")
    prof = aegis.profiles.HernquistProfile(a=1.0, M=1.0)
    np.random.seed(42)
    pos, vel, mass = prof.sample_particles(N, "pos_vel_m")
    part0 = fmdj.data.Particles(pos=pos, vel=vel, mass=mass)

    cfg = fmdj.SimConfig()
    cfg.force.kernel.softening = soft
    cfg.centered = False

    tend = prof.tcirc(1.0) * ntc
    part = fmdj.time_integration.simulate.jit(part0, tend=tend, nsteps=nsteps, cfg=cfg)
    part0b = fmdj.time_integration.simulate.jit(part, tstart=tend, tend=0.0, nsteps=nsteps, cfg=cfg)

    eps = (
        2.0
        * jnp.linalg.norm(part0.apos() - part0b.apos(), axis=-1)
        / (jnp.linalg.norm(part0.apos(), axis=-1) + jnp.linalg.norm(part0b.apos(), axis=-1))
    )
    eps = jax.block_until_ready(eps)

    np.savez_compressed(path, eps=np.asarray(eps))
    return eps


def plot_hist(N: int = int(1e4), soft: float = 1e-1, dsteps: int = 100, title: str | None = None) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for ntc in ntc_values:
        eps = get_err_estim(N=N, soft=soft, ntc=ntc, dsteps=dsteps)
        ax.hist(
            np.log10(np.asarray(eps) + 1e-10),
            bins=np.linspace(-9, 0, 101),
            density=True,
            alpha=0.5,
            label=fr"{ntc:g} $t_c$",
        )

    ax.set_xlabel(r"log10($\epsilon_x$)")
    ax.set_ylabel(r"d$N$/dlog10($\epsilon_x$)")
    ax.legend(loc="upper right")
    ax.set_title(title if title is not None else fr"N={N:g}, soft={soft:g}, dt={1 / dsteps:g}")
    fig.tight_layout()

    path = LOG_DIR / f"backwards_{setup_name(N, soft, dsteps)}.pdf"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"Wrote {path}")
    plt.close(fig)

def plot_line(ax, N: int = int(1e4), soft: float = 1e-1, dsteps: int = 100, title: str | None = None) -> None:
    perc = []
    for ntc in ntc_values:
        eps = get_err_estim(N=N, soft=soft, ntc=ntc, dsteps=dsteps)
        perc.append(np.percentile(eps, 90))
    label = title if title is not None else fr"N={N:g}, soft={soft:g}, dt={1 / dsteps:g}"
    ax.plot(ntc_values, perc, label=label, marker="o")

    ax.set_xlabel(r"$t / t_c$")
    ax.set_ylabel(r"$\epsilon_x$ (90th percentile)")

if __name__ == "__main__":
    plot_hist(N=int(1e4), soft=0.05, dsteps=100, title=r"N=$10^4$, soft=0.05, dt=0.01")
    plot_hist(N=int(1e4), soft=1e-1, dsteps=100, title=r"N=$10^4$, soft=0.1, dt=0.01")
    plot_hist(N=int(1e4), soft=1e-1, dsteps=200, title=r"N=$10^4$, soft=0.1, dt=0.005")
    plot_hist(N=int(1e5), soft=1e-1, dsteps=100, title=r"N=$10^5$, soft=0.1, dt=0.01")
    plot_hist(N=int(1e5), soft=1e-2, dsteps=100, title=r"N=$10^5$, soft=0.01, dt=0.01")
    plot_hist(N=int(1e5), soft=5e-2, dsteps=100, title=r"N=$10^5$, soft=0.05, dt=0.01")

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    plot_line(ax, N=int(1e4), soft=1e-1, dsteps=100, title=r"N=$10^4$, soft=0.1")
    # plot_line(ax, N=int(1e4), soft=1e-1, dsteps=10, title=r"N=$10^4$, soft=0.1, dt=0.1")
    # plot_line(ax, N=int(1e4), soft=1e-1, dsteps=200, title=r"N=$10^4$, soft=0.1, dt=0.005")
    plot_line(ax, N=int(1e5), soft=1e-1, dsteps=100, title=r"N=$10^5$, soft=0.1")
    plot_line(ax, N=int(1e6), soft=1e-1, dsteps=100, title=r"N=$10^6$, soft=0.1")
    plot_line(ax, N=int(1e5), soft=5e-2, dsteps=100, title=r"N=$10^5$, soft=0.05")
    # plot_line(ax, N=int(1e5), soft=1e-2, dsteps=100, title=r"N=$10^5$, soft=0.01, dt=0.01")
    plot_line(ax, N=int(1e6), soft=5e-2, dsteps=100, title=r"N=$10^6$, soft=0.01")
    ax.loglog()
    ax.legend(loc="upper left")
    plt.axhline(1., linestyle="dashed", color="black")
    fig.tight_layout()
    fig.savefig(LOG_DIR / f"backwards_vs_time.pdf", dpi=200, bbox_inches="tight")
    plt.close(fig)