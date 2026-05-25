from dataclasses import replace
from pathlib import Path
import argparse
import time

import aegis
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fmdj.config import FMMConfig, OpeningByAngle, PlummerKernel, UnitConfig
from fmdj.data import PosMass
from fmdj.fmm import direct_summation, fast_multipole_method
from jztree.config import TreeConfig


N_PARTICLES = int(1e6)
P_VALUES = (2, 3, 4, 5)
THETAS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float32)
# THETAS = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], dtype=np.float32)
SETUP = "hernquist"  # "hernquist" or "gaus"
REFERENCE = "direct"  # "direct" or "analytic"

HERNQUIST_A = 1.0
HERNQUIST_MASS = 1.0
GAUS_SCALE = 1.0
GAUS_MASS = 1.0
SOFTENING = 1e-2
SEED = 0

RUN_LOOPS = 10

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs" / "opening_performance"


def make_hernquist_particles() -> PosMass:
    np.random.seed(SEED)
    prof = aegis.profiles.HernquistProfile(a=HERNQUIST_A, M=HERNQUIST_MASS)
    pos, _, mass = prof.sample_particles(
        N_PARTICLES,
        result="pos_vel_m",
        rpmin=1e-6 * HERNQUIST_A,
        ramax=1e6 * HERNQUIST_A,
    )
    pos = jnp.asarray(pos, dtype=jnp.float32)
    mass = jnp.asarray(mass, dtype=pos.dtype)
    if mass.ndim == 0:
        mass = jnp.full(pos.shape[0], mass, dtype=pos.dtype)
    return PosMass(pos=pos, mass=mass, num=N_PARTICLES, num_total=N_PARTICLES)


def make_gaus_particles() -> PosMass:
    key = jax.random.PRNGKey(SEED)
    pos = jax.random.normal(key, (N_PARTICLES, 3), dtype=jnp.float32) * GAUS_SCALE
    mass = jnp.full(N_PARTICLES, GAUS_MASS / N_PARTICLES, dtype=pos.dtype)
    return PosMass(pos=pos, mass=mass, num=N_PARTICLES, num_total=N_PARTICLES)


def hernquist_force_reference(pos: jax.Array, G: float) -> jax.Array:
    r = jnp.linalg.norm(pos, axis=-1)
    r_safe = jnp.clip(r, 1e-30, None)
    acc_mag = G * HERNQUIST_MASS / (r_safe + HERNQUIST_A) ** 2
    return -(pos / r_safe[:, None]) * acc_mag[:, None]


def gaussian_force_reference(pos: jax.Array, G: float) -> jax.Array:
    # For a spherical Gaussian density with total mass M and per-coordinate
    # standard deviation sigma, M(<r) = M * [erf(x) - 2*x*exp(-x^2)/sqrt(pi)]
    # where x = r / (sqrt(2)*sigma).
    r = jnp.linalg.norm(pos, axis=-1)
    r_safe = jnp.clip(r, 1e-30, None)
    x = r_safe / (jnp.sqrt(2.0) * GAUS_SCALE)
    enclosed_mass = GAUS_MASS * (
        jax.scipy.special.erf(x) - 2.0 * x * jnp.exp(-(x * x)) / jnp.sqrt(jnp.pi)
    )
    acc_mag = G * enclosed_mass / (r_safe * r_safe)
    return -(pos / r_safe[:, None]) * acc_mag[:, None]


def setup_paths(setup: str, reference: str) -> tuple[Path, Path]:
    stem = f"{setup}_{reference}"
    return LOG_DIR / f"{stem}.npz", LOG_DIR / f"{stem}.png"


def direct_reference_path(setup: str) -> Path:
    return LOG_DIR / f"{setup}_direct_force_reference.npz"


def make_particles(setup: str) -> PosMass:
    if setup == "hernquist":
        return make_hernquist_particles()
    if setup == "gaus":
        return make_gaus_particles()
    raise ValueError(f"Unknown setup {setup!r}. Expected 'hernquist' or 'gaus'.")


def force_reference(setup: str, pos: jax.Array, G: float) -> jax.Array:
    if setup == "hernquist":
        return hernquist_force_reference(pos, G)
    if setup == "gaus":
        return gaussian_force_reference(pos, G)
    raise ValueError(f"Unknown setup {setup!r}. Expected 'hernquist' or 'gaus'.")


def direct_force_reference(setup: str, part: PosMass, cfg_fmm: FMMConfig, G: float, recompute: bool = False) -> jax.Array:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = direct_reference_path(setup)

    if path.exists() and not recompute:
        print(f"Loading cached direct force reference from {path}")
        with np.load(path) as data:
            return jnp.asarray(data["force_ref"])

    print(f"Calculating direct force reference with kahan summation and writing {path}")
    t0 = time.perf_counter()
    force_ref = direct_summation.jit(part, kernel=cfg_fmm.kernel, kahan=True, G=G).force()
    force_ref = jax.block_until_ready(force_ref)
    np.savez(
        path,
        force_ref=np.asarray(force_ref),
        setup=np.asarray(setup),
        n_particles=np.asarray(N_PARTICLES, dtype=np.int64),
        softening=np.asarray(SOFTENING, dtype=np.float64),
    )
    print(f"Direct reference done after {time.perf_counter() - t0:.2f}s")
    return force_ref


def get_force_reference(setup: str, reference: str, part: PosMass, cfg_fmm: FMMConfig, G: float, recompute_direct: bool = False) -> jax.Array:
    if reference == "analytic":
        return jax.block_until_ready(force_reference(setup, part.pos, G))
    if reference == "direct":
        return direct_force_reference(setup, part, cfg_fmm, G, recompute=recompute_direct)
    raise ValueError(f"Unknown reference {reference!r}. Expected 'direct' or 'analytic'.")


def relative_force_error(force: jax.Array, force_ref: jax.Array) -> jax.Array:
    denom = jnp.linalg.norm(force_ref, axis=-1)
    denom = jnp.clip(denom, 1e-30, None)
    return jnp.linalg.norm(force - force_ref, axis=-1) / denom


def time_fmm(part: PosMass, cfg_fmm: FMMConfig, G: float) -> tuple[float, jax.Array]:
    compiled = fast_multipole_method.jit.lower(part, cfg_fmm=cfg_fmm, G=G).compile()

    def call():
        return jax.block_until_ready(compiled(part))

    result = call()
    t0 = time.perf_counter()
    for _ in range(RUN_LOOPS):
        result = call()
    dt = (time.perf_counter() - t0) / RUN_LOOPS
    return dt, result.force()


def run_benchmark(setup: str, reference: str, recompute_direct: bool = False) -> dict[str, np.ndarray]:
    part = make_particles(setup)
    cfg_tree = TreeConfig(mass_centered=False, alloc_fac_nodes=2.0)
    cfg_fmm = FMMConfig(
        kernel=PlummerKernel(softening=SOFTENING),
        tree=cfg_tree,
        kahan_summation=False,
        alloc_fac_ilist=1024,
        opening=OpeningByAngle(),
        p_extra_m2l=1,
    )
    G = UnitConfig().G()

    force_ref = get_force_reference(setup, reference, part, cfg_fmm, G, recompute_direct=recompute_direct)

    times = np.empty((len(P_VALUES), len(THETAS)), dtype=np.float64)
    err_median = np.empty_like(times)
    err_p90 = np.empty_like(times)
    err_p99 = np.empty_like(times)

    for ip, p in enumerate(P_VALUES):
        for itheta, theta in enumerate(THETAS):
            cfg_fmm_p = replace(
                cfg_fmm,
                p=p,
                opening=OpeningByAngle(theta=float(theta)),
                p_extra_m2l=1 if p<5 else 0
            )
            dt, force = time_fmm(part, cfg_fmm_p, G)
            rerr = relative_force_error(force, force_ref)

            times[ip, itheta] = dt
            err_median[ip, itheta] = float(jnp.percentile(rerr, 50))
            err_p90[ip, itheta] = float(jnp.percentile(rerr, 90))
            err_p99[ip, itheta] = float(jnp.percentile(rerr, 99))
            print(
                f"p={p}, theta={theta:.2f}: "
                f"{1e3 * dt:.1f} ms, "
                f"p90 rel. force error={err_p90[ip, itheta]:.3e}"
            )

    return {
        "p_values": np.asarray(P_VALUES, dtype=np.int32),
        "thetas": THETAS,
        "setup": np.asarray(setup),
        "reference": np.asarray(reference),
        "times": times,
        "err_median": err_median,
        "err_p90": err_p90,
        "err_p99": err_p99,
        "n_particles": np.asarray(N_PARTICLES, dtype=np.int64),
        "softening": np.asarray(SOFTENING, dtype=np.float64),
        "hernquist_a": np.asarray(HERNQUIST_A, dtype=np.float64),
        "hernquist_mass": np.asarray(HERNQUIST_MASS, dtype=np.float64),
        "gaus_scale": np.asarray(GAUS_SCALE, dtype=np.float64),
        "gaus_mass": np.asarray(GAUS_MASS, dtype=np.float64),
        "run_loops": np.asarray(RUN_LOOPS, dtype=np.int32),
    }


def load_or_run(
    setup: str,
    reference: str,
    recompute: bool = False,
    recompute_direct: bool = False,
) -> dict[str, np.ndarray]:
    result_file, _ = setup_paths(setup, reference)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if result_file.exists() and not recompute and not recompute_direct:
        print(f"Loading cached benchmark results from {result_file}")
        with np.load(result_file) as data:
            return {key: data[key] for key in data.files}

    print(f"Running {setup} benchmark with {reference} reference and writing {result_file}")
    results = run_benchmark(setup, reference, recompute_direct=recompute_direct)
    np.savez(result_file, **results)
    return results


def plot_results(results: dict[str, np.ndarray]) -> None:
    setup = str(results["setup"]) if "setup" in results else SETUP
    reference = str(results["reference"]) if "reference" in results else REFERENCE
    _, plot_file = setup_paths(setup, reference)

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

    for ip, p in enumerate(results["p_values"]):
        ax.loglog(
            1e3 * results["times"][ip],
            results["err_p90"][ip],
            "o-",
            label=f"p={p}{'+1' if p < 5 else ''}",
        )
        for x, y, theta in zip(1e3 * results["times"][ip], results["err_p90"][ip], results["thetas"]):
            ax.annotate(f"{theta:.1f}", (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("relative force error (90% percentile)")
    ax.set_title(f"Opening criterion performance: {setup}, {reference} reference")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_file, dpi=200)
    print(f"Wrote {plot_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", choices=("hernquist", "gaus"), default=SETUP)
    parser.add_argument("--reference", choices=("direct", "analytic"), default=REFERENCE)
    parser.add_argument("--recompute", action="store_true", help="Ignore the cached npz file and rerun the benchmark.")
    parser.add_argument("--recompute-direct", action="store_true", help="Recalculate the cached direct force reference.")
    args = parser.parse_args()

    results = load_or_run(
        args.setup,
        args.reference,
        recompute=args.recompute,
        recompute_direct=args.recompute_direct,
    )
    plot_results(results)


if __name__ == "__main__":
    main()
