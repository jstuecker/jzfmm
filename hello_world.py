import argparse
from functools import partial
import time
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "--movie", action="store_true",
    help="Render a movie without opening an interactive window",
)
parser.add_argument(
    "--rewind", action="store_true",
    help="Integrate backward to the initial state after integrating forward",
)
args = parser.parse_args()

matplotlib.use("Agg" if args.movie else "TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from jzfmm.config import DKDLatticeConfig, FMMConfig, PlummerKernel, SimConfig, TreeConfig
from jzfmm.time_integration import simulate
from jzfmm_utils.ics import gaussian_text
from jztree.tools import log

cfg_fmm = FMMConfig(kernel=PlummerKernel(softening=4e-2))
cfg_fmm.tree = TreeConfig(alloc_fac_nodes=1.5)
cfg = SimConfig(force=cfg_fmm)
cfg.integrator = DKDLatticeConfig()

p0 = gaussian_text(
    "jz-tree",
    N=1024 * 1024,
    spacing=0.45,
    blob_sigma=0.06,
    velocity_dispersion=0.05,
    angular_velocity=0.15,
    mass=1e7,
)
IMAGE_SHAPE = (288*2, 512*2)
XBOUNDS = (-10., 10.)
YBOUNDS = (-10.*9./16, 10.*9./16)

def particle_image(pos):
    ny, nx = IMAGE_SHAPE
    ix = ((pos[:, 0] - XBOUNDS[0]) * nx / (XBOUNDS[1] - XBOUNDS[0])).astype(jnp.int32)
    iy = ((pos[:, 1] - YBOUNDS[0]) * ny / (YBOUNDS[1] - YBOUNDS[0])).astype(jnp.int32)
    valid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    flat_index = jnp.clip(iy * nx + ix, 0, nx * ny - 1)
    counts = jnp.bincount(flat_index, weights=valid.astype(pos.dtype), length=nx * ny)
    return jnp.log1p(counts.reshape(IMAGE_SHAPE))

@jax.jit
def initial_image(p):
    return particle_image(p.pos)

@partial(jax.jit, static_argnames=("cfg",), donate_argnums=(0,))
def image_step(p, ts, *, cfg):
    p = simulate(p, ts=ts, cfg=cfg)
    return p, particle_image(p.pos)

def simulate_images(p, tend, nout, steps_per_output, cfg, tstart=0., rewind=False):
    """Advance particles on the GPU while yielding only rendered images."""
    time_dtype = p.pos.dtype
    yield initial_image(p)
    nframes = nout * (2 if rewind else 1)
    for iframe in range(nframes):
        if iframe < nout:
            i0, i1 = iframe, iframe + 1
        else:
            i0 = 2 * nout - iframe
            i1 = i0 - 1
        t0 = jnp.asarray(tstart + i0 * tend / nout, dtype=time_dtype)
        t1 = jnp.asarray(tstart + i1 * tend / nout, dtype=time_dtype)
        ts = jnp.linspace(t0, t1, steps_per_output + 1, dtype=time_dtype)
        frame_start = time.perf_counter()
        p, rendered = image_step(p, ts, cfg=cfg)
        rendered.block_until_ready()
        log(
            "Reached output {} ({:.2f}s for {} steps)",
            iframe + 1,
            time.perf_counter() - frame_start,
            steps_per_output,
            level=1,
            cfg_log=cfg.logging,
        )
        yield rendered

fig = plt.figure(figsize=(10., 6.), facecolor="black")
ax = fig.add_axes((0., 0., 1., 1.), facecolor="black")
ax.set_axis_off()
image = ax.imshow(
    np.asarray(initial_image(p0)),
    origin="lower",
    extent=(*XBOUNDS, *YBOUNDS),
    interpolation="nearest",
    cmap="cubehelix",
    vmax=6.5
)

print(np.max(initial_image(p0)))

def update(rendered):
    image.set_data(np.asarray(rendered))
    return [image]

sim_iter = simulate_images(
    p0, tend=10., nout=400, steps_per_output=1, cfg=cfg,
    rewind=args.rewind,
)

ani = FuncAnimation(fig, update, frames=sim_iter, blit=True, interval=40, repeat=True,
                    cache_frame_data=False)

if args.movie:
    plt.close()
    import os
    os.makedirs("output", exist_ok=True)

    ani.save("output/nbody_simulation.mp4", dpi=200)
else:
    plt.show()
