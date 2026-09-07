import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import time
from pathlib import Path
import subprocess
import tempfile
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
    "--only-forward", action="store_true",
    help="Stop after the forward integration instead of rewinding",
)
args = parser.parse_args()

if args.movie:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation


def movie_writer():
    """Find an MP4 encoder and check it before doing any simulation work."""
    if not FFMpegWriter.isAvailable():
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError(
                "Movie export requires FFmpeg. Install it on your system or run "
                "`pip install imageio-ffmpeg` in this Python environment."
            ) from exc
    writer = FFMpegWriter(
        fps=20, codec="libx264",
        extra_args=["-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"],
    )
    with tempfile.TemporaryDirectory(prefix="jzfmm-encoder-") as directory:
        command = [
            writer.bin_path(), "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
            "-video_size", "2x2", "-i", "pipe:0", "-frames:v", "1",
            "-c:v", writer.codec, *writer.extra_args, str(Path(directory) / "probe.mp4"),
        ]
        try:
            subprocess.run(command, input=bytes(12), capture_output=True, check=True, timeout=15)
        except (OSError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "stderr", b"") or b""
            raise RuntimeError(
                "FFmpeg could not encode an H.264 MP4. Use an FFmpeg build with libx264 "
                "support (for example from imageio-ffmpeg). " + detail.decode(errors="replace")
            ) from exc
    return writer


IMAGE_SHAPE = (576, 1024)
try:
    writer = movie_writer() if args.movie else None
    fig = plt.figure(figsize=(IMAGE_SHAPE[1] / 100, IMAGE_SHAPE[0] / 100),
                     dpi=100, facecolor="black")
    if not args.movie and fig.canvas.required_interactive_framework is None:
        raise RuntimeError("The selected Matplotlib backend cannot open an interactive window.")
except (ImportError, RuntimeError, OSError) as exc:
    hint = "" if args.movie else (
        " Install a GUI toolkit (e.g. `pip install PySide6`, or Ubuntu's python3-tk "
        "for the matching system Python) and use a graphical session. "
        "You can select a backend with MPLBACKEND=QtAgg, or use --movie without a display."
    )
    parser.exit(1, f"{exc}{hint}\n")

from jzfmm.config import DKDLatticeConfig, FMMConfig, PlummerKernel, SimConfig, TreeConfig
from jzfmm.time_integration import simulate
from jzfmm_utils.ics import gaussian_text
from jztree.tools import log

cfg_fmm = FMMConfig(kernel=PlummerKernel(softening=4e-2))
cfg_fmm.tree = TreeConfig(alloc_fac_nodes=1.5)
cfg = SimConfig(force=cfg_fmm)
cfg.integrator = DKDLatticeConfig()

p0 = gaussian_text(
    "JZ-FMM",
    N=1024 * 1024,
    spacing=0.45,
    blob_sigma=0.06,
    velocity_dispersion=0.05,
    angular_velocity=0.15,
    mass=1e7,
)
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
    """Advance particles while rendering the previous frame on the host."""
    time_dtype = p.pos.dtype
    nframes = nout * (2 if rewind else 1)

    def compute_frame(p, iframe):
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
        rendered = np.asarray(rendered)
        return p, rendered, time.perf_counter() - frame_start

    # JAX's FFI dispatch occupies the calling thread for much of a step. Run it
    # in a worker so the GPU advances the next frame while Matplotlib renders
    # the current host image on the main thread.
    rendered_initial = np.asarray(initial_image(p))
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(compute_frame, p, 0)
        yield rendered_initial

        for iframe in range(nframes):
            p, rendered, frame_time = future.result()
            if iframe + 1 < nframes:
                future = executor.submit(compute_frame, p, iframe + 1)

            log(
                "Reached output {} ({:.2f}s for {} steps)",
                iframe + 1,
                frame_time,
                steps_per_output,
                level=1,
                cfg_log=cfg.logging,
            )
            yield rendered

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

def update(rendered):
    image.set_data(rendered)
    return [image]

def animation_frames():
    # ``image_step`` donates its particle input. Give every repeated animation
    # a fresh device copy while retaining ``p0`` as the restart state.
    p = jax.tree.map(lambda x: x.copy(), p0)
    return simulate_images(
        p, tend=8., nout=400, steps_per_output=1, cfg=cfg,
        rewind=not args.only_forward,
    )

if args.movie:
    Path("output").mkdir(exist_ok=True)
    # Stream frames directly; no animation cache or Pillow fallback is needed.
    with writer.saving(fig, "output/nbody_simulation.mp4", dpi=100):
        for rendered in animation_frames():
            update(rendered)
            writer.grab_frame()
    plt.close(fig)
else:
    ani = FuncAnimation(fig, update, frames=animation_frames, blit=fig.canvas.supports_blit,
                        interval=1, repeat=True, cache_frame_data=False)
    plt.show()
