import numpy as np
import jax
import jax.numpy as jnp
from jzfmm.data import PosMass, Particles
from jztree.comm import get_rank_info
from jztree.jax_ext import tree_map_by_len

def pad_pytree(x, num, num_pad, float_val=jnp.nan, int_val=0):
    def pad(xi):
        if xi.dtype.kind == "f":
            val = float_val
        else:
            val = int_val
        
        return jnp.pad(xi, [(0, num_pad)] + [(0,0)]*(xi.ndim - 1), constant_values=val)

    return tree_map_by_len(pad, x, num)

def gaussian_blob(N, scale=1.0, mass=1., seed=0, npad=0):
    rank, ndev, axis_name = get_rank_info()

    pos = jax.random.normal(jax.random.PRNGKey(seed), (N,3), dtype=jnp.float32) * scale
    posmass = PosMass(pos=pos, mass=mass/N, num=N, num_total=ndev*N)

    if npad > 0:
        return pad_pytree(posmass, N, npad)
    else:
        return posmass

_BITMAP_FONT = {
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "j": ("00100", "00000", "00100", "00100", "00100", "10100", "01100"),
    "z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "-": ("000", "000", "000", "111", "000", "000", "000"),
    "t": ("00100", "11111", "00100", "00100", "00100", "00100", "00011"),
    "r": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "e": ("01110", "10001", "10000", "11110", "10000", "10001", "01110"),
    "f": ("00111", "00100", "11110", "00100", "00100", "00100", "00100"),
    "m": ("00000", "11011", "10101", "10101", "10101", "10101", "10101"),
}

def gaussian_text(
    text="jz-tree", N=1024 * 128, spacing=0.45, blob_sigma=0.06,
    velocity_dispersion=0., angular_velocity=0., mass=1., seed=0,
):
    """Place Gaussian particle blobs on the active pixels of a bitmap string."""
    rng = np.random.default_rng(seed)
    pixels = []
    xoffset = 0
    for character in text:
        glyph = _BITMAP_FONT[character]
        for row, line in enumerate(glyph):
            pixels.extend(
                (xoffset + column, -row)
                for column, active in enumerate(line) if active == "1"
            )
        xoffset += len(glyph[0]) + 1

    centers = np.asarray(pixels, dtype=float)
    centers -= 0.5 * (centers.min(axis=0) + centers.max(axis=0))
    centers *= spacing

    blob_ids = np.arange(N) % len(centers)
    rng.shuffle(blob_ids)
    pos = np.zeros((N, 3))
    pos[:, :2] = centers[blob_ids]
    pos += rng.normal(scale=blob_sigma, size=pos.shape)
    vel = rng.normal(scale=velocity_dispersion, size=pos.shape)
    pos -= np.mean(pos, axis=0)
    vel[:, 0] -= angular_velocity * pos[:, 1]
    vel[:, 1] += angular_velocity * pos[:, 0]
    vel -= np.mean(vel, axis=0)

    return Particles(
        pos=jnp.asarray(pos),
        vel=jnp.asarray(vel),
        mass=jnp.full(N, mass / N),
    )

def hernquist(N, a=1., M=1., anisotropy=0., seed=None):
    import aegis

    if seed is not None:
        np.random.seed(seed)
    prof = aegis.profiles.HernquistProfile(a=a, M=M, anisotropy=anisotropy)
    pos, vel, mass = prof.sample_particles(N, result="pos_vel_m", rpmin=1e-6*a, ramax=1e6*a)
    return Particles(pos=pos, mass=mass, vel=vel)

def discodj_sim(res, zsort=False):
    from discodj_examples.simulations import disco_sim
    pos = disco_sim(res=res, res_pm=res)[1].reshape(-1,3)
    if zsort:
        pos = jztree.tree.zsort(pos)[0]

    return PosMass(pos=pos, mass=1. / res**3)
discodj_sim.jit = jax.jit(discodj_sim, static_argnames=("res", "zsort"))
