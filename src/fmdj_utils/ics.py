import numpy as np
import jax
import jax.numpy as jnp
import fmdj
import aegis

def gaussian_blob(N, scale=1.0, mass=1., seed=0, zsort=False):
    pos = jax.random.normal(jax.random.PRNGKey(seed), (N,3), dtype=jnp.float32) * scale
    if zsort:
        pos, isort = fmdj.ztree.pos_zorder_sort(pos)
    mass0 = jnp.ones(len(pos), dtype=pos.dtype) * (mass/N)
    return fmdj.data.PosMass(pos, mass0)

def hernquist(N, a=1., M=1., anisotropy=0., seed=None):
    if seed is not None:
        np.random.seed(seed)
    prof = aegis.profiles.HernquistProfile(a=a, M=M, anisotropy=anisotropy)
    pos, vel, mass = prof.sample_particles(N, result="pos_vel_m", rpmin=1e-6*a, ramax=1e6*a)
    return fmdj.data.Particles(pos, mass, vel)