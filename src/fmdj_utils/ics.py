import numpy as np
import jax
import jax.numpy as jnp
import aegis
from fmdj.data import PosMass, Particles

def pad_pytree(x, num, float_val=jnp.nan, int_val=0):
    def pad(xi):
        if xi.dtype.kind == "f":
            val = float_val
        else:
            val = int_val
        
        return jnp.pad(xi, [(0, num)] + [(0,0)]*(xi.ndim - 1), constant_values=val)

    return jax.tree.map(pad, x)

def gaussian_blob(N, scale=1.0, mass=1., seed=0, zsort=False, npad=0):
    pos = jax.random.normal(jax.random.PRNGKey(seed), (N,3), dtype=jnp.float32) * scale
    if zsort:
        pos, isort = jztree.ztree.pos_zorder_sort(pos)
    mass0 = jnp.ones(len(pos), dtype=pos.dtype) * (mass/N)
    posmass = PosMass(pos, mass0)

    if npad > 0:
        return pad_pytree(posmass, npad)
    else:
        return posmass

def hernquist(N, a=1., M=1., anisotropy=0., seed=None):
    if seed is not None:
        np.random.seed(seed)
    prof = aegis.profiles.HernquistProfile(a=a, M=M, anisotropy=anisotropy)
    pos, vel, mass = prof.sample_particles(N, result="pos_vel_m", rpmin=1e-6*a, ramax=1e6*a)
    return Particles(pos, mass, vel)

def discodj_sim(res, zsort=False):
    from discodj_examples.simulations import disco_sim
    pos = disco_sim(res=res, res_pm=res)[1].reshape(-1,3)
    if zsort:
        pos = jztree.ztree.pos_zorder_sort(pos)[0]

    mass = jnp.ones(len(pos), dtype=pos.dtype) / res**3
    return PosMass(pos, mass)
discodj_sim.jit = jax.jit(discodj_sim, static_argnames=("res", "zsort"))