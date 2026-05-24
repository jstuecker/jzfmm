import numpy as np
import jax
import jax.numpy as jnp
import aegis
from fmdj.data import PosMass, Particles
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

def hernquist(N, a=1., M=1., anisotropy=0., seed=None):
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
