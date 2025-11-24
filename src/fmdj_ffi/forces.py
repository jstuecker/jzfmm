from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from fmdj_cuda import ffi_forces

jax.ffi.register_ffi_target("ForceAndPotential", ffi_forces.ForceAndPotential(), platform="CUDA")
jax.ffi.register_ffi_target("BwdForceAndPotential", ffi_forces.BwdForceAndPotential(), platform="CUDA")

def force_and_potential_fwd(xm, block_size=64, softening=1e-2, kahan=False):
    assert xm.dtype == jnp.float32
    assert xm.shape[-1] == 4
    assert xm.ndim >= 2
    
    assert softening > 0, "Epsilon must be positive to deal with self-interaction."

    out_type = jax.ShapeDtypeStruct(xm.shape, xm.dtype)
    fphi = jax.ffi.ffi_call("ForceAndPotential", (out_type,))(
        xm, block_size=np.uint64(block_size), epsilon=np.float32(softening), kahan=kahan)[0]
    return fphi, xm

def force_and_potential_bwd(block_size, softening, kahan, xm, gfphi):
    out_type = jax.ShapeDtypeStruct(xm.shape, xm.dtype)
    gxm = jax.ffi.ffi_call("BwdForceAndPotential", (out_type,))(
        gfphi, xm, block_size=np.uint64(block_size), epsilon=np.float32(softening), kahan=kahan)[0]
    return gxm,

@partial(jax.custom_vjp, nondiff_argnames=("block_size", "softening", "kahan"))
def force_and_potential(xm, block_size=64, softening=1e-2, kahan=False):
    return force_and_potential_fwd(xm, block_size, softening, kahan)[0]
force_and_potential.defvjp(force_and_potential_fwd, force_and_potential_bwd)
force_and_potential.jit = jax.jit(force_and_potential, static_argnames=("block_size", "softening", "kahan"))

# ------------------------------------------------------------------------------------------------ #
#                             Some reference implemetations for testing                            #
# ------------------------------------------------------------------------------------------------ #

def potential_pure_jax(x, m=1., softening=1e-2):
    rij2 = jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    rinv = jnp.where(rij2 > 0, 1. / jnp.sqrt(rij2 + softening**2), 0.)
    
    return -jnp.sum(rinv * jnp.broadcast_to(m, x.shape[:-1])[None,:], axis=1)
potential_pure_jax.jit = jax.jit(potential_pure_jax)

def force_pure_jax(x, m=1., softening=1e-2):
    dx = x[:, None] - x[None, :]
    rij2 = jnp.sum(dx ** 2, axis=-1, keepdims=True)
    rinv = jnp.where(rij2 > 0, 1. / jnp.sqrt(rij2 + softening**2), 0.)
    
    return -jnp.sum(dx * rinv**3 * jnp.broadcast_to(m, x.shape[:-1])[None,:,None], axis=1)
force_pure_jax.jit = jax.jit(force_pure_jax)

def force_and_potential_pure_jax(x, m=1., softening=1e-2):
    phi = potential_pure_jax(x, m, softening)
    f = force_pure_jax(x, m, softening)
    
    return jnp.concatenate([f, phi[:,None]], axis=-1)
force_and_potential_pure_jax.jit = jax.jit(force_and_potential_pure_jax)