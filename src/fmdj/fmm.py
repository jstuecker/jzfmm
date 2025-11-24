import jax
import jax.numpy as jnp
from . import multipoles
from . import config

# ================================== Some utility methods ======================================== #

def cumsum_starting_with_zero(x):
    return jnp.pad(jnp.cumsum(x), (1, 0))

def offset_sum(num):
    cs = jnp.cumsum(num, axis=0)
    return cs - num, cs[-1]

def scatter_masked(y, value, mask, offset=0, get_offsets=False):
    """Emulates y[offset:offset+sum(mask)] = value[mask], but jit-compatible."""
    off, num = offset_sum(mask.astype(jnp.int32))
    off_masked = jnp.where(mask & (offset+off >= 0), offset+off, y.shape[0])

    ynew = y.at[off_masked].set(value)
    if get_offsets:
        return ynew, num + offset, off + offset
    else:
        return ynew, num + offset

# ------------------------------------------------------------------------------------------------ #
#                                              New FMM                                             #
# ------------------------------------------------------------------------------------------------ #

def new_fmm_potential(pos, mass, cfg : config.Config, return_sorted=False):
    import fmdj_ffi as cj
    import fmdj_ffi.cj_new_tree as cnt
    import fmdj.new_tree as nt

    if mass is None:
        mass = jnp.ones((pos.shape[0],), dtype=pos.dtype)
    elif jnp.shape(mass) != jnp.shape(pos)[:-1]:
        mass = jnp.broadcast_to(mass, pos.shape[:-1])

    posz, isortz = cj.tree.pos_zorder_sort(pos)
    particlesz = nt.Particles(pos=posz, mass=mass[isortz])

    th = nt.build_tree_hierarchy(particlesz, cfg)
    loc, ilist = nt.evaluate_interaction_hierarchy(th, cfg=cfg)

    parent = th[0].icoarse_of_fine()
    phi_loc = multipoles.evaluate_local_potential(loc[parent], posz - th[0].mp.center()[parent])

    fphi = cnt.cj_new_force_and_pot(particlesz, th[0], ilist, cfg=cfg)
    
    phiz = fphi[:,3] + phi_loc

    if return_sorted:
        return particlesz.pos, particlesz.mass, isortz, phiz
    else:
        return jnp.zeros_like(phiz).at[isortz].set(phiz)
new_fmm_potential.jit = jax.jit(new_fmm_potential, static_argnames=("cfg", "return_sorted"))

def new_fmm_fphi(pos, mass, cfg : config.Config, return_sorted=False):
    import fmdj_ffi as cj
    import fmdj_ffi.cj_new_tree as cnt
    import fmdj.new_tree as nt

    if mass is None:
        mass = jnp.ones((pos.shape[0],), dtype=pos.dtype)
    elif jnp.shape(mass) != jnp.shape(pos)[:-1]:
        mass = jnp.broadcast_to(mass, pos.shape[:-1])

    posz, isortz = cj.tree.pos_zorder_sort(pos)
    particlesz = nt.Particles(pos=posz, mass=mass[isortz])

    th = nt.build_tree_hierarchy(particlesz, cfg)
    loc, ilist = nt.evaluate_interaction_hierarchy(th, cfg=cfg)

    parent = th[0].icoarse_of_fine()
    fphi_loc = multipoles.evaluate_local_fphi(loc[parent], particlesz.pos - th[0].mp.center()[parent])

    fphi = cnt.cj_new_force_and_pot(particlesz, th[0], ilist, cfg=cfg) + fphi_loc

    if return_sorted:
        return particlesz.pos, particlesz.mass, isortz, fphi
    else:
        fphi_unsorted = jnp.zeros_like(fphi).at[isortz].set(fphi)
        return fphi_unsorted
new_fmm_fphi.jit = jax.jit(new_fmm_fphi, static_argnames=("cfg", "return_sorted"))

def get_force_and_potential(pos, mass, cfg : config.Config, separately=False):
    if cfg.fmm is None: # Use direct summation
        import fmdj_ffi as cj
        xm = jnp.concatenate([pos, mass[:,None]], axis=-1)
        fphi = cj.forces.force_and_potential(xm, softening=cfg.softening, kahan=True) * cfg.G()
    else:
        fphi = new_fmm_fphi(pos, mass, cfg=cfg) * cfg.G()

    if separately:
        return fphi[:,0:3], fphi[:,3]
    else:
        return fphi
get_force_and_potential.jit = jax.jit(get_force_and_potential, static_argnames=("cfg", "separately"))