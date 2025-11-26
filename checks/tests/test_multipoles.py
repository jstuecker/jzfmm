import jax.numpy as jnp
import fmdj
from fmdj_jaxonly.jaxonly_multipoles import multipoles_from_particles_jax, coarsen_multipoles_jax
from fmdj.data import PosMass, TreePlane

def test_tree_multipoles(pos_mass_z: PosMass, tree_hierarchy: list[TreePlane], cfg: fmdj.Config):
    mp_base = multipoles_from_particles_jax.jit(tree_hierarchy[0], pos_mass_z, cfg=cfg)
    mp_cuda = fmdj.multipoles.multipoles_from_particles.jit(tree_hierarchy[0], pos_mass_z, cfg=cfg)
    assert jnp.allclose(tree_hierarchy[0].npart, mp_base.get(0))
    
    for i in range(mp_base.values.shape[1]):
        assert jnp.allclose(mp_base.get(i), mp_cuda.get(i), rtol=1e-3)
    assert jnp.allclose(mp_base.center(), mp_cuda.center(), rtol=1e-6, equal_nan=True)

    mp_coarse = coarsen_multipoles_jax.jit(mp_base, tree_hierarchy[1], cfg=cfg)
    mp_coarse2 = fmdj.multipoles.coarsen_multipoles.jit(mp_cuda, tree_hierarchy[1], cfg=cfg)
    assert jnp.allclose(tree_hierarchy[1].npart, mp_coarse.get(0))
    for i in range(mp_base.values.shape[1]):
        assert jnp.allclose(mp_coarse.get(i), mp_coarse2.get(i), rtol=1e-3, atol=1e-4)