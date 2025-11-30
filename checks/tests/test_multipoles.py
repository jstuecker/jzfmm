import jax.numpy as jnp
import fmdj
from fmdj_jaxonly.jaxonly_multipoles import multipoles_from_particles_jax, coarsen_multipoles_jax
from fmdj.data import PosMass, TreePlane
import pytest

# def test_multipoles(pos_mass_z: PosMass, tree_hierarchy: list[TreePlane], cfg: fmdj.Config):
#     mp_base = multipoles_from_particles_jax.jit(tree_hierarchy[0], pos_mass_z, cfg=cfg)
#     mp_cuda = fmdj.multipoles.multipoles_from_particles.jit(tree_hierarchy[0], pos_mass_z, cfg=cfg)
#     assert jnp.allclose(tree_hierarchy[0].npart, mp_base.get(0))
    
#     for i in range(mp_base.values.shape[1]):
#         assert mp_base.get(i) == pytest.approx(mp_cuda.get(i), rel=1e-3, abs=1e-6)
#     mp_base.center() == pytest.approx(mp_cuda.center(), rel=1e-6, nan_ok=True)

#     mp_coarse = coarsen_multipoles_jax.jit(mp_base, tree_hierarchy[1], cfg=cfg)
#     assert tree_hierarchy[1].npart == pytest.approx(mp_coarse.get(0))
#     for i in range(mp_base.values.shape[1]):
#         assert mp_coarse.get(i) == pytest.approx(mp_coarse2.get(i), rel=1e-3, abs=1e-4)