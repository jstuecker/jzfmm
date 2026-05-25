import jax.numpy as jnp
import jax
import aegis
import pytest
from fmdj.config import SimConfig
from fmdj.external_potential import NFWPotential

def test_nfw_acc():
    cfg = SimConfig()

    host = aegis.profiles.NFWProfile(conc=6., m200c=1e12)
    cfg.external_potential = NFWPotential(host.rs, host.rhoc)

    pos = jax.random.normal(jax.random.PRNGKey(0), (10,3)) * 200.

    phi1 = cfg.external_potential.potential(pos, cfg=cfg)
    phi2 = host.potential(jnp.linalg.norm(pos, axis=-1))

    acc1 = cfg.external_potential.acceleration(pos, cfg=cfg)
    r = jnp.linalg.norm(pos, axis=-1)
    acc2 = host.accr(r)[:,None] * (pos / r[:,None])
    
    assert phi1 == pytest.approx(phi2, rel=1e-5)
    assert acc1 == pytest.approx(acc2, rel=1e-5)
