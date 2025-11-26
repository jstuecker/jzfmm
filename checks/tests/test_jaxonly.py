import fmdj_jaxonly.jaxonly_fmm
import numpy as np
import jax.numpy as jnp
import fmdj
import fmdj_jaxonly.jaxonly_multipoles as jmp

def test_expand_interactions():
    nnodes = 3
    npart_per_node = 2

    ilist = fmdj.data.InteractionList(
        ispl = jnp.arange(nnodes+1)*2, # [0, 2, 4, 6] each node has 2 interactions
        iother=jnp.array([0, 1, 2, 1, 1, 2]),
        nfilled = 8
    )

    spl = jnp.arange(nnodes)*npart_per_node # [0, 2, 4] each node has 2 particles

    inew = fmdj_jaxonly.jaxonly_fmm.expand_interactions.jit(ilist, spl, size_children=6, size_new_ilist=14)
    
    assert jnp.all(inew.ispl == jnp.array([0,  4,  8, 10, 12, 12, 12]))
    assert jnp.all(inew.iother == jnp.array([0,  1,  2,  3,  
                                             0,  1,  2,  3,  
                                             2,  3,  2,  3, 
                                             14, 14]))

def test_shift_mp_to_mp_circuit():
    mp0 = np.random.uniform(-0.1, 0.1, (20))

    x1 = np.random.uniform(-1, 1, (3))
    x2 = np.random.uniform(-1, 1, (3))
    x3 = -(x1 + x2)

    mp0 = np.random.uniform(-0.1, 0.1, (20))
    
    mp = mp0
    for dx in [x1, x2, x3]:
        mp = jmp.shift_multipoles(mp, dx, p=3)

    assert np.allclose(mp, mp0, rtol=1e-3), "Shifted local multipoles do not match original multipoles after three shifts."
