import fmdj
import pytest
import jax

@pytest.mark.parametrize("ilist_fac", [512, 256, 128])
def test_ilist_size_exception(tree, ilist_fac):
    """Here we create a scenario where the interaction list is too small and check that an error is raised."""
    octree, posz, massz, isortz = tree
    try:
        ilist, nilist = fmdj.fmm.build_interaction_list.jit(octree, ilist_fac=ilist_fac)
        assert ilist_fac == 512, "Expected MemoryError for small ilist_fac not raised"
        assert nilist <= len(ilist), f"nfilled {nilist} exceeds ilist size {len(ilist)} for ilist_fac={ilist_fac}"
    except (MemoryError, jax.errors.JaxRuntimeError) as err:
        assert ilist_fac in (128, 256), "Unexpected RuntimeError raised"