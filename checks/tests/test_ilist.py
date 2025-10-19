import fmdj
import pytest

@pytest.mark.parametrize("ilist_fac", [512, 256, 128])
def test_ilist_size_err(tree, ilist_fac):
    octree, posz, massz, isortz = tree
    err, (ilist, nilist) = fmdj.fmm.build_interaction_list.jit(octree, ilist_fac=ilist_fac)
    err.throw()
    
    assert nilist <= len(ilist), f"nfilled {nilist} exceeds ilist size {len(ilist)} for ilist_fac={ilist_fac}"
    
    