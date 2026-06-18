import os
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.20")

import jax
import pytest
from jax.sharding import AxisType, PartitionSpec as P

from fmdj.config import FMMConfig
from fmdj.fmm import _fmm_dual_walk, fast_multipole_method, leaf_leaf_summation
from fmdj.multipoles import build_multipole_hierarchy
from jztree.jax_ext import shard_map_constructor
from jztree.tree import zsort_and_tree
from jztree_utils import ics
import jztree as jz


def get_mesh(ndev=-1):
    return jax.make_mesh((ndev,), ("gpus",), axis_types=(AxisType.Auto,))


def leaf_leaf_with_tree(partz, th, ilist, cfg_fmm):
    return leaf_leaf_summation(partz, th.splits_leaf_to_part(), ilist, cfg_fmm=cfg_fmm)


@pytest.mark.shrink_in_quick(keep_index=2)
@pytest.mark.parametrize("N", (int(1e6), int(3e6), int(1e7), int(3e7), int(1e8)))
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def bench_distr_hernquist(jax_bench, N):
    cfg = Config()
    cfg.tree.alloc_fac_nodes = 1.8
    cfg.fmm.alloc_fac_ilist = 64.

    ndev = jax.device_count()
    mesh = get_mesh(ndev)

    part = ics.hernquist_posmass.smap(mesh, jit=True)(
        N, a=1.0, total_mass=1.0, seed=0, npad=int(0.5 * N), rmax=100.0
    )

    jb = jax_bench(jit_rounds=4, jit_warmup=1)
    with jz.stats.statistics() as st:
        jb.measure(
            fn_jit=fast_multipole_method.smap(mesh, jit=True),
            part=part,
            cfg_fmm=cfg_fmm,
            result="locz",
            tag=f"ndev{ndev}",
        )
        st = st.reduce_multi_host()
        if jax.process_index() == 0:
            print(f"Allocation suggestions for N={N}, ndev={ndev}")
            st.print_suggestions(cfg_fmm)

@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def bench_distr_steps(jax_bench):
    N = int(1e7)
    cfg = Config()
    cfg.tree.alloc_fac_nodes = 1.8
    cfg.fmm.alloc_fac_ilist = 64.

    ndev = jax.device_count()
    mesh = get_mesh(ndev)

    jb = jax_bench(jit_rounds=4, jit_warmup=1)

    part = ics.hernquist_posmass.smap(mesh, jit=True)(
        N, a=1.0, total_mass=1.0, seed=0, npad=int(0.5 * N), rmax=100.0
    )

    partz, th = jb.measure(fn_jit=zsort_and_tree.smap(mesh, jit=True),
        part=part, cfg_tree=cfg_fmm.tree, tag=f"zsort_tree_ndev{ndev}"
    )[1]

    build_multipoles = shard_map_constructor(
        build_multipole_hierarchy, in_specs=(P(-1), P(-1), P(-1), None),
        out_specs=P(-1), static_argnames=("cfg_fmm",)
    )(mesh, jit=True)
    mph = jb.measure(fn_jit=build_multipoles,
        th=th, pos=partz.pos, mp=partz.mass, cfg_fmm=cfg_fmm, tag=f"multipoles_ndev{ndev}"
    )[1]

    dual_walk = shard_map_constructor(
        _fmm_dual_walk, in_specs=(P(-1), P(-1), None), out_specs=P(-1), static_argnames=("cfg_fmm",)
    )(mesh, jit=True)
    loc_node, ilist = jb.measure(fn_jit=dual_walk,
        th=th, mph=mph, cfg_fmm=cfg_fmm, tag=f"node2node_ndev{ndev}"
    )[1]

    leaf_leaf = shard_map_constructor(
        leaf_leaf_with_tree, in_specs=(P(-1), P(-1), P(-1), None),
        out_specs=P(-1), static_argnames=("cfg_fmm",)
    )(mesh, jit=True)
    jb.measure(fn_jit=leaf_leaf,
        partz=partz, th=th, ilist=ilist, cfg_fmm=cfg_fmm, tag=f"leaf2leaf_ndev{ndev}"
    )

    jb.measure(fn_jit=fast_multipole_method.smap(mesh, jit=True),
        part=partz, cfg_fmm=cfg_fmm, th=th, result="locz", tag=f"totalzz_ndev{ndev}"
    )

@pytest.mark.skip_in_quick
@pytest.mark.parametrize("p, pex", ((2, 1), (3, 0), (3, 1), (4, 0), (4, 1), (5,0)))
@pytest.mark.skipif(jax.device_count() <= 1, reason="Requires multiple devices")
def bench_distr_p(jax_bench, p, pex):
    N = int(1e7)
    cfg = Config(fmm=FMMConfig(p=p, p_extra_m2l=pex))
    cfg.tree.alloc_fac_nodes = 1.8
    cfg.fmm.alloc_fac_ilist = 64.

    ndev = jax.device_count()
    mesh = get_mesh(ndev)

    part = ics.hernquist_posmass.smap(mesh, jit=True)(
        N, a=1.0, total_mass=1.0, seed=0, npad=int(0.5 * N), rmax=100.0
    )

    jb = jax_bench(jit_rounds=4, jit_warmup=1)
    jb.measure(fn_jit=fast_multipole_method.smap(mesh, jit=True),
        part=part, cfg_fmm=cfg_fmm, result="locz", tag=f"ndev{ndev}"
    )
