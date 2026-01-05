from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
import jax
import jax.numpy as jnp
from typing import Tuple
from .tools import conditional_callback

#Some global variables

mesh = jax.sharding.Mesh(jax.devices(), ('gpus',), axis_types=(AxisType.Auto))
sharding = NamedSharding(mesh, P('gpus'))
ndev = len(jax.devices())

def global_splits(n, axis_name="gpus"):
    alln = jax.lax.all_gather(n, axis_name)
    return jnp.pad(jnp.cumsum(alln), (1,0), constant_values=0)

def pytree_len(x):
    """Returns the leading axis of the first leaf of a pytree"""
    leaves = jax.tree_util.tree_leaves(x)

    return len(leaves[0])

def empty_like(x, float_val=jnp.nan, int_val=0):
    def empty_el(xi):
        if xi.dtype.kind == "f":
            return jnp.full_like(xi, fill_value=float_val)
        else:
            return jnp.full_like(xi, fill_value=int_val)

    return jax.tree.map(empty_el, x)

def all_to_all_with_splits(x, ispl, output=None, axis_name="gpus", verify=True, copy_self=True):
    """all_to_all communication with data-dependent communication volume
    
    We send to rank i: x[ispl[i]:ispl[i+1]] 
    all received values will be inserted continguously into output (starting at 0).
    
    x: jnp.ndarray or pytree. If it is a pytree the communication will be applied over the leading
       dimensions of all leaves (undefined behaviour if some leaves have different lengths)
    output: jnp.ndarray or pytree. If x is a pytree output needs to be of identical structure.
            If not provided, we use a copy of x filled with jnp.nan (or 0 for integers)
    verify: If True, throws an error if output buffer is too small. Otherwise out-of-range values
            will simply be discarded.
    copy_self: Extract self-send data and copy it directly (surprisingly this is faster)
    """
    if output is None:
        output = empty_like(x)

    out_size = pytree_len(output)

    input_offsets = ispl[:-1]
    send_sizes = ispl[1:] - ispl[:-1]
    recv_sizes = jax.lax.all_to_all(send_sizes, axis_name, 0, 0, tiled=True)
    output_offsets = jnp.cumsum(recv_sizes) - recv_sizes

    if verify:
        def myerr(need, have):
            raise MemoryError(f"The receiving buffer is too small, need={need}, have={have}")
        need = jnp.sum(recv_sizes)
        recv_sizes = recv_sizes + conditional_callback(need > out_size, myerr, need, out_size)

    if copy_self:
        # avoid communication for self i/o
        rank = jax.lax.axis_index("gpus")
        iout = jnp.arange(out_size) #+ output_offsets[rank]
        iin = jnp.arange(out_size) + input_offsets[rank] - output_offsets[rank]
        mask = (iout >= output_offsets[rank]) & (iout < output_offsets[rank] + send_sizes[rank])
        
        send_sizes = send_sizes.at[rank].set(0)
        recv_sizes = recv_sizes.at[rank].set(0)

        def copy(xi, outi):
            mask_rs = jnp.reshape(mask, (len(mask),) + (1,)*(xi.ndim -1))
            return jnp.where(mask_rs, xi[iin], outi)

        output = jax.tree.map(copy, x, output)

    # funnily jax.lax.ragged_all_to_all wants to know the output_offsets on the
    # sending GPU rather than the receiving one... So we need to communicate the offsets
    output_offsets = jax.lax.all_to_all(output_offsets, axis_name, 0, 0, tiled=True)

    def comm(xi, outi):
        return jax.lax.ragged_all_to_all(
            xi, outi, input_offsets, send_sizes, output_offsets, recv_sizes, axis_name=axis_name
        )

    return jax.tree.map(comm, x, output)

def get_rank_info() -> Tuple[int, int, str]:
    mesh = jax.sharding.get_abstract_mesh()
    if len(mesh.axis_names) == 0:
        return 0, 1, None
    assert len(mesh.axis_names) == 1, "Assuming only a single sharded axis"
    axis_name = mesh.axis_names[0]

    rank = jax.lax.axis_index(axis_name)
    ndev = jax.lax.axis_size(axis_name)

    return rank, ndev, axis_name

def send_to_right(x, axis_name, invalid_val=0):
    rank = jax.lax.axis_index(axis_name)
    ndev = jax.lax.axis_size(axis_name)

    xin = jax.lax.ppermute(x, axis_name, [(i, i+1) for i in range(0,ndev-1)])
    xin = jnp.where(rank == 0, invalid_val, xin)

    return xin

def send_to_left(x, axis_name, invalid_val=0):
    rank = jax.lax.axis_index(axis_name)
    ndev = jax.lax.axis_size(axis_name)

    xin = jax.lax.ppermute(x, axis_name, [(i, i-1) for i in range(1,ndev)])
    xin = jnp.where(rank == ndev-1, invalid_val, xin)

    return xin