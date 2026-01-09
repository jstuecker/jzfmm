from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
import jax
import jax.numpy as jnp
from typing import Tuple
from .tools import conditional_callback, cumsum_starting_with_zero

# ------------------------------------------------------------------------------------------------ #
#                                        General Device Info                                       #
# ------------------------------------------------------------------------------------------------ #

def get_rank_info() -> Tuple[int, int, str]:
    mesh = jax.sharding.get_abstract_mesh()
    if len(mesh.axis_names) == 0:
        return 0, 1, None
    assert len(mesh.axis_names) == 1, "Assuming only a single sharded axis"
    axis_name = mesh.axis_names[0]

    rank = jax.lax.axis_index(axis_name)
    ndev = jax.lax.axis_size(axis_name)

    return rank, ndev, axis_name

# ------------------------------------------------------------------------------------------------ #
#                                       Tiny Helper Functions                                      #
# ------------------------------------------------------------------------------------------------ #

def pytree_len(x):
    """Returns the leading axis of the first leaf of a pytree"""
    leaves = jax.tree_util.tree_leaves(x)

    return len(leaves[0])

def value_for_dtype(dtype, float_val=jnp.nan, int_val=0):
    if dtype.kind == "f":
        return float_val
    else:
        return int_val

def empty_like(x, float_val=jnp.nan, int_val=0):
    def empty_el(xi):
        return jnp.full_like(xi, fill_value=value_for_dtype(xi.dtype, float_val, int_val))

    return jax.tree.map(empty_el, x)

def invalidate(x, mask, invalid_float=jnp.nan, invalid_int=0):
    def inv(x):
        mask_rs = jnp.reshape(mask, mask.shape + (1,)*(x.ndim-1))
        return jnp.where(mask_rs, value_for_dtype(x.dtype, invalid_float, invalid_int), x)
    
    return jax.tree.map(inv, x)

# ------------------------------------------------------------------------------------------------ #
#                                  Simple Communication Directives                                 #
# ------------------------------------------------------------------------------------------------ #

def global_splits(n, axis_name="gpus"):
    alln = jax.lax.all_gather(n, axis_name)
    return jnp.pad(jnp.cumsum(alln), (1,0), constant_values=0)

def send_to_right(x, axis_name, invalid_float=jnp.nan, invalid_int=0):
    rank = jax.lax.axis_index(axis_name)
    ndev = jax.lax.axis_size(axis_name)

    xin = jax.lax.ppermute(x, axis_name, [(i, i+1) for i in range(0,ndev-1)])
    xin = invalidate(xin, rank == 0, invalid_float, invalid_int)

    return xin

def send_to_left(x, axis_name, invalid_float=jnp.nan, invalid_int=0):
    rank = jax.lax.axis_index(axis_name)
    ndev = jax.lax.axis_size(axis_name)

    xin = jax.lax.ppermute(x, axis_name, [(i, i-1) for i in range(1,ndev)])
    xin = invalidate(xin, rank == ndev-1, invalid_float, invalid_int)

    return xin

def get_pos(x):
    if isinstance(x, jax.typing.ArrayLike):
        return x
    else: # assume x is a pytree with .pos attribute
        return x.pos

def shift_particles_left(x, nsend, max_send, npart):
    rank, ndev, axis_name = get_rank_info()
    
    # Validate that send buffer is large enough
    def send_size_err(nsend, max_send):
        raise ValueError(f"Cannot fit {nsend} particles into buffer of size {max_send}!")
    npart = npart + conditional_callback(
        nsend >= max_send, send_size_err, nsend, max_send,
    )

    # Validate that particle array has enough free space
    nget = send_to_left(nsend, axis_name)
    def array_size_err(nhave, nget, nsend, nmax):
        raise MemoryError(
            f"Cannot shift particles: have={nhave}, get={nget}, send={nsend}, max={nmax}."
            "(fix: larger allocaction)"
        )
    npart = npart + conditional_callback((
        npart + nget - nsend >= pytree_len(x)), array_size_err, npart, nget, nsend, pytree_len(x)
    )

    # Send the particles
    x_get = send_to_left(jax.tree.map(lambda v: v[0:max_send], x), axis_name, invalid_float=jnp.nan)

    # Delete the particles that were send
    iar = jnp.arange(pytree_len(x))
    x = invalidate(x, iar < nsend)
    x = jax.tree.map(lambda v: jnp.roll(v, -nsend, axis=0), x)

    # Insert the received particles
    idx = jnp.arange(max_send)
    idx = jnp.where(idx < nget, npart - nsend + idx, pytree_len(x)) # discard indices beyond nadd
    x = jax.tree.map(lambda u,v: u.at[idx].set(v), x, x_get)

    return x, npart + nget - nsend

# ------------------------------------------------------------------------------------------------ #
#                                     All To All communication                                     #
# ------------------------------------------------------------------------------------------------ #

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
    dev_spl = cumsum_starting_with_zero(recv_sizes)
    output_offsets = dev_spl[:-1]

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

    return jax.tree.map(comm, x, output), dev_spl

def dynamic_all_gather(x, nsend, output=None, axis_name="gpus", verify=True):
    """An all-gather where each task may send different amounts.
    
    returns output, dev_spl -- where output[dev_spl[i]:dev_spl[i+1]] contains rank i's input
    """
    if output is None:
        output = empty_like(x)
    
    rank = jax.lax.axis_index(axis_name)
    ndev = jax.lax.axis_size(axis_name)

    nrecv = jax.lax.all_gather(nsend, axis_name)
    dev_spl = cumsum_starting_with_zero(nrecv)

    if verify:
        out_size = pytree_len(output)
        def recv_buffer_err(need, out_size):
            raise MemoryError(f"The receive buffer (size: {out_size}) is to small (need: {need})")
        dev_spl = dev_spl + conditional_callback(
            dev_spl[-1] >= out_size, recv_buffer_err, dev_spl[-1], out_size
        )

    input_off = jnp.zeros(ndev, jnp.int32)
    nsend = jnp.full(ndev, nsend, dtype=jnp.int32)
    output_off = jnp.full(ndev, dev_spl[rank])

    def comm(xi, outi):
        return jax.lax.ragged_all_to_all(
            xi, outi, input_off, nsend, output_off, nrecv, axis_name=axis_name
        )

    return jax.tree.map(comm, x, output), dev_spl

def arange_for_comm(irank: jnp.ndarray, data, num=None, axis_name="gpus"):
    rank = jax.lax.axis_index(axis_name)
    ndev = jax.lax.axis_size(axis_name)

    if num is not None:
        irank = jnp.where(jnp.arange(len(irank), dtype=irank.dtype) < num, irank, ndev)
    isort = jnp.argsort(irank)
    dev_spl = jnp.searchsorted(irank[isort], jnp.arange(ndev+1, dtype=irank.dtype), side="left")

    datasort = jax.tree.map(lambda d: d[isort], data)

    return datasort, dev_spl