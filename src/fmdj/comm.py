from jax.sharding import PartitionSpec as P, NamedSharding, AxisType
import jax
import jax.numpy as jnp
from typing import Tuple

#Some global variables

mesh = jax.sharding.Mesh(jax.devices(), ('gpus',), axis_types=(AxisType.Auto))
sharding = NamedSharding(mesh, P('gpus'))
ndev = len(jax.devices())

def ragged_all_to_all_through_buf(operand, output, input_offsets, send_sizes, output_offsets, recv_sizes, *, axis_name, buf_size=1024):
    """Does the same as jax.lax.ragged_all_to_all, but works on CPU and needs a buffer"""
    def comm(icom, output):
        igpu, ioff = jnp.indices((ndev, buf_size))
        inoff = input_offsets[igpu] + icom*buf_size + ioff
        xbuf = operand[inoff]

        xrecv = jax.lax.all_to_all(xbuf, axis_name, 0, 0, tiled=True)

        valid = icom*buf_size + ioff < recv_sizes[igpu]
        outoff = output_offsets[igpu] + icom*buf_size + ioff
        outoff = jnp.where(valid, outoff, output.size)
        output = output.at[outoff].set(xrecv)

        return output
    
    max_size = jax.lax.pmax(jnp.max(send_sizes), "gpus")
    ncomm = (max_size + buf_size - 1) // buf_size

    output = jax.lax.fori_loop(0, ncomm, comm, jax.lax.pvary(output, axis_name))

    return output

from jax.experimental import io_callback
def conditional_callback(flag, f, *args, **kwargs):
    """Calls a device function f, only if the flag is True. Useful for raising exceptions that 
    truly stop the execution of a jitted program

    returns an integer that is set to 0 if the callback is not triggered. This can be used to force
    the jax graph to resolve the condition before continuing the graph, e.g. as in
    """

    res = jax.lax.cond(
        flag, 
        lambda : io_callback(f, jax.ShapeDtypeStruct((), jnp.int32), *args, **kwargs),
        lambda : jnp.int32(0)
    )

    return res

def global_splits(n, axis_name="gpus"):
    alln = jax.lax.all_gather(n, axis_name)
    return jnp.pad(jnp.cumsum(alln), (1,0), constant_values=0)

def all_to_all_with_splits(x, ispl, output, axis_name="gpus", buf_size=1024, mode="auto", verify=True, copy_self=True):
    """all_to_all communication where we send to rank i: x[ispl[i]:ispl[i+1]]
    
    output: array where outputs are written to
    mode: can be "auto", "ragged" or "buffered". "ragged" will use `jax.lax.ragged_all_to_all`.
        "auto" will use "ragged", but fall back to "buffered" when `jax.lax.ragged_all_to_all` 
        is not available
    buf_size: size of the buffer per GPU -- only used in "buffered" mode.
    verify: If True, throws an error if output buffer is too small. Otherwise out-of-range values
            will simply be discarded.
    """

    input_offsets = ispl[:-1]
    send_sizes = ispl[1:] - ispl[:-1]
    recv_sizes = jax.lax.all_to_all(send_sizes, axis_name, 0, 0, tiled=True)
    output_offsets = jnp.cumsum(recv_sizes) - recv_sizes

    if verify:
        def myerr(need, have):
            raise MemoryError(f"The receiving buffer is too small, need={need}, have={have}")
        need = jnp.sum(recv_sizes)
        have = len(output)
        recv_sizes = recv_sizes + conditional_callback(need > have, myerr, need, have)

    if copy_self:
        # avoid communication for self i/o
        rank = jax.lax.axis_index("gpus")
        iout = jnp.arange(len(output)) #+ output_offsets[rank]
        iin = jnp.arange(len(output)) + input_offsets[rank] - output_offsets[rank]
        sel = (iout >= output_offsets[rank]) & (iout < output_offsets[rank] + send_sizes[rank])
        sel = jnp.reshape(sel, (len(sel),) + (1,)*(x.ndim -1))

        output = jnp.where(sel, x[iin], output)
        send_sizes = send_sizes.at[rank].set(0)
        recv_sizes = recv_sizes.at[rank].set(0)

    if mode in ("auto", "ragged"):
        try:
            # funnily jax.lax.ragged_all_to_all wants to know the output_offsets on the
            # sending GPU rather than the receiving one... So we need to communicate the offsets
            output_offsets = jax.lax.all_to_all(output_offsets, axis_name, 0, 0, tiled=True)
            return jax.lax.ragged_all_to_all(x, output, input_offsets, send_sizes, output_offsets, 
                                             recv_sizes, axis_name=axis_name)
        except jax.errors.JaxRuntimeError as e:
            print("Cannout use jax.lax.ragged_all_to_all (likely because not implemented for CPU) :")
            if mode == "ragged":
                raise e
            else:
                print(e)
                print(f"Falling back to less efficient buffered approach with buf_size={buf_size}")
                mode = "buffered"
    if mode == "buffered":
        return ragged_all_to_all_through_buf(x, output, input_offsets, send_sizes, output_offsets, 
                                             recv_sizes, axis_name=axis_name, buf_size=buf_size)


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