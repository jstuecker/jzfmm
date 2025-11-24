import jax
from jax.experimental import io_callback
import jax.numpy as jnp
from .config import Config
import inspect
import os

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

def _callsite(depth=2, *, shorten=True):
    """
    depth=2 -> caller of `log` (this -> log -> caller).
    """
    frame = inspect.currentframe()
    # Walk back `depth` frames safely
    for _ in range(depth):
        frame = frame.f_back if frame is not None else None
    if frame is None:
        return "<unknown:0>"

    info = inspect.getframeinfo(frame)
    path = info.filename
    if shorten:
        try:
            path = os.path.relpath(path)
        except Exception:
            pass
        path = os.path.basename(path)  # or keep full relpath if you prefer
    return f"{path}:{info.lineno}"

def log(txt, 
        *args, 
        level : int = 0,
        cfg : Config | None = None,
        ordered : bool = False,
        partitioned : bool = False,
        **kwargs):
    if cfg is not None:
        if level > cfg.logging.level:
            return
        if cfg.logging.show_loc:
            txt = f"[{_callsite()}] {txt}"

    jax.debug.print(txt, *args, ordered=ordered, partitioned=partitioned, **kwargs)


# ------------------------------------------------------------------------------------------------ #
#                               Some frequently used helper functions                              #
# ------------------------------------------------------------------------------------------------ #

def cumsum_starting_with_zero(x):
    return jnp.pad(jnp.cumsum(x), (1, 0))

def offset_sum(num):
    cs = jnp.cumsum(num, axis=0)
    return cs - num, cs[-1]

def masked_prefix_sum(mask):
    off, n = offset_sum(mask)
    off_masked = jnp.where(mask, off, len(mask))
    return off_masked, n

def inverse_of_splits(ispl, size):
    """given [0, 4, 7] returns [0,0,0,0,1,1,1] for size=7"""
    mask = jnp.zeros(size, dtype=jnp.int32).at[ispl].add(1)
    return jnp.cumsum(mask) - 1