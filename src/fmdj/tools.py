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