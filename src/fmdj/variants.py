"""This module provides a mechanism to register and select variants of operations. 

This makes it possible to switch between different implementations of the same operation or
to override the default behavior based on runtime conditions or user preferences."""

from dataclasses import dataclass
from contextlib import contextmanager
import os
import jax
from functools import wraps

@dataclass
class Variant:
    name: str
    fn: callable
    priority: int = 0               # higher wins
    only_if: callable = lambda: True  # runtime predicate

_variants: dict[str, dict[str,Variant]] = {}
_overrides: dict[str, str] = {}     # op_name -> variant_name

def has_gpu():
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False

def has_pkg(modname: str) -> bool:
    try:
        __import__(modname)
        return True
    except Exception:
        return False

def register_variant(op: str, *, name: str, fn, priority: int = 0, only_if=lambda: True, clear_caches=True):
    if clear_caches:
        jax.clear_caches()

    if priority is None:
        vardict = _variants.get(op, {})
        max_priority = max((v.priority for v in vardict.values()), default=0)
        priority = max_priority + 1

    vardict = _variants.setdefault(op, {})
    vardict[name] = Variant(name, fn, priority, only_if)

def variant(op: str, *, name: str = "user", priority: int | None = None, only_if=lambda: True, clear_caches=True):
    """Registers a variant for the operation `op` with the given name and priority."""
    def deco(fn):
        register_variant(op, name=name, fn=fn, priority=priority, only_if=only_if, clear_caches=clear_caches)
        return fn
    return deco

def select(op: str):
    """Selects the best variant for the operation `op`."""

    # First get all useable variants of the operation
    vardict = _variants.get(op, {})
    candidates = {k:v for k,v in vardict.items() if v.only_if()}
    if len(candidates) == 0:
        raise RuntimeError(f"No usable variants for {op!r}. Registered: { [v.name for v in _variants.get(op, [])] }")

    # Now check if there is an active override
    ov = _overrides.get(op)
    if not ov: # Can also override via environment variables
        ov = os.getenv(f"FMDJ_OVERRIDE_{op.upper()}")
    if ov:
        if ov in candidates:
            return candidates[ov].fn
        else:
            raise RuntimeError(f"Override {ov!r} for {op!r} not found in registered variants: { [v.name for v in candidates.values()] }")

    # otherwise, we select the highest priority variant
    best = max(candidates, key=lambda name: candidates[name].priority)
    return candidates[best].fn

@contextmanager
def algorithm(clear_caches="enter_and_exit", **overrides):
    """A context manager that selects specific variants for the target operations.
    This allows to override the default behavior of operations in a specific context.
    E.g. use like
    with algorithm(potential="gpu"):
        # Do something

    clear_caches: Can be "enter", "exit", "enter_and_exit" or False. Wheter to clear jax's global 
        caches. This may be necessary to ensure that all externally jitted functions get recompiled
        and use the correct variants. However, this may be overkill in most cases and therefore
        you can control whether/when the caches are cleared. If you do, you should make sure to 
        recompile any jitted functions that depend on the selected variants.
    """

    old = dict(_overrides)
    _overrides.update(overrides)
    try:
        if clear_caches in ("enter", "both"):
            jax.clear_caches()
        yield
    finally:
        _overrides.clear()
        _overrides.update(old)
        if clear_caches in ("exit", "both"):
            jax.clear_caches()