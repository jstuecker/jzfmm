from importlib.metadata import entry_points
import os, warnings
from .variants import vm

_DEFAULT_ALLOW = {"custom_jax"}   # Plugins that are automatically enabled if available
_enabled = set()

def available_plugins() -> dict[str, object]:
    return {ep.name: ep for ep in entry_points(group="fmdj.plugins")}

def enable_plugins(names: list[str] | None = None, *, include_default=True, strict=True, verbose=True):
    eps = available_plugins()
    allow = set(names or [])
    if include_default: allow |= _DEFAULT_ALLOW
    for name in allow:
        ep = eps.get(name)
        if not ep:
            msg = f"[fmdj] plugin '{name}' not found among {list(eps)}"
            if strict: raise RuntimeError(msg)
            warnings.warn(msg); continue
        obj = ep.load()  # can be module or callable
        if callable(obj): obj()  # e.g. register()
        _enabled.add(name)
        if verbose:
            print_diagnostics()

def enabled_plugins() -> tuple[str, ...]:
    return tuple(sorted(_enabled))

def print_diagnostics():
    print("Available plugins:", tuple(available_plugins().keys()))
    print("Enabled plugins:", enabled_plugins())
    vm.print_available_variants()