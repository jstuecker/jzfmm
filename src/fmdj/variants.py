"""This module provides a mechanism to register and select variants of operations. 

This makes it possible to switch between different implementations of the same operation or
to override the default behavior based on runtime conditions or user preferences."""

from dataclasses import dataclass, replace, field
import jax
from typing import Callable, TypeVar, ParamSpec  # Python 3.10+, or typing_extensions
from enum import Enum, auto

def has_gpu():
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False

class HashableDict(dict):
    """Hashable by content; recomputes hash each time (no cache)."""

    def __hash__(self):
        # Canonicalize as a sorted tuple of (key, value) pairs.
        def key_fn(k):
            if isinstance(k, Enum):
                return (0, k.__class__.__qualname__, k.name)
            return (1, type(k).__qualname__, repr(k))
        items = tuple(sorted(self.items(), key=lambda kv: key_fn(kv[0])))
        # NOTE: values must be hashable; otherwise this raises TypeError (good).
        return hash(items)

    def __eq__(self, other):
        if not isinstance(other, dict):
            return NotImplemented
        return dict.__eq__(self, other)

@dataclass(frozen=True)
class Variant:
    fn: callable
    applicable: callable = lambda cfg: True
    tag : str = ""

# ============================= Define Variants that can be overriden ==============================

class V(Enum):
    ilist_node_to_node : int = auto()
    ilist_node_to_leaf : int = auto()
    ilist_leaf_to_node : int = auto()
    ilist_leaf_to_leaf : int = auto()
    test_function : int = auto()

    multipoles_from_particles : int = auto()
    coarsen_multipoles : int = auto()
    evaluate_plane_interactions : int = auto()

# ============================== Helper classes for managing Variants ==============================

@dataclass(unsafe_hash=True)
class VariantConfig:
    tags : tuple[str] = ("user", "cuda", "base")
    variants : HashableDict = field(default_factory=HashableDict)
    verbose : int = 0

class VariantLine(HashableDict):
    def set(self, var : Variant | Callable, tag="user"):
        if callable(var):
            var = Variant(var, tag=tag)
        if not isinstance(var, Variant):
            raise TypeError(f"Value must be callable or a Variant, got {type(var)}")
        if tag is None:
            tag = var.tag
        self[tag] = var

    def select(self, cfg, require=False) -> Variant | None:
        all_variants = self
        applicable_variants = {vtag: v for vtag, v in all_variants.items() if v.applicable(cfg)}

        if cfg.verbose >= 3:
            print(f"-- All variants: {tuple(vtag for vtag in all_variants)}")
            print(f"-- Applicable variants: {tuple(vtag for vtag in applicable_variants)}")
        
        for tag in cfg.tags:
            if tag in applicable_variants:
                if cfg.verbose >= 3:
                    print(f"-- Selected variant: {tag}")
                return replace(applicable_variants[tag], tag=tag)
        
        if require:
            raise ValueError(f"No applicable variant found for tags {cfg.tags} in variants {tuple(all_variants.keys())}")
        else:
            return None

class VariantManager(HashableDict):
    """Manages a dictionary of variants for each allowed function that allows variants"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for v in V:
            self[v] = VariantLine()

    def register_variants(self, vman : "VariantManager"):
        for v in self.keys():
            if v in vman.keys():
                if isinstance(vman[v], VariantLine):
                    self[v].update(vman[v])
                else:
                    raise TypeError(f"Unknown variant type {type(vman[v])} for field '{v}'")

    def resolve_all_variants(self, cfg: VariantConfig, require_all=True) -> HashableDict:
        """Returns an object containing all the variants for a given config."""
        if not isinstance(cfg, VariantConfig):
            raise TypeError("config must be an instance of BaseConfig or a subclass")

        selected_variants = HashableDict()
        for k,vdict in self.items():
            if cfg.verbose >= 4:
                print("Checking variants of {k}")
            selected_variants[k] = vdict.select(cfg, require=require_all)
        
        if cfg.verbose >= 3:
            print("Selected variants:")
            for field_name, variant in selected_variants.items():
                if variant is None:
                    print(f"  {field_name}: None")
                else:
                    print(f"  {field_name}: {variant.tag}")
        
        return selected_variants
    
    def config_with_variants(self, cfg : VariantConfig) -> VariantConfig:
        """If variants are not set, return a new config with resolved variants."""
        cfg = replace(cfg, variants=self.resolve_all_variants(cfg))
        
        return cfg
    
    def print_available_variants(self, cfg : VariantConfig = None):
        print("Available Variants:")
        for k,vline in self.items():
            if cfg is not None:
                vsel = vline.select(cfg).tag
                print(f"-- {k.name}: -> {tuple(vline.keys())} -> {vsel}")
            else:
                print(f"-- {k.name}: -> {tuple(vline.keys())}")
            

# ============================== Helper classes for managing Variants ==============================

from typing import Any, TypeVar, Callable
from functools import wraps
import inspect

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T", bound=Callable[..., Any])

def make_dispatcher(var : VariantLine, base_func: T, add_jit=True) -> T:
    """Create a dispatcher function that selects the appropriate variant based on the config."""
    var[TAG_BASE] = Variant(base_func)

    @wraps(base_func)
    def wrapper(*args, cfg=None, **kwargs):
        variant = var.select(cfg, require=True)
        if cfg.verbose >= 2:
            print(f"Using variant: {variant.tag} for function {base_func.__name__}")

        return variant.fn(*args, cfg=cfg, **kwargs)
    
    wrapper.__signature__ = inspect.signature(base_func)

    if add_jit:
        wrapper.jit = jax.jit(wrapper, static_argnames=("cfg",))

    return wrapper

# ====================================== Global Variables ==========================================

vm = VariantManager()

TAG_BASE = "base"


def _test_function(x, cfg : VariantConfig = None):
    print("Tracing Test function, returns 0")
    return x*0.

test_function = make_dispatcher(vm[V.test_function], _test_function, add_jit=True)