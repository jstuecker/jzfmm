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

class VariantsNew(Enum):
    ilist_node_to_node : int = auto()
    ilist_node_to_leaf : int = auto()
    ilist_leaf_to_node : int = auto()
    ilist_leaf_to_leaf : int = auto()
    direct_summation_force : int = auto()

@dataclass(unsafe_hash=True)
class Variants:
    ilist_node_to_node : Variant = None
    ilist_node_to_leaf : Variant = None
    ilist_leaf_to_node : Variant = None
    ilist_leaf_to_leaf : Variant = None
    direct_summation_force : Variant = None

# ============================== Helper classes for managing Variants ==============================

@dataclass(unsafe_hash=True)
class VariantConfigNew:
    tags : tuple[str] = ("cuda", "base")
    variants : HashableDict = field(default_factory=HashableDict)
    verbose : int = 0

@dataclass(frozen=True)
class VariantConfig:
    tags : tuple[str] = ("cuda", "base")
    variants : Variants = None
    verbose : int = 0

class VariantLineNew(HashableDict):
    def add(self, var : Variant, tag=None):
        if not isinstance(var, Variant):
            raise TypeError(f"Value must be a Variant, got {type(var)}")
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


@dataclass(unsafe_hash=True)
class VariantDict():
    """Groups a bunch of variants that may differ by their tags"""
    v : dict[str, Variant] = field(default_factory=dict)

    def __getitem__(self, key):
        return self.v[key]
    def __setitem__(self, key, value):
        if not isinstance(value, Variant):
            raise TypeError(f"Value must be a Variant, got {type(value)}")
        self.v[key] = value

    def select(self, cfg : VariantConfig, require=False) -> Variant | None:
        assert isinstance(cfg, VariantConfig), "cfg must be inherited from VariantConfig"

        all_variants = self.v
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

class VariantManager(Variants):
    """Manages a dictionary of variants for each allowed function that allows variants"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name in self.__dataclass_fields__:
            if getattr(self, field_name) is None:
                setattr(self, field_name, VariantDict())

    def register_variants(self, var : "VariantManager"):
        data_fields = Variants.__dataclass_fields__
        for tag in data_fields:
            if tag in var.__dataclass_fields__:
                vdict = getattr(self, tag)
                new_var = getattr(var, tag)
                if new_var is None:
                    continue
                elif isinstance(new_var, VariantDict):
                    vdict.v.update(new_var.v)
                else:
                    raise TypeError(f"Unknown variant type {type(new_var)} for field '{tag}'")

    def resolve_all_variants(self, cfg: VariantConfig, require_all=True) -> Variants:
        """Returns an object containing all the variants for a given config."""
        data_fields = Variants.__dataclass_fields__
        if not isinstance(cfg, VariantConfig):
            raise TypeError("config must be an instance of BaseConfig or a subclass")

        selected_variants = {}
        for field_name in data_fields:
            if cfg.verbose >= 4:
                print("Checking variants of {field_name}")
            vdict : VariantDict = getattr(self, field_name)
            selected_variants[field_name] = vdict.select(cfg, require=require_all)
        
        if cfg.verbose >= 3:
            print("Selected variants:")
            for field_name, variant in selected_variants.items():
                if variant is None:
                    print(f"  {field_name}: None")
                else:
                    print(f"  {field_name}: {variant.tag}")
        
        return Variants(**selected_variants)
    
    def config_with_variants(self, cfg : VariantConfig) -> VariantConfig:
        """If variants are not set, return a new config with resolved variants."""
        if cfg.variants is None:
            cfg = replace(cfg, variants=self.resolve_all_variants(cfg))
        return cfg
    
    def print_available_variants(self):
        print("Available Variants:")
        data_fields = Variants.__dataclass_fields__
        for field_name in data_fields:
            all_variants = getattr(self, field_name)
            print(f"-- {field_name}: -> {tuple(all_variants.v.keys())}")

class VariantManagerNew(HashableDict):
    """Manages a dictionary of variants for each allowed function that allows variants"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for v in VariantsNew:
            self[v] = VariantLineNew()

    def register_variants(self, vman : "VariantManagerNew"):
        for v in self.keys():
            if v in vman.keys():
                if isinstance(vman[v], VariantLineNew):
                    self[v].update(vman[v])
                else:
                    raise TypeError(f"Unknown variant type {type(vman[v])} for field '{v}'")

    def resolve_all_variants(self, cfg: VariantConfig, require_all=True) -> Variants:
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
        if cfg.variants is None:
            cfg = replace(cfg, variants=self.resolve_all_variants(cfg))
        return cfg
    
    def print_available_variants(self):
        print("Available Variants:")
        for k,vline in self.items():
            print(f"-- {k.name}: -> {tuple(vline.keys())}")

# ============================== Helper classes for managing Variants ==============================

from typing import Any, TypeVar, Callable
from functools import wraps
import inspect

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T", bound=Callable[..., Any])

def make_dispatcher(var : VariantLineNew, base_func: T, add_jit=True) -> T:
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

vm = VariantManagerNew()

TAG_BASE = "base"