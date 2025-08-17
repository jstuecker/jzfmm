"""This module provides a mechanism to register and select variants of operations. 

This makes it possible to switch between different implementations of the same operation or
to override the default behavior based on runtime conditions or user preferences."""

from dataclasses import dataclass, replace, field
import jax

def has_gpu():
    try:
        return any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        return False

@dataclass(frozen=True)
class Variant:
    fn: callable
    applicable: callable = lambda cfg: True
    tag : str = ""

# ============================= Define Variants that can be overriden ==============================

@dataclass(unsafe_hash=True)
class Variants:
    ilist_leaf_to_leaf : Variant = None
    testfunc : Variant = None

# ============================== Helper classes for managing Variants ==============================

@dataclass(frozen=True)
class VariantConfig:
    tags : tuple[str] = ("cj", "ref")
    variants : Variants = None
    verbose : int = 0

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

        if cfg.verbose >= 2:
            print(f"-- All variants: {tuple(vtag for vtag in all_variants)}")
            print(f"-- Applicable variants: {tuple(vtag for vtag in applicable_variants)}")
        
        for tag in cfg.tags:
            if tag in applicable_variants:
                if cfg.verbose >= 2:
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
        """Resolve the variants for a given config.
        
        verbose : 0: no printing at all
                1: print warnings in likely error cases
                2: print warnings in likely correct cases that may be confusing
                3: useful output for understanding what happened
                4: a lot of debug output
        """
        data_fields = Variants.__dataclass_fields__
        if not isinstance(cfg, VariantConfig):
            raise TypeError("config must be an instance of BaseConfig or a subclass")

        selected_variants = {}
        for field_name in data_fields:
            if cfg.verbose >= 2:
                print("Checking variants of {field_name}")
            vdict : VariantDict = getattr(self, field_name)
            selected_variants[field_name] = vdict.select(cfg, require=require_all)
        
        if cfg.verbose >= 2:
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

vm = VariantManager()

TAG_REF = "ref"