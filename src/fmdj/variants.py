"""This module provides a mechanism to register and select variants of operations. 

This makes it possible to switch between different implementations of the same operation or
to override the default behavior based on runtime conditions or user preferences."""

from dataclasses import dataclass, replace

@dataclass(frozen=True)
class Variant:
    fn: callable
    applicable: callable = lambda cfg: True
    tag : str = ""

@dataclass(unsafe_hash=True)
class Variants:
    ilist_leaf_to_leaf : Variant | list[Variant] = None
    testfunc : Variant | list[Variant] = None

class VariantDict(Variants):
    def __init__(self, *args, **kwargs):
        # Initialize all uninitialized fields to empty lists
        super().__init__(*args, **kwargs)
        for field_name in self.__dataclass_fields__:
            if getattr(self, field_name) is None:
                setattr(self, field_name, {})

@dataclass(frozen=True)
class VariantConfig:
    tags : tuple[str] = ("cj", "ref")
    variants : Variants = None

class VariantManager():
    def __init__(self):
        self.variants = VariantDict()

    def register_variants(self, var : Variants | VariantDict):
        for tag in self.variants.__dataclass_fields__:
            if tag in var.__dataclass_fields__:
                vdict = getattr(self.variants, tag)
                new_var = getattr(var, tag)
                if new_var is None:
                    continue
                elif isinstance(new_var, dict):
                    vdict.update(new_var)
                elif isinstance(new_var, Variant):
                    vdict[tag] = new_var
                else:
                    raise TypeError(f"Unknown variant type {type(new_var)} for field '{tag}'")

    def resolve_variants(self, cfg: VariantConfig, verbose=1) -> Variants:
        """Resolve the variants for a given config.
        
        verbose : 0: no printing at all
                1: print warnings in likely error cases
                2: print warnings in likely correct cases that may be confusing
                3: useful output for understanding what happened
                4: a lot of debug output
        """

        if not isinstance(cfg, VariantConfig):
            raise TypeError("config must be an instance of BaseConfig or a subclass")

        selected_variants = {}
        for field_name in self.variants.__dataclass_fields__:
            all_variants = getattr(self.variants, field_name)
            if verbose >= 4:
                print(f"All variants for {field_name}: {tuple(all_variants.keys())}")
            applicable_variants = {vtag: v for vtag, v in all_variants.items() if v.applicable(cfg)}
            # for vname, v in all_variants.items():
            #     if v.applicable(cfg):
            #         if verbose >= 2 and v.tag in applicable_variants:
            #             print(f"Warning: Multiple variants with tag {v.tag} for {field_name}. Using the most recent definition.")
            #         applicable_variants[v.tag] = v
            if verbose >= 4:
                print(f"Applicable variants for {field_name}: {tuple(vtag for vtag in applicable_variants)}")
            
            for tag in cfg.tags:
                if tag in applicable_variants:
                    if verbose >= 4:
                        print(f"Selected variant for {field_name}: {applicable_variants[tag].tag}")
                    selected_variants[field_name] = replace(applicable_variants[tag], tag=tag)
                    break
            
            if not field_name in selected_variants:
                selected_variants[field_name] = None
                if verbose >= 1:
                    print(f"Warning: No applicable variant found for {field_name}")
                    print(f"-- All variants: {tuple(all_variants.keys())}")
                    print(f"-- Applicable variants: {tuple(applicable_variants.keys())}")
                    print(f"-- Looking for tags: {cfg.tags}")
            
        if verbose >= 3:
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
            cfg = replace(cfg, variants=self.resolve_variants(cfg))
        return cfg

vm = VariantManager()