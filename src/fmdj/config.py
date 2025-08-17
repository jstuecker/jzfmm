from dataclasses import dataclass, field, is_dataclass, replace
import jax

@dataclass(frozen=True)
class Variant:
    tag: str
    fn: callable
    applicable: callable = lambda cfg: True

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
class OpeningBarnesAndHut:
    opening_angle : float = 0.8

@dataclass(frozen=True)
class OpeningRelative:
    relative_accuracy : float = 0.01

@dataclass(frozen=True)
class BaseConfig:
    tags : tuple[str] = ("cj", "ref")

@dataclass(frozen=True)
class Config(BaseConfig):
    opening: OpeningBarnesAndHut | OpeningRelative = OpeningBarnesAndHut()

class VariantManager():
    def __init__(self):
        self.variants = VariantDict()

    def register_variants(self, var : Variants | VariantDict):
        for key in self.variants.__dataclass_fields__:
            if key in var.__dataclass_fields__:
                vdict = getattr(self.variants, key)
                new_var = getattr(var, key)
                if new_var is None:
                    continue
                elif isinstance(new_var, dict):
                    vdict.update(new_var)
                elif isinstance(new_var, Variant):
                    vdict[new_var.tag] = new_var
                else:
                    raise TypeError(f"Unknown variant type {type(new_var)} for field '{key}'")

    def resolve_variants(self, cfg: BaseConfig, verbose=1) -> Variants:
        """Resolve the variants for a given config.
        
        verbose : 0: no printing at all
                1: print warnings in likely error cases
                2: print warnings in likely correct cases that may be confusing
                3: useful output for understanding what happened
                4: a lot of debug output
        """

        if not isinstance(cfg, BaseConfig):
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
                    selected_variants[field_name] = applicable_variants[tag]
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

vm = VariantManager()