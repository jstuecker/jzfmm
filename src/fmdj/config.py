from dataclasses import dataclass, field, is_dataclass
import jax

def static_dataclass(cls):
    """A decorator marking all fields of a dataclass static for JAXs"""
    cls = dataclass(cls)
    cls = jax.tree_util.register_dataclass(cls, data_fields=[], 
                                           meta_fields=cls.__dataclass_fields__.keys())
    return cls

def flatten_dataclass(obj):
    return [getattr(obj, key) for key in obj.__dataclass_fields__]
def unflatten_dataclass(cls, values):
    return cls(*values)

def flatten_nested_dataclass(obj):
    res = []
    fields = obj.__dataclass_fields__
    for name in fields:
        if is_dataclass(fields[name].type):
            res.extend(flatten_nested_dataclass(getattr(obj, name)))
        else:
            res.append(getattr(obj, name))
    return res
def unflatten_nested_dataclass(cls, values):
    fields = cls.__dataclass_fields__
    args = []
    for name in fields:
        field = fields[name]
        if is_dataclass(field.type):
            sub_cls = field.type
            sub_fields = cls.__dataclass_fields__
            sub_values, values = values[:len(sub_fields)], values[len(sub_fields):]
            args.append(unflatten_nested_dataclass(sub_cls, sub_values))
        else:
            args.append(values.pop(0))
    return cls(*args)

def nested_dataclass(cls):
    """A decorator that allows nested dataclasses to be flattened and unflattened."""
    cls = dataclass(cls)
    jax.tree_util.register_pytree_node(cls, 
        flatten_func=lambda obj: ((), flatten_nested_dataclass(obj)),
        unflatten_func=lambda values, _: unflatten_nested_dataclass(cls, values))

    return cls

@static_dataclass
class Variants:
    ilist_leaf_to_leaf : str = "none"
    testfunc : str = "none"

@static_dataclass
class FMMConfig:
    opening_criterion : str = "barnes_and_hut"
    opening_angle : float = 0.8

@nested_dataclass
class Config:
    fmm : FMMConfig = field(default_factory=FMMConfig)
    variants : Variants = field(default_factory=Variants)

def default_config() -> Config:
    """Returns a default configuration object."""
    return Config()