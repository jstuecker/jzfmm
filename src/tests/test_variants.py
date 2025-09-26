import fmdj
import jax.numpy as jnp
from fmdj.variants import V, Variant, VariantLine

def f1(x):
    return x + 1
def f2(x):
    return x * 2

def test_hashes():
    v1 = Variant(f1, tag="f1")
    v2 = Variant(f2, tag="f2")
    v1b = Variant(f1, tag="f1")

    assert hash(v1) == hash(v1b)
    assert hash(v1) != hash(v2)

    vline = VariantLine()
    h0 = hash(vline)
    vline[V.direct_summation_force] = v1
    h1 = hash(vline)
    vline[V.direct_summation_force] = v2
    h2 = hash(vline)
    vline[V.direct_summation_force] = v1
    h1b = hash(vline)

    vline2 = VariantLine()
    vline2[V.direct_summation_force] = v1
    h1c = hash(vline2)

    assert h0 != h1
    assert h1 != h2
    assert h1 == h1b
    assert h1 == h1c

def test_retracing():
    def f1(x, cfg=None):
        print("Tracing my own variant!")
        return x*0.+1.

    def f2(x, cfg=None):
        print("Tracing my Second variant!")
        return x*0.+2.

    x = jnp.zeros((1,))
    cfg = fmdj.Config()

    fmdj.variants.vm[fmdj.variants.V.test_function].set(f1)
    assert fmdj.variants.test_function.jit(x, cfg=cfg) == 1.

    fmdj.variants.vm[fmdj.variants.V.test_function].set(f2)
    assert fmdj.variants.test_function(x, cfg=cfg) == 2., "Resolves to f2!"
    assert fmdj.variants.test_function.jit(x, cfg=cfg) == 1., "Resolve to f1, since retracing doesn't work!"

    fmdj.variants.vm[fmdj.variants.V.test_function].set(f1)
    cfg1 = fmdj.variants.vm.config_with_variants(fmdj.Config())
    assert cfg1 == fmdj.variants.vm.config_with_variants(fmdj.Config()), "Check config hash makes sense"
    assert fmdj.variants.test_function.jit(x, cfg=cfg1) == 1.

    fmdj.variants.vm[fmdj.variants.V.test_function].set(f2)
    cfg2 = fmdj.variants.vm.config_with_variants(fmdj.Config())
    assert cfg2 != cfg1, "Configs should be different, because of changed variants"

    assert fmdj.variants.test_function.jit(x, cfg=cfg2) == 2., "Should resolve to f2!"

    fmdj.variants.vm.print_available_variants(cfg)

    fmdj.variants.vm[fmdj.variants.V.test_function].set(f2, tag="other_one")