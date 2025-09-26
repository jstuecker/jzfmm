from fmdj.variants import VariantsNew as Variants, Variant, VariantLineNew

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

    vline = VariantLineNew()
    h0 = hash(vline)
    vline[Variants.direct_summation_force] = v1
    h1 = hash(vline)
    vline[Variants.direct_summation_force] = v2
    h2 = hash(vline)
    vline[Variants.direct_summation_force] = v1
    h1b = hash(vline)

    vline2 = VariantLineNew()
    vline2[Variants.direct_summation_force] = v1
    h1c = hash(vline2)

    assert h0 != h1
    assert h1 != h2
    assert h1 == h1b
    assert h1 == h1c