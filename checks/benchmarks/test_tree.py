import jax
import jax.numpy as jnp

def fft(x): # An example function we want to profile
    return jnp.fft.ifftn(jnp.fft.fftn(x))

def test_fft(jax_bench):  # jax_bench is a fixture that creates a JaxBench object.
    x = jnp.ones((256, 256, 256), dtype=jnp.float32)

    jb = jax_bench(jit_rounds=20, jit_warmup=1, eager_rounds=10, eager_warmup=1)
    jb.measure(fn=fft, fn_jit=jax.jit(fft), x=x)