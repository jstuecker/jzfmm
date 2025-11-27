import os
import jax
import jax.numpy as jnp
import fmdj
import numpy as np
import matplotlib.pyplot as plt

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_numpy_dtype_promotion", "strict")

jax.config.update("jax_compilation_cache_dir", "logs/cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

eps = 1e-2
N = int(1024*1024)

pos0 = jax.random.normal(jax.random.PRNGKey(0), (N,3), dtype=jnp.float32) * 1.0
mass0 = jnp.ones(len(pos0), dtype=pos0.dtype)
part = fmdj.data.PosMass(pos0, mass0)

import time
t0 = time.time()

phi0 = fmdj.fmm.direct_force_and_potential.jit(part.posm(), softening=eps, kahan=True)[:,3]

for p in (1,2,3,4,5):
    cfg = fmdj.Config(softening=eps)
    cfg.fmm.kahan_summation = True
    cfg.fmm.p = p

    phi_new = fmdj.fmm.fmm_force_and_potential.jit(part, cfg)[:,3]

    plt.hist(np.log10(np.abs((phi_new - phi0)/phi0)), bins=np.linspace(-7,0), label=f'p={p}', alpha=0.5,
             color="C%d"%(p-1), edgecolor='black')

    print(time.time() - t0)

plt.xlim(-7, -1)
plt.legend()
plt.xlabel("log10(relative error)")
plt.ylabel("count")

plt.title("Potential Error Distribution")
plt.savefig("logs/potential_error_distribution.pdf")
plt.show()