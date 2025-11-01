import os
import jax
import jax.numpy as jnp
import fmdj
import numpy as np
import custom_jax as cj
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

import time
t0 = time.time()

phi0 = cj.forces.force_and_potential.jit(pos0, mass0, softening=eps, kahan=True)[:,3]

# for p in (1,2,3,4,5):
for p in (1,2,3,4,5):
    config = fmdj.Config(tags=("base",), softening=eps, p=p, verbose=2)
    config_cj = fmdj.Config(tags=("cuda", "base"), softening=eps, p=p, verbose=2)
    config_cj.tree.p = config_cj.p
    config_cj.tree.kahan_summation = True
    # config_cj.tree.ilist_alloc_fac = 2056
    # config_cj.opening.opening_angle = 0.6

    # phi_jax = fmdj.fmm.fast_multipole_potential.jit(pos0, mass0, config)
    # phi_cj = fmdj.fmm.fast_multipole_potential.jit(pos0, mass0, config_cj)
    phi_new = fmdj.fmm.new_fmm.jit(pos0, mass0, config_cj)

    # plt.hist(np.log10(np.abs((phi_jax - phi0)/phi0)), bins=np.linspace(-7,1), label=f'p={p}', alpha=0.8)
    plt.hist(np.log10(np.abs((phi_new - phi0)/phi0)), bins=np.linspace(-7,-1), label=f'p={p}', alpha=0.5,
             color="C%d"%(p-1), edgecolor='black')

    print(time.time() - t0)

plt.xlim(-7, -1)
plt.legend()
plt.xlabel("log10(relative error)")
plt.ylabel("count")

plt.title("Potential Error Distribution")
plt.savefig("logs/potential_error_distribution.pdf")
plt.show()