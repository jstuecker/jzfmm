import jax
import jax.numpy as jnp
import fmdj
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_compilation_cache_dir", "logs/cache")

eps = 1e-2
N = int(1024*1024)

pos0 = jax.random.normal(jax.random.PRNGKey(0), (N,3), dtype=jnp.float32) * 1.0
mass0 = jnp.ones(len(pos0), dtype=pos0.dtype)
xm = jnp.concatenate([pos0, mass0[:,None]], axis=-1)

import time
t0 = time.time()

fphi = fmdj.fmm.direct_force_and_potential.jit(xm, softening=eps, kahan=True)

for p in (1,2,3,4,5):
    cfg = fmdj.Config(softening=eps)
    cfg.fmm.p = p
    cfg.fmm.kahan_summation = True

    fphi_new = fmdj.fmm.new_fmm_fphi.jit(pos0, mass0, cfg)

    rel_err = jnp.linalg.norm(fphi_new[:,:3] - fphi[:,:3], axis=-1)/jnp.linalg.norm(fphi[:,:3], axis=-1)

    plt.hist(np.log10(rel_err), bins=np.linspace(-7,0), label=f'p={p}', alpha=0.5,
             color="C%d"%(p-1), edgecolor='black')

    print(f"p={p} done, {time.time() - t0:.2f}s")

plt.xlim(-7, -1)
plt.legend()
plt.xlabel("log10(relative error)")
plt.ylabel("count")

plt.title("Force Error Distribution")
plt.savefig("logs/force_error_distribution.pdf")
plt.show()