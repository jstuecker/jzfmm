import jax
import jax.numpy as jnp
import fmdj
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_compilation_cache_dir", "logs/cache")

eps = 1e-2
N = int(512*1024)

pos0 = jax.random.normal(jax.random.PRNGKey(0), (N,3), dtype=jnp.float32) * 1.0
mass0 = jnp.ones(len(pos0), dtype=pos0.dtype)
part = fmdj.data.PosMass(pos0, mass0)

import time
t0 = time.time()

cfg = fmdj.Config(softening=eps)
cfg.fmm.kahan_summation = True

fphi = fmdj.fmm.direct_force_and_potential.jit(part.posm(), softening=eps, kahan=True) * cfg.G()

print(f"Direct sum. done, {time.time() - t0:.2f}s")

fig, axs = plt.subplots(1,2, figsize=(12,5))

for p in (1,2,3,4,5):
    cfg.fmm.p = p

    loc: fmdj.data.LocalExpansion = fmdj.fmm.fast_multipole_method.jit(part, cfg)

    rel_err = jnp.linalg.norm(loc.force() - fphi[:,:3], axis=-1)/jnp.linalg.norm(fphi[:,:3], axis=-1)

    axs[0].hist(np.log10(rel_err), bins=np.linspace(-7,0), label=f'p={p}', alpha=0.5,
             color="C%d"%(p-1), edgecolor='black')
    axs[1].hist(np.log10(np.abs((loc.potential() - fphi[:,3])/fphi[:,3])), bins=np.linspace(-7,0), label=f'p={p}', alpha=0.5,
             color="C%d"%(p-1), edgecolor='black')

    print(f"p={p} done, {time.time() - t0:.2f}s")

axs[0].set_title("Force Error Distribution")
axs[0].set_xlim(-6, 0)
axs[0].set_ylabel("count")

axs[1].set_title("Potential Error Distribution")
axs[1].set_xlim(-7, -1)

for ax in axs:
    ax.legend()
    ax.set_xlabel("log10(relative error)")

plt.savefig("logs/fphi_error_distribution.pdf")
plt.show()