import jax
import jax.numpy as jnp
import fmdj
import numpy as np
import custom_jax as cj
import matplotlib.pyplot as plt

N = int(3e4)

pos0 = jax.random.normal(jax.random.PRNGKey(0), (N,3), dtype=jnp.float32) * 0.05
mass0 = jnp.ones(len(pos0), dtype=jnp.float32)

import time
t0 = time.time()

for p in (1,2,3,4):
    phi_direct = fmdj.multipoles.potential_direct_sum(pos0, mass0)
    phi = fmdj.fmm.fast_multipole_potential.jit(pos0, mass0, return_sorted=False, p=p, use_cj=True)

    err = np.abs((phi - phi_direct)/phi_direct)
    plt.hist(np.log10(err), bins=np.linspace(-7,-2), label=f'p={p}', alpha=0.5)
    print(time.time() - t0)

plt.legend()
plt.xlabel("log10(relative error)")
plt.ylabel("count")

plt.title("Potential Error Distribution")
plt.show()