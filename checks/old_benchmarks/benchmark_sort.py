import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from fmdj.utility import Timer
import numpy as np
import time
import cupy as cp

def f_argsort(x):
    return jnp.argsort(x[::-1]**2 + 2*x + 1)
f_argsort.jit = jax.jit(f_argsort)

def f_sort(x):
    return jnp.sort(x[::-1]**2 + 2*x + 1)
f_sort.jit = jax.jit(f_sort)

def f_lexargsort(x):
    return jnp.lexsort([x, x, x[::-1]**2 + 2*x + 1])
f_lexargsort.jit = jax.jit(f_lexargsort)

def f_lexsort(x):
    return jax.lax.sort([x[::-1]**2 + 2*x + 1, x, x], num_keys=3)
f_lexsort.jit = jax.jit(f_lexsort)

timer = Timer(loops=5)

for N in int(1e3), int(1e4), int(1e5), int(1e6), int(1e7), int(1e8):
    timer.set_tag(N=N, a=1.)
    x = jnp.linspace(-1, 1, N).block_until_ready()
    i1 = timer.timeit_jit(f_argsort.jit, x)
    i2 = timer.timeit_jit(f_sort.jit, x)
    i3 = timer.timeit_jit(f_lexargsort.jit, x)
    r4 = timer.timeit_jit(f_lexsort.jit, x)

ax = timer.plot_timings("N", save="logs/sort_timings.pdf")
# plt.show()

# ============= simplified version for jax github ==============
plt.figure()

Ns = [int(1e3), int(1e4), int(1e5), int(3e5), int(1e6), int(3e6), int(1e7), int(3e7)]

tcupy_sort, tcupy_argsort, tjax_sort, tjax_argsort = np.zeros((4,len(Ns),40))

for i,N in enumerate(Ns):
    values = jax.random.randint(jax.random.PRNGKey(0), N, 0, 2**30, dtype=jnp.int32).block_until_ready()
    values_cp = cp.from_dlpack(values.__dlpack__())
    for j in range(40):
        cp.cuda.Device(0).synchronize()
        t0 = time.time()
        sorted_idx = cp.sort(values_cp)
        cp.cuda.Device(0).synchronize()
        t1 = time.time()
        sorted_idx2 = cp.argsort(values_cp)
        cp.cuda.Device(0).synchronize()
        t2 = time.time()
        sorted_idx = jnp.sort(values).block_until_ready()
        t3 = time.time()
        sorted_idx2 = jnp.argsort(values).block_until_ready()
        t4 = time.time()
        
        tcupy_sort[i,j], tcupy_argsort[i,j], tjax_sort[i,j], tjax_argsort[i,j] = t1-t0, t2-t1, t3-t2, t4-t3

# Plot average over 40 iterations (excluding the first run which might compile some code)
print(np.mean(tcupy_sort[:,1:], axis=1))
plt.plot(Ns, np.mean(tcupy_sort[:,1:], axis=1)*1e3, label="cupy sort", marker="o")
plt.plot(Ns, np.mean(tcupy_argsort[:,1:], axis=1)*1e3, label="cupy argsort", marker="o")
plt.plot(Ns, np.mean(tjax_sort[:,1:], axis=1)*1e3, label="jax sort", ls="dashed", marker="o")
plt.plot(Ns, np.mean(tjax_argsort[:,1:], axis=1)*1e3, label="jax argsort", ls="dashed", marker="o")
plt.ylabel("Time (ms)")
plt.xlabel("N")
plt.legend()
plt.loglog()
plt.grid()

# plt.show()
plt.savefig("logs/sort_vs_cupy.pdf", bbox_inches='tight')
# plt.show()