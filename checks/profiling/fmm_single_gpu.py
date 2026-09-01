import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import io
import matplotlib.pyplot as plt

from pytest_jax_bench import JaxBench
import jax
import jax.numpy as jnp
import time


def get_part(n, mode="uniform", seed=0):
	from jztree_utils import ics
	import jzfmm
	from jztree.data import PosMass
	if mode == "grid":
		ng = int(np.cbrt(n))
		xi = jnp.arange(ng, dtype=jnp.float32) * (1.0 / ng)
		pos = jnp.stack(jnp.meshgrid(xi, xi, xi, indexing="ij"), axis=-1).reshape(-1, 3)
		return PosMass(pos=pos, mass=1.0 / len(pos), num=len(pos), num_total=len(pos))
	elif mode == "uniform":
		return ics.uniform_particles(int(n), seed=seed)
	elif mode == "normal":
		return ics.gaussian_particles(int(n), seed=seed)
	elif mode == "hernquist":
		return ics.hernquist_posmass(int(n), seed=seed, a=1.0, total_mass=1.0)
	elif mode == "discodj":
		res = int(np.cbrt(n))
		return ics.discodj_particles.jit(res, boxsize=1.0 * res)
	else:
		raise ValueError(f"Unknown mode {mode}")


def bench_distributions():
	import jzfmm
	from jztree.data import PosMass

	jb = JaxBench(jit_rounds=10, jit_warmup=1)

	t0 = time.time()

	ns = np.logspace(5, 8, 7)
	ns = ((np.cbrt(ns) // 4) * 4) ** 3
	res = dict(n=ns, discodj=[], grid=[], uniform=[], normal=[], hernquist=[])

	for mode in "discodj", "grid", "uniform", "normal", "hernquist":
		for n in ns:
			print(mode, n, time.time()-t0)

			part = get_part(n, mode=mode)

			softening = 0.1 * float(n) ** (-1.0 / 3.0)
			cfg_fmm = jzfmm.FMMConfig(
				kernel=jzfmm.PlummerKernel(softening=softening),
			)
			cfg_fmm.tree.alloc_fac_nodes = 1.2
			cfg_fmm.alloc_fac_ilist = 64.

			timing, loc = jb.measure(
				fn_jit=jzfmm.fmm.fast_multipole_method.jit,
				part=part,
				cfg_fmm=cfg_fmm,
				write=False,
			)
			res[mode].append(timing.jit_mean_ms)

	return res


os.makedirs("out", exist_ok=True)

if os.path.exists("out/fmm_distributions.npz"):
# if False:
	res = np.load("out/fmm_distributions.npz")
else:
	res = bench_distributions()
	np.savez("out/fmm_distributions.npz", **res)

plt.figure(figsize=(5,3.5))
plt.loglog(res["n"], res["grid"], label="grid", marker="o")
plt.loglog(res["n"], res["uniform"], label="uniform", marker="o")
plt.loglog(res["n"], res["normal"], label="normal", marker="o")
plt.loglog(res["n"], res["discodj"], label="cosm. (no wrapping)", marker="o")
plt.loglog(res["n"], res["hernquist"], label="hernquist", marker="o")

# nlin = np.logspace(6.5, 8.)
# plt.loglog(nlin, nlin / 3.5e5, ls="dashed", color="black")
# plt.annotate("linear scaling", (1.2e7, 2.2e1))

nlin = np.logspace(6.5, 8.)
plt.loglog(nlin, nlin/2.0e5, ls="dashed", color="black")
plt.annotate("linear scaling", (1.5e7, 4.5e1))

plt.legend()
plt.xlabel("N")
plt.ylabel("Time [ms]")

plt.savefig("out/fmm_distributions.pdf", bbox_inches="tight")
