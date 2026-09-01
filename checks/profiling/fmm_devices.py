import os
import numpy as np
import matplotlib.pyplot as plt

from pytest_jax_bench import JaxBench
import jax
import jzfmm
from jztree_utils import ics


jax.distributed.initialize()

def bench_ndev(ndevices):
	jb = JaxBench(jit_rounds=10, jit_warmup=2)

	ns = np.logspace(5, 8, 7)[::-1]

	ndevs = 2**np.arange(0, 10)
	ndevs = ndevs[ndevs <= ndevices][::-1]

	res = dict(ndevs=ndevs, ns=ns)

	for ndev in ndevs:
		mesh = jax.make_mesh((ndev,), axis_names=("gpus",), axis_types=jax.sharding.AxisType.Explicit)
		res[str(ndev)] = []
		res["std" + str(ndev)] = []
		for n in ns:
			if jax.process_index() == 0:
				print(ndev, n)

			softening = 0.1 * (ndev * float(n)) ** (-1.0 / 3.0)

			cfg_fmm = jzfmm.FMMConfig(
				kernel=jzfmm.PlummerKernel(softening=softening),
			)

			if ndev > 1:
				part = ics.uniform_particles.smap(mesh, jit=True)(int(n), npad=int(max(n*0.2, 1e5)))
				f = jzfmm.fmm.fast_multipole_method.smap(mesh, jit=True)
				timing = jb.measure(
					fn_jit=f, part=part, cfg_fmm=cfg_fmm, write=False, result="locz"
				)[0]
			else:
				part = ics.uniform_particles(int(n))
				timing = jb.measure(
					fn_jit=jzfmm.fmm.fast_multipole_method.jit, part=part, cfg_fmm=cfg_fmm,
					write=False, result="locz"
				)[0]

			res[str(ndev)].append(timing.jit_mean_ms)
			res["std" + str(ndev)].append(timing.jit_std_ms)

	return res


os.makedirs("out", exist_ok=True)

ndevices = jax.device_count()
# ndevices = 64
fname = f"out/fmm_devices_{ndevices}.npz"

# if not os.path.exists(fname):
if ndevices == jax.device_count():
	res = bench_ndev(ndevices)
	if jax.process_index() == 0:
		np.savez(fname, **res)
else:
	res = np.load(fname)


if jax.process_index() == 0:
	plt.figure(figsize=(5,3.5))
	# for ndev in (64,32,16,8,4,2,1):
	for ndev in (1,2,4,8,16,32,64):
		if ndev not in res["ndevs"]:
			continue
		color = plt.get_cmap("viridis")(np.log2(ndev) / 6)
		label = f"{ndev} GPU{'s' if ndev > 1 else ''}"
		plt.loglog(res["ns"], res[str(ndev)], label=label, marker="o", color=color)
		print(ndev, res[str(ndev)][0])
	plt.legend(ncol = 2)
	plt.xlabel("N per GPU")
	plt.ylabel("Time [ms]")

	nlin = np.logspace(6.5, 8.)
	plt.loglog(nlin, nlin/2.0e5, ls="dashed", color="black")
	plt.annotate("linear scaling", (1.5e7, 4.5e1))

	plt.savefig("out/fmm_devices.pdf", bbox_inches="tight")
