import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import replace
from fmdj_utils.ics import gaussian_blob
from fmdj.data import LocalExpansion
from fmdj.config import Config, PlummerKernel
from fmdj.fmm import direct_summation, fast_multipole_method

part = gaussian_blob(N=int(512*1024), scale=1.0, mass=1.)

cfg = Config(kernel=PlummerKernel(softening=1e-2))
cfg.fmm.kahan_summation = True
cfg.fmm.p_extra_m2l = 1
# cfg.tree.mass_centered = True

def rerr_force(a: LocalExpansion, b: LocalExpansion):
    return jnp.linalg.norm(a.force() - b.force(), axis=-1)/jnp.linalg.norm(b.force(), axis=-1)
def rerr_potential(a: LocalExpansion, b: LocalExpansion):
    return jnp.abs((a.potential() - b.potential())/b.potential())
def rel_mom_cons(a: LocalExpansion):
    f = a.force() * part.mass[:,None]
    return jnp.abs(jnp.sum(f) / jnp.linalg.norm(f))

t0 = time.time()

loc_ref = direct_summation.jit(part, kernel=cfg.kernel, kahan=True, G=cfg.G())

print(f"Direct sum. done, {time.time() - t0:.2f}s, rel. mom. cons = {rel_mom_cons(loc_ref):.2e}")

fig, axs = plt.subplots(1,2, figsize=(12,5))

def hist(ax, rerr, p):
    ax.hist(np.log10(rerr), bins=np.linspace(-7,0), label=f'p={p}', alpha=0.5, color="C%d"%(p-1), edgecolor='black')

for p in (1,2,3,4,5): # 
    loc = fast_multipole_method.jit(part, cfg=replace(cfg, fmm=replace(cfg.fmm, p=p)))

    hist(axs[0], rerr_force(loc, loc_ref), p)
    hist(axs[1], rerr_potential(loc, loc_ref), p)

    print(f"p={p} done, {time.time() - t0:.2f}s, rel. mom. cons = {rel_mom_cons(loc):.2e}")

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
