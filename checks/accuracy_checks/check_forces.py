import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import replace
from fmdj_utils.ics import gaussian_blob
from fmdj.data import LocalExpansion
from fmdj.config import DirectSummationConfig, FMMConfig, PlummerKernel, UnitConfig
from fmdj.fmm import direct_summation, fast_multipole_method

part = gaussian_blob(N=int(1024*1024), scale=1.0, mass=1.)

cfg_fmm = FMMConfig(
    kernel=PlummerKernel(softening=1e-2),
    kahan_summation=False,
    p_extra_m2l=0,
)
cfg_direct = DirectSummationConfig(kernel=cfg_fmm.kernel, kahan_summation=True)
G = UnitConfig().G()

def rerr_force(a: LocalExpansion, b: LocalExpansion):
    return jnp.linalg.norm(a.force() - b.force(), axis=-1)/jnp.linalg.norm(b.force(), axis=-1)
def rerr_potential(a: LocalExpansion, b: LocalExpansion):
    return jnp.abs((a.potential() - b.potential())/b.potential())
def rel_mom_cons(a: LocalExpansion):
    mass = jnp.broadcast_to(part.mass, part.pos.shape[:-1])
    f = a.force() * mass[:, None]
    return jnp.abs(jnp.sum(f) / jnp.linalg.norm(f))

t0 = time.time()

loc_ref = direct_summation.jit(part, cfg_direct=cfg_direct, G=G)

print(f"Direct sum. done, {time.time() - t0:.2f}s, rel. mom. cons = {rel_mom_cons(loc_ref):.2e}")

fig, axs = plt.subplots(1,2, figsize=(6.5,3.0))

def hist(ax, rerr, p):
    ax.hist(np.log10(rerr), bins=np.linspace(-8, 0, num=81), label=f'p={p}', alpha=0.6, color="C%d"%(p-1), edgecolor='black', density=True, histtype="stepfilled")

for p in (1,2,3,4,5): # 
    loc = fast_multipole_method.jit(part, cfg_fmm=replace(cfg_fmm, p=p), G=G)

    hist(axs[0], rerr_force(loc, loc_ref), p)
    hist(axs[1], rerr_potential(loc, loc_ref), p)

    print(f"p={p} done, {time.time() - t0:.2f}s, rel. mom. cons = {rel_mom_cons(loc):.2e}")

# axs[0].set_title(r"$|\vec{F}_\mathrm{FMM} - \vec{F}_\mathrm{ref}| / |\vec{F}_\mathrm{ref}|$")
axs[0].set_title("Force")
axs[0].set_xlim(-5.2, 0.2)
axs[0].set_ylabel(r"dn/dlog $\epsilon$")

# axs[1].set_title("r"$|\phi_\mathrm{FMM} - \phi_\mathrm{ref}| / |\phi_\mathrm{ref}|$"")
axs[1].set_title("Potential")
axs[1].set_xlim(-7.0, 0)

for ax in axs:
    ax.set_xlabel("log10(relative error)")

axs[1].legend()

plt.savefig("logs/fphi_error_distribution.pdf", bbox_inches="tight")
plt.show()
