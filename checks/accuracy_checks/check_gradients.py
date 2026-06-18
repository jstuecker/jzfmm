import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import replace
from fmdj_utils.ics import gaussian_blob
from fmdj.data import PosMass
from fmdj.fmm import direct_summation, fast_multipole_method
from fmdj.config import DirectSummationConfig, FMMConfig, PlummerKernel, UnitConfig
from jztree.config import TreeConfig

part = gaussian_blob(N=int(1e6), scale=1.0, mass=1.)
part.mass = part.mass * jnp.ones(part.pos.shape[:-1], jnp.float32)
part.num = None
part.num_total = None

cfg_fmm = FMMConfig(
    kernel=PlummerKernel(softening=1e-2),
    tree=TreeConfig(mass_centered=False),
    kahan_summation=False,
)
cfg_direct = DirectSummationConfig(kernel=cfg_fmm.kernel, kahan_summation=True)
G = UnitConfig().G()

def rerr_pos(a: PosMass, b: PosMass):
    return jnp.linalg.norm(a.pos - b.pos, axis=-1)/jnp.linalg.norm(b.pos, axis=-1)
def rerr_mass(a: PosMass, b: PosMass):
    return jnp.abs((a.mass - b.mass)/b.mass)

def direct(part):
    return direct_summation(part, cfg_direct=cfg_direct, G=G)
def fmm(part, p):
    return fast_multipole_method(part, cfg_fmm=replace(cfg_fmm, p=p), G=G)

t0 = time.time()

gposm_phi_ref = jax.jit(jax.grad(lambda part: direct(part).potential().sum()))(part)
gposm_accabs_ref = jax.jit(jax.grad(lambda part: jnp.abs(direct(part).force()).sum()))(part)
gposm_acc_ref = jax.jit(jax.grad(lambda part: direct(part).force().sum()))(part)

print(f"Direct sum. done, {time.time() - t0:.2f}s")

fig, axs = plt.subplots(2,2, figsize=(6.5,6.))

def hist(ax, rerr, p):
    ax.hist(np.log10(rerr), bins=np.linspace(-7, 0, num=57), label=f'p={p}', alpha=0.6, color="C%d"%(p-1), edgecolor='black', density=True, histtype="stepfilled")

for p in (1,2,3,4,5):
    gposm_phi_fmm = jax.jit(jax.grad(lambda part: fmm(part, p).potential().sum()))(part)
    gposm_accabs_fmm = jax.jit(jax.grad(lambda part: jnp.abs(fmm(part, p).force()).sum()))(part)

    hist(axs[0,0], rerr_pos(gposm_accabs_fmm, gposm_accabs_ref), p)
    hist(axs[0,1], rerr_mass(gposm_accabs_fmm, gposm_accabs_ref), p)
    hist(axs[1,0], rerr_pos(gposm_phi_fmm, gposm_phi_ref), p)
    hist(axs[1,1], rerr_mass(gposm_phi_fmm, gposm_phi_ref), p)

    print(f"p={p} done, {time.time() - t0:.2f}s")

axs[0,0].set_title(r"d$\sum |\vec{F}|$/d $\vec{x}$")
axs[0,1].set_title(r"d$\sum |\vec{F}|$/d $m$")
axs[1,0].set_title(r"d$\sum \phi$/d $\vec{x}$")
axs[1,1].set_title(r"d$\sum \phi$/d $m$")

axs[0,0].set_ylim(0, 1.75)

for ax in axs[0,0], axs[1,0]:
    ax.set_xlim(-5.2, 0.2)

for ax in axs[1,:]:
    ax.set_xlabel("log10(relative error)")
for ax in axs[:,0]:
    ax.set_ylabel(r"dn/dlog $\epsilon$")
axs[1,1].legend()

plt.savefig("logs/grad_error_distribution.pdf", bbox_inches="tight")
plt.show()
