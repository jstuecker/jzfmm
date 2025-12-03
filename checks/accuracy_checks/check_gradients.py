import jax
import jax.numpy as jnp
import fmdj
import numpy as np
import matplotlib.pyplot as plt
import time
from dataclasses import replace
from fmdj_utils.ics import gaussian_blob

part = gaussian_blob(N=int(512*1024), scale=1.0, mass=1.)

fmm_cfg = fmdj.config.FMMConfig(kahan_summation=True, multipoles_around_com=True)
cfg = fmdj.Config(softening=1e-2, fmm=fmm_cfg)

def rerr_pos(a: fmdj.data.PosMass, b: fmdj.data.PosMass):
    return jnp.linalg.norm(a.pos - b.pos, axis=-1)/jnp.linalg.norm(b.pos, axis=-1)
def rerr_mass(a: fmdj.data.PosMass, b: fmdj.data.PosMass):
    return jnp.abs((a.mass - b.mass)/b.mass)

def direct(part):
    loc = fmdj.fmm.direct_force_and_potential(part, softening=cfg.softening, kahan=True) * cfg.G()
    return fmdj.data.LocalExpansion(loc)
def fmm(part, p):
    return fmdj.fmm.fast_multipole_method(part, cfg=replace(cfg, fmm=replace(cfg.fmm, p=p)))

t0 = time.time()

gposm_phi_ref = jax.jit(jax.grad(lambda part: direct(part).potential().sum()))(part)
gposm_acc_ref = jax.jit(jax.grad(lambda part: jnp.abs(direct(part).force()).sum()))(part)

print(f"Direct sum. done, {time.time() - t0:.2f}s")

fig, axs = plt.subplots(2,2, figsize=(10,9.5))

def hist(ax, rerr, p):
    ax.hist(np.log10(rerr), bins=np.linspace(-7,0), label=f'p={p}', alpha=0.5, color="C%d"%(p-1), edgecolor='black')

for p in (2,3,4,5):
    gposm_phi_fmm = jax.jit(jax.grad(lambda part: fmm(part, p).potential().sum()))(part)
    gposm_acc_fmm = jax.jit(jax.grad(lambda part: jnp.abs(fmm(part, p).force()).sum()))(part)

    hist(axs[0,0], rerr_pos(gposm_acc_fmm, gposm_acc_ref), p)
    hist(axs[0,1], rerr_mass(gposm_acc_fmm, gposm_acc_ref), p)
    hist(axs[1,0], rerr_pos(gposm_phi_fmm, gposm_phi_ref), p)
    hist(axs[1,1], rerr_mass(gposm_phi_fmm, gposm_phi_ref), p)

    print(f"p={p} done, {time.time() - t0:.2f}s")

axs[0,0].set_title("d|Acc|/dPos")
axs[0,1].set_title("d|Acc|/dMass")
axs[1,0].set_title("dPhi/dPos")
axs[1,1].set_title("dPhi/dMass")

for ax in axs.flatten():
    ax.legend()
    ax.set_xlabel("log10(relative error)")

plt.savefig("logs/grad_error_distribution.pdf")
plt.show()