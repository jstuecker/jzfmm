import numpy as np
import aegis
import jax
import jax.numpy as jnp
import fmdj
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from fmdj_utils.plots import time_in_years
from matplotlib.animation import FuncAnimation
from fmdj.external_potential import MilkyWayPotential
from jztree.config import TreeConfig

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--show", action="store_true", help="Visualize on the fly")
parser.add_argument("--dm", action="store_true", help="Include DM halo in potential")
args = parser.parse_args()

# ------------------------------------------------------------------------------------------------ #
#                                        Initial Conditions                                        #
# ------------------------------------------------------------------------------------------------ #

nfw = aegis.profiles.NFWProfile(5, 1e12)

r = 10.**jax.random.uniform(jax.random.key(0), int(1e5), jnp.float32, 0., jnp.log10(20.))
xy = aegis.numerics.sample.random_direction(len(r), ndim=2)
pos = jnp.stack([xy[...,0], xy[...,1], jax.random.normal(jax.random.key(1), len(xy))*0.1], axis=-1)*r[:,None]

pot = MilkyWayPotential()
accr = jnp.linalg.norm(pot.acceleration(pos, cfg=fmdj.SimConfig()), axis=-1)
vcirc = jnp.sqrt(jnp.clip(accr * r, 0., None))
vel = jnp.cross(pos, jnp.array((0.,0.,1.))) * (vcirc/r)[:,None]
vel = vel + jax.random.normal(jax.random.key(2), vel.shape) * 10.

# We choose effectively mass-less particles, let dynamics be driven by external potential
part = fmdj.data.Particles(pos=jnp.array(pos), vel=jnp.array(vel), mass=1e-1)

# ------------------------------------------------------------------------------------------------ #
#                                             Plotting                                             #
# ------------------------------------------------------------------------------------------------ #

def myplot(t: float, p: fmdj.data.Particles, previous=None, skip=10, dm=True):
    if previous is None:
        fig, ax = plt.subplots(1, 1, figsize=[6.5, 6])
        ax = plt.gca()

        # make figure and plot frame black and font white
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_xlabel("x [kpc]", color="white")
        ax.set_ylabel("y [kpc]", color="white")

        ax.tick_params(colors="white", which="both")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        title = ax.text(0.5,0.95, f't={time_in_years(t)/1e6:.0f} Myr', transform=ax.transAxes, 
                        ha="center", color="white")
        
        if dm:
            xi = np.linspace(-20., 20., 100)
            x,y = np.meshgrid(xi,xi)
            rho = np.clip(nfw.density(np.sqrt(x**2+y**2)), 0, 1e8)
            plt.pcolormesh(x, y, np.log10(rho), shading="auto", cmap="cubehelix", alpha=0.5)
        
        color = plt.get_cmap("afmhot")(0.5 + 0.5*np.random.uniform(size=len(p.pos)))[::skip] # 0.4 + 0.6 * 

        s1 = ax.scatter(p.apos()[::skip,0], p.apos()[::skip,1], s=4, alpha=0.1, label=f't={t:.2f}', c = color)
    else:
        fig, ax, s1, title = previous

        title.set_text(f't={time_in_years(t)/1e6:.0f} Myr')
        
        s1.set_offsets(jnp.array([p.apos()[::skip,0], p.apos()[::skip,1]]).T)
    
    return fig, ax, s1, title

dm = args.dm

previous = myplot(0, part, dm=dm)
fig, ax, s1, title = previous
fig.subplots_adjust(left=0.1, right=0.96, top=0.96, bottom=0.09)

def update(t_and_p):
    global previous
    previous = myplot(*t_and_p, previous=previous, skip=10)
    fig, ax, s1, title = previous
    return [s1,title]

# ------------------------------------------------------------------------------------------------ #
#                                         Run and Visualize                                        #
# ------------------------------------------------------------------------------------------------ #

cfg_fmm = fmdj.FMMConfig(
    kernel=fmdj.PlummerKernel(softening=1e-2),
    tree=TreeConfig(mass_centered=False, alloc_fac_nodes=2.0),
)
cfg = fmdj.SimConfig(force=cfg_fmm)
cfg.external_potential = MilkyWayPotential()
if not dm:
    cfg.external_potential.halo = None

sim_iter = fmdj.time_integration.simulate_with_outputs(
    part, tend=nfw.tcirc(20.)*0.5, nout=300, steps_per_output=20, cfg=cfg,
)

# # Use the generator as the frames iterable
ani = FuncAnimation(fig, update, frames=sim_iter, blit=True, interval=40, repeat=True,
                    cache_frame_data=False)

if args.show:
    plt.show()
else:
    plt.close()

    ani.save(f"output/milkyway{'_dm' if dm else '_nodm'}.mp4", writer="ffmpeg", fps=30, dpi=400)
