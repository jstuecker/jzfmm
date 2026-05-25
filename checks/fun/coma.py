import numpy as np
import aegis
import jax
import jax.numpy as jnp
import fmdj
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import os
from fmdj_utils.plots import plot_particles_inset, time_in_years
from matplotlib.animation import FuncAnimation
from jztree.config import TreeConfig
import argparse

os.makedirs("output", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--show", action="store_true", help="Visualize on the fly")
parser.add_argument("--dm", action="store_true", help="Include DM halo in potential")
args = parser.parse_args()

# ------------------------------------------------------------------------------------------------ #
#                                     Setup initial conditions                                     #
# ------------------------------------------------------------------------------------------------ #

coma = aegis.profiles.NFWProfile(5, 5e14)

gpos, gvel, gmass = coma.sample_particles(40, "pos_vel_m", ramax=1e3)
gmass = gmass * 1e-3

file = "output/coma_part.npz"
if os.path.exists(file):
    print("Loading Initial Conditions...")
    pdict = np.load(file)
    part = fmdj.data.Particles(pos = pdict["pos"], mass = pdict["mass"], vel = pdict["vel"])
    igal = pdict["igal"]
else:
    print("Generating Initial Conditions...")
    fac = 10**np.random.uniform(-1.,0.3, len(gpos))
    concs = np.random.uniform(10.,30., len(fac))
    pos, vel, mass, igal = [], [], [], []
    for i in range(len(gpos)):
        nfw = aegis.profiles.NFWProfile(concs[i], m200c=gmass[i]*fac[i])
        p,v,m = nfw.sample_particles(int(10000*fac[i]), "pos_vel_m", ramax=nfw.r200c)
        pos.append(p + gpos[i])
        vel.append(v + gvel[i])
        mass.append(m)
        igal.append(np.full(len(p), i))

    part = fmdj.data.Particles(
        pos = jnp.concatenate(pos),
        mass = jnp.concatenate(mass),
        vel = jnp.concatenate(vel)
    )
    igal = jnp.concatenate(igal)

    np.savez(file, pos=part.pos, mass=part.mass, vel=part.vel, igal=igal)

# ------------------------------------------------------------------------------------------------ #
#                                            Define plot                                           #
# ------------------------------------------------------------------------------------------------ #

def myplot(t: float, p: fmdj.data.Particles, previous=None, skip=10, dm=True):
    if previous is None:
        fig, ax = plt.subplots(1, 1, figsize=[6.5, 6])
        ax = plt.gca()

        # make figure and plot frame black and font white
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
        
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        ax.set_xlabel("x [kpc]", color="white")
        ax.set_ylabel("y [kpc]", color="white")

        ax.tick_params(colors="white", which="both")
        for spine in ax.spines.values():
            spine.set_edgecolor("white")

        title = ax.text(0.5,0.95, f't={time_in_years(t)/1e9:.1f} Gyr', transform=ax.transAxes, 
                        ha="center", color="white")
        
        if dm:
            xi = np.linspace(-1000., 1000., 256)
            x,y = np.meshgrid(xi,xi)
            rho = coma.density(np.sqrt(x**2+y**2))
            plt.pcolormesh(x, y, np.log10(rho), shading="auto", cmap="cubehelix", alpha=0.5)
        
        color = plt.get_cmap("afmhot")(0.4 + 0.5 * (igal / len(gpos)))

        s1 = ax.scatter(p.apos()[::skip,0], p.apos()[::skip,1], s=1, alpha=0.1, label=f't={t:.2f}', c = color[::skip])

        fig.subplots_adjust(left=0.14, right=0.96, top=0.96, bottom=0.09)
    else:
        fig, ax, s1, title = previous

        title.set_text(f't={time_in_years(t)/1e9:.1f} Gyr')
        
        s1.set_offsets(jnp.array([p.apos()[::skip,0], p.apos()[::skip,1]]).T)
    
    return fig, ax, s1, title

dm = args.dm

previous = myplot(0, part, dm=dm)
fig, ax, s1, title = previous

def update(t_and_p):
    global previous
    previous = myplot(*t_and_p, previous=previous, skip=10)
    fig, ax, s1, title = previous
    return [s1,title]

cfg_fmm = fmdj.FMMConfig(
    kernel=fmdj.PlummerKernel(softening=0.1),
    tree=TreeConfig(mass_centered=False, alloc_fac_nodes=2.0),
)
cfg = fmdj.SimConfig(force=cfg_fmm)

if dm:
    cfg.external_potential = fmdj.external_potential.NFWPotential(coma.rs, coma.rhoc)

sim_iter = fmdj.time_integration.simulate_with_outputs(
    part, tend=coma.tcirc(150.)*2.5, nout=300, steps_per_output=20, cfg=cfg,
)

ani = FuncAnimation(fig, update, frames=sim_iter, blit=True, interval=40, repeat=True,
                    cache_frame_data=False)

if args.show:
    plt.show()
else:
    plt.close()

    ani.save(f"output/coma{'_dm' if dm else '_nodm'}.mp4", writer="ffmpeg", fps=30, dpi=200)
