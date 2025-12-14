import fmdj
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import aegis

import argparse

import matplotlib
matplotlib.use("TkAgg")

from matplotlib.animation import FuncAnimation
import fmdj_utils.plots

parser = argparse.ArgumentParser()
parser.add_argument("--show", action="store_true", help="Visualize on the fly")
args = parser.parse_args()

print(args.show)

prof = aegis.profiles.NFWProfile(conc=10., r200c=10.)
pos0, vel0, m = prof.sample_particles(1024*128, result="pos_vel_m", rpmin=1e-3, ramax=10.)

p0 = fmdj.data.Particles(jnp.array(pos0), jnp.array(m), jnp.array(vel0))
cfg = fmdj.config.Config()
cfg.softening = 1e-2

host = aegis.profiles.NFWProfile(conc=6., m200c=1e12)
cfg.external_potential = fmdj.external_potential.NFWPotential(host.rs, host.rhoc)

p0.cpos = jnp.array((150.,0.,0.))
p0.cvel = jnp.array((0.,host.vcirc(150.)*0.9,0.))

fig, ax, axins, s1, s2, title = fmdj_utils.plots.plot_particles_inset(0., p0, skip=10)

def update(t_and_p):
    fmdj_utils.plots.plot_particles_inset(*t_and_p, previous=(fig, ax, axins, s1, s2, title), skip=10)
    return [s1,s2,title]

sim_iter = fmdj.time_integration.simulate_with_outputs(
    p0, tend=host.tcirc(150.)*2., nout=200, steps_per_output=20, cfg=cfg,
)

# Use the generator as the frames iterable
ani = FuncAnimation(fig, update, frames=sim_iter, blit=True, interval=40, repeat=True,
                    cache_frame_data=False)

if args.show:
    plt.show()
else:
    plt.close()
    import os
    os.makedirs("output", exist_ok=True)

    ani.save("output/nbody_simulation.mp4", dpi=200)