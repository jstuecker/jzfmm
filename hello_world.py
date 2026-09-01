import argparse
from dataclasses import replace

import aegis
import jax.numpy as jnp
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from jzfmm.data import Particles
from jzfmm.config import FMMConfig, KDKConfig, PlummerKernel, SimConfig
from jzfmm.external_potential import NFWPotential
from jzfmm.time_integration import find_center, force_and_potential, simulate_with_outputs
from jzfmm_utils.plots import plot_particles_inset

parser = argparse.ArgumentParser()
parser.add_argument("--show", action="store_true", help="Visualize on the fly")
args = parser.parse_args()

prof = aegis.profiles.NFWProfile(conc=10., r200c=10.)
pos0, vel0, m = prof.sample_particles(1024*128, result="pos_vel_m", rpmin=1e-3, ramax=10.)

host = aegis.profiles.NFWProfile(conc=6., m200c=1e12)
pos0 = jnp.array(pos0) + jnp.array((150., 0., 0.))
vel0 = jnp.array(vel0) + jnp.array((0., host.vcirc(150.) * 0.9, 0.))

p0 = Particles(pos=pos0, mass=jnp.array(m), vel=vel0)
cfg_fmm = FMMConfig(kernel=PlummerKernel(softening=1e-2))
cfg = SimConfig(force=cfg_fmm)
cfg.integrator = KDKConfig()
cfg.external_potential = NFWPotential(host.rs, host.rhoc)
p0 = replace(p0, loc=force_and_potential.jit(p0, cfg=cfg))

fig, ax, axins, s1, s2, title = plot_particles_inset(
    0., p0, skip=10, center=find_center.jit(p0, npot=50, nbind=200)[0]
)

def update(t_and_p):
    t, p = t_and_p
    plot_particles_inset(
        t, p, previous=(fig, ax, axins, s1, s2, title),
        skip=10, center=find_center.jit(p, npot=50, nbind=200)[0],
    )
    return [s1,s2,title]

sim_iter = simulate_with_outputs(
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
