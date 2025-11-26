import matplotlib.pyplot as plt
from fmdj.data import Particles
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import jax.numpy as jnp

def plot_particles_inset(t: float, p: Particles, previous=None, skip=1):
    if previous is None:
        fig, ax = plt.subplots(1, 1, figsize=[6, 6])
        axin = inset_axes(ax, width="40%", height="40%", loc="lower left")

        ax.set_xticks([-200, -100, 0, 100, 200])
        ax.set_yticks([-200, -100, 0, 100, 200])
        ax.set_xlim(-200, 200)
        ax.set_ylim(-200, 200)

        axin.set_xlim(-10, 10)
        axin.set_ylim(-10, 10)
        axin.set_title("20 kpc")

        axin.set_xticks([])
        axin.set_yticks([])

        title = ax.text(0.5,0.95, f't={time_in_years(t)/1e9:.1f} Gyr', transform=ax.transAxes, 
                        ha="center")

        ax.scatter(0, 0, s=20, c="black", marker="o")
        
        s1 = ax.scatter(p.apos()[::skip,0], p.apos()[::skip,1], s=1, alpha=0.05, label=f't={t:.2f}')
        s2 = axin.scatter(p.pos[:,0], p.pos[:,1], s=4, alpha=0.01, label=f't={t:.2f}')
    else:
        fig, ax, axin, s1, s2, title = previous

        title.set_text(f't={time_in_years(t)/1e9:.1f} Gyr')
        
        s1.set_offsets(jnp.array([p.apos()[::skip,0], p.apos()[::skip,1]]).T)
        s2.set_offsets(jnp.array([p.pos[:,0], p.pos[:,1]]).T)

    return fig, ax, axin, s1, s2, title

def time_in_years(t) -> float:
    si_parsec, si_year = 3.085677581491367e+16, 31536000.0
    to_year = (1e3*si_parsec) / (1e3)  / si_year
    return t * to_year