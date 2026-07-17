import os
import aegis
import jax.numpy as jnp
import fmdj
from fmdj.config import FMMConfig, PlummerKernel, QuarticPlummerKernel, SimConfig
from fmdj.time_integration import simulate_with_outputs
import numpy as np
import matplotlib.pyplot as plt

# Parameters:
Ns = [int(1e7), int(1e6), int(1e5), int(1e4)]
steps_per_tcirc = 100
target_times = (0, 1, 4, 16) # times we want to plot in units of tcirc
softening = 2e-3

prof = aegis.profiles.HernquistProfile(a=1., M=1.)
tcirc = prof.tcirc(r=1.)
rbins = np.logspace(-2.2, 1.2)

def relaxation_radius(t_in_tcirc, N):
    rtest = np.logspace(-3,3,10000)
    trel = prof.two_body_relaxation_time(rtest, N, modeN="Ntot", lam=None, rsoft=softening, rmax=1e6, rnorm=1e6)
    rrel = np.interp(tcirc*t_in_tcirc/(np.pi), trel, rtest)
    return rrel

# ------------------------------------------------------------------------------------------------ #
#                                      Profile cache helpers                                       #
# ------------------------------------------------------------------------------------------------ #

cache_dir = "logs/hernquist_sim"
os.makedirs(cache_dir, exist_ok=True)

def cache_path(N, t):
    return os.path.join(cache_dir, f"profile_N{N:g}_t{t:g}.npy")

def save_profile(pp, N, t):
    np.save(cache_path(N, t), pp.to_dict(), allow_pickle=True)

def load_profile(N, t):
    d = np.load(cache_path(N, t), allow_pickle=True).item()
    return aegis.profiles.ParticleProfile.from_dict(d)

# ------------------------------------------------------------------------------------------------ #
#                                          Run simulation                                          #
# ------------------------------------------------------------------------------------------------ #

for N in Ns:
    all_cached = all(os.path.exists(cache_path(N, t)) for t in target_times)

    if not all_cached:
        print(f"[N={N:g}] Sampling initial conditions...")
        np.random.seed(42)
        pos, vel, mass = prof.sample_particles(ntot=N, result="pos_vel_m")
        part = fmdj.data.Particles(pos=jnp.array(pos), mass=jnp.array(mass), vel=jnp.array(vel))

        prof_0 = aegis.profiles.ParticleProfile((pos, vel, mass), rbins=rbins)
        save_profile(prof_0, N, 0)

        sim_targets = sorted(t for t in target_times if t > 0)
        if sim_targets:
            max_t = max(sim_targets)
            cfg = SimConfig(force=FMMConfig(kernel=PlummerKernel(softening=softening)))
            print(f"[N={N:g}] Running simulation to t = {max_t} tcirc...")
            sim_iter = simulate_with_outputs(
                part, tend=max_t * tcirc, nout=max_t,
                steps_per_output=steps_per_tcirc, cfg=cfg,
            )
            for i, (_, p_out) in enumerate(sim_iter):
                t = i  # i=0 is IC (tstart), i=1 is tcirc, i=2 is 2*tcirc, ...
                if t in sim_targets:
                    print(f"[N={N:g}]   Saving profile at t = {t} tcirc")
                    prof_t = aegis.profiles.ParticleProfile(
                        (np.array(p_out.pos), np.array(p_out.vel), np.array(p_out.mass)),
                        rbins=rbins,
                    )
                    save_profile(prof_t, N, t)

profiles = {(N, t): load_profile(N, t) for N in Ns for t in target_times}

# ------------------------------------------------------------------------------------------------ #
#                                               Plots                                              #
# ------------------------------------------------------------------------------------------------ #

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(4.5, 4.5),
    gridspec_kw={"height_ratios": [3, 1]},
    sharex=True,
)

ri = profiles[(Ns[0], 0)].ri
rho_analytic = prof.density(ri)
rho_norm = float(np.interp(1.0, ri, rho_analytic))  # analytic density at r = a = 1
rho_analytic_normed = rho_analytic / rho_norm
colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
color_for_N = {N: colors[j] for j, N in enumerate(Ns)}
ls_for_t = {t: ls for t, ls in zip(target_times, ["-", "--", "-.", ":"])}

plot_times = [t for t in target_times if t != 0]

# expected number of analytic particles per bin (N-dependent)
_shell_vol = (4/3) * np.pi * (rbins[1:]**3 - rbins[:-1]**3)

for j, N in enumerate(Ns):
    color = color_for_N[N]
    mask_resolved = (rho_analytic * _shell_vol * N) >= 40
    # label on the first plotted time so the legend shows N
    first = True
    for t in plot_times:
        ls = ls_for_t[t]
        rho = profiles[(N, t)].density(ri)
        ratio = rho / rho_analytic
        rho_normed = rho / rho_norm
        label = rf"$N=10^{int(np.log10(N))}$" if first else None
        first = False

        rrelax = relaxation_radius(t, N)
        mask_lo = (ri <= rrelax) & mask_resolved
        mask_hi = (ri >= rrelax) & mask_resolved

        ax_top.loglog(ri[mask_lo], rho_normed[mask_lo], color=color, alpha=0.5, ls=ls)
        ax_top.loglog(ri[mask_hi], rho_normed[mask_hi], color=color, alpha=0.8, ls=ls, label=label)
        ax_bot.semilogx(ri[mask_lo], ratio[mask_lo], color=color, alpha=0.5, ls=ls)
        ax_bot.semilogx(ri[mask_hi], ratio[mask_hi], color=color, alpha=0.8, ls=ls)

        rho_at_rrelax = np.interp(rrelax, ri, rho_normed)
        ratio_at_rrelax = np.interp(rrelax, ri, ratio)
        ax_top.plot(rrelax, rho_at_rrelax, "o", color=color)
        ax_bot.plot(rrelax, ratio_at_rrelax, "o", color=color)

ax_top.loglog(ri, rho_analytic_normed, ls="-", color="black")
ax_top.set_ylabel(r"$\rho\,/\,\rho_a$")

from matplotlib.lines import Line2D
ax_top.legend(handles=[
    *ax_top.get_legend_handles_labels()[0],
    Line2D([0], [0], color="black", ls="-", label="analytic"),
    *[Line2D([0], [0], color="gray", ls=ls_for_t[t], label=rf"$t = {t}\,t_c$")
      for t in plot_times],
    Line2D([0], [0], marker="o", color="black", linestyle="none", label="relaxation radius"),
], ncols=2)

ax_bot.axhline(1., ls="dashed", color="black")
ax_bot.set_ylim(0, 2)
ax_bot.set_xlabel("r/a")
ax_bot.set_ylabel("ratio")

fig.tight_layout()
plt.savefig("logs/hernquist_sim/hernquist_sim.pdf", bbox_inches="tight")
