API Reference
=============

This reference lists the supported interfaces for configuring and running
jz-fmm. Lower-level implementation functions are intentionally omitted; users
interested in those details can consult the source code directly.

Particle data
-------------

The core data structures describe simulation particles and the potential,
force, and tidal-field values calculated for them.

.. automodule:: jzfmm.data
   :members: Particles, LocalExpansion
   :member-order: bysource

Configuration
-------------

Configuration objects select the interaction kernel, opening criterion, force
solver, units, and integration method. They are static arguments to compiled
JAX functions, so modifying them generally triggers recompilation. Config
variables are not differentiable and should not be used as optimization
parameters for :func:`jax.grad` or :func:`jax.vjp`.

Configurations are composable dataclasses: specialized objects can be nested
to describe a complete simulation. For example, this configuration combines a
custom tree, interaction kernel, opening criterion, and integrator::

   import jzfmm
   from jztree.config import TreeConfig

   cfg = jzfmm.SimConfig(
       force=jzfmm.FMMConfig(
           tree=TreeConfig(max_leaf_size=64),
           kernel=jzfmm.PlummerKernel(softening=0.01),
           opening=jzfmm.OpeningByAngle(theta=0.7),
           p=6,
       ),
       integrator=jzfmm.KDKConfig(),
   )

Most configurations have sensible defaults, so only parameters relevant to a
particular simulation need to be changed. Individual settings can conveniently
be modified after construction, including settings in nested configs::

   cfg = jzfmm.SimConfig()
   cfg.force.kernel.softening = 0.1
   cfg.force.p = 6

   # Replace an entire nested config to select a different force solver.
   cfg.force = jzfmm.DirectSummationConfig(
       kernel=jzfmm.PlummerKernel(softening=0.1)
   )

Config hashes include the values of nested configuration objects. Configs
should therefore not be modified after being added to a dictionary, set, or
other hash-based container.

.. automodule:: jzfmm.config
   :members: KernelConfig, PlummerKernel, Plummer2DKernel,
      SoftenedDistanceKernel, OpeningCriterionConfig, OpeningByAngle,
      PotentialField, UnitConfig, IntegratorConfig, DKDConfig, KDKConfig,
      DKDLatticeConfig, DirectSummationConfig, FMMConfig, SimConfig
   :member-order: bysource

Force evaluation
----------------

These functions evaluate interactions using either the fast multipole method
or direct summation.

.. automodule:: jzfmm.fmm
   :members: direct_summation, fast_multipole_method
   :member-order: bysource

Time integration
----------------

These functions calculate forces, advance particles, and run simulations.

.. automodule:: jzfmm.time_integration
   :members: force_and_potential, find_center, timestep, simulate,
      simulate_with_outputs
   :member-order: bysource

External potentials
-------------------

External potential fields can be attached to a simulation through
:class:`jzfmm.config.SimConfig`.

.. automodule:: jzfmm.external_potential
   :members: NFWPotential, HernquistPotential, UniformAcceleration,
      DiskPotential, MilkyWayPotential
   :member-order: bysource

Loss functions
--------------

.. automodule:: jzfmm.loss
   :members: maximum_mean_discrepancy
   :member-order: bysource

Additional helpers
------------------

The :mod:`jzfmm_utils` package contains optional helpers for creating initial
conditions and plotting particle distributions. These conveniences are not
part of the documented core API.
