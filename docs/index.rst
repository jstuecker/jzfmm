jz-fmm documentation
====================

**jz-fmm** (*JAX z-order tree, Fast Multipole Method*) is a GPU-native FMM
implementation for fast and differentiable gravitational N-body simulations.
It builds on `jz-tree <https://jstuecker.github.io/jztree/>`_ for GPU-native
tree construction, traversal, and distributed communication. Its high-level
interface is written in JAX, while performance-critical operations use a CUDA
backend through JAX's foreign function interface.

Features
--------

* **GPU-native force evaluation:** the fast multipole method on one or many
  GPUs, including multi-host execution, with isolated boundary conditions.
* **Differentiable simulations:** gradients with respect to particle positions,
  velocities, and masses through force calculations and time integration.
* **Reproducible forces:** bit-perfect reproducibility of repeated FMM
  calculations with the same inputs and configuration on the same hardware.
* **Reversible time integration:** bit-perfect backwards integration with the
  integer-lattice integrator, alongside a standard floating-point integrator.
* **Configurable accuracy:** selectable multipole orders in two and three
  dimensions, with single- and double-precision calculations.
* **Extensibility:** custom external potentials and support for adding new
  interaction kernels beyond gravitational N-body applications.
* **Convenient Python/JAX interface:** compose force calculations, integrators,
  and external potentials to customize N-body simulations in a few lines of
  Python, while retaining JIT-compiled GPU performance.
* **Reference calculations:** direct summation for checking approximate forces.

Start with the :doc:`quickstart`, see :doc:`multi_gpu_guide` for distributed
execution, or consult :doc:`jax_compatibility` for function-specific support.

Performance and scaling
-----------------------

The benchmarks below show force accuracy versus evaluation time and weak
scaling across multiple GPUs. **jz-fmm** delivers world-class performance and demonstrates the 
potential of a fully GPU-native approach to N-body simulations.

.. container:: benchmark-grid

   .. container:: benchmark-panel

      .. image:: _static/hernquist_performance_comparison.png
         :alt: Force accuracy versus evaluation time for jz-fmm, GADGET-4, and PKDGRAV3

      **Accuracy and performance.** Comparison with GADGET-4 (G4) and
      PKDGRAV3 for a single force evaluation in a Hernquist sphere with :math:`4 \times 10^7` particles.
      Labels give the opening angle; jz-fmm curves show different multipole
      expansion orders. All codes were benchmarked on one node with four
      NVIDIA A100 GPUs and a 32-core CPU; a fairer hardware allocation for the
      CPU-only GADGET-4 comparison would use approximately 4--8 times as many
      CPU cores.

   .. container:: benchmark-panel

      .. image:: _static/fmm_devices.png
         :alt: jz-fmm execution time per GPU across one to 64 GPUs

      **Multi-GPU scaling.** Weak scaling from one to 64 GPUs for uniform
      particle distributions. In the GPU-saturating regime, efficiency
      decreases by less than a factor of two from one to 64 devices.

Project links
-------------

* `Source repository <https://github.com/jstuecker/jzfmm>`_
* `jz-tree documentation <https://jstuecker.github.io/jztree/>`_
* `Jens Stücker's homepage <https://jstuecker.github.io/>`_

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation.md
   quickstart.md
   multi_gpu_guide.md
   developer_guide.md
   api.rst
   jax_compatibility.rst
   known_issues.md
   attribution.md
   changelog.md
