JAX compatibility and helpers
=============================

Compatibility badges
--------------------

The API reference uses badges to summarize whether a function produces the
correct result under common JAX transformations. ``Shard map`` means that a
natural split of the input produces the corresponding global result;
``Local only`` means that each shard is treated as an independent problem.
Amber badges indicate a documented restriction, blue badges indicate expected
but untested support, and grey badges indicate unsupported behavior.

.. _jit-helper:

The ``.jit`` helper
-------------------

Some functions provide a ``.jit`` attribute containing a preconfigured
:func:`jax.jit` wrapper. It marks configuration and other compile-time options
as static arguments, so it can be called with the same arguments as the
original function::

   result = function.jit(*args, **kwargs)

The helper is optional convenience; functions marked as JIT-compatible can
also be wrapped in a user-defined :func:`jax.jit` transformation.

.. _smap-helper:

The ``.smap`` helper
--------------------

Selected distributed functions provide a ``.smap`` constructor. Given a JAX
device mesh, it returns a callable with the appropriate input partitioning and
static arguments::

   mapped_function = function.smap(mesh, jit=True)
   result = mapped_function(*args, **kwargs)

Setting ``jit=True`` additionally wraps the mapped function in :func:`jax.jit`.
The constructor accepts keyword arguments when calling the returned function
and caches wrappers created for the same mesh. It is intended for jz-fmm's
distributed particle layout, where particle arrays are split over all mesh
axes while configuration arguments remain static.
