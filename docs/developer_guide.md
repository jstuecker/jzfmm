# Developer Guide

This guide describes the structure of the **jz-fmm** repository and the usual
workflow for modifying its Python, JAX, and CUDA components. It is intended for
contributors who want to change the implementation rather than only call its
public interface.

The guide first covers the repository layout, generated FFI bindings, and the
validation tools used throughout the project. More detailed notes on adding
interaction kernels follow at the end.

## Repository structure

The main repository has the following structure:

```text
├── CMakeLists.txt
├── checks
│   ├── accuracy_checks
│   ├── benchmarks
│   ├── fun
│   ├── gradient_descent
│   ├── profiling
│   └── tests
├── docs
├── examples
├── hello_world.py
├── pyproject.toml
└── src
    ├── jzfmm
    ├── jzfmm_cuda
    │   ├── common
    │   └── generated
    └── jzfmm_utils
```

The top-level `CMakeLists.txt` and `pyproject.toml` configure the Python package
and compile the CUDA extension modules. The remaining directories have the
following roles:

- `src/jzfmm` contains the public Python interface and the JAX implementation.
- `src/jzfmm_cuda` contains CUDA kernels and the generated JAX FFI bindings.
- `src/jzfmm_utils` contains optional initial-condition and plotting helpers.
- `checks/tests` contains correctness and regression tests.
- `checks/benchmarks` contains repeatable performance benchmarks.
- `checks/accuracy_checks` contains more expensive numerical experiments.
- `checks/profiling` contains scripts for inspecting performance and CUDA
  resource usage.
- `checks/fun` and `checks/gradient_descent` contain larger applications and
  research experiments rather than unit tests.
- `examples` contains small user-facing examples, while `docs` contains this
  Sphinx documentation.

### Python modules

The core Python package is split into a few comparatively small modules:

```text
src/jzfmm
├── __init__.py
├── config.py
├── data.py
├── external_potential.py
├── fmm.py
├── loss.py
├── multipoles.py
└── time_integration.py
```

- `config.py` defines composable configuration dataclasses. Configuration
  objects are static arguments to compiled JAX functions.
- `data.py` defines particle data and local potential expansions.
- `fmm.py` implements direct summation and the FMM tree walk, including custom
  derivative rules and multi-device communication.
- `multipoles.py` constructs and translates Cartesian multipole expansions.
- `time_integration.py` implements force evaluation, integrator steps,
  simulations, and reversible backward integration.
- `external_potential.py` contains external analytic potential fields.
- `loss.py` contains distributional losses built on the force solvers.

Tree construction, particle ordering, interaction lists, and common
multi-device operations are provided by
[jz-tree](https://jstuecker.github.io/jztree/). Changes to those general tree
algorithms should normally be made in jz-tree rather than duplicated here.

### CUDA modules

The handwritten CUDA implementation is organized as follows:

```text
src/jzfmm_cuda
├── _generate_ffi.py
├── common
│   ├── data.cuh
│   ├── iterators.cuh
│   └── math.cuh
├── fmm.cuh
├── multipoles.cuh
├── opening.cuh
├── pair_summation.cuh
├── radial_kernels.cuh
└── generated
    ├── ffi_fmm.cu
    ├── ffi_multipoles.cu
    └── ffi_pair_summation.cu
```

The `.cuh` files contain the implementation. Files under `generated` expose
selected CUDA kernels through JAX's foreign function interface and are created
by `_generate_ffi.py` using
[jax_ffi_gen](https://github.com/jstuecker/jax_ffi_gen).

Do not modify the generated `.cu` files manually. Modify the corresponding
header and `_generate_ffi.py`, regenerate the bindings, and rebuild the package.

### Why the CUDA implementation uses templates

The CUDA code uses templates heavily for dimensions, floating-point types,
multipole orders, radial kernels, and opening criteria. In particular, the
sizes and indices of multipole arrays are known at compile time. Loops can be
unrolled and array accesses remain static, allowing intermediate multipole data
to stay in registers instead of spilling into slower local memory. For the
central FMM operations, this can improve performance by an order of magnitude.

The price is a large matrix of template instances and correspondingly long
CUDA compilation times. Only a deliberately limited set of combinations is
generated. Supporting a template-valued option outside that set requires
changing the template matrix in `_generate_ffi.py`, regenerating the bindings,
and compiling the CUDA extensions again.

## Generated FFI bindings

`_generate_ffi.py` parses the CUDA signatures and defines template instances,
grid sizes, shared-memory sizes, output initialization, and expressions for
parameters that can be inferred from array shapes. It then produces the
nanobind modules under `src/jzfmm_cuda/generated`.

The template ranges near the top of the file intentionally limit how much code
NVCC must instantiate. At the time of writing, the important defaults include:

```python
dimensions = (2, 3)
direct_summation_dimensions = (2, 3, 4, 5, 6)
float_types = ("float", "double")
p_instance_values = (1, 2, 3, 4, 5, 6, 7)
p_m2l_instance_values = (1, 2, 3, 4, 5, 6, 7)
p_l2l_instance_values = (1, 2, 3, 4, 5, 6, 7)
```

An additional filter currently restricts expensive two-dimensional FMM
instances to `p <= 5`, while three-dimensional instances extend through
`p = 7`. These ranges are an explicit engineering tradeoff rather than a
fundamental limitation of the formulas.

A developer may want to adapt this matrix in either direction. For example:

- Higher-dimensional FMM instances require extending `dimensions` and the
  remaining dimension-specific compile-time multipole-indexing helpers in
  `multipoles.cuh`. The general formulation is intended to work in higher
  dimensions, but FMM dimensions above three have not been tested.
- During development, removing `p = 6` and `p = 7`, omitting two-dimensional
  instances, or compiling only `float` or only `double` can substantially
  reduce compilation time.
- `float_types_direct` separately controls the formats instantiated by direct
  and leaf-to-leaf pair summation, while `direct_summation_dimensions` controls
  its supported dimensions.

Reducing the matrix also reduces what the resulting extension can execute. If
a requested dimension, order, dtype, kernel ID, or opening-criterion ID has no
compiled instance, the package must be regenerated and rebuilt with that
combination enabled.

The Python call in `jzfmm.fmm` should closely match the handwritten CUDA kernel
signature: input buffers become positional arguments, while static scalar and
template parameters are passed as keyword attributes. The generated host code
performs the repetitive type dispatch and JAX FFI registration boilerplate.

When adding an entirely new CUDA entry point rather than a template instance,
the usual sequence is:

1. Write and test the CUDA kernel in a `.cuh` file.
2. Select it in `_generate_ffi.py` and define its launch configuration,
   template instances, and inferred parameters.
3. Regenerate the `.cu` binding.
4. Import its extension module and register the FFI target in Python.
5. Define output `ShapeDtypeStruct` objects and call it with `jax.ffi.ffi_call`.
6. Add JIT, autodiff, and multi-device tests appropriate to the operation.

Input/output aliasing and buffer donation can improve memory use, but should
only be introduced after correctness is established. Any custom derivative
rule must be tested independently of the forward calculation.

## Unit tests and benchmarks

Always run tests from the `checks` directory. The `--quick` option skips slow
tests and reduces selected parametrized tests to a representative case:

```bash
cd checks
pytest --quick
```

Focused runs are usually faster while developing:

```bash
pytest tests/test_fmm.py --quick
pytest tests/test_gradients.py --quick
pytest tests/test_sim.py --quick
pytest tests/test_external_potential.py --quick
```

The main test modules cover:

- `test_fmm.py`: force accuracy, dimensions, kernels, padding, result modes,
  reproducibility, and numerical scale;
- `test_gradients.py`: FMM primitives, complete forces, and simulation
  gradients;
- `test_sim.py`: integration and reversibility;
- `test_external_potential.py`: external fields; and
- `test_distr_fmm.py` and `test_distr_sim.py`: distributed execution.

Distributed tests require an appropriate multi-GPU or multi-process launch.
They should not be interpreted as single-GPU failures when the required device
setup is unavailable.

Benchmarks also use pytest and
[pytest-jax-bench](https://github.com/jstuecker/pytest-jax-bench):

```bash
pytest benchmarks/bench_fmm.py --quick
pytest benchmarks/bench_sim.py --quick
```

Benchmark results and plots are written below `checks/benchmarks/.results`.
Keep compilation time separate from steady-state execution time, perform warmup
calls, and synchronize device results before reporting timings.

The scripts under `accuracy_checks` answer broader numerical questions and may
be substantially more expensive than unit tests. For example:

```bash
python accuracy_checks/check_forces.py
python accuracy_checks/opening_performance.py
```

Profiling scripts under `checks/profiling` separate FMM stages and inspect the
register and local-memory use of compiled CUDA kernels. They are particularly
useful after changing derivative recurrences, multipole orders, or CUDA launch
parameters.

## Building the documentation

Build the documentation locally with:

```bash
cd docs
make html
```

The generated HTML is written to `docs/_build/html`.

## Adding new interaction kernels

At its core, the FMM efficiently evaluates the convolution of a weighted point
distribution with a radial interaction kernel. For particles at positions
{math}`x_j` with weights {math}`m_j`, this has the form

```{math}
\Phi(x) = \sum_j m_j K\!\left(\lVert x-x_j \rVert\right).
```

The gravitational potential is one important example, but the same machinery
can be useful for other applications involving radial interactions between
points. Forces are obtained from spatial derivatives of the resulting field.

Adding a kernel requires coordinated changes to its Python configuration, CUDA
implementation, and generated template matrix. Interaction kernels are defined
jointly by `jzfmm.config.KernelConfig` and
`src/jzfmm_cuda/radial_kernels.cuh`. The same implementation is then used by
direct pair summation, leaf-to-leaf interactions, and multipole evaluation.

### 1. Add the Python configuration

Create a dataclass derived from `KernelConfig`. Its `kind_id()` selects the
compiled CUDA specialization, while `params(dtype)` creates the
one-dimensional runtime parameter array consumed by that implementation:

```python
@dataclass(unsafe_hash=True, slots=True)
class MyKernel(KernelConfig):
    scale: float = 1.0

    def kind_id(self) -> int:
        return 3

    def params(self, dtype: jax.typing.DTypeLike = jnp.float32) -> jax.Array:
        return jnp.asarray([self.scale], dtype=dtype)
```

The configuration must remain hashable because it is passed as a static
argument to JAX. The ID must agree with the corresponding constant in the CUDA
header and must be included in the generated template matrix. A kernel may
have multiple runtime parameters: `params()` packs them into the array in a
fixed order, and the CUDA `make_params()` function must unpack them in the same
order. The example uses a single parameter only for simplicity.

### 2. Implement the CUDA radial derivatives

Add a matching constant and `RadialKernel` specialization to
`radial_kernels.cuh`:

```cpp
static constexpr int RADIAL_KERNEL_MY_KERNEL = 3;

template<>
struct RadialKernel<RADIAL_KERNEL_MY_KERNEL> {
    template<typename tvec>
    struct Params {
        tvec scale;
    };

    template<typename tvec>
    __device__ __forceinline__ static Params<tvec>
    make_params(const tvec* params) {
        return Params<tvec>{params[0]};
    }

    template<int p, typename tvec>
    __device__ __forceinline__ static void r2_derivative_coeffs(
        tvec scaled_r2,
        Params<tvec> params,
        Vec<p + 1, tvec>& coeffs,
        int scale_exp = 0
    ) {
        // Fill coeffs[0] ... coeffs[p].
    }
};
```

The derivative convention is important. With

```text
s = r² / R²,  R = 2^scale_exp,
```

the routine must return

```text
coeffs[n] = 2ⁿ dⁿ/dsⁿ K(R sqrt(s)).
```

The scaling strategy keeps intermediate values numerically safe over a large
dynamic range. Follow the existing Plummer, two-dimensional Plummer, and
softened-distance implementations carefully when handling `scale_exp`.

Finally, add the new specialization to the switch in
`evaluate_radial_kernel_derivatives`. The generic `radial_kernel_value`
function then also makes it available to pair summation.

### 3. Instantiate the generated bindings

Add the new ID to `radial_kernel_instance_values` in
`src/jzfmm_cuda/_generate_ffi.py`:

```python
radial_kernel_instance_values = (0, 1, 2, 3)
```

This instantiates the kernel for direct summation and the leaf-to-leaf forward
and backward kernels. Regenerate the bindings from the repository root:

```bash
python src/jzfmm_cuda/_generate_ffi.py
```

Then rebuild the editable installation. Setting `JZFMM_GENERATE=1` makes CMake
run the generator during configuration as well:

```bash
JZFMM_GENERATE=1 uv pip install -e . --no-build-isolation
```

Using `CUDAARCHS=native` explicitly is useful if it is not already the default
for your build:

```bash
CUDAARCHS=native JZFMM_GENERATE=1 uv pip install -e . --no-build-isolation
```

### 4. Validate the kernel

A new kernel should be tested at several levels:

- Compare direct summation with an independent analytic or JAX reference.
- Compare the FMM result with direct summation in every supported dimension.
- Test both potential and force, not only one of them.
- Test reverse-mode gradients with finite differences or
  `jax.test_util.check_grads`.
- Test multiple floating-point formats and physical scales where applicable.
- Benchmark both direct and FMM evaluations. A mathematically simple recurrence
  is not necessarily cheap once register pressure and higher multipole orders
  are considered.

`checks/tests/test_fmm.py::test_other_kernels` and
`checks/tests/test_gradients.py::test_force_gradients` are useful starting
points. Larger accuracy sweeps belong under `checks/accuracy_checks`.
