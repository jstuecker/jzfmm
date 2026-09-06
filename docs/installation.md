# Installation

## Requirements

Right now **jz-fmm** is only supported on NVIDIA GPUs with CUDA 12 or CUDA 13
compatibility. The installed NVIDIA driver must support the selected CUDA
version.

Future updates may include CPU and AMD GPU support.

## Via pip

The easiest way to install **jz-fmm** is from a pre-built wheel. Select the
extra matching the CUDA version used by JAX. For CUDA 13, use

```bash
pip install "jzfmm[cuda13]"
```

and for CUDA 12, use

```bash
pip install "jzfmm[cuda12]"
```

The supported Python versions will follow the available JAX and binary-wheel
versions. Building from source may work outside the published wheel range, but
JAX itself also supports only a limited range of Python and CUDA versions.

## Build from source

Clone the repository and enter its root directory:

```bash
git clone https://github.com/jstuecker/jzfmm.git
cd jzfmm
```

**jz-fmm** builds on [jz-tree](https://jstuecker.github.io/jztree/), which must
be installed for the same CUDA version. A source build additionally requires a
C++ compiler, CMake 3.24 or newer, and the CUDA compiler `nvcc`.

Check the installed GPU and driver with

```bash
nvidia-smi
```

The build targets the locally detected GPU by default using
`CUDAARCHS=native`. To target a particular compute capability explicitly, set
`CUDAARCHS` before building. For example, an NVIDIA A100 has compute capability
8.0:

```bash
export CUDAARCHS=80
```

`CUDAARCHS=all` builds for all supported architectures. This can substantially
increase compilation time and the size of the compiled module.

### CUDA 13

CUDA 13 can be installed entirely through Python packages because NVIDIA's
CUDA 13 packages include `nvcc`. In an activated virtual environment, install
the dependencies and then **jz-fmm**:

```bash
pip install "jztree[cuda13]"
pip install "jax[cuda13]" "scikit-build-core>=0.11" "nanobind>=2.9.2" "cmake>=3.24"
pip install -e ".[cuda13]" --no-build-isolation
```

```{note}
Keep `--no-build-isolation` for editable installations so the build can locate
the CUDA and JAX packages in the active environment. With this option, build
dependencies must already be installed; the prerequisite commands above install
them explicitly.
```

### CUDA 12

The CUDA 12 packages distributed through PyPI for Linux do not include the
`nvcc` compiler driver. A CUDA 12 source build therefore needs either a
conda-provided or system-provided CUDA toolkit.

For a self-contained conda environment, install the CUDA compiler and runtime
libraries from conda-forge:

```bash
conda create --name jzfmm-cu12 python=3.12
conda activate jzfmm-cu12
conda install -c conda-forge pip cuda-nvcc cuda-version=12 cudnn nccl \
    libcufft cuda-cupti libcublas libcusparse
pip install --upgrade "jax[cuda12-local]"
pip install "scikit-build-core>=0.11" "nanobind>=2.9.2" "cmake>=3.24"
pip install "jztree[cuda12]"
pip install -e . --no-build-isolation
```

Choose a CUDA 12 minor version compatible with the installed NVIDIA driver.

### Existing system CUDA installation

If a compatible CUDA toolkit is already installed or provided by a cluster
module, first verify that its compiler is available:

```bash
nvcc --version
```

Then install JAX for the local toolkit and build **jz-fmm**:

```bash
pip install --upgrade "jax[cuda13-local]"  # or jax[cuda12-local]
pip install "scikit-build-core>=0.11" "nanobind>=2.9.2" "cmake>=3.24"
pip install -e . --no-build-isolation
```

Avoid mixing a system CUDA toolkit with incompatible CUDA packages installed in
the same Python environment.

## Verify the installation

Import **jz-fmm** and ask JAX which accelerators it detects:

```bash
python -c "import jax, jzfmm; print(jax.devices())"
```

The output should list at least one CUDA device.
