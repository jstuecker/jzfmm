# JZ-FMM
Fast multipoles implemented with JAX and CUDA.

# Installation

## Dependencies

Install the `jz-tree` repository first.

## Prerequisites

Use a virtual Python or conda environment for installation.

Which CUDA version you can use depends on the GPU drivers that are installed on your system. If you install the newest CUDA libraries via pip/conda they will not always support your possibly outdated driver version. Therefore, as a first step it is important to get aware of the maximal CUDA version that you can install. For this check your driver version with
```
nvidia-smi
```
If you have the option, it is best to install the newest driver version your GPU supports. You can then check [here](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html) (in Table 3) what is the maximum CUDA version you can use. If you install (directly or indirectly) a newer version, you will get unpredictable errors when running jax.

## Installation with CUDA 13 wheels

If your driver supports the latest CUDA version (note this will not be the case on many GPU clusters), the simplest way to install is:
```bash
uv pip install -e ".[cuda13,dev]" --no-build-isolation
```
Double-check which CUDA versions were installed with `uv pip list` and look for a line like:
```bash
[...]
nvidia-cuda-crt     13.1.80
[...]
```
E.g. this version of CUDA will only work if the driver version is ">=590.44.01". You can force specific CUDA versions by manually defining versions for all the packages, e.g.
```
uv pip install "nvidia-cuda-crt==13.0.88" "nvidia-cuda-cupti==13.0.85" "nvidia-cuda-nvcc==13.0.88" # ... and so on
```
Installing with a pip-installed CUDA works only with CUDA >= 13 because NVIDIA's pip packages include `nvcc` starting with CUDA 13. To install CUDA 12, use conda or local CUDA libraries.

## Conda (miniforge) Installation (Recommended for CUDA12)
Install miniforge (or another conda distribution) / setup an environment / activate it (skip steps as appropriate)
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-Linux-x86_64.sh -b
rm Miniforge3-Linux-x86_64.sh
eval "$(~/miniforge3/bin/conda shell.bash hook)" # Or wherever you installed miniforge
conda init
conda create --name cu12
conda activate cu12
```
Install prerequisites via conda and pip:
```bash
export CONDA_OVERRIDE_CUDA="12.9"  # Use a version that fits your needs
conda install pip
conda install -c conda-forge cuda-nvcc cuda-version=12 cudnn nccl libcufft cuda-cupti libcublas libcusparse
uv pip install scikit-build-core nanobind "cmake==3.24"
uv pip install --upgrade "jax[cuda12-local]"
```
Verify that the installation fits your drivers
```
conda list
uv pip list
```
Finally install the code with
```
uv pip install -e . --no-build-isolation
```
Add the `[dev]` optional dependencies if you would like to run tests or use the optional development features.
