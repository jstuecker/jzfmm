# Local CUDA build used for this experiment

The new `ffi_fmm` binary was built from commit
`d6b6eea` with CUDA 12.9.86 and GCC 14.3.1, targeting the GTX 1070's sm_61.
Both opening kinds are in the same extension. Unchanged multipole and pair
summation extensions come from the installed jzfmm 1.0.1 CUDA12 package.
`build_metadata.json` records extension hashes. No system compiler, CUDA
installation, or existing Python environment was modified.

## Isolated prerequisites

An environment was created with uv at
`/home/bender/Work/toolchains/jzfmm-build-env`, using Python 3.12.14. Installed
build packages: nanobind 3.0.1, cmake 4.4.3, ninja 1.13.2, jax-ffi-gen 0.6.1;
pytest 9.1.1 for tests. A `.pth` file adds the existing runtime's site-packages
at `/home/bender/.venvs/jzfmm/lib/python3.12/site-packages` (JAX/jaxlib 0.11.1,
jztree 1.1.0, jzfmm 1.0.1 and CUDA12 runtime libraries).

CUDA nvcc 12.9.86, cudart 12.9.79 and CCCL 12.9.27 archives were downloaded from
[NVIDIA's redistribution archive](https://developer.download.nvidia.com/compute/cuda/redist/)
and SHA256-checked against `redistrib_12.9.1.json`. Their directories are combined
through symlinks at `/home/bender/Work/toolchains/cuda12/toolkit`, including
`lib64 -> lib`.

The official [Arch GCC14 archive](https://archive.archlinux.org/packages/g/gcc14/)
package `gcc14-14.3.1+r416+g44d5743651c4-2-x86_64.pkg.tar.zst` was unpacked under
`/home/bender/Work/toolchains/gcc14`. A local `libgcc_s.so` symlink points to
`/usr/lib/libgcc_s.so.1` to support linking from the relocated compiler.

This machine's newer glibc declares six math functions noexcept, conflicting
with CUDA12's declarations. Only the local toolchain copy of
`include/crt/math_functions.h` was patched: `noexcept(true)` was added to the
six extern declarations of rsqrt, rsqrtf, sinpi, sinpif, cospi and cospif.
Original header backup and downloaded archives are retained. No mathematical
implementation or device optimization flags were changed by that compatibility
patch. GCC16/Clang22 on the host were not used for the CUDA build.

## Generate and compile

From the repository root, with the above local paths:

```bash
/home/bender/Work/toolchains/jzfmm-build-env/bin/python src/jzfmm_cuda/_generate_ffi.py

CUDAARCHS=61 /home/bender/Work/toolchains/jzfmm-build-env/bin/cmake \
  -S . -B build-softened -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSKBUILD_PROJECT_NAME=jzfmm \
  -DCMAKE_MAKE_PROGRAM=/home/bender/Work/toolchains/jzfmm-build-env/bin/ninja \
  -DPython_EXECUTABLE=/home/bender/Work/toolchains/jzfmm-build-env/bin/python \
  -DCMAKE_CUDA_COMPILER=/home/bender/Work/toolchains/cuda12/toolkit/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/home/bender/Work/toolchains/gcc14/usr/bin/g++-14 \
  -DCMAKE_CXX_COMPILER=/home/bender/Work/toolchains/gcc14/usr/bin/g++-14 \
  -Dnanobind_DIR=/home/bender/Work/toolchains/jzfmm-build-env/lib/python3.12/site-packages/nanobind/cmake

/home/bender/Work/toolchains/jzfmm-build-env/bin/cmake \
  --build build-softened --target ffi_fmm -j 2
```

This CMake/nanobind combination emitted a filename with a duplicated `.so`
suffix. The local benchmark backend directory contains a correctly named
`ffi_fmm.cpython-312-x86_64-linux-gnu.so` symlink to that build product, alongside
symlinks to the unchanged installed `ffi_multipoles` and `ffi_pair_summation`
extensions and a copy of `jzfmm_cuda/__init__.py` declaring CUDA_MAJOR=12.

The benchmark environment used:

```bash
export PYTHONPATH=/home/bender/Work/benchmarks/jzfmm-softened-2026-09-12/backend:/home/bender/Work/repos/jzfmm-fork/src
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
```

Run the commands in README.md with
`/home/bender/Work/toolchains/jzfmm-build-env/bin/python`. On a standard compatible
CUDA development system, building/installing the full package normally is also
sufficient; the local symlinks and header compatibility patch are specific to
this machine. CUDA profiling requires device/CUPTI access.
