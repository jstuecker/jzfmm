# Wheel packages

The root `pyproject.toml` remains the source-build entry point, including compilation of the CUDA backend. The projects here separate the Python package from the CUDA-specific wheels:

- `jzfmm`: `jzfmm` and `jzfmm_utils`, depending on `jztree>=1.1.0`.
- `jzfmm-cu12`: compiled `jzfmm_cuda` modules, depending on `jztree-cu12>=1.1.0`.
- `jzfmm-cu13`: compiled `jzfmm_cuda` modules, depending on `jztree-cu13>=1.1.0`.

The main wheel's `cuda12` and `cuda13` extras select the matching backend at exactly the same jz-fmm version. Install only one CUDA variant in an environment: both variants provide `jzfmm_cuda`.

These are wheel-build projects and use sources from the repository root. Use the root project for source distributions. Keep package versions, descriptions and license copies synchronized when preparing releases.

## Building

Run from the repository root with Docker available. Start with the main wheel and one CUDA backend:

```bash
BUILD_JOBS=4 ./packaging/run-wheel-builds.sh main
BUILD_PYTHONS=3.12 BUILD_JOBS=4 ./packaging/run-wheel-builds.sh cu13
```

Then build the remaining matrix with:

```bash
BUILD_JOBS=4 ./packaging/run-wheel-builds.sh all
```

This produces one Python wheel, CUDA 13 wheels for Python 3.11–3.14, and CUDA 12 wheels for Python 3.11–3.13. All CUDA architectures supported by the compiler are included. `BUILD_JOBS` sets compiler parallelism; reduce it if memory is tight. No GPU is required for building. Runtime verification is a separate step.

The builders pin JAX/jaxlib and CUDA plugin/PJRT packages to 0.8.3, scikit-build-core to 0.12.2, nanobind to 2.9.2, and CMake to 4.3.1. CUDA 12 uses a conda-provided 12.9 toolkit; CUDA 13 uses PyPI CUDA packages. Both builders use manylinux_2_28. Full resolved dependencies are recorded in the cached environments; base images and transitive dependencies are not fully pinned. Build dependencies do not require installing jz-tree or jz-fmm.

## Resuming and locating artifacts

After interruption, rerun the same command. Completed wheels are checksum-verified and skipped; compiled object files and completed raw wheels are retained. Keep `packaging/output/` and `packaging/.build-cache/` intact. Do not change build inputs while a build is running. Committing unchanged files does not invalidate the cache, but editing sources or packaging files does. Run only one builder per checkout.

Logs are written under `packaging/output/<mode>/logs/`, with `BUILD`, `DONE`, and `SKIP` progress messages. Each successful invocation writes `packaging/output/<mode>/latest.json` with the selected artifact paths and SHA-256 checksums. A subset run lists only that subset. Use only validated manifests when preparing an upload, not a broad glob over old output directories. The builder rejects wheels missing the project license.

Docker images and build environments are reused. `REBUILD_IMAGE=1` rebuilds the image; changing its ID or `BUILD_JAX_VERSION` selects a fresh environment. Source changes select new build directories. The containers run as the host UID with explicit writable package-cache and conda-registry locations.

Run CPU-only builder tests from `checks`:

```bash
python -m unittest discover -s ../packaging/tests -v
python test_packaging_metadata.py
```

Nothing in these scripts commits, tags, installs the resulting wheels, or uploads them. Before publication, check metadata and licenses, run `twine check --strict`, and test the wheels with the corresponding published jz-tree packages on the intended runtimes.
