# Known issues

(cuda-graph-r535)=
## JAX 0.10.2: CUDA graph failure with NVIDIA R535

On a cluster with A100 GPUs, NVIDIA driver **535.274.02**, and CUDA toolkit
**12.9.86**, distributed workloads with JAX **0.10.2** failed with:

```text
Failed to add memset node to a CUDA graph:
CUDA_ERROR_INVALID_VALUE: invalid argument
```

The failure was reproduced independently of jz-tree and jz-fmm, both in pure
JAX and in a standalone CUDA test involving graph memset on virtual-memory
allocations. This points to a driver/CUDA-graph compatibility issue, not our
custom kernels; it does **not** imply that all CUDA 12 installations are affected.

### Workaround

Set this before starting Python on each process:

```bash
export XLA_FLAGS="--xla_gpu_enable_command_buffer="
```

If `XLA_FLAGS` already contains required settings, retain them when adding
this option. It disables XLA command buffers, but retains JIT compilation and
GPU execution. With this workaround, selected four-GPU FMM, gradient,
simulation, kNN, and FoF consistency checks passed. Measured execution times
were approximately 1–2% slower than the older working JAX baseline.

### Scope

JAX 0.8.3 and 0.9.2 passed the selected checks with default settings on this
cluster. The tested single-GPU JAX 0.10.2 generator also passed, but this is not
a guarantee for all single-GPU workloads. No newer driver release has yet been
verified to resolve the issue.

JAX **0.11.1** encountered a separate distributed `RaggedAllToAll` problem:
the command-buffer workaround alone is **not a validated solution for 0.11.1**.
