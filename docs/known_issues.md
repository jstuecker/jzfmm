# Known issues

JAX is evolving rapidly, and newer versions that we have not yet tested may
occasionally introduce compatibility problems. We aim to provide fixes promptly
when these arise. Many compute clusters also use older NVIDIA drivers, adding another layer of compatibility challenges with newer JAX and CUDA versions. Some issues originate in JAX, CUDA, or drivers and are outside
our direct control; known cases and available workarounds are listed below.
See the [JAX changelog](https://docs.jax.dev/en/latest/changelog.html) for upstream
release notes.

(jax-0-11-distributed)=
## JAX 0.11.1: multi-GPU failures on an older CUDA driver stack

Tested on 6 September 2026 with **JAX 0.11.1, CUDA 12.9, NVIDIA driver 535.274.02, NCCL 2.29.3, and four A100 GPUs**. Other configurations may behave differently; these settings are not required for every installation.

### Symptoms

You may see either error:

```text
Failed to add memset node to a CUDA graph:
CUDA_ERROR_INVALID_VALUE
```

```text
RaggedAllToAll fallback to NCCL is not allowed
```

After enabling the NCCL fallback, larger distributed calculations may instead hang during particle redistribution or tree construction, even when small communication tests pass. A hang alone is not sufficient to identify this issue.

### Validated workaround

Set these variables before starting Python on every process, preferably in the cluster job script:

```bash
export XLA_FLAGS="--xla_gpu_enable_command_buffer= --xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true"
export NCCL_CUMEM_ENABLE=0
```

Preserve any other required settings already present in `XLA_FLAGS`. The three options address separate problems:

- `--xla_gpu_enable_command_buffer=` disables command buffers to avoid the [CUDA graph error](#cuda-graph-r535).
- `--xla_gpu_allow_ragged_all_to_all_nccl_send_recv_fallback=true` allows XLA to implement variable-sized GPU-to-GPU exchanges using NCCL sends and receives when other implementations are unavailable.
- `NCCL_CUMEM_ENABLE=0` disables NCCL's `cuMem*` memory-allocation path, avoiding the large-exchange stall on this setup. See [NVIDIA's description of this option](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-cumem-enable).

JIT compilation and GPU peer-to-peer communication remain enabled; these options do not change the numerical algorithm. The compact four-GPU FMM, gradient, simulation, kNN, and FoF consistency suite passed. The largest measured slowdown was approximately **1.5%** versus paired JAX 0.8.2 controls.

### Scope

The graph failure was reproduced with standalone CUDA code, and the communication stall with pure JAX, without jz-tree or jz-fmm. No library-kernel changes were required. The precise change triggering the stall has not been identified, and no particular newer driver release has yet been verified to eliminate both problems.

For JAX 0.10.2, disabling command buffers alone passed the selected suite on this cluster, as described below. The full recipe above was validated for 0.11.1.

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

[NetKet issue #2248](https://github.com/netket/netket/issues/2248) reports the same
error and command-buffer workaround with JAX 0.10.2 on H100 GPUs. This is a
related report, not confirmation of the driver-specific cause identified here.

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

See JAX's [XLA flag instructions](https://docs.jax.dev/en/latest/xla_flags.html)
for setting `XLA_FLAGS`, and its
[GPU performance tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html)
for documentation of the command-buffer flag.

### Scope

JAX 0.8.3 and 0.9.2 passed the selected checks with default settings on this
cluster. The tested single-GPU JAX 0.10.2 generator also passed, but this is not
a guarantee for all single-GPU workloads. No newer driver release has yet been
verified to resolve the issue.

For JAX **0.11.1**, use the [complete validated workaround above](#jax-0-11-distributed), which also addresses the ragged all-to-all rejection and communication stall on this cluster.
