# Multi-GPU guide

This guide shows how to evaluate forces and run a **jz-fmm** simulation across
multiple GPUs. We first use four GPUs controlled by one Python process, which
is convenient for interactive development. The same calculation is then
adapted to multiple hosts, and finally extended with distributed snapshot
output.

The complete example is available as a
{download}`Python script <../examples/multi_gpu_simulation.py>`.

## A brief `shard_map` recap

The distributed implementation runs inside a JAX
[`shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html). Code in
the mapped function sees the arrays stored on one GPU and can explicitly
communicate with the other shards. This local view is useful for particle
distributions because every device can have a different number of valid
particles in fixed-size padded arrays.

**jz-fmm** uses the `shard_map` helpers provided by
[jz-tree](https://jstuecker.github.io/jztree/multi_gpu_guide.html). That guide
contains a more detailed introduction to the programming model. The
{ref}`.smap helper <smap-helper>` constructs a mapped function with the expected
partitioning and static arguments.

## Single-host multi-GPU execution

In the single-host case, one Python process controls all local GPUs. Start by
creating a one-dimensional mesh containing every visible device:

```python
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, PartitionSpec as P

import jztree
import jzfmm

mesh = jax.sharding.Mesh(jax.devices(), ("gpus",), axis_types=(AxisType.Auto,))
print(mesh)
```

Example output on four GPUs:

```text
Mesh('gpus': 4, axis_types=(Auto,))
```

Here, `P("gpus")` would partition an ordinary array along the named mesh axis.
jz-tree additionally accepts `P(-1)`, meaning that particle arrays are split
over every axis of the supplied mesh. This keeps the code independent of the
mesh-axis names and also works for meshes with more than one axis.

### Create distributed particle data

For simplicity, we generate 20 Gaussian clumps per GPU, with random centers
throughout the same volume and a different random key on each device.

```python
def make_gaussian_clumps(npart, npad, nclumps=20, seed=0):
    rank, ndev, _ = jztree.comm.get_rank_info()
    key = jax.random.fold_in(jax.random.key(seed), rank)
    key_center, key_bulk, key_label, key_pos, key_vel = jax.random.split(key, 5)

    centers = 30.0 * jax.random.normal(key_center, (nclumps, 3))
    bulk_velocities = -2.0 * centers + 20.0 * jax.random.normal(key_bulk, (nclumps, 3))
    labels = jax.random.randint(key_label, (npart,), 0, nclumps)
    pos = centers[labels] + jax.random.normal(key_pos, (npart, 3))
    vel = bulk_velocities[labels] + 20.0 * jax.random.normal(key_vel, (npart, 3))

    particles = jzfmm.data.Particles(
        pos=pos,
        vel=vel,
        mass=jnp.full(npart, 1.0 / (npart * ndev), dtype=pos.dtype),
        num=jnp.asarray(npart, dtype=jnp.int32),
        num_total=npart * ndev,
    )
    return jztree.data.pad_particles(particles, npad)


make_gaussian_clumps.smap = jztree.jax_ext.shard_map_constructor(
    make_gaussian_clumps,
    in_specs=(None, None, None, None),
    out_specs=P(-1),
    static_argnames=("npart", "npad", "nclumps", "seed"),
)
```

`num` is the number of valid particles in the local padded array and may change at various points of the simulation, while
`num_total` is the total number of valid particles on all devices and considered static. Both are
required for distributed particle data. Padding provides temporary capacity
when tree construction redistributes particles between devices.

```python
particles = make_gaussian_clumps.smap(mesh, jit=True)(
    npart=10_000_000,
    npad=2_000_000,
    nclumps=20,
    seed=7,
)

print(particles.pos.shape)
print(particles.num)
print(particles.num_total)
```

On four GPUs, the arrays outside the mapped function have an additional
leading device axis:

```text
(4, 12000000, 3)
[10000000 10000000 10000000 10000000]
40000000
```

Invalid entries beyond `num` are padding and must not be interpreted as
particles. If redistribution exhausts this capacity, increase `npad`. This is
separate from the allocation factors in `FMMConfig`, which control internal
tree, interaction-list, and communication buffers.

### Evaluate distributed forces

`fast_multipole_method.smap` evaluates the global particle interaction while
returning one result shard per device. The FMM performs the required
communication internally.

```python
cfg = jzfmm.SimConfig(units=jzfmm.UnitConfig(mass_in_msol=1.0e12))
loc = jzfmm.fmm.fast_multipole_method.smap(mesh, jit=True)(
    particles,
    cfg_fmm=cfg.force,
    G=cfg.units.G(),
)
loc.values.block_until_ready()

print(loc.values.shape)
```

```text
(4, 12000000, 4)
```

The last axis contains the potential and its three spatial derivatives.

The local expansion has the same padded particle layout as the input. Its
potential and force values are globally correct: they include source particles
held by every device, not only particles in the local shard.

### Run the simulation

The simulation interface differs from the single-GPU case only in the use of
`.smap`:

```python
ts = jnp.linspace(0.0, 0.2, 21)
particles = jzfmm.time_integration.simulate.smap(mesh, jit=True)(particles, ts=ts, cfg=cfg)
particles.pos.block_until_ready()
```

Sharded code typically takes longer to compile than single-GPU code; expect
roughly 10 seconds for the first simulation call on the tested setup. Later calls with the same
particle shapes, number of time steps, and configuration reuse the compiled
program. `simulate` calls the distributed FMM at every step because it is
executing inside the mapped context.

Keeping the complete simulation inside `shard_map` is preferable to repeatedly
moving data between global and device-local views. Ordinary JAX operations can
be used inside the mapped function, while global reductions can be expressed
with collectives such as `jax.lax.psum`.

## Multi-host multi-GPU execution

In a multi-host run, several Python processes jointly control the global device
mesh. Call `jax.distributed.initialize()` before querying devices or performing
any other JAX computation:

```python
import jax

jax.distributed.initialize()

mesh = jax.sharding.Mesh(
    jax.devices(),
    ("gpus",),
    axis_types=(jax.sharding.AxisType.Auto,),
)
```

The remainder of the Python program is unchanged. `jax.devices()` describes
the global device set after distributed initialization, and every process must
execute the same mapped operations in the same order.

If you encounter an error, check the [known issues](known_issues.md) page for possible workarounds.

Launch details depend on the cluster. A common arrangement is one process per
GPU. **Assign several CPU cores to each process:** communication can be very
slow with too few CPU cores. For example, request eight CPUs per task in the
batch allocation and launch with:

```bash
srun --cpus-per-task=8 python multi_gpu_simulation.py
```

The batch allocation must provide the matching numbers of tasks, GPUs, and CPU
cores. A single process controlling several GPUs is useful for interactive
development, but one process per GPU will often provide better communication
performance on a cluster.

See [jz-tree's example Slurm script](https://jstuecker.github.io/jztree/multi_gpu_guide.html#multi-host-execution-and-performance)
for a complete batch allocation example. With the downloaded example, run:

```bash
srun --cpus-per-task=8 python multi_gpu_simulation.py
```

For one Python process controlling all local GPUs, use
`python multi_gpu_simulation.py --single-host`. If your Slurm launch uses
`--gpus-per-task=1` and exposes only one GPU per process, pass
`--local-device-id=0` to the script.

## Writing distributed snapshots

Gathering a 40-million-particle snapshot onto one host would require additional
memory and communication. Instead, each device can write its valid local shard
through a host callback. The files belonging to one output time together form
the complete snapshot.

The callback below writes a `.npz` file with keys `pos` and `vel`.
Including both the snapshot index and global device rank prevents
different processes from writing to the same file.

```python
from pathlib import Path

import numpy as np
from jax.experimental import io_callback

output_dir = Path("snapshots")


def write_snapshot(snapshot, rank, pos, vel, num):
    output_dir.mkdir(parents=True, exist_ok=True)
    num = int(num)
    filename = output_dir / f"snapshot_{int(snapshot):04d}_rank_{int(rank):04d}.npz"
    np.savez(filename, pos=pos[:num], vel=vel[:num])


def simulate_and_write(particles, tstart, tend, nout, steps_per_output, cfg):
    rank, _, _ = jztree.comm.get_rank_info()

    def step(i, particles):
        t0 = tstart + i * (tend - tstart) / nout
        t1 = tstart + (i + 1) * (tend - tstart) / nout
        ts = jnp.linspace(t0, t1, steps_per_output + 1)
        particles = jzfmm.time_integration.simulate(particles, ts=ts, cfg=cfg)
        io_callback(
            write_snapshot, None, i + 1, rank,
            particles.pos, particles.vel, particles.num,
        )
        return particles

    return jax.lax.fori_loop(0, nout, step, particles)


simulate_and_write.smap = jztree.jax_ext.shard_map_constructor(
    simulate_and_write,
    in_specs=(P(-1), None, None, None, None, None),
    out_specs=P(-1),
    static_argnames=("nout", "steps_per_output", "cfg"),
)
```

Continue from the previous simulation's final time, writing five snapshots
spaced by 100 integration steps.
The entire loop runs inside the jitted `shard_map`:

```python
particles = simulate_and_write.smap(mesh, jit=True)(
    particles, tstart=0.2, tend=1.2, nout=5, steps_per_output=100, cfg=cfg,
)
particles.pos.block_until_ready()
jax.effects_barrier()
```

On four NVIDIA A100 GPUs, expect roughly 80 seconds for this example, including
compilation and snapshot output. This estimate is based on the measured
interval timings; runtime depends on the system and filesystem.

Only the final particle state is returned; snapshots are written after each
interval. `jax.effects_barrier()` waits for outstanding callbacks before the
program exits. The resulting files can be loaded with
`np.load`; for example, snapshot 4 on rank 2 is stored in
`snapshot_0004_rank_0002.npz`.

For a serious large simulation, use a format supporting compression, such as
HDF5, and include additional data such as masses, particle IDs, snapshot time,
units, and simulation parameters.
