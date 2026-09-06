"""Run the multi-GPU guide, including distributed NumPy snapshots.

Multi-process Slurm job:
    srun --cpus-per-task=8 python multi_gpu_simulation.py
One process controlling all local GPUs:
    python multi_gpu_simulation.py --single-host

When Slurm exposes only one GPU per process (--gpus-per-task=1), also pass
--local-device-id=0. Otherwise leave JAX's device selection automatic.
"""

import argparse
import time

import jax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-host", action="store_true")
    parser.add_argument("--local-device-id", type=int)
    args = parser.parse_args()
    if not args.single_host:
        local_ids = None if args.local_device_id is None else [args.local_device_id]
        jax.distributed.initialize(local_device_ids=local_ids)

    start = time.perf_counter()
    import jax.numpy as jnp
    from jax.sharding import AxisType, PartitionSpec as P

    import jztree
    import jzfmm

    mesh = jax.sharding.Mesh(jax.devices(), ("gpus",), axis_types=(AxisType.Auto,))
    print(mesh)

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

    particles = make_gaussian_clumps.smap(mesh, jit=True)(
        npart=10_000_000,
        npad=2_000_000,
        nclumps=20,
        seed=7,
    )

    print(particles.pos.shape)
    print(particles.num_total)

    cfg = jzfmm.SimConfig(units=jzfmm.UnitConfig(mass_in_msol=1.0e12))
    loc = jzfmm.fmm.fast_multipole_method.smap(mesh, jit=True)(
        particles,
        cfg_fmm=cfg.force,
        G=cfg.units.G(),
    )
    loc.values.block_until_ready()

    print(loc.values.shape)

    ts = jnp.linspace(0.0, 0.2, 21)
    particles = jzfmm.time_integration.simulate.smap(mesh, jit=True)(particles, ts=ts, cfg=cfg)
    particles.pos.block_until_ready()

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

    particles = simulate_and_write.smap(mesh, jit=True)(
        particles, tstart=0.2, tend=1.2, nout=5, steps_per_output=100, cfg=cfg,
    )
    particles.pos.block_until_ready()
    jax.effects_barrier()

    print(
        f"[process {jax.process_index()}] Finished in {time.perf_counter() - start:.1f} s "
        "(including initial conditions, compilation, and output).",
        flush=True,
    )
    if not args.single_host:
        from jax.experimental import multihost_utils
        multihost_utils.sync_global_devices("simulation_finished")
        jax.distributed.shutdown()


if __name__ == "__main__":
    main()
