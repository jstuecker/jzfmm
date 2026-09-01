import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import argparse
import functools
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import NamedTuple

import aegis
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jztree.config import LoggingConfig, TreeConfig

import jzfmm
from aegis.profiles.analytical import RvirOfMvir


POSITION_UNIT = 1e3  # kpc per Mpc
VELOCITY_UNIT = 1e3  # km/s
MASS_UNIT = 1e14
LOSS_SOFTENING = 4.0
SAMPLING_RADIUS_MAX = 10.0
LOG10_MASS_MIN = 9.5
LOG10_MASS_MAX = 11.5
TARGET_LOG10_MASS_MIN = 10.0
TARGET_LOG10_MASS_MAX = 11.0
CONCENTRATION_MIN = 1.0
CONCENTRATION_MAX = 20.0
FIXED_CONCENTRATION = 8.0
TARGET_CONCENTRATION_MIN = 6.0
TARGET_CONCENTRATION_MAX = 10.0
LOG10_CONCENTRATION_MIN = np.log10(CONCENTRATION_MIN)
LOG10_CONCENTRATION_MAX = np.log10(CONCENTRATION_MAX)
R200_MASS_FACTOR = RvirOfMvir(1.0)
MAX_COM_POSITION = 5.0  # Mpc
MAX_COM_VELOCITY = 5.0  # 1000 km/s
MASS_WEIGHT = 10.0
EARLY_STOP_RTOL = 1e-3
EARLY_STOP_LOSS_RATIO = 100.0
EARLY_STOP_IMPROVEMENT_FACTOR = 2.0
GYR_TO_SIMULATION_TIME = 1.022712165
HOST_MASS = 1e15
HOST_SCALE_RADIUS = 351.2  # kpc
HOST_VIRIAL_RADIUS = RvirOfMvir(HOST_MASS) * POSITION_UNIT
ORBIT_PERICENTRE_MIN = 0.1 * HOST_VIRIAL_RADIUS
ORBIT_APOCENTRE_MAX = HOST_VIRIAL_RADIUS
LATTICE_POSITION_LIMIT = 20_000.0  # kpc
LATTICE_VELOCITY_LIMIT = 10_000.0  # km/s
LATTICE_DX = LATTICE_POSITION_LIMIT / np.iinfo(np.int32).max
LATTICE_DV = LATTICE_VELOCITY_LIMIT / np.iinfo(np.int32).max
DEFAULT_GRAVITATIONAL_SOFTENING = 4.0  # kpc
FMM_ACCURACY_LEVELS = (
    (5, 0.8),
    (6, 0.7),
    (7, 0.6),
)
FMM_ALLOC_FAC_ILIST = 150
DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[0] / "logs"
)


def mass_from_exp(
    exp_m,
    log10_mass_min=LOG10_MASS_MIN,
    log10_mass_max=LOG10_MASS_MAX,
):
    log10_mass = (
        log10_mass_min
        + (log10_mass_max - log10_mass_min) * jax.nn.sigmoid(exp_m)
    )
    return 10.0**log10_mass
mass_from_exp.jit = jax.jit(mass_from_exp)


def concentration_from_log10(log10_concentration):
    return 10.0**log10_concentration
concentration_from_log10.jit = jax.jit(concentration_from_log10)


def radius_from_mass_concentration(
    exp_m,
    log10_concentration,
    log10_mass_min=LOG10_MASS_MIN,
    log10_mass_max=LOG10_MASS_MAX,
):
    halo_mass = mass_from_exp(exp_m, log10_mass_min, log10_mass_max)
    r200 = R200_MASS_FACTOR * jnp.cbrt(halo_mass)
    return r200 / concentration_from_log10(log10_concentration)
radius_from_mass_concentration.jit = jax.jit(
    radius_from_mass_concentration
)


def duplication_loss(parameters, log10_mass_min, log10_mass_max):
    scale_radius = radius_from_mass_concentration.jit(
        parameters.exp_m,
        parameters.log10_concentration,
        log10_mass_min,
        log10_mass_max,
    )
    displacement = (
        parameters.position[:, None, :] - parameters.position[None, :, :]
    )
    count = parameters.position.shape[0]
    distance = jnp.sqrt(
        jnp.sum(displacement**2, axis=-1) + jnp.eye(count)
    )
    overlap = jax.nn.relu(
        1.0
        - distance
        / (scale_radius[:, None] + scale_radius[None, :])
    )
    pair_mask = jnp.triu(jnp.ones((count, count)), k=1)
    number_of_pairs = max(count * (count - 1) // 2, 1)
    return jnp.sum(pair_mask * overlap**2) / number_of_pairs


duplication_loss.jit = jax.jit(duplication_loss)


@jax.tree_util.register_dataclass
@dataclass
class Parameters:
    exp_m: jax.Array
    log10_concentration: jax.Array
    position: jax.Array
    velocity: jax.Array

    @classmethod
    def from_structured(cls, parameters):
        return cls(
            exp_m=jnp.asarray(parameters["exp_m"]),
            log10_concentration=jnp.asarray(
                parameters["log10_concentration"]
            ),
            position=jnp.asarray(parameters["position"]),
            velocity=jnp.asarray(parameters["velocity"]),
        )


class StepRecord(NamedTuple):
    step: int
    restart: int
    parameters: Parameters
    loss: float
    gradient_norm: float
    linesearch_evaluations: int
    total_evaluations: int
    elapsed_seconds: float
    step_seconds: float


def default_sim_config() -> jzfmm.SimConfig:
    cfg = jzfmm.SimConfig(
        force=jzfmm.FMMConfig(
            kernel=jzfmm.PlummerKernel(
                softening=DEFAULT_GRAVITATIONAL_SOFTENING
            ),
        ),
        external_potential=jzfmm.external_potential.HernquistPotential(
            a=HOST_SCALE_RADIUS,
            mass=HOST_MASS,
        ),
        integrator=jzfmm.DKDLatticeConfig(
            dx=LATTICE_DX,
            dv=LATTICE_DV,
            int_dtype=jnp.int32,
        ),
    )
    cfg.force.tree.alloc_fac_nodes = 1.8
    return cfg


def sim_config_from_dict(values) -> jzfmm.SimConfig:
    force_values = values["force"].copy()
    force_values["tree"] = force_values["tree"].copy()
    force_values["tree"] = TreeConfig(**force_values["tree"])
    force_values["kernel"] = jzfmm.PlummerKernel(
        **force_values["kernel"]
    )
    force_values["opening"] = jzfmm.OpeningByAngle(
        **force_values["opening"]
    )

    integrator_values = values["integrator"].copy()
    int_dtype = integrator_values.pop("int_dtype")
    integrator = jzfmm.DKDLatticeConfig(
        **integrator_values,
        int_dtype=getattr(jnp, int_dtype),
    )
    return jzfmm.SimConfig(
        force=jzfmm.FMMConfig(**force_values),
        units=jzfmm.UnitConfig(**values["units"]),
        logging=LoggingConfig(**values["logging"]),
        external_potential=jzfmm.external_potential.HernquistPotential(
            **values["external_potential"]
        ),
        integrator=integrator,
    )


@dataclass(frozen=True)
class Config:
    N: int = 100_000
    Nhaloes: int = 3
    mode: str = "sim"
    integration_time_gyr: float = 1.0
    integration_steps: int = 20
    sim_config: jzfmm.SimConfig = field(default_factory=default_sim_config)
    loss_mode: str = "distance"
    loss_softening: float = LOSS_SOFTENING
    log10_mass_min: float = LOG10_MASS_MIN
    log10_mass_max: float = LOG10_MASS_MAX
    target_log10_mass_min: float = TARGET_LOG10_MASS_MIN
    target_log10_mass_max: float = TARGET_LOG10_MASS_MAX
    deduplication_weight: float | None = None
    vary_concentration: bool = False
    fix_mass: bool = False
    optimizer: str = "lbfgs"
    adam_learning_rate: float = 0.01
    max_steps: int = 2000
    patience: int = 20
    early_stop: bool = False
    max_restarts: int = 0
    perturbation_scale: float = 0.05
    perturbation_seed: int = 0
    loss_rtol: float = 1e-5
    memory_size: int = 10
    max_linesearch_steps: int = 5
    print_every: int = 10

    target_parameter_seed: int = 0
    initial_parameter_seed: int = 42
    initial_parameters_file: str | None = None
    target_particle_seed: int = 37
    model_particle_seed: int = 0
    output_directory: str = str(DEFAULT_OUTPUT_DIRECTORY)
    output_prefix: str | None = None
    output_id: int | None = None
    output_id_width: int = 0

    @classmethod
    def from_json(cls, filename):
        with Path(filename).open() as file:
            values = json.load(file)
        values["sim_config"] = sim_config_from_dict(
            values["sim_config"]
        )
        return cls(**values)

    def to_json(self, filename):
        values = asdict(self)
        values["sim_config"]["integrator"]["int_dtype"] = np.dtype(
            self.sim_config.integrator.int_dtype
        ).name
        with Path(filename).open("w") as file:
            json.dump(values, file, indent=2)
            file.write("\n")


def sample_parameters(
    Nhaloes: int,
    seed: int,
    vary_concentration: bool = False,
    log10_mass_min: float = LOG10_MASS_MIN,
    log10_mass_max: float = LOG10_MASS_MAX,
    sample_log10_mass_min: float = TARGET_LOG10_MASS_MIN,
    sample_log10_mass_max: float = TARGET_LOG10_MASS_MAX,
    pericentre_min: float | None = None,
    apocentre_min: float | None = None,
    apocentre_max: float = MAX_COM_POSITION * POSITION_UNIT,
) -> Parameters:
    mass_seed, concentration_seed, phase_space_seed = (
        np.random.SeedSequence(seed).spawn(3)
    )
    mass_rng = np.random.default_rng(mass_seed)
    concentration_rng = np.random.default_rng(concentration_seed)
    log10_mass = mass_rng.uniform(
        sample_log10_mass_min,
        sample_log10_mass_max,
        Nhaloes,
    )
    if vary_concentration:
        log10_concentration = concentration_rng.uniform(
            np.log10(TARGET_CONCENTRATION_MIN),
            np.log10(TARGET_CONCENTRATION_MAX),
            Nhaloes,
        )
    else:
        log10_concentration = np.full(
            Nhaloes,
            np.log10(FIXED_CONCENTRATION),
        )
    concentration = 10.0**log10_concentration
    scale_radius = RvirOfMvir(10.0**log10_mass) / concentration

    np.random.seed(phase_space_seed.generate_state(1)[0])
    host = aegis.profiles.HernquistProfile(
        a=HOST_SCALE_RADIUS,
        M=HOST_MASS,
    )
    position = np.empty((Nhaloes, 3))
    velocity = np.empty((Nhaloes, 3))
    for i in range(Nhaloes):
        while True:
            candidate_position, candidate_velocity = host.sample_particles(
                1,
                result="pos_vel",
                rpmin=pericentre_min,
                ramin=apocentre_min,
                ramax=apocentre_max,
            )
            candidate_position = candidate_position[0] / POSITION_UNIT
            if i == 0 or np.all(
                np.linalg.norm(position[:i] - candidate_position, axis=1)
                > scale_radius[:i] + scale_radius[i]
            ):
                position[i] = candidate_position
                velocity[i] = candidate_velocity[0] / VELOCITY_UNIT
                break

    mass_fraction = (
        (log10_mass - log10_mass_min)
        / (log10_mass_max - log10_mass_min)
    )
    return Parameters(
        exp_m=jnp.asarray(np.log(mass_fraction / (1.0 - mass_fraction))),
        log10_concentration=jnp.asarray(log10_concentration),
        position=jnp.asarray(position),
        velocity=jnp.asarray(velocity),
    )


def structured_parameters(parameters: Parameters) -> np.ndarray:
    exp_m = np.asarray(parameters.exp_m)
    log10_concentration = np.asarray(parameters.log10_concentration)
    position = np.asarray(parameters.position)
    velocity = np.asarray(parameters.velocity)

    result = np.empty(
        exp_m.shape,
        dtype=[
            ("exp_m", exp_m.dtype),
            ("log10_concentration", log10_concentration.dtype),
            ("position", position.dtype, (3,)),
            ("velocity", velocity.dtype, (3,)),
        ],
    )
    result["exp_m"] = exp_m
    result["log10_concentration"] = log10_concentration
    result["position"] = position
    result["velocity"] = velocity
    return result


@functools.lru_cache
def sample_base_particles(N: int, seed: int):
    np.random.seed(seed)
    profile = aegis.profiles.HernquistProfile(a=1.0, M=1.0)
    pos, vel, mass = profile.sample_particles(
        N,
        result="pos_vel_m",
        ramax=SAMPLING_RADIUS_MAX,
    )
    return (
        np.asarray(pos),
        np.asarray(vel),
        np.asarray(mass / np.sum(mass)),
    )


def map_particles(
    parameters: Parameters,
    base_particles,
    config: Config,
    outputs: int | None = None,
):
    """Map parameters to particles, optionally yielding snapshots in Gyr."""
    pos, vel, mass = [], [], []
    scale_radius = radius_from_mass_concentration.jit(
        parameters.exp_m,
        parameters.log10_concentration,
        config.log10_mass_min,
        config.log10_mass_max,
    )
    halo_mass = mass_from_exp.jit(
        parameters.exp_m,
        config.log10_mass_min,
        config.log10_mass_max,
    )
    for i, (base_pos, base_vel, base_mass) in enumerate(base_particles):
        pos.append(
            jnp.asarray(base_pos) * scale_radius[i] * POSITION_UNIT
            + parameters.position[i] * POSITION_UNIT
        )
        vel.append(
            jnp.asarray(base_vel)
            * jnp.sqrt(
                halo_mass[i] / (scale_radius[i] * POSITION_UNIT)
            )
            + parameters.velocity[i] * VELOCITY_UNIT
        )
        mass.append(
            jnp.asarray(base_mass) * halo_mass[i]
        )

    particles = jzfmm.data.Particles(
        pos=jnp.concatenate(pos),
        vel=jnp.concatenate(vel),
        mass=jnp.concatenate(mass),
    )
    if config.mode == "ic":
        if outputs is not None:
            raise ValueError("outputs is only available in simulation mode")
        return particles
    if config.mode != "sim":
        raise ValueError(f"Unknown mode: {config.mode}")

    simulation_time = (
        config.integration_time_gyr * GYR_TO_SIMULATION_TIME
    )
    if outputs is None:
        times = np.linspace(
            0.0,
            simulation_time,
            config.integration_steps + 1,
            dtype=np.float32,
        )
        return jzfmm.time_integration.simulate(
            particles,
            ts=times,
            cfg=config.sim_config,
        )

    if outputs < 2:
        raise ValueError("outputs must include at least the initial and final states")
    number_of_intervals = outputs - 1
    if config.integration_steps % number_of_intervals != 0:
        raise ValueError(
            f"outputs={outputs} is incompatible with "
            f"integration_steps={config.integration_steps}; "
            "outputs - 1 must divide integration_steps"
        )
    simulation_outputs = jzfmm.time_integration.simulate_with_outputs(
        particles,
        tend=simulation_time,
        nout=number_of_intervals,
        steps_per_output=(
            config.integration_steps // number_of_intervals
        ),
        cfg=config.sim_config,
    )
    return (
        (time / GYR_TO_SIMULATION_TIME, output_particles)
        for time, output_particles in simulation_outputs
    )


map_particles.jit = jax.jit(
    lambda parameters, base_particles, config: map_particles(
        parameters,
        base_particles,
        config,
    ),
    static_argnames=("config",),
)


def get_particles(
    parameters: Parameters,
    config: Config,
    seed: int,
    outputs: int | None = None,
):
    """Generate base particles and map them, optionally at multiple times."""
    if not isinstance(parameters, Parameters):
        parameters = Parameters.from_structured(parameters)

    base_particles = [
        sample_base_particles(config.N, seed + i)
        for i in range(len(parameters.exp_m))
    ]
    if outputs is None:
        return map_particles.jit(parameters, base_particles, config)
    return map_particles(
        parameters,
        base_particles,
        config,
        outputs=outputs,
    )


def run(config: Config):
    run_start_time = time.perf_counter()
    target_parameters = sample_parameters(
        config.Nhaloes,
        config.target_parameter_seed,
        vary_concentration=config.vary_concentration,
        log10_mass_min=config.log10_mass_min,
        log10_mass_max=config.log10_mass_max,
        sample_log10_mass_min=config.target_log10_mass_min,
        sample_log10_mass_max=config.target_log10_mass_max,
        pericentre_min=ORBIT_PERICENTRE_MIN,
        apocentre_max=ORBIT_APOCENTRE_MAX,
    )
    if config.initial_parameters_file is None:
        initial_parameters = sample_parameters(
            config.Nhaloes,
            config.initial_parameter_seed,
            vary_concentration=config.vary_concentration,
            log10_mass_min=config.log10_mass_min,
            log10_mass_max=config.log10_mass_max,
            sample_log10_mass_min=config.target_log10_mass_min,
            sample_log10_mass_max=config.target_log10_mass_max,
            pericentre_min=ORBIT_PERICENTRE_MIN,
            apocentre_max=ORBIT_APOCENTRE_MAX,
        )
    else:
        with np.load(config.initial_parameters_file) as previous_run:
            previous_history = previous_run["history"]
            if len(previous_history) == 0:
                raise ValueError(
                    f"{config.initial_parameters_file} has an empty history"
                )
            best_step = np.argmin(previous_history["loss"])
            initial_parameters = Parameters.from_structured(
                previous_history["parameters"][best_step]
            )
        if len(initial_parameters.exp_m) != config.Nhaloes:
            raise ValueError(
                f"{config.initial_parameters_file} contains "
                f"{len(initial_parameters.exp_m)} haloes, but "
                f"Nhaloes={config.Nhaloes}"
            )
        print(
            f"Initialized parameters from "
            f"{config.initial_parameters_file}, step {best_step}"
        )
    if config.fix_mass or not config.vary_concentration:
        initial_parameters = replace(
            initial_parameters,
            exp_m=(
                target_parameters.exp_m
                if config.fix_mass
                else initial_parameters.exp_m
            ),
            log10_concentration=(
                target_parameters.log10_concentration
                if not config.vary_concentration
                else initial_parameters.log10_concentration
            ),
        )

    target_base = [
        sample_base_particles(config.N, config.target_particle_seed + i)
        for i in range(config.Nhaloes)
    ]
    model_base = [
        sample_base_particles(config.N, config.model_particle_seed + i)
        for i in range(config.Nhaloes)
    ]

    target = map_particles.jit(target_parameters, target_base, config)
    target = replace(target, mass=target.mass / MASS_UNIT)

    if config.loss_mode == "plummer":
        kernel = jzfmm.PlummerKernel(softening=config.loss_softening)
        normalize = False
    elif config.loss_mode == "distance":
        kernel = jzfmm.SoftenedDistanceKernel(
            softening=config.loss_softening
        )
        normalize = True
    else:
        raise ValueError(f"Unknown loss mode: {config.loss_mode}")

    fmm_config = jzfmm.config.FMMConfig(
        kernel=kernel,
        remove_self_interaction=False,
    )

    def loss(parameters):
        if config.fix_mass or not config.vary_concentration:
            parameters = replace(
                parameters,
                exp_m=(
                    jax.lax.stop_gradient(parameters.exp_m)
                    if config.fix_mass
                    else parameters.exp_m
                ),
                log10_concentration=(
                    jax.lax.stop_gradient(
                        parameters.log10_concentration
                    )
                    if not config.vary_concentration
                    else parameters.log10_concentration
                ),
            )

        valid = (
            jnp.all(jnp.isfinite(parameters.exp_m))
            & jnp.all(jnp.isfinite(parameters.log10_concentration))
            & jnp.all(
                parameters.log10_concentration
                >= LOG10_CONCENTRATION_MIN
            )
            & jnp.all(
                parameters.log10_concentration
                <= LOG10_CONCENTRATION_MAX
            )
            & jnp.all(jnp.isfinite(parameters.position))
            & jnp.all(jnp.isfinite(parameters.velocity))
            & jnp.all(jnp.abs(parameters.position) <= MAX_COM_POSITION)
            & jnp.all(jnp.abs(parameters.velocity) <= MAX_COM_VELOCITY)
        )

        parameters = replace(
            parameters,
            exp_m=jnp.nan_to_num(parameters.exp_m),
            log10_concentration=jnp.clip(
                jnp.nan_to_num(parameters.log10_concentration),
                LOG10_CONCENTRATION_MIN,
                LOG10_CONCENTRATION_MAX,
            ),
            position=jnp.clip(
                jnp.nan_to_num(parameters.position),
                -MAX_COM_POSITION,
                MAX_COM_POSITION,
            ),
            velocity=jnp.clip(
                jnp.nan_to_num(parameters.velocity),
                -MAX_COM_VELOCITY,
                MAX_COM_VELOCITY,
            ),
        )

        particles = map_particles(parameters, model_base, config)
        particles = replace(particles, mass=particles.mass / MASS_UNIT)
        mmd = jzfmm.loss.maximum_mean_discrepancy(
            part=particles,
            part_target=target,
            cfg_fmm=fmm_config,
            normalize=normalize,
        )
        if config.deduplication_weight is None:
            prior_loss = 0.0
        else:
            prior_loss = (
                config.deduplication_weight
                * duplication_loss.jit(
                    parameters,
                    config.log10_mass_min,
                    config.log10_mass_max,
                )
            )
        if config.loss_mode == "distance":
            model_mass = jnp.sum(particles.mass)
            target_mass = jnp.sum(target.mass)
            mass_loss = jnp.log10(model_mass / target_mass) ** 2
            value = (
                mmd / POSITION_UNIT
                + MASS_WEIGHT * mass_loss
                + prior_loss
            )
        else:
            value = mmd + prior_loss

        return jnp.where(
            valid,
            value,
            jnp.asarray(jnp.inf, dtype=value.dtype),
        )

    loss.jit = jax.jit(loss)
    optimal_loss = float(
        jax.block_until_ready(loss.jit(target_parameters))
    )
    print(f"Optimal loss: {optimal_loss:.3e}")

    parameters = initial_parameters
    if config.optimizer == "adam":
        optimizer = optax.adam(config.adam_learning_rate)
        opt_state = optimizer.init(parameters)
        initial_value_and_grad = jax.jit(jax.value_and_grad(loss))
        value, grad = jax.block_until_ready(
            initial_value_and_grad(parameters)
        )
        value = float(value)

        @jax.jit
        def step(parameters, state, grad):
            updates, state = optimizer.update(
                grad,
                state,
                parameters,
            )
            parameters = optax.apply_updates(parameters, updates)
            value, grad = jax.value_and_grad(loss)(parameters)
            return parameters, state, value, grad

    elif config.optimizer == "lbfgs":
        linesearch = optax.scale_by_zoom_linesearch(
            max_linesearch_steps=config.max_linesearch_steps,
            initial_guess_strategy="one",
            tol=3e-7
        )
        optimizer = optax.lbfgs(
            memory_size=config.memory_size,
            linesearch=linesearch,
        )
        opt_state = optimizer.init(parameters)
        value_and_grad = optax.value_and_grad_from_state(loss)

        @jax.jit
        def step(parameters, state):
            value, grad = value_and_grad(parameters, state=state)
            updates, state = optimizer.update(
                grad,
                state,
                parameters,
                value=value,
                grad=grad,
                value_fn=loss,
            )
            return optax.apply_updates(parameters, updates), state

    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    history = []
    global_best_loss = np.inf
    best_parameters = None
    attempt_best_loss = np.inf
    stalled = 0
    early_stop_reference = None
    restart = 0
    restart_rng = np.random.default_rng(config.perturbation_seed)
    total_evaluations = 1
    initialization_seconds = time.perf_counter() - run_start_time
    start_time = time.perf_counter()

    try:
        for i in range(config.max_steps):
            step_start = time.perf_counter()
            if config.optimizer == "adam":
                parameters, opt_state, value, grad = step(
                    parameters,
                    opt_state,
                    grad,
                )
                linesearch_evaluations = 1
            else:
                parameters, opt_state = step(parameters, opt_state)
                value = optax.tree.get(opt_state, "value")
                grad = optax.tree.get(opt_state, "grad")
                linesearch_evaluations = int(
                    optax.tree.get(opt_state, "num_linesearch_steps")
                )
            gradient_norm = optax.global_norm(grad)
            value, gradient_norm, parameters = jax.block_until_ready(
                (value, gradient_norm, parameters)
            )

            value = float(value)
            total_evaluations += linesearch_evaluations

            history.append(
                StepRecord(
                    step=i,
                    restart=restart,
                    parameters=jax.tree.map(np.asarray, parameters),
                    loss=value,
                    gradient_norm=float(gradient_norm),
                    linesearch_evaluations=linesearch_evaluations,
                    total_evaluations=total_evaluations,
                    elapsed_seconds=time.perf_counter() - start_time,
                    step_seconds=time.perf_counter() - step_start,
                )
            )

            if not np.isfinite(value):
                print(f"Stopping at step {i}: non-finite loss")
                break

            loss_rtol = (
                EARLY_STOP_RTOL
                if config.early_stop
                else config.loss_rtol
            )
            attempt_improvement = (
                not np.isfinite(attempt_best_loss)
                or value
                < attempt_best_loss
                - loss_rtol
                * max(abs(attempt_best_loss), np.finfo(float).tiny)
            )
            attempt_best_loss = min(attempt_best_loss, value)
            if value < global_best_loss:
                global_best_loss = value
                best_parameters = parameters
            if attempt_improvement:
                stalled = 0
            else:
                stalled += 1

            if i % config.print_every == 0:
                print(
                    f"Step {i}, eval {total_evaluations}: "
                    f"loss={value:.3e} "
                    f"t={history[-1].elapsed_seconds:.2f}s"
                )

            if stalled >= config.patience:
                if (
                    config.early_stop
                    and attempt_best_loss
                    >= EARLY_STOP_LOSS_RATIO * optimal_loss
                ):
                    if early_stop_reference is None:
                        early_stop_reference = attempt_best_loss
                        stalled = 0
                        print(
                            f"Early-stop reference at step {i}: "
                            f"loss={attempt_best_loss:.3e} "
                            f"({attempt_best_loss / optimal_loss:.1f}x "
                            "optimal)"
                        )
                        continue

                    improvement_factor = (
                        early_stop_reference / attempt_best_loss
                    )
                    if (
                        improvement_factor
                        < EARLY_STOP_IMPROVEMENT_FACTOR
                    ):
                        print(
                            f"Early stopping at step {i}: loss only "
                            f"improved by {improvement_factor:.2f}x "
                            "between patience triggers"
                        )
                        break

                    early_stop_reference = attempt_best_loss
                    stalled = 0
                    print(
                        f"Early-stop progress at step {i}: loss "
                        f"improved by {improvement_factor:.2f}x"
                    )
                    continue

                if restart >= config.max_restarts:
                    print(
                        f"Converged at step {i}: no relative loss "
                        f"improvement greater than {loss_rtol:g} "
                        f"for {config.patience} steps after the final "
                        "restart"
                    )
                    break

                parameters = jax.tree.map(
                    lambda values: values
                    + config.perturbation_scale
                    * jnp.asarray(
                        restart_rng.normal(size=values.shape),
                        dtype=values.dtype,
                    ),
                    best_parameters,
                )
                parameters = replace(
                    parameters,
                    exp_m=(
                        best_parameters.exp_m
                        if config.fix_mass
                        else parameters.exp_m
                    ),
                    log10_concentration=(
                        best_parameters.log10_concentration
                        if not config.vary_concentration
                        else parameters.log10_concentration
                    ),
                    position=jnp.clip(
                        parameters.position,
                        -MAX_COM_POSITION,
                        MAX_COM_POSITION,
                    ),
                    velocity=jnp.clip(
                        parameters.velocity,
                        -MAX_COM_VELOCITY,
                        MAX_COM_VELOCITY,
                    ),
                )
                opt_state = optimizer.init(parameters)
                if config.optimizer == "adam":
                    value, grad = jax.block_until_ready(
                        initial_value_and_grad(parameters)
                    )
                    value = float(value)
                    total_evaluations += 1
                restart += 1
                attempt_best_loss = np.inf
                early_stop_reference = None
                stalled = 0
                print(
                    f"Restart {restart}/{config.max_restarts} at step {i}: "
                    f"perturbing best loss {global_best_loss:.3e}"
                )
    except KeyboardInterrupt:
        print("\nInterrupted by user; saving completed steps")

    if not history:
        print("No completed steps; nothing to save")
        return None

    history = jax.tree.map(lambda *values: np.asarray(values), *history)
    parameter_history = structured_parameters(history.parameters)
    history_fields = tuple(
        name for name in history._fields if name != "parameters"
    )
    structured_history = np.empty(
        len(history.loss),
        dtype=[
            ("parameters", parameter_history.dtype, (config.Nhaloes,)),
            *[
                (name, np.asarray(getattr(history, name)).dtype)
                for name in history_fields
            ],
        ],
    )
    structured_history["parameters"] = parameter_history
    for name in history_fields:
        structured_history[name] = getattr(history, name)

    output_directory = Path(config.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_prefix = config.mode if config.output_prefix is None else config.output_prefix
    if config.output_id is None:
        run_number = 0
        while (
            (output_directory / f"{output_prefix}_{run_number}.npz").exists()
            or (output_directory / f"{output_prefix}_{run_number}.json").exists()
        ):
            run_number += 1
    else:
        run_number = config.output_id

    run_label = f"{run_number:0{config.output_id_width}d}"
    output_path = output_directory / f"{output_prefix}_{run_label}.npz"
    config_path = output_directory / f"{output_prefix}_{run_label}.json"
    if output_path.exists() or config_path.exists():
        raise FileExistsError(
            f"Output ID {run_number} already exists for mode "
            f"{output_prefix!r} in {output_directory}"
        )

    np.savez(
        output_path,
        target_parameters=structured_parameters(target_parameters),
        initial_parameters=structured_parameters(initial_parameters),
        history=structured_history,
        optimal_loss=optimal_loss,
        initialization_seconds=initialization_seconds,
        optimization_seconds=float(history.elapsed_seconds[-1]),
        total_run_seconds=(
            initialization_seconds + float(history.elapsed_seconds[-1])
        ),
    )
    config.to_json(config_path)

    print(f"Saved {output_path}")
    print(f"Saved {config_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--N", type=int, default=Config.N)
    parser.add_argument(
        "--Nhaloes",
        type=int,
        default=Config.Nhaloes,
    )
    parser.add_argument(
        "--ic",
        action="store_true",
        help="Compare initial conditions without running the simulation",
    )
    parser.add_argument(
        "--integration_time_gyr",
        type=float,
        default=Config.integration_time_gyr,
    )
    parser.add_argument(
        "--integration_steps",
        type=int,
        default=Config.integration_steps,
    )
    parser.add_argument(
        "--sim_softening",
        type=float,
        default=DEFAULT_GRAVITATIONAL_SOFTENING,
        help="Gravitational Plummer softening in kpc",
    )
    parser.add_argument(
        "--loss_softening",
        type=float,
        default=Config.loss_softening,
        help="Loss-kernel softening in kpc",
    )
    parser.add_argument(
        "--log10_mass_min",
        type=float,
        default=Config.log10_mass_min,
    )
    parser.add_argument(
        "--log10_mass_max",
        type=float,
        default=Config.log10_mass_max,
    )
    parser.add_argument(
        "--target_log10_mass_min",
        type=float,
        default=Config.target_log10_mass_min,
    )
    parser.add_argument(
        "--target_log10_mass_max",
        type=float,
        default=Config.target_log10_mass_max,
    )
    parser.add_argument(
        "--accurate",
        type=int,
        choices=range(len(FMM_ACCURACY_LEVELS)),
        default=0,
        metavar="LEVEL",
        help="FMM accuracy: 0=(p=5, theta=0.8), "
        "1=(p=6, theta=0.7), 2=(p=7, theta=0.6)",
    )
    parser.add_argument(
        "--loss_plummer",
        action="store_true",
        help="Use the Plummer loss instead of the distance loss",
    )
    parser.add_argument(
        "--loss_dedup",
        type=float,
        default=Config.deduplication_weight,
        help="Weight of the optional component-overlap penalty",
    )
    parser.add_argument(
        "--vary-conc",
        action="store_true",
        help="Sample and infer concentrations instead of fixing c=8",
    )
    parser.add_argument(
        "--fix_mass",
        action="store_true",
        help="Fix masses to their target values",
    )
    parser.add_argument(
        "--adam",
        action="store_true",
        help="Use Adam instead of L-BFGS",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for Adam",
    )
    parser.add_argument(
        "--initial_parameter_seed",
        type=int,
        default=Config.initial_parameter_seed,
    )
    parser.add_argument(
        "--target_parameter_seed",
        type=int,
        default=Config.target_parameter_seed,
    )
    parser.add_argument(
        "--output_id",
        type=int,
        default=None,
        help="Explicit output ID; by default use the first free integer",
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default=Config.output_directory,
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Output filename prefix; by default use the run mode",
    )
    parser.add_argument(
        "--output_id_width",
        type=int,
        default=0,
        help="Zero-padding width for an explicit output ID",
    )
    parser.add_argument(
        "--initial_parameters",
        type=str,
        default=Config.initial_parameters_file,
        metavar="PATH",
        help="Initialize from the best-loss parameters in a previous .npz run",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=Config.max_steps,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=Config.patience,
        help="Steps without sufficient loss improvement before stopping",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Stop far-from-optimal runs without factor-two progress",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=Config.max_restarts,
        help="Number of perturb-and-restart attempts after stalls",
    )
    parser.add_argument(
        "--perturbation_scale",
        type=float,
        default=Config.perturbation_scale,
        help="Gaussian perturbation scale in optimizer coordinates",
    )
    parser.add_argument(
        "--perturbation_seed",
        type=int,
        default=Config.perturbation_seed,
    )
    args = parser.parse_args()
    if args.restarts < 0:
        parser.error("--restarts must be non-negative")
    if args.output_id is not None and args.output_id < 0:
        parser.error("--output_id must be non-negative")
    if args.output_id_width < 0:
        parser.error("--output_id_width must be non-negative")
    if args.perturbation_scale < 0:
        parser.error("--perturbation_scale must be non-negative")
    if args.log10_mass_min >= args.log10_mass_max:
        parser.error("--log10_mass_min must be below --log10_mass_max")
    if args.target_log10_mass_min >= args.target_log10_mass_max:
        parser.error(
            "--target_log10_mass_min must be below "
            "--target_log10_mass_max"
        )
    if not (
        args.log10_mass_min < args.target_log10_mass_min
        and args.target_log10_mass_max < args.log10_mass_max
    ):
        parser.error("target mass range must lie inside inference bounds")
    if args.loss_softening <= 0:
        parser.error("--loss_softening must be positive")
    sim_config = default_sim_config()
    fmm_order, opening_angle = FMM_ACCURACY_LEVELS[args.accurate]
    sim_config = replace(
        sim_config,
        force=replace(
            sim_config.force,
            p=fmm_order,
            alloc_fac_ilist=FMM_ALLOC_FAC_ILIST,
            opening=replace(
                sim_config.force.opening,
                theta=opening_angle,
            ),
            kernel=replace(
                sim_config.force.kernel,
                softening=args.sim_softening,
            ),
        ),
    )
    run(
        Config(
            N=args.N,
            Nhaloes=args.Nhaloes,
            mode="ic" if args.ic else "sim",
            integration_time_gyr=args.integration_time_gyr,
            integration_steps=args.integration_steps,
            sim_config=sim_config,
            loss_mode="plummer" if args.loss_plummer else "distance",
            loss_softening=args.loss_softening,
            log10_mass_min=args.log10_mass_min,
            log10_mass_max=args.log10_mass_max,
            target_log10_mass_min=args.target_log10_mass_min,
            target_log10_mass_max=args.target_log10_mass_max,
            deduplication_weight=args.loss_dedup,
            vary_concentration=args.vary_conc,
            fix_mass=args.fix_mass,
            optimizer="adam" if args.adam else "lbfgs",
            adam_learning_rate=(
                Config.adam_learning_rate
                if args.learning_rate is None
                else args.learning_rate
            ),
            target_parameter_seed=args.target_parameter_seed,
            initial_parameter_seed=args.initial_parameter_seed,
            output_directory=args.output_directory,
            output_prefix=args.output_prefix,
            output_id=args.output_id,
            output_id_width=args.output_id_width,
            initial_parameters_file=args.initial_parameters,
            max_steps=args.steps,
            patience=args.patience,
            early_stop=args.early_stop,
            max_restarts=args.restarts,
            perturbation_scale=args.perturbation_scale,
            perturbation_seed=args.perturbation_seed,
        )
    )
