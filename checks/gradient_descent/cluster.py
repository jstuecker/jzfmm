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

import fmdj
from aegis.profiles.analytical import RvirOfMvir


POSITION_UNIT = 1e3  # kpc per Mpc
VELOCITY_UNIT = 1e3  # km/s
MASS_UNIT = 1e14
LOSS_SOFTENING = 10.0
SAMPLING_RADIUS_MAX = 10.0
LOG10_MASS_MIN = 11.5
LOG10_MASS_MAX = 13.5
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
MASS_WEIGHT = 1.0
GYR_TO_SIMULATION_TIME = 1.022712165
HOST_MASS = 1e14
HOST_SCALE_RADIUS = 163.0  # kpc
LATTICE_POSITION_LIMIT = 10_000.0  # kpc
LATTICE_VELOCITY_LIMIT = 10_000.0  # km/s
LATTICE_DX = LATTICE_POSITION_LIMIT / np.iinfo(np.int32).max
LATTICE_DV = LATTICE_VELOCITY_LIMIT / np.iinfo(np.int32).max
DEFAULT_GRAVITATIONAL_SOFTENING = 1.0  # kpc
FMM_ACCURACY_LEVELS = (
    (5, 0.8),
    (6, 0.7),
    (7, 0.6),
)
FMM_ALLOC_FAC_ILIST = 150
DEFAULT_OUTPUT_DIRECTORY = (
    Path(__file__).resolve().parents[0] / "logs"
)


def mass_from_exp(exp_m):
    log10_mass = (
        LOG10_MASS_MIN
        + (LOG10_MASS_MAX - LOG10_MASS_MIN) * jax.nn.sigmoid(exp_m)
    )
    return 10.0**log10_mass
mass_from_exp.jit = jax.jit(mass_from_exp)


def concentration_from_log10(log10_concentration):
    return 10.0**log10_concentration
concentration_from_log10.jit = jax.jit(concentration_from_log10)


def radius_from_mass_concentration(exp_m, log10_concentration):
    halo_mass = mass_from_exp(exp_m)
    r200 = R200_MASS_FACTOR * jnp.cbrt(halo_mass)
    return r200 / concentration_from_log10(log10_concentration)
radius_from_mass_concentration.jit = jax.jit(
    radius_from_mass_concentration
)


def duplication_loss(parameters):
    scale_radius = radius_from_mass_concentration.jit(
        parameters.exp_m,
        parameters.log10_concentration,
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


def default_sim_config() -> fmdj.SimConfig:
    cfg = fmdj.SimConfig(
        force=fmdj.FMMConfig(
            kernel=fmdj.PlummerKernel(
                softening=DEFAULT_GRAVITATIONAL_SOFTENING
            ),
        ),
        external_potential=fmdj.external_potential.HernquistPotential(
            a=HOST_SCALE_RADIUS,
            mass=HOST_MASS,
        ),
        integrator=fmdj.DKDLatticeConfig(
            dx=LATTICE_DX,
            dv=LATTICE_DV,
            int_dtype=jnp.int32,
        ),
    )
    cfg.force.tree.alloc_fac_nodes = 1.8
    return cfg


def sim_config_from_dict(values) -> fmdj.SimConfig:
    force_values = values["force"].copy()
    force_values["tree"] = force_values["tree"].copy()
    force_values["tree"].setdefault("alloc_fac_nodes", 1.2)
    force_values["tree"] = TreeConfig(**force_values["tree"])
    force_values["kernel"] = fmdj.PlummerKernel(
        **force_values["kernel"]
    )
    force_values["opening"] = fmdj.OpeningByAngle(
        **force_values["opening"]
    )

    integrator_values = values["integrator"].copy()
    int_dtype = integrator_values.pop("int_dtype")
    integrator = fmdj.DKDLatticeConfig(
        **integrator_values,
        int_dtype=getattr(jnp, int_dtype),
    )
    return fmdj.SimConfig(
        force=fmdj.FMMConfig(**force_values),
        units=fmdj.UnitConfig(**values["units"]),
        logging=LoggingConfig(**values["logging"]),
        external_potential=fmdj.external_potential.HernquistPotential(
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
    sim_config: fmdj.SimConfig = field(default_factory=default_sim_config)
    loss_mode: str = "distance"
    deduplication_weight: float | None = None
    vary_concentration: bool = False
    fix_mass: bool = False
    optimizer: str = "lbfgs"
    adam_learning_rate: float = 0.01
    max_steps: int = 2000
    patience: int = 20
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

    @classmethod
    def from_json(cls, filename):
        with Path(filename).open() as file:
            values = json.load(file)
        if "mode" not in values:
            values["mode"] = "ic"
        values.pop("max_learning_rate", None)
        values.pop("max_direction_norm", None)
        values.pop("full_matrix_learning_rate", None)
        values.pop("full_matrix_decay", None)
        values.pop("full_matrix_epsilon", None)
        values.pop("initial_parameters_file", None)
        values.pop("fix_structure", None)
        values.pop("constant_radius", None)
        values.pop("fix_radius", None)
        sim_values = values.pop("sim_config", None)
        if sim_values is not None:
            values["sim_config"] = sim_config_from_dict(sim_values)
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
) -> Parameters:
    rng = np.random.default_rng(seed)
    log10_mass = rng.uniform(12.0, 13.0, Nhaloes)
    if vary_concentration:
        log10_concentration = rng.uniform(
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
    position = np.empty((Nhaloes, 3))
    for i in range(Nhaloes):
        while True:
            candidate = rng.normal(0.0, 1.0, 3)
            if i == 0 or np.all(
                np.linalg.norm(position[:i] - candidate, axis=1)
                > scale_radius[:i] + scale_radius[i]
            ):
                position[i] = candidate
                break

    mass_fraction = (
        (log10_mass - LOG10_MASS_MIN)
        / (LOG10_MASS_MAX - LOG10_MASS_MIN)
    )
    return Parameters(
        exp_m=jnp.asarray(np.log(mass_fraction / (1.0 - mass_fraction))),
        log10_concentration=jnp.asarray(log10_concentration),
        position=jnp.asarray(position),
        velocity=jnp.asarray(rng.normal(0.0, 0.4, (Nhaloes, 3))),
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
    )
    halo_mass = mass_from_exp.jit(parameters.exp_m)
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

    particles = fmdj.data.Particles(
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
        return fmdj.time_integration.simulate(
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
    simulation_outputs = fmdj.time_integration.simulate_with_outputs(
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
    target_parameters = sample_parameters(
        config.Nhaloes,
        config.target_parameter_seed,
        vary_concentration=config.vary_concentration,
    )
    if config.initial_parameters_file is None:
        initial_parameters = sample_parameters(
            config.Nhaloes,
            config.initial_parameter_seed,
            vary_concentration=config.vary_concentration,
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
        kernel = fmdj.PlummerKernel(softening=LOSS_SOFTENING)
        normalize = False
    elif config.loss_mode == "distance":
        kernel = fmdj.SoftenedDistanceKernel(
            softening=LOSS_SOFTENING
        )
        normalize = True
    else:
        raise ValueError(f"Unknown loss mode: {config.loss_mode}")

    fmm_config = fmdj.config.FMMConfig(
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
        mmd = fmdj.loss.maximum_mean_discrepancy(
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
                * duplication_loss.jit(parameters)
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
    restart = 0
    restart_rng = np.random.default_rng(config.perturbation_seed)
    total_evaluations = 1
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

            attempt_improvement = (
                not np.isfinite(attempt_best_loss)
                or value
                < attempt_best_loss
                - config.loss_rtol
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
                if restart >= config.max_restarts:
                    print(
                        f"Converged at step {i}: no relative loss "
                        f"improvement greater than {config.loss_rtol:g} "
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
    output_prefix = config.mode
    run_number = 0
    while (
        (output_directory / f"{output_prefix}_{run_number}.npz").exists()
        or (output_directory / f"{output_prefix}_{run_number}.json").exists()
    ):
        run_number += 1

    output_path = output_directory / f"{output_prefix}_{run_number}.npz"
    config_path = output_directory / f"{output_prefix}_{run_number}.json"

    np.savez(
        output_path,
        target_parameters=structured_parameters(target_parameters),
        initial_parameters=structured_parameters(initial_parameters),
        history=structured_history,
        optimal_loss=optimal_loss,
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
    if args.perturbation_scale < 0:
        parser.error("--perturbation_scale must be non-negative")
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
            deduplication_weight=args.loss_dedup,
            vary_concentration=args.vary_conc,
            fix_mass=args.fix_mass,
            optimizer="adam" if args.adam else "lbfgs",
            adam_learning_rate=(
                Config.adam_learning_rate
                if args.learning_rate is None
                else args.learning_rate
            ),
            initial_parameter_seed=args.initial_parameter_seed,
            initial_parameters_file=args.initial_parameters,
            max_steps=args.steps,
            patience=args.patience,
            max_restarts=args.restarts,
            perturbation_scale=args.perturbation_scale,
            perturbation_seed=args.perturbation_seed,
        )
    )
