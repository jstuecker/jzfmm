import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

import argparse
import functools
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import NamedTuple

import aegis
import jax
import jax.numpy as jnp
import numpy as np
import optax

import fmdj
from aegis.profiles.analytical import RvirOfMvir


POSITION_UNIT = 1e3  # kpc per Mpc
VELOCITY_UNIT = 1e3  # km/s
MASS_UNIT = 1e14
SOFTENING = 10.0
SAMPLING_RADIUS_MAX = 10.0
LOG10_MASS_MIN = 11.5
LOG10_MASS_MAX = 13.5
SCALE_RADIUS_MIN = 0.01
SCALE_RADIUS_MAX = 0.2
MASS_WEIGHT = 1.0
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


def radius_from_exp(exp_rs):
    return (
        SCALE_RADIUS_MIN
        + (SCALE_RADIUS_MAX - SCALE_RADIUS_MIN) * jax.nn.sigmoid(exp_rs)
    )
radius_from_exp.jit = jax.jit(radius_from_exp)


def duplication_loss(parameters):
    scale_radius = radius_from_exp.jit(parameters.exp_rs)
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
    exp_rs: jax.Array
    position: jax.Array
    velocity: jax.Array

    @classmethod
    def from_structured(cls, parameters):
        return cls(
            exp_m=jnp.asarray(parameters["exp_m"]),
            exp_rs=jnp.asarray(parameters["exp_rs"]),
            position=jnp.asarray(parameters["position"]),
            velocity=jnp.asarray(parameters["velocity"]),
        )


class StepRecord(NamedTuple):
    step: int
    parameters: Parameters
    loss: float
    gradient_norm: float
    linesearch_evaluations: int
    total_evaluations: int
    elapsed_seconds: float
    step_seconds: float


@dataclass(frozen=True)
class Config:
    N: int = 100_000
    Nhaloes: int = 3
    loss_mode: str = "distance"
    deduplication_weight: float | None = None
    max_steps: int = 2000
    patience: int = 15
    loss_rtol: float = 1e-5
    memory_size: int = 10
    max_linesearch_steps: int = 4
    print_every: int = 10

    target_parameter_seed: int = 0
    initial_parameter_seed: int = 42
    target_particle_seed: int = 37
    model_particle_seed: int = 0
    output_directory: str = str(DEFAULT_OUTPUT_DIRECTORY)

    @classmethod
    def from_json(cls, filename):
        with Path(filename).open() as file:
            return cls(**json.load(file))

    def to_json(self, filename):
        with Path(filename).open("w") as file:
            json.dump(asdict(self), file, indent=2)
            file.write("\n")


def sample_parameters(Nhaloes: int, seed: int) -> Parameters:
    rng = np.random.default_rng(seed)
    log10_mass = rng.uniform(12.0, 13.0, Nhaloes)
    concentration = rng.uniform(5.0, 10.0, Nhaloes)
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
    radius_fraction = (
        (scale_radius - SCALE_RADIUS_MIN)
        / (SCALE_RADIUS_MAX - SCALE_RADIUS_MIN)
    )
    return Parameters(
        exp_m=jnp.asarray(np.log(mass_fraction / (1.0 - mass_fraction))),
        exp_rs=jnp.asarray(
            np.log(radius_fraction / (1.0 - radius_fraction))
        ),
        position=jnp.asarray(position),
        velocity=jnp.asarray(rng.normal(0.0, 0.4, (Nhaloes, 3))),
    )


def structured_parameters(parameters: Parameters) -> np.ndarray:
    exp_m = np.asarray(parameters.exp_m)
    exp_rs = np.asarray(parameters.exp_rs)
    position = np.asarray(parameters.position)
    velocity = np.asarray(parameters.velocity)

    result = np.empty(
        exp_m.shape,
        dtype=[
            ("exp_m", exp_m.dtype),
            ("exp_rs", exp_rs.dtype),
            ("position", position.dtype, (3,)),
            ("velocity", velocity.dtype, (3,)),
        ],
    )
    result["exp_m"] = exp_m
    result["exp_rs"] = exp_rs
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
) -> fmdj.data.Particles:
    pos, vel, mass = [], [], []
    scale_radius = radius_from_exp.jit(parameters.exp_rs)
    halo_mass = mass_from_exp.jit(parameters.exp_m)
    for i, (base_pos, base_vel, base_mass) in enumerate(base_particles):
        pos.append(
            jnp.asarray(base_pos) * scale_radius[i] * POSITION_UNIT
            + parameters.position[i] * POSITION_UNIT
        )
        vel.append(
            jnp.asarray(base_vel) + parameters.velocity[i] * VELOCITY_UNIT
        )
        mass.append(
            jnp.asarray(base_mass) * halo_mass[i]
        )

    return fmdj.data.Particles(
        pos=jnp.concatenate(pos),
        vel=jnp.concatenate(vel),
        mass=jnp.concatenate(mass),
    )


map_particles.jit = jax.jit(map_particles, static_argnames=("config",))


def get_particles(
    parameters: Parameters,
    config: Config,
    seed: int,
) -> fmdj.data.Particles:
    if not isinstance(parameters, Parameters):
        parameters = Parameters.from_structured(parameters)

    base_particles = [
        sample_base_particles(config.N, seed + i)
        for i in range(len(parameters.exp_m))
    ]
    return map_particles.jit(parameters, base_particles, config)


def run(config: Config):
    target_parameters = sample_parameters(
        config.Nhaloes, config.target_parameter_seed
    )
    initial_parameters = sample_parameters(
        config.Nhaloes, config.initial_parameter_seed
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
        kernel = fmdj.PlummerKernel(softening=SOFTENING)
        normalize = False
    elif config.loss_mode == "distance":
        kernel = fmdj.SoftenedDistanceKernel(softening=SOFTENING)
        normalize = True
    else:
        raise ValueError(f"Unknown loss mode: {config.loss_mode}")

    fmm_config = fmdj.config.FMMConfig(
        kernel=kernel,
        remove_self_interaction=False,
    )

    def loss(parameters):
        particles = map_particles.jit(parameters, model_base, config)
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
            return (
                mmd / POSITION_UNIT
                + MASS_WEIGHT * mass_loss
                + prior_loss
            )
        return mmd + prior_loss

    loss.jit = jax.jit(loss)
    optimal_loss = float(
        jax.block_until_ready(loss.jit(target_parameters))
    )
    print(f"Optimal loss: {optimal_loss:.3e}")

    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=config.max_linesearch_steps,
        initial_guess_strategy="one",
    )
    optimizer = optax.lbfgs(
        memory_size=config.memory_size,
        linesearch=linesearch,
    )
    parameters = initial_parameters
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

    history = []
    best_loss = np.inf
    stalled = 0
    total_evaluations = 1
    start_time = time.perf_counter()

    for i in range(config.max_steps):
        step_start = time.perf_counter()
        parameters, opt_state = step(parameters, opt_state)

        value = optax.tree.get(opt_state, "value")
        grad = optax.tree.get(opt_state, "grad")
        gradient_norm = optax.global_norm(grad)
        value, gradient_norm, parameters = jax.block_until_ready(
            (value, gradient_norm, parameters)
        )

        value = float(value)
        linesearch_evaluations = int(
            optax.tree.get(opt_state, "num_linesearch_steps")
        )
        total_evaluations += linesearch_evaluations

        history.append(
            StepRecord(
                step=i,
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

        if (
            not np.isfinite(best_loss)
            or value
            < best_loss
            - config.loss_rtol * max(abs(best_loss), np.finfo(float).tiny)
        ):
            best_loss = value
            stalled = 0
        else:
            best_loss = min(best_loss, value)
            stalled += 1

        if i % config.print_every == 0:
            print(
                f"Step {i}, eval {total_evaluations}: "
                f"loss={value:.3e} t={history[-1].elapsed_seconds:.2f}s"
            )

        if stalled >= config.patience:
            print(
                f"Converged at step {i}: no relative loss improvement "
                f"greater than {config.loss_rtol:g} for "
                f"{config.patience} steps"
            )
            break

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
    run_number = 0
    while (
        (output_directory / f"run_{run_number}.npz").exists()
        or (output_directory / f"run_{run_number}.json").exists()
    ):
        run_number += 1

    output_path = output_directory / f"run_{run_number}.npz"
    config_path = output_directory / f"run_{run_number}.json"

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
        "--initial_parameter_seed",
        type=int,
        default=Config.initial_parameter_seed,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=Config.max_steps,
    )
    args = parser.parse_args()
    run(
        Config(
            N=args.N,
            Nhaloes=args.Nhaloes,
            loss_mode="plummer" if args.loss_plummer else "distance",
            deduplication_weight=args.loss_dedup,
            initial_parameter_seed=args.initial_parameter_seed,
            max_steps=args.steps,
        )
    )
