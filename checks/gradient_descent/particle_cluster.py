import argparse
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import aegis
import jax
import jax.numpy as jnp
import numpy as np
import optax

import cluster as cl
import fmdj


DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "logs"
TARGET_WEIGHT = 1.0
MASS_WEIGHT = 10.0
HERNQUIST_WEIGHT = 1.0
HUBBLE_CONSTANT = 70.0  # km/s/Mpc
BASE_HERNQUIST = aegis.profiles.HernquistProfile(a=1.0, M=1.0)
BASE_VCIRC_AT_RS = float(BASE_HERNQUIST.vcirc(np.asarray([1.0]))[0])


@jax.tree_util.register_dataclass
@dataclass
class ParticleParameters:
    log10_mass: jax.Array
    position: jax.Array  # Mpc
    velocity: jax.Array  # 1000 km/s

    @classmethod
    def from_structured(cls, parameters):
        return cls(
            log10_mass=jnp.asarray(parameters["log10_mass"]),
            position=jnp.asarray(parameters["position"]),
            velocity=jnp.asarray(parameters["velocity"]),
        )


def get_particles(parameters):
    if not isinstance(parameters, ParticleParameters):
        parameters = ParticleParameters.from_structured(parameters)
    particle_mass = jnp.broadcast_to(
        10.0**parameters.log10_mass,
        (len(parameters.position),),
    )
    return fmdj.data.Particles(
        pos=parameters.position * cl.POSITION_UNIT,
        vel=parameters.velocity * cl.VELOCITY_UNIT,
        mass=particle_mass,
    )


@dataclass(frozen=True)
class Config:
    N: int = 1_000
    mode: str = "sim"
    integration_time_gyr: float = 1.0
    integration_steps: int = 10
    sim_softening: float = 10.0
    loss_softening: float = cl.LOSS_SOFTENING
    log10_mass_min: float = 12.0
    log10_mass_max: float = 13.0
    accuracy: int = 1
    redshift_space: bool = False
    fixed_masses: bool = False
    merging: bool = True
    max_steps: int = 2000
    optimizer: str = "lbfgs"
    learning_rate: float = 0.01
    learning_rate_reduction: float = 0.25
    minimum_learning_rate: float = 1e-5
    memory_size: int = 10
    max_linesearch_steps: int = 5
    epoch_steps: int = 10
    target_weight: float = TARGET_WEIGHT
    mass_weight: float = MASS_WEIGHT
    hernquist_weight: float = HERNQUIST_WEIGHT
    patience: int = 20
    loss_rtol: float = 1e-5
    snapshot_every_epochs: int = 10
    target_parameter_seed: int = 0
    initial_parameter_seed: int = 42
    initial_parameters_file: str | None = None
    target_particle_seed: int = 37
    model_particle_seed: int = 0
    initial_particle_seed: int = 0
    prior_particle_seed: int = 19
    remap_seed: int = 0
    output_directory: str = str(DEFAULT_OUTPUT_DIRECTORY)

    def to_json(self, filename):
        with Path(filename).open("w") as file:
            json.dump(asdict(self), file, indent=2)
            file.write("\n")

    @classmethod
    def from_json(cls, filename):
        with Path(filename).open() as file:
            return cls(**json.load(file))


def simulation_config(config):
    sim_config = cl.default_sim_config()
    order, opening = cl.FMM_ACCURACY_LEVELS[config.accuracy]
    return replace(
        sim_config,
        force=replace(
            sim_config.force,
            p=order,
            alloc_fac_ilist=cl.FMM_ALLOC_FAC_ILIST,
            tree=replace(
                sim_config.force.tree,
                alloc_fac_nodes=3.0,
            ),
            opening=replace(
                sim_config.force.opening,
                theta=opening,
            ),
            kernel=replace(
                sim_config.force.kernel,
                softening=config.sim_softening,
            ),
        ),
    )


def simulate(parameters, config):
    particles = get_particles(parameters)
    if config.mode == "ic":
        return particles
    if config.mode != "sim":
        raise ValueError(f"Unknown mode: {config.mode}")
    times = np.linspace(
        0.0,
        config.integration_time_gyr * cl.GYR_TO_SIMULATION_TIME,
        config.integration_steps + 1,
        dtype=np.float32,
    )
    return fmdj.time_integration.simulate(
        particles,
        ts=times,
        cfg=simulation_config(config),
    )


def observed_position(particles, redshift_space=False):
    if not redshift_space:
        return particles.pos
    return particles.pos.at[:, 2].add(
        particles.vel[:, 2]
        / HUBBLE_CONSTANT
        * cl.POSITION_UNIT
    )


def mass_weighted_com(parameters, particle_mass):
    total_mass = jnp.sum(particle_mass)
    position = jnp.sum(
        particle_mass[:, None] * parameters.position,
        axis=0,
    ) / total_mass
    velocity = jnp.sum(
        particle_mass[:, None] * parameters.velocity,
        axis=0,
    ) / total_mass
    return position, velocity


def sample_host_particles(N, seed):
    np.random.seed(seed)
    host = aegis.profiles.HernquistProfile(
        a=cl.HOST_SCALE_RADIUS,
        M=cl.HOST_MASS,
    )
    position, velocity = host.sample_particles(
        N,
        result="pos_vel",
        ramax=cl.MAX_COM_POSITION * cl.POSITION_UNIT,
    )
    return ParticleParameters(
        log10_mass=jnp.zeros(N),
        position=jnp.asarray(position / cl.POSITION_UNIT),
        velocity=jnp.asarray(velocity / cl.VELOCITY_UNIT),
    )


def sample_log10_mass(seed, minimum, maximum):
    mass_seed = np.random.SeedSequence(seed).spawn(3)[0]
    return np.random.default_rng(mass_seed).uniform(minimum, maximum)


def make_halo_particles(log10_mass, position, velocity, N, seed):
    base_position, base_velocity, base_mass = cl.sample_base_particles(
        N,
        seed,
    )
    halo_mass = 10.0**log10_mass
    scale_radius = (
        cl.R200_MASS_FACTOR
        * np.cbrt(halo_mass)
        / cl.FIXED_CONCENTRATION
    )
    return fmdj.data.Particles(
        pos=(
            jnp.asarray(base_position)
            * scale_radius
            * cl.POSITION_UNIT
            + position * cl.POSITION_UNIT
        ),
        vel=(
            jnp.asarray(base_velocity)
            * jnp.sqrt(halo_mass / (scale_radius * cl.POSITION_UNIT))
            + velocity * cl.VELOCITY_UNIT
        ),
        mass=jnp.asarray(base_mass) * halo_mass,
    )


def structured_parameters(parameters):
    position = np.asarray(parameters.position)
    velocity = np.asarray(parameters.velocity)
    result = np.empty(
        (),
        dtype=[
            ("log10_mass", np.asarray(parameters.log10_mass).dtype,
             np.asarray(parameters.log10_mass).shape),
            ("position", position.dtype, position.shape),
            ("velocity", velocity.dtype, velocity.shape),
        ],
    )
    result["log10_mass"] = np.asarray(parameters.log10_mass)
    result["position"] = position
    result["velocity"] = velocity
    return result


def remap_particles(parameters, target_average_mass, rng):
    """Vectorized fixed-N merge/split remapping between optimizer epochs."""
    mass = 10.0 ** np.asarray(parameters.log10_mass)
    position = np.asarray(parameters.position).copy()
    velocity = np.asarray(parameters.velocity).copy()

    low_mass = mass < 0.1 * target_average_mass
    low_donor = np.flatnonzero(low_mass)
    requested = np.maximum(
        np.floor(mass / target_average_mass).astype(np.int64) - 1,
        0,
    )
    parent_order = np.argsort(-mass)
    requested_source = np.repeat(
        parent_order,
        requested[parent_order],
    )

    if len(requested_source) >= len(low_donor):
        extra_donor = np.flatnonzero(
            (requested == 0) & ~low_mass
        )
        extra_donor = extra_donor[np.argsort(mass[extra_donor])]
        donor = np.concatenate([low_donor, extra_donor])
        number = min(len(requested_source), len(donor))
        source = requested_source[:number]
        donor = donor[:number]
    else:
        number = len(low_donor)
        donor = low_donor
        extra_number = number - len(requested_source)
        source_candidate = np.flatnonzero(~low_mass)
        source_candidate = source_candidate[
            np.argsort(-mass[source_candidate])
        ]
        if len(source_candidate) == 0:
            return parameters, 0
        extra_source = np.resize(source_candidate, extra_number)
        source = np.concatenate([requested_source, extra_source])

    if number == 0:
        return parameters, 0

    children_per_parent = np.bincount(source, minlength=len(mass))
    donor_mass_per_parent = np.bincount(
        source,
        weights=mass[donor],
        minlength=len(mass),
    )
    family_total_mass = mass + donor_mass_per_parent
    family_mass = family_total_mass / (1 + children_per_parent)
    donor_position_per_parent = np.zeros((len(mass), 3))
    donor_velocity_per_parent = np.zeros((len(mass), 3))
    np.add.at(
        donor_position_per_parent,
        source,
        mass[donor, None] * position[donor],
    )
    np.add.at(
        donor_velocity_per_parent,
        source,
        mass[donor, None] * velocity[donor],
    )
    family_position = (
        mass[:, None] * position + donor_position_per_parent
    ) / family_total_mass[:, None]
    family_velocity = (
        mass[:, None] * velocity + donor_velocity_per_parent
    ) / family_total_mass[:, None]
    parent = np.flatnonzero(children_per_parent)
    member = np.concatenate([parent, donor])
    family = np.concatenate([parent, source])

    total_mass = np.sum(mass)
    scale_radius = (
        cl.R200_MASS_FACTOR
        * np.cbrt(total_mass)
        / cl.FIXED_CONCENTRATION
    )
    scale_radius_kpc = scale_radius * cl.POSITION_UNIT
    velocity_scale = np.sqrt(total_mass / scale_radius_kpc)
    vcirc_at_rs = BASE_VCIRC_AT_RS * velocity_scale
    scatter_position = rng.normal(size=(len(member), 3))
    scatter_velocity = rng.normal(size=(len(member), 3))
    family_count = 1 + children_per_parent
    position_sum = np.zeros((len(mass), 3))
    velocity_sum = np.zeros((len(mass), 3))
    np.add.at(position_sum, family, scatter_position)
    np.add.at(velocity_sum, family, scatter_velocity)
    scatter_position -= position_sum[family] / family_count[family, None]
    scatter_velocity -= velocity_sum[family] / family_count[family, None]

    mass[member] = family_mass[family]
    position[member] = (
        family_position[family]
        + 1e-3 * scale_radius * scatter_position
    )
    velocity[member] = (
        family_velocity[family]
        + 1e-3
        * vcirc_at_rs
        / cl.VELOCITY_UNIT
        * scatter_velocity
    )
    log10_mass = np.log10(mass).astype(
        np.asarray(parameters.log10_mass).dtype
    )
    log10_mass += np.log10(
        total_mass / np.sum(10.0**log10_mass)
    )
    return (
        ParticleParameters(
            log10_mass=jnp.asarray(log10_mass),
            position=jnp.asarray(position),
            velocity=jnp.asarray(velocity),
        ),
        number,
    )


def run(config):
    target_phase_space = cl.sample_parameters(
        1,
        config.target_parameter_seed,
        vary_concentration=False,
    )
    target_log10_mass = sample_log10_mass(
        config.target_parameter_seed,
        config.log10_mass_min,
        config.log10_mass_max,
    )
    initial_log10_mass = sample_log10_mass(
        config.initial_parameter_seed,
        config.log10_mass_min,
        config.log10_mass_max,
    )
    target_particles = make_halo_particles(
        target_log10_mass,
        target_phase_space.position[0],
        target_phase_space.velocity[0],
        config.N,
        config.target_particle_seed,
    )
    target_mass = 10.0**target_log10_mass
    target_average_mass = float(target_mass / config.N)
    target_parameters = ParticleParameters(
        log10_mass=jnp.log10(target_particles.mass),
        position=target_particles.pos / cl.POSITION_UNIT,
        velocity=target_particles.vel / cl.VELOCITY_UNIT,
    )
    reference_particles = make_halo_particles(
        target_log10_mass,
        target_phase_space.position[0],
        target_phase_space.velocity[0],
        config.N,
        config.model_particle_seed,
    )
    reference_parameters = ParticleParameters(
        log10_mass=(
            jnp.log10(target_average_mass)
            if config.fixed_masses
            else jnp.log10(reference_particles.mass)
        ),
        position=reference_particles.pos / cl.POSITION_UNIT,
        velocity=reference_particles.vel / cl.VELOCITY_UNIT,
    )
    target_particles = simulate(target_parameters, config)
    target = fmdj.data.PosMass(
        pos=observed_position(
            target_particles,
            config.redshift_space,
        ),
        mass=target_particles.mass / cl.MASS_UNIT,
    )

    if config.initial_parameters_file is None:
        initial_parameters = sample_host_particles(
            config.N,
            config.initial_particle_seed,
        )
        initial_mass = 10.0**initial_log10_mass
        initial_parameters.log10_mass = jnp.asarray(
            jnp.log10(initial_mass / config.N)
            if config.fixed_masses
            else jnp.full(
                config.N,
                jnp.log10(initial_mass / config.N),
            )
        )
    else:
        with np.load(config.initial_parameters_file) as previous_run:
            if "best_parameters" not in previous_run:
                raise ValueError(
                    f"{config.initial_parameters_file} does not contain "
                    "best_parameters"
                )
            values = previous_run["best_parameters"]
            initial_parameters = ParticleParameters.from_structured(values)
            previous_best_step = int(previous_run["best_step"])
        if (
            initial_parameters.position.shape != (config.N, 3)
            or initial_parameters.velocity.shape != (config.N, 3)
        ):
            raise ValueError(
                f"{config.initial_parameters_file} contains "
                f"{len(initial_parameters.position)} particles, but N={config.N}"
            )
        expected_mass_shape = () if config.fixed_masses else (config.N,)
        if initial_parameters.log10_mass.shape != expected_mass_shape:
            mode = "fixed" if config.fixed_masses else "variable"
            raise ValueError(
                f"{config.initial_parameters_file} is incompatible with "
                f"the requested {mode}-mass mode"
            )
        print(
            f"Initialized parameters from "
            f"{config.initial_parameters_file}, step {previous_best_step}"
        )
    log10_average_mass = np.log10(target_average_mass)

    base_position, base_velocity, base_mass = cl.sample_base_particles(
        config.N,
        config.prior_particle_seed,
    )
    base_position = jnp.asarray(base_position)
    base_velocity = jnp.asarray(base_velocity)
    base_mass = jnp.asarray(base_mass)

    distance_config = fmdj.DirectSummationConfig(
        kernel=fmdj.SoftenedDistanceKernel(
            softening=config.loss_softening
        ),
        kahan_summation=True,
        remove_self_interaction=False,
    )

    def loss_terms(parameters):
        position_limit = cl.LATTICE_POSITION_LIMIT / cl.POSITION_UNIT
        velocity_limit = cl.LATTICE_VELOCITY_LIMIT / cl.VELOCITY_UNIT
        valid = (
            jnp.all(jnp.isfinite(parameters.log10_mass))
            & jnp.all(
                parameters.log10_mass >= log10_average_mass - 4.0
            )
            & jnp.all(
                parameters.log10_mass <= log10_average_mass + 2.0
            )
            & jnp.all(jnp.isfinite(parameters.position))
            & jnp.all(
                jnp.abs(parameters.position) <= position_limit
            )
            & jnp.all(jnp.isfinite(parameters.velocity))
            & jnp.all(
                jnp.abs(parameters.velocity) <= velocity_limit
            )
        )
        parameters = replace(
            parameters,
            log10_mass=jnp.clip(
                jnp.nan_to_num(parameters.log10_mass),
                log10_average_mass - 4.0,
                log10_average_mass + 2.0,
            ),
            position=jnp.clip(
                jnp.nan_to_num(parameters.position),
                -position_limit,
                position_limit,
            ),
            velocity=jnp.clip(
                jnp.nan_to_num(parameters.velocity),
                -velocity_limit,
                velocity_limit,
            ),
        )
        particle_mass = jnp.broadcast_to(
            10.0**parameters.log10_mass,
            (config.N,),
        )
        total_mass = jnp.sum(particle_mass)
        com_position, com_velocity = mass_weighted_com(
            parameters,
            particle_mass,
        )

        evolved_particles = simulate(parameters, config)
        particles = fmdj.data.PosMass(
            pos=observed_position(
                evolved_particles,
                config.redshift_space,
            ),
            mass=evolved_particles.mass / cl.MASS_UNIT,
        )
        target_loss = fmdj.loss.maximum_mean_discrepancy(
            particles,
            target,
            cfg_fmm=distance_config,
            normalize=True,
        ) / cl.POSITION_UNIT
        mass_loss = jnp.log10(total_mass / target_mass) ** 2

        scale_radius = (
            cl.R200_MASS_FACTOR
            * jnp.cbrt(total_mass)
            / cl.FIXED_CONCENTRATION
        )
        scale_radius_kpc = scale_radius * cl.POSITION_UNIT
        velocity_scale = jnp.sqrt(total_mass / scale_radius_kpc)
        vcirc_at_rs = BASE_VCIRC_AT_RS * velocity_scale
        omega = vcirc_at_rs / scale_radius_kpc

        centered_position = (
            parameters.position - com_position
        ) * cl.POSITION_UNIT
        centered_velocity = (
            parameters.velocity - com_velocity
        ) * cl.VELOCITY_UNIT
        phase_position = jnp.concatenate(
            [centered_position, centered_velocity / omega],
            axis=1,
        )

        reference_position = base_position * scale_radius_kpc
        reference_velocity = base_velocity * velocity_scale
        reference_position -= jnp.sum(
            base_mass[:, None] * reference_position,
            axis=0,
        )
        reference_velocity -= jnp.sum(
            base_mass[:, None] * reference_velocity,
            axis=0,
        )
        reference_phase_position = jnp.concatenate(
            [reference_position, reference_velocity / omega],
            axis=1,
        )
        phase_particles = fmdj.data.PosMass(
            pos=phase_position,
            mass=particle_mass / cl.MASS_UNIT,
        )
        reference_phase_particles = fmdj.data.PosMass(
            pos=reference_phase_position,
            mass=base_mass,
        )
        hernquist_loss = fmdj.loss.maximum_mean_discrepancy(
            phase_particles,
            reference_phase_particles,
            cfg_fmm=distance_config,
            normalize=True,
        ) / cl.POSITION_UNIT
        total_loss = (
            config.target_weight * target_loss
            + config.mass_weight * mass_loss
            + config.hernquist_weight * hernquist_loss
        )
        invalid = jnp.asarray(jnp.inf, dtype=total_loss.dtype)
        return (
            jnp.where(valid, total_loss, invalid),
            tuple(
                jnp.where(valid, term, invalid)
                for term in (target_loss, mass_loss, hernquist_loss)
            ),
        )

    def loss(parameters):
        return loss_terms(parameters)[0]

    loss_terms_jit = jax.jit(loss_terms)
    parameters = initial_parameters
    if config.optimizer == "adam":
        optimizer = optax.adam(1.0)
        optimizer_state = optimizer.init(parameters)
        current_learning_rate = config.learning_rate
        minimum_learning_rate = min(
            config.learning_rate,
            config.minimum_learning_rate,
        )

        @jax.jit
        def step(parameters, optimizer_state, learning_rate):
            gradient = jax.grad(loss)(parameters)
            updates, optimizer_state = optimizer.update(
                gradient,
                optimizer_state,
                parameters,
            )
            updates = jax.tree.map(
                lambda update: learning_rate * update,
                updates,
            )
            parameters = optax.apply_updates(parameters, updates)
            (value, terms), gradient = jax.value_and_grad(
                loss_terms,
                has_aux=True,
            )(parameters)
            return (
                parameters,
                optimizer_state,
                value,
                terms,
                gradient,
                jnp.asarray(1),
            )

    elif config.optimizer == "lbfgs":
        current_learning_rate = np.nan
        linesearch = optax.scale_by_zoom_linesearch(
            max_linesearch_steps=config.max_linesearch_steps,
            max_learning_rate=0.1,
            initial_guess_strategy="one",
            tol=3e-7,
        )
        optimizer = optax.lbfgs(
            memory_size=config.memory_size,
            linesearch=linesearch,
        )
        optimizer_state = optimizer.init(parameters)
        value_and_grad = optax.value_and_grad_from_state(loss)

        @jax.jit
        def step(parameters, optimizer_state):
            value, gradient = value_and_grad(
                parameters,
                state=optimizer_state,
            )
            updates, optimizer_state = optimizer.update(
                gradient,
                optimizer_state,
                parameters,
                value=value,
                grad=gradient,
                value_fn=loss,
            )
            parameters = optax.apply_updates(parameters, updates)
            return parameters, optimizer_state

    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    optimal_loss, optimal_terms = jax.block_until_ready(
        loss_terms_jit(reference_parameters)
    )
    initial_loss, initial_terms = jax.block_until_ready(
        loss_terms_jit(initial_parameters)
    )
    print(
        f"Target parameters (particle seed "
        f"{config.model_particle_seed}): "
        f"loss={float(optimal_loss):.3e}, "
        f"target={float(optimal_terms[0]):.3e}, "
        f"mass={float(optimal_terms[1]):.3e}, "
        f"Hernquist={float(optimal_terms[2]):.3e}"
    )
    print(
        "Initial parameters: "
        f"loss={float(initial_loss):.3e}, "
        f"target={float(initial_terms[0]):.3e}, "
        f"mass={float(initial_terms[1]):.3e}, "
        f"Hernquist={float(initial_terms[2]):.3e}"
    )

    history = []
    snapshots = [
        (0, 0, structured_parameters(initial_parameters))
    ]
    remap_rng = np.random.default_rng(config.remap_seed)
    total_evaluations = 1
    best_loss = float(initial_loss)
    best_step = 0
    best_parameters = structured_parameters(initial_parameters)
    stalled = 0
    start_time = time.perf_counter()
    try:
        for step_number in range(config.max_steps):
            completed_steps = step_number + 1
            if config.optimizer == "adam":
                (
                    parameters,
                    optimizer_state,
                    value,
                    terms,
                    gradient,
                    linesearch_evaluations,
                ) = step(
                    parameters,
                    optimizer_state,
                    jnp.asarray(current_learning_rate),
                )
            else:
                parameters, optimizer_state = step(
                    parameters,
                    optimizer_state,
                )
                value = optax.tree.get(optimizer_state, "value")
                gradient = optax.tree.get(optimizer_state, "grad")
                linesearch_evaluations = optax.tree.get(
                    optimizer_state,
                    "num_linesearch_steps",
                )
                terms = loss_terms_jit(parameters)[1]
            value, gradient, terms, parameters = jax.block_until_ready(
                (value, gradient, terms, parameters)
            )
            linesearch_evaluations = int(linesearch_evaluations)
            total_evaluations += linesearch_evaluations
            history.append(
                (
                    completed_steps,
                    completed_steps // config.epoch_steps,
                    float(value),
                    float(terms[0]),
                    float(terms[1]),
                    float(terms[2]),
                    float(optax.global_norm(gradient)),
                    linesearch_evaluations,
                    total_evaluations,
                    current_learning_rate,
                    time.perf_counter() - start_time,
                )
            )
            evaluated_parameters = structured_parameters(parameters)
            value_float = float(value)
            if (
                not np.isfinite(value_float)
                or not np.all(np.isfinite(np.asarray(terms)))
            ):
                print(
                    f"Stopping at step {completed_steps}: "
                    "non-finite loss"
                )
                break
            improvement = (
                not np.isfinite(best_loss)
                or value_float
                < best_loss
                - config.loss_rtol
                * max(abs(best_loss), np.finfo(float).tiny)
            )
            if value_float < best_loss:
                best_loss = value_float
                best_step = completed_steps
                best_parameters = evaluated_parameters.copy()
            stalled = 0 if improvement else stalled + 1
            epoch_finished = (
                completed_steps % config.epoch_steps == 0
            )
            epoch_number = completed_steps // config.epoch_steps
            converged = stalled >= config.patience
            learning_rate_reduced = False
            remapped = 0
            if (
                epoch_finished
                and epoch_number % config.snapshot_every_epochs == 0
            ):
                snapshots.append(
                    (epoch_number, completed_steps, evaluated_parameters)
                )
            if (
                converged
                and config.optimizer == "adam"
                and current_learning_rate > minimum_learning_rate
            ):
                previous_learning_rate = current_learning_rate
                current_learning_rate = max(
                    minimum_learning_rate,
                    current_learning_rate
                    * config.learning_rate_reduction,
                )
                parameters = ParticleParameters.from_structured(
                    best_parameters
                )
                optimizer_state = optimizer.init(parameters)
                stalled = 0
                converged = False
                learning_rate_reduced = True
                print(
                    f"Reduced Adam learning rate from "
                    f"{previous_learning_rate:g} to "
                    f"{current_learning_rate:g}; restored best step "
                    f"{best_step}"
                )
            if (
                epoch_finished
                and config.merging
                and not config.fixed_masses
                and not converged
                and not learning_rate_reduced
                and completed_steps < config.max_steps
            ):
                parameters, remapped = remap_particles(
                    parameters,
                    target_average_mass,
                    remap_rng,
                )
                if remapped:
                    optimizer_state = optimizer.init(parameters)
            if epoch_finished or converged:
                learning_rate_text = (
                    f"lr={current_learning_rate:g}, "
                    if config.optimizer == "adam"
                    else ""
                )
                print(
                    f"Step {completed_steps}, eval {total_evaluations}: "
                    f"loss={float(value):.3e}, "
                    f"target={float(terms[0]):.3e}, "
                    f"mass={float(terms[1]):.3e}, "
                    f"Hernquist={float(terms[2]):.3e}, "
                    f"remapped={remapped}, "
                    f"{learning_rate_text}"
                    f"t={history[-1][-1]:.2f}s"
                )
            if converged:
                print(
                    f"Converged at step {completed_steps}: no relative loss "
                    f"improvement greater than {config.loss_rtol:g} "
                    f"for {config.patience} steps"
                )
                break
    except KeyboardInterrupt:
        print("\nInterrupted by user; saving completed steps")

    if not history:
        print("No completed steps; nothing to save")
        return None

    structured_history = np.empty(
        len(history),
        dtype=[
            ("step", np.int64),
            ("epoch", np.int64),
            ("loss", np.float64),
            ("target_loss", np.float64),
            ("mass_loss", np.float64),
            ("hernquist_loss", np.float64),
            ("gradient_norm", np.float64),
            ("linesearch_evaluations", np.int64),
            ("total_evaluations", np.int64),
            ("learning_rate", np.float64),
            ("elapsed_seconds", np.float64),
        ],
    )
    for index, values in enumerate(history):
        structured_history[index] = values

    parameter_dtype = structured_parameters(initial_parameters).dtype
    structured_snapshots = np.empty(
        len(snapshots),
        dtype=[
            ("epoch", np.int64),
            ("step", np.int64),
            ("parameters", parameter_dtype),
        ],
    )
    for index, values in enumerate(snapshots):
        structured_snapshots[index] = values

    output_directory = Path(config.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_prefix = f"particle_{config.mode}"
    run_number = 0
    while (
        (output_directory / f"{output_prefix}_{run_number}.npz").exists()
        or (output_directory / f"{output_prefix}_{run_number}.json").exists()
        or (
            output_directory
            / f"{output_prefix}_{run_number}_snapshots.npz"
        ).exists()
    ):
        run_number += 1
    output_path = output_directory / f"{output_prefix}_{run_number}.npz"
    config_path = output_directory / f"{output_prefix}_{run_number}.json"
    snapshot_path = (
        output_directory / f"{output_prefix}_{run_number}_snapshots.npz"
    )
    np.savez(
        output_path,
        target_parameters=structured_parameters(target_parameters),
        reference_parameters=structured_parameters(reference_parameters),
        initial_parameters=structured_parameters(initial_parameters),
        best_parameters=best_parameters,
        best_step=best_step,
        best_loss=best_loss,
        history=structured_history,
        optimal_loss=float(optimal_loss),
        optimal_terms=np.asarray(optimal_terms),
    )
    np.savez(snapshot_path, snapshots=structured_snapshots)
    config.to_json(config_path)
    print(f"Saved {output_path}")
    print(f"Saved {snapshot_path}")
    print(f"Saved {config_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-N", "--N", type=int, default=Config.N)
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
        default=Config.sim_softening,
        help="Gravitational Plummer softening in kpc",
    )
    parser.add_argument(
        "--loss_softening",
        type=float,
        default=Config.loss_softening,
        help="Softened-distance loss softening in kpc",
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
        "--accurate",
        type=int,
        choices=range(len(cl.FMM_ACCURACY_LEVELS)),
        default=Config.accuracy,
        metavar="LEVEL",
        help="FMM accuracy level",
    )
    parser.add_argument(
        "--redshift_space",
        action="store_true",
        help="Evaluate the final target loss in LOS redshift space",
    )
    parser.add_argument(
        "--fixed-masses",
        action="store_true",
        help="Infer one shared particle mass and disable particle merging",
    )
    parser.add_argument(
        "--no-merging",
        action="store_true",
        help="Infer individual particle masses without particle merging",
    )
    parser.add_argument("--steps", type=int, default=Config.max_steps)
    parser.add_argument(
        "--patience",
        type=int,
        default=Config.patience,
        help="Steps without sufficient loss improvement before stopping",
    )
    parser.add_argument(
        "--snapshot_every_epochs",
        type=int,
        default=Config.snapshot_every_epochs,
        help="Save pre-remap parameters every this many epochs",
    )
    parser.add_argument(
        "--epoch_steps",
        type=int,
        default=Config.epoch_steps,
        help="Optimizer steps between particle remappings",
    )
    parser.add_argument(
        "--adam",
        action="store_true",
        help="Use Adam instead of L-BFGS",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=Config.learning_rate,
    )
    parser.add_argument(
        "--minimum_learning_rate",
        type=float,
        default=Config.minimum_learning_rate,
    )
    parser.add_argument(
        "--learning_rate_reduction",
        type=float,
        default=Config.learning_rate_reduction,
    )
    parser.add_argument(
        "--target_weight",
        type=float,
        default=Config.target_weight,
    )
    parser.add_argument(
        "--mass_weight",
        type=float,
        default=Config.mass_weight,
    )
    parser.add_argument(
        "--hernquist_weight",
        type=float,
        default=Config.hernquist_weight,
    )
    parser.add_argument(
        "--target_parameter_seed",
        type=int,
        default=Config.target_parameter_seed,
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
        help="Initialize from best_parameters in a previous .npz run",
    )
    parser.add_argument(
        "--initial_particle_seed",
        type=int,
        default=Config.initial_particle_seed,
    )
    args = parser.parse_args()
    if args.patience < 1:
        parser.error("--patience must be at least 1")
    if args.snapshot_every_epochs < 1:
        parser.error("--snapshot_every_epochs must be at least 1")
    if args.epoch_steps < 1:
        parser.error("--epoch_steps must be at least 1")
    if args.learning_rate <= 0 or args.minimum_learning_rate <= 0:
        parser.error("learning rates must be positive")
    if args.sim_softening <= 0 or args.loss_softening <= 0:
        parser.error("softenings must be positive")
    if args.log10_mass_min >= args.log10_mass_max:
        parser.error("--log10_mass_min must be below --log10_mass_max")
    if not 0 < args.learning_rate_reduction < 1:
        parser.error("--learning_rate_reduction must lie between 0 and 1")
    if min(
        args.target_weight,
        args.mass_weight,
        args.hernquist_weight,
    ) < 0:
        parser.error("loss weights must be non-negative")
    run(
        Config(
            N=args.N,
            mode="ic" if args.ic else "sim",
            integration_time_gyr=args.integration_time_gyr,
            integration_steps=args.integration_steps,
            sim_softening=args.sim_softening,
            loss_softening=args.loss_softening,
            log10_mass_min=args.log10_mass_min,
            log10_mass_max=args.log10_mass_max,
            accuracy=args.accurate,
            redshift_space=args.redshift_space,
            fixed_masses=args.fixed_masses,
            merging=not args.no_merging and not args.fixed_masses,
            max_steps=args.steps,
            patience=args.patience,
            snapshot_every_epochs=args.snapshot_every_epochs,
            epoch_steps=args.epoch_steps,
            optimizer="adam" if args.adam else "lbfgs",
            learning_rate=args.learning_rate,
            minimum_learning_rate=args.minimum_learning_rate,
            learning_rate_reduction=args.learning_rate_reduction,
            target_weight=args.target_weight,
            mass_weight=args.mass_weight,
            hernquist_weight=args.hernquist_weight,
            target_parameter_seed=args.target_parameter_seed,
            initial_parameter_seed=args.initial_parameter_seed,
            initial_parameters_file=args.initial_parameters,
            initial_particle_seed=args.initial_particle_seed,
        )
    )
