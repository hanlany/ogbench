from __future__ import annotations

import os
import pickle
import random
import time
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from absl import app, flags

try:
    from impls.utils.datasets import Dataset
    from impls.utils.flax_utils import TrainState
    from impls.utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb
    from latent.train.autoencoder import reconstruction_metrics
    from latent.train.dynamics import LatentDynamics
    from latent.train.train_autoencoder import (
        DEFAULT_DATASET_DIR,
        NormalizationStats,
        average_metrics,
        config_to_dict,
        create_batch_sampler,
        create_validation_batches,
        normalize_observations,
        parse_dims,
        to_float_dict,
        validate_observations,
        write_json,
    )
    from latent.validate.evaluate_autoencoder import load_checkpoint as load_autoencoder_checkpoint
    from latent.validate.evaluate_autoencoder import restore_autoencoder_state
    from ogbench import make_env_and_datasets
except ImportError as e:  # pragma: no cover - exercised by direct script misuse.
    raise ImportError(
        'Could not import OGBench dynamics training dependencies. Run from the repository root with '
        '`PYTHONPATH=. python latent/train/train_dynamics.py ...`, or install the repo in editable mode.'
    ) from e


CHECKPOINT_SCHEMA_VERSION = 1
FLAGS = flags.FLAGS

_FLAG_KWARGS = {'allow_override': True}

flags.DEFINE_string('run_group', 'LatentDynamics', 'Run group.', **_FLAG_KWARGS)
flags.DEFINE_integer('seed', 0, 'Random seed.', **_FLAG_KWARGS)
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.', **_FLAG_KWARGS)
flags.DEFINE_string('dataset_dir', DEFAULT_DATASET_DIR, 'Directory to save/load OGBench datasets.', **_FLAG_KWARGS)
flags.DEFINE_string('dataset_path', None, 'Optional path to the training dataset file.', **_FLAG_KWARGS)
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.', **_FLAG_KWARGS)
flags.DEFINE_string(
    'restore_path',
    None,
    'Path to a checkpoint file or directory containing params_<step>.pkl.',
    **_FLAG_KWARGS,
)
flags.DEFINE_integer(
    'restore_step',
    None,
    'Checkpoint step to restore when restore_path is a directory.',
    **_FLAG_KWARGS,
)
flags.DEFINE_enum('wandb_mode', 'online', ['online', 'offline', 'disabled'], 'Weights & Biases mode.', **_FLAG_KWARGS)
flags.DEFINE_string('ae_checkpoint_path', None, 'Path to a trained autoencoder checkpoint.', **_FLAG_KWARGS)

flags.DEFINE_list(
    'hidden_dims', ['512', '512'], 'Comma-separated hidden dimensions for the dynamics MLP.', **_FLAG_KWARGS
)
flags.DEFINE_enum('activation', 'gelu', ['elu', 'gelu', 'relu', 'swish', 'tanh'], 'MLP activation.', **_FLAG_KWARGS)
flags.DEFINE_bool('layer_norm', False, 'Whether to use layer normalization after hidden dense layers.', **_FLAG_KWARGS)
flags.DEFINE_float('dropout_rate', 0.0, 'Dropout rate for hidden layers.', **_FLAG_KWARGS)
flags.DEFINE_enum(
    'prediction_type',
    'absolute',
    ['absolute', 'residual'],
    'Predict an absolute next latent or a residual added to the current latent.',
    **_FLAG_KWARGS,
)
flags.DEFINE_float('lr', 3e-4, 'Learning rate.', **_FLAG_KWARGS)
flags.DEFINE_integer('batch_size', 1024, 'Batch size.', **_FLAG_KWARGS)
flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.', **_FLAG_KWARGS)
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.', **_FLAG_KWARGS)
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.', **_FLAG_KWARGS)
flags.DEFINE_integer(
    'prefetch_batches', 2, 'Number of host batches to prefetch; 0 disables prefetching.', **_FLAG_KWARGS
)
flags.DEFINE_integer(
    'validation_batches',
    16,
    'Number of fixed validation batches to evaluate. Set to 0 for a deterministic full validation sweep.',
    **_FLAG_KWARGS,
)
flags.DEFINE_integer('validation_batch_size', None, 'Validation batch size. Defaults to batch_size.', **_FLAG_KWARGS)
flags.DEFINE_integer(
    'encoder_batch_size',
    8192,
    'Batch size for offline AE encoding before dynamics training.',
    **_FLAG_KWARGS,
)
flags.DEFINE_list(
    'rollout_horizons',
    ['1'],
    'Recursive training horizons. A single horizon 1 preserves one-step training.',
    **_FLAG_KWARGS,
)
flags.DEFINE_integer(
    'rollout_horizon1_steps',
    0,
    'Curriculum steps restricted to horizon 1 before the horizon-5 phase.',
    **_FLAG_KWARGS,
)
flags.DEFINE_integer(
    'rollout_horizon5_steps',
    0,
    'End step of the horizon-5 curriculum phase; later steps sample rollout_horizons.',
    **_FLAG_KWARGS,
)

flags.DEFINE_float('recon_weight', 1.0, 'Weight for decoded next-observation reconstruction loss.', **_FLAG_KWARGS)
flags.DEFINE_float('latent_weight', 1.0, 'Weight for latent prediction loss.', **_FLAG_KWARGS)
flags.DEFINE_float('xy_weight', 0.0, 'Weight for decoded next-position loss in original units.', **_FLAG_KWARGS)
flags.DEFINE_float(
    'xy_tolerance',
    1.0,
    'Original-unit position tolerance used to scale the decoded XY loss.',
    **_FLAG_KWARGS,
)
flags.DEFINE_enum(
    'latent_loss_mode',
    'l2',
    ['l2', 'gramian'],
    'Latent objective: pure squared L2 error or the annealed controllability-Gramian loss.',
    **_FLAG_KWARGS,
)
flags.DEFINE_integer(
    'gramian_warmup_steps',
    100000,
    'Steps over which to anneal from L2 to Gramian latent loss.',
    **_FLAG_KWARGS,
)
flags.DEFINE_float(
    'gramian_diag_eps', 1e-4, 'Diagonal regularizer added to each controllability Gramian.', **_FLAG_KWARGS
)
flags.DEFINE_bool(
    'differentiate_gramian',
    False,
    'Whether gradients should flow through the Gramian matrix.',
    **_FLAG_KWARGS,
)


@dataclass(frozen=True)
class DynamicsModelConfig:
    hidden_dims: tuple[int, ...]
    latent_dim: int
    action_dim: int
    activation: str
    layer_norm: bool
    dropout_rate: float
    prediction_type: str = 'absolute'


@dataclass(frozen=True)
class DynamicsTrainingConfig:
    run_group: str
    seed: int
    env_name: str
    dataset_dir: str
    dataset_path: str | None
    save_dir: str
    restore_path: str | None
    restore_step: int | None
    wandb_mode: str
    ae_checkpoint_path: str | None
    lr: float
    batch_size: int
    train_steps: int
    log_interval: int
    save_interval: int
    prefetch_batches: int
    validation_batches: int
    validation_batch_size: int
    encoder_batch_size: int
    rollout_horizons: tuple[int, ...] = (1,)
    rollout_horizon1_steps: int = 0
    rollout_horizon5_steps: int = 0


@dataclass(frozen=True)
class DynamicsLossConfig:
    recon_weight: float
    latent_weight: float
    gramian_warmup_steps: int
    gramian_diag_eps: float
    differentiate_gramian: bool
    latent_loss_mode: str = 'l2'
    xy_weight: float = 0.0
    xy_tolerance: float = 1.0


def create_configs(latent_dim=-1, action_dim=-1):
    """Create explicit configs from CLI flags."""
    validation_batch_size = FLAGS.validation_batch_size or FLAGS.batch_size
    training_config = DynamicsTrainingConfig(
        run_group=FLAGS.run_group,
        seed=FLAGS.seed,
        env_name=FLAGS.env_name,
        dataset_dir=FLAGS.dataset_dir,
        dataset_path=FLAGS.dataset_path,
        save_dir=FLAGS.save_dir,
        restore_path=FLAGS.restore_path,
        restore_step=FLAGS.restore_step,
        wandb_mode=FLAGS.wandb_mode,
        ae_checkpoint_path=FLAGS.ae_checkpoint_path,
        lr=FLAGS.lr,
        batch_size=FLAGS.batch_size,
        train_steps=FLAGS.train_steps,
        log_interval=FLAGS.log_interval,
        save_interval=FLAGS.save_interval,
        prefetch_batches=FLAGS.prefetch_batches,
        validation_batches=FLAGS.validation_batches,
        validation_batch_size=validation_batch_size,
        encoder_batch_size=FLAGS.encoder_batch_size,
        rollout_horizons=parse_dims(FLAGS.rollout_horizons, 'rollout_horizons'),
        rollout_horizon1_steps=FLAGS.rollout_horizon1_steps,
        rollout_horizon5_steps=FLAGS.rollout_horizon5_steps,
    )
    model_config = DynamicsModelConfig(
        hidden_dims=parse_dims(FLAGS.hidden_dims, 'hidden_dims'),
        latent_dim=latent_dim,
        action_dim=action_dim,
        activation=FLAGS.activation,
        layer_norm=FLAGS.layer_norm,
        dropout_rate=FLAGS.dropout_rate,
        prediction_type=FLAGS.prediction_type,
    )
    loss_config = DynamicsLossConfig(
        recon_weight=FLAGS.recon_weight,
        latent_weight=FLAGS.latent_weight,
        gramian_warmup_steps=FLAGS.gramian_warmup_steps,
        gramian_diag_eps=FLAGS.gramian_diag_eps,
        differentiate_gramian=FLAGS.differentiate_gramian,
        latent_loss_mode=FLAGS.latent_loss_mode,
        xy_weight=FLAGS.xy_weight,
        xy_tolerance=FLAGS.xy_tolerance,
    )
    return training_config, model_config, loss_config


def validate_training_config(config: DynamicsTrainingConfig):
    """Fail fast on invalid training settings."""
    if config.ae_checkpoint_path is None:
        raise ValueError('ae_checkpoint_path is required.')
    positive_ints = {
        'batch_size': config.batch_size,
        'train_steps': config.train_steps,
        'log_interval': config.log_interval,
        'save_interval': config.save_interval,
        'validation_batch_size': config.validation_batch_size,
        'encoder_batch_size': config.encoder_batch_size,
    }
    for name, value in positive_ints.items():
        if value <= 0:
            raise ValueError(f'{name} must be positive, got {value}.')
    if config.validation_batches < 0:
        raise ValueError(f'validation_batches must be non-negative, got {config.validation_batches}.')
    if config.prefetch_batches < 0:
        raise ValueError(f'prefetch_batches must be non-negative, got {config.prefetch_batches}.')
    if config.lr <= 0:
        raise ValueError(f'lr must be positive, got {config.lr}.')
    if not config.rollout_horizons:
        raise ValueError('rollout_horizons must not be empty.')
    if tuple(sorted(set(config.rollout_horizons))) != config.rollout_horizons:
        raise ValueError(f'rollout_horizons must be strictly increasing and unique, got {config.rollout_horizons}.')
    if config.rollout_horizons[0] != 1:
        raise ValueError(f'rollout_horizons must begin with 1, got {config.rollout_horizons}.')
    if config.rollout_horizon1_steps < 0:
        raise ValueError(f'rollout_horizon1_steps must be non-negative, got {config.rollout_horizon1_steps}.')
    if config.rollout_horizon5_steps < config.rollout_horizon1_steps:
        raise ValueError(
            'rollout_horizon5_steps must be greater than or equal to rollout_horizon1_steps, got '
            f'{config.rollout_horizon5_steps} < {config.rollout_horizon1_steps}.'
        )
    if config.rollout_horizon5_steps > config.train_steps:
        raise ValueError(
            f'rollout_horizon5_steps must not exceed train_steps, got '
            f'{config.rollout_horizon5_steps} > {config.train_steps}.'
        )
    if config.rollout_horizon5_steps > config.rollout_horizon1_steps and 5 not in config.rollout_horizons:
        raise ValueError('rollout_horizons must contain 5 when the horizon-5 curriculum phase is enabled.')


def validate_model_config(config: DynamicsModelConfig):
    """Fail fast on invalid dynamics model settings."""
    if config.latent_dim <= 0:
        raise ValueError(f'latent_dim must be positive, got {config.latent_dim}.')
    if config.action_dim <= 0:
        raise ValueError(f'action_dim must be positive, got {config.action_dim}.')
    if config.dropout_rate < 0 or config.dropout_rate >= 1:
        raise ValueError(f'dropout_rate must be in [0, 1), got {config.dropout_rate}.')
    if config.prediction_type not in ('absolute', 'residual'):
        raise ValueError(f'prediction_type must be one of (absolute, residual), got {config.prediction_type!r}.')


def validate_loss_config(config: DynamicsLossConfig):
    """Fail fast on invalid dynamics loss settings."""
    if config.recon_weight < 0:
        raise ValueError(f'recon_weight must be non-negative, got {config.recon_weight}.')
    if config.latent_weight < 0:
        raise ValueError(f'latent_weight must be non-negative, got {config.latent_weight}.')
    if config.xy_weight < 0:
        raise ValueError(f'xy_weight must be non-negative, got {config.xy_weight}.')
    if config.xy_tolerance <= 0:
        raise ValueError(f'xy_tolerance must be positive, got {config.xy_tolerance}.')
    if config.recon_weight == 0 and config.latent_weight == 0 and config.xy_weight == 0:
        raise ValueError('At least one of recon_weight, latent_weight, or xy_weight must be positive.')
    if config.latent_loss_mode not in ('l2', 'gramian'):
        raise ValueError(f'latent_loss_mode must be one of (l2, gramian), got {config.latent_loss_mode!r}.')
    if config.gramian_warmup_steps < 0:
        raise ValueError(f'gramian_warmup_steps must be non-negative, got {config.gramian_warmup_steps}.')
    if config.gramian_diag_eps <= 0:
        raise ValueError(f'gramian_diag_eps must be positive, got {config.gramian_diag_eps}.')


def validate_actions(actions, split_name):
    """Validate and cast vector actions."""
    actions = np.asarray(actions)
    if actions.ndim != 2:
        raise ValueError(
            f'Only vector actions are supported for now. '
            f'Expected {split_name} actions with shape (N, action_dim), got {actions.shape}.'
        )
    if len(actions) == 0:
        raise ValueError(f'Expected non-empty {split_name} actions.')
    return actions.astype(np.float32, copy=False)


def make_transition_dataset(raw_dataset, stats: NormalizationStats, split_name: str):
    """Create normalized transition data with explicit next observations."""
    for key in ('observations', 'actions', 'next_observations'):
        if key not in raw_dataset:
            raise ValueError(f'{split_name} dataset is missing required key {key}.')

    raw_observations = validate_observations(raw_dataset['observations'], f'{split_name} observations')
    raw_next_observations = validate_observations(raw_dataset['next_observations'], f'{split_name} next_observations')
    actions = validate_actions(raw_dataset['actions'], split_name)

    if len(raw_observations) != len(raw_next_observations) or len(raw_observations) != len(actions):
        raise ValueError(
            f'Expected aligned {split_name} transitions, got observations={len(raw_observations)}, '
            f'next_observations={len(raw_next_observations)}, actions={len(actions)}.'
        )
    if raw_observations.shape[-1] != raw_next_observations.shape[-1]:
        raise ValueError(
            f'Expected matching observation dimensions for {split_name}, got '
            f'{raw_observations.shape[-1]} and {raw_next_observations.shape[-1]}.'
        )
    if stats.mean.shape != (raw_observations.shape[-1],) or stats.std.shape != (raw_observations.shape[-1],):
        raise ValueError(
            f'Normalization stats shape must match obs_dim={raw_observations.shape[-1]}, '
            f'got mean={stats.mean.shape}, std={stats.std.shape}.'
        )

    fields = dict(
        observations=normalize_observations(raw_observations, stats),
        actions=actions,
        next_observations=normalize_observations(raw_next_observations, stats),
        raw_next_observations=raw_next_observations,
    )
    if 'terminals' in raw_dataset:
        terminals = np.asarray(raw_dataset['terminals'], dtype=np.float32).reshape(-1)
        if len(terminals) != len(raw_observations):
            raise ValueError(
                f'Expected aligned {split_name} terminals, got terminals={len(terminals)} and '
                f'observations={len(raw_observations)}.'
            )
        fields['terminals'] = terminals
    return Dataset.create(**fields)


def load_transition_datasets(config: DynamicsTrainingConfig, stats: NormalizationStats):
    """Load OGBench transition datasets using the AE checkpoint normalization."""
    train_dataset, val_dataset = make_env_and_datasets(
        config.env_name,
        dataset_dir=config.dataset_dir,
        dataset_path=config.dataset_path,
        compact_dataset=False,
        dataset_only=True,
    )
    train_dataset = make_transition_dataset(train_dataset, stats, 'training')
    if val_dataset is None:
        return train_dataset, None
    return train_dataset, make_transition_dataset(val_dataset, stats, 'validation')


def predict_latents(ae_state, observations, batch_size: int):
    """Encode observations with a frozen AE in deterministic batches."""
    latents = []
    for start in range(0, len(observations), batch_size):
        batch = jnp.asarray(observations[start : start + batch_size], dtype=jnp.float32)
        batch_latents = ae_state(batch, method='encode', deterministic=True)
        latents.append(np.asarray(jax.device_get(batch_latents), dtype=np.float32))
    return np.concatenate(latents, axis=0)


def encode_transition_dataset(dataset, ae_state, batch_size: int):
    """Cache normalized transitions in latent space for dynamics training."""
    latents = predict_latents(ae_state, dataset['observations'], batch_size)
    next_latents = predict_latents(ae_state, dataset['next_observations'], batch_size)
    fields = dict(
        observations=latents,
        latents=latents,
        actions=np.asarray(dataset['actions'], dtype=np.float32),
        next_observations=np.asarray(dataset['next_observations'], dtype=np.float32),
        next_latents=next_latents,
        raw_next_observations=np.asarray(dataset['raw_next_observations'], dtype=np.float32),
    )
    if 'terminals' in dataset:
        fields['terminals'] = np.asarray(dataset['terminals'], dtype=np.float32)
    return Dataset.create(**fields)


class RolloutDataset:
    """Indexed episode-safe recursive rollout views over encoded transitions."""

    def __init__(self, dataset, horizon: int):
        if horizon <= 0:
            raise ValueError(f'horizon must be positive, got {horizon}.')
        if 'terminals' not in dataset:
            raise ValueError('Rollout training requires terminal markers.')
        terminals = np.asarray(dataset['terminals']).reshape(-1)
        ends = np.flatnonzero(terminals > 0) + 1
        if len(ends) == 0 or ends[-1] != dataset.size:
            raise ValueError('Rollout training data must mark the final transition as terminal.')
        starts = np.concatenate([[0], ends[:-1]])
        candidates = [np.arange(start, end - horizon + 1, dtype=np.int64) for start, end in zip(starts, ends)]
        candidates = [indices for indices in candidates if len(indices) > 0]
        if not candidates:
            raise ValueError(f'No episodes are long enough for rollout horizon {horizon}.')

        self.dataset = dataset
        self.horizon = horizon
        self.valid_starts = np.concatenate(candidates)
        self.size = len(self.valid_starts)

    def get_subset(self, idxs):
        """Gather rollout windows using positions within valid_starts."""
        starts = self.valid_starts[np.asarray(idxs)]
        offsets = np.arange(self.horizon, dtype=np.int64)
        indices = starts[:, None] + offsets[None, :]
        return {
            'latents': self.dataset['latents'][starts],
            'actions': self.dataset['actions'][indices],
            'next_latents': self.dataset['next_latents'][indices],
            'next_observations': self.dataset['next_observations'][indices],
            'raw_next_observations': self.dataset['raw_next_observations'][indices],
        }

    def sample(self, batch_size, idxs=None):
        """Sample episode-safe rollout windows."""
        if idxs is None:
            idxs = np.random.randint(self.size, size=batch_size)
        return self.get_subset(idxs)


def rollout_horizon_for_step(step: int, config: DynamicsTrainingConfig) -> int:
    """Select the deterministic curriculum horizon for a training step."""
    if len(config.rollout_horizons) == 1:
        return config.rollout_horizons[0]
    if step <= config.rollout_horizon1_steps:
        return 1
    if step <= config.rollout_horizon5_steps:
        return 5
    rng = np.random.default_rng(config.seed + step)
    return int(rng.choice(config.rollout_horizons))


def create_train_state(seed, model_config: DynamicsModelConfig, training_config: DynamicsTrainingConfig):
    """Initialize dynamics train state."""
    model_def = LatentDynamics(
        hidden_dims=model_config.hidden_dims,
        latent_dim=model_config.latent_dim,
        activation=model_config.activation,
        layer_norm=model_config.layer_norm,
        dropout_rate=model_config.dropout_rate,
        prediction_type=model_config.prediction_type,
    )
    rng = jax.random.PRNGKey(seed)
    params = model_def.init(
        rng,
        jnp.zeros((1, model_config.latent_dim), dtype=jnp.float32),
        jnp.zeros((1, model_config.action_dim), dtype=jnp.float32),
    )['params']
    tx = optax.adam(training_config.lr)
    return TrainState.create(model_def, params, tx=tx)


def compute_gramian_alpha(step, warmup_steps: int):
    """Return the interpolation weight from latent L2 loss to Gramian loss."""
    if warmup_steps <= 0:
        return jnp.asarray(1.0, dtype=jnp.float32)
    return jnp.clip(jnp.asarray(step, dtype=jnp.float32) / float(warmup_steps), 0.0, 1.0)


def apply_dynamics(state, params, latents, actions, deterministic: bool, rng=None):
    """Apply dynamics params with optional dropout RNG."""
    if rng is None:
        return state(latents, actions, params=params, deterministic=deterministic)
    return state(latents, actions, params=params, deterministic=deterministic, rngs={'dropout': rng})


def apply_dynamics_rollout(state, params, initial_latents, actions, deterministic: bool, rng=None):
    """Recursively propagate predicted latents without teacher forcing."""
    time_major_actions = jnp.swapaxes(actions, 0, 1)
    if rng is None:
        step_rngs = None
    else:
        step_rngs = jax.random.split(rng, actions.shape[1])

    def deterministic_step(latents, step_actions):
        next_latents = apply_dynamics(
            state,
            params,
            latents,
            step_actions,
            deterministic=deterministic,
        )
        return next_latents, next_latents

    def stochastic_step(latents, inputs):
        step_actions, step_rng = inputs
        next_latents = apply_dynamics(
            state,
            params,
            latents,
            step_actions,
            deterministic=deterministic,
            rng=step_rng,
        )
        return next_latents, next_latents

    if step_rngs is None:
        _, predictions = jax.lax.scan(deterministic_step, initial_latents, time_major_actions)
    else:
        _, predictions = jax.lax.scan(stochastic_step, initial_latents, (time_major_actions, step_rngs))
    return jnp.swapaxes(predictions, 0, 1)


def controllability_gramian_metrics(
    state,
    params,
    latents,
    actions,
    errors,
    gramian_diag_eps: float,
    differentiate_gramian: bool,
):
    """Compute the paper's weighted controllability Gramian latent loss."""

    def single_jacobians(latent, action):
        def predict_from_latent(latent_input):
            return state(latent_input[None, :], action[None, :], params=params, deterministic=True)[0]

        def predict_from_action(action_input):
            return state(latent[None, :], action_input[None, :], params=params, deterministic=True)[0]

        return jax.jacrev(predict_from_latent)(latent), jax.jacrev(predict_from_action)(action)

    a_mats, b_mats = jax.vmap(single_jacobians)(latents, actions)
    ab_mats = jnp.matmul(a_mats, b_mats)
    gramians = jnp.matmul(ab_mats, jnp.swapaxes(ab_mats, -1, -2))
    gramians = 0.5 * (gramians + jnp.swapaxes(gramians, -1, -2))

    eye = jnp.eye(errors.shape[-1], dtype=errors.dtype)
    regularized = gramians + jnp.asarray(gramian_diag_eps, dtype=errors.dtype) * eye
    if not differentiate_gramian:
        regularized = jax.lax.stop_gradient(regularized)

    solved = jnp.linalg.solve(regularized, errors[..., None])[..., 0]
    energy_per_example = jnp.sum(errors * solved, axis=-1)
    eigvals = jnp.linalg.eigvalsh(regularized)
    eig_min = jnp.mean(jnp.min(eigvals, axis=-1))
    eig_max = jnp.mean(jnp.max(eigvals, axis=-1))
    condition = jnp.mean(jnp.max(eigvals, axis=-1) / jnp.maximum(jnp.min(eigvals, axis=-1), 1e-12))

    return {
        'gramian/energy': jnp.mean(energy_per_example),
        'gramian/eig_min': eig_min,
        'gramian/eig_max': eig_max,
        'gramian/condition': condition,
    }


def dynamics_loss(
    state,
    ae_state,
    params,
    batch,
    mean,
    std,
    alpha,
    recon_weight,
    latent_weight,
    xy_weight,
    xy_tolerance,
    gramian_diag_eps,
    differentiate_gramian,
    latent_loss_mode,
    deterministic,
    rng=None,
):
    """Compute dynamics loss and scalar diagnostics."""
    is_rollout = batch['actions'].ndim == 3
    if is_rollout:
        if latent_loss_mode != 'l2':
            raise ValueError('Recursive rollout training currently supports latent_loss_mode=l2 only.')
        pred_latents = apply_dynamics_rollout(
            state,
            params,
            batch['latents'],
            batch['actions'],
            deterministic=deterministic,
            rng=rng,
        )
        rollout_horizon = batch['actions'].shape[1]
    else:
        pred_latents = apply_dynamics(
            state,
            params,
            batch['latents'],
            batch['actions'],
            deterministic=deterministic,
            rng=rng,
        )
        rollout_horizon = 1
    pred_next_observations = ae_state(pred_latents, method='decode', deterministic=True)

    recon = reconstruction_metrics(batch['next_observations'], pred_next_observations)
    raw_pred_next_observations = pred_next_observations * std + mean
    raw_recon = reconstruction_metrics(batch['raw_next_observations'], raw_pred_next_observations)
    raw_xy_errors = raw_pred_next_observations[..., :2] - batch['raw_next_observations'][..., :2]
    xy_mse = jnp.mean(raw_xy_errors**2)
    xy_loss = xy_mse / jnp.asarray(xy_tolerance, dtype=xy_mse.dtype) ** 2

    errors = batch['next_latents'] - pred_latents
    latent_l2 = jnp.mean(jnp.sum(errors**2, axis=-1))
    latent_mse = jnp.mean(errors**2)
    latent_rmse = jnp.sqrt(latent_mse)
    if latent_loss_mode == 'l2':
        # Keep the logging schema stable without paying for per-example Jacobians,
        # eigendecompositions, or linear solves in the pure-L2 baseline.
        alpha = jnp.asarray(0.0, dtype=latent_l2.dtype)
        gramian = {
            'gramian/energy': jnp.asarray(0.0, dtype=latent_l2.dtype),
            'gramian/eig_min': jnp.asarray(0.0, dtype=latent_l2.dtype),
            'gramian/eig_max': jnp.asarray(0.0, dtype=latent_l2.dtype),
            'gramian/condition': jnp.asarray(0.0, dtype=latent_l2.dtype),
        }
        latent_loss = latent_l2
    elif latent_loss_mode == 'gramian':
        gramian = controllability_gramian_metrics(
            state,
            params,
            batch['latents'],
            batch['actions'],
            errors,
            gramian_diag_eps,
            differentiate_gramian,
        )
        latent_loss = (1.0 - alpha) * latent_l2 + alpha * gramian['gramian/energy']
    else:
        raise ValueError(f'Unsupported latent_loss_mode {latent_loss_mode!r}.')
    loss = recon_weight * recon['mse'] + latent_weight * latent_loss + xy_weight * xy_loss
    metrics = {
        'loss': loss,
        'latent/loss': latent_loss,
        'latent/l2': latent_l2,
        'latent/mse': latent_mse,
        'latent/rmse': latent_rmse,
        'latent/error_norm': jnp.mean(jnp.linalg.norm(errors, axis=-1)),
        'latent/pred_norm': jnp.mean(jnp.linalg.norm(pred_latents, axis=-1)),
        'xy/loss': xy_loss,
        'xy/mse': xy_mse,
        'xy/rmse': jnp.sqrt(xy_mse),
        'xy/mean_error': jnp.mean(jnp.linalg.norm(raw_xy_errors, axis=-1)),
        'rollout/horizon': jnp.asarray(rollout_horizon, dtype=latent_l2.dtype),
        'gramian/alpha': alpha,
    }
    metrics.update({f'recon/{key}': value for key, value in recon.items()})
    metrics.update({f'original/{key}': value for key, value in raw_recon.items()})
    metrics.update(gramian)
    return metrics


@partial(jax.jit, static_argnames=('gramian_warmup_steps', 'differentiate_gramian', 'latent_loss_mode'))
def train_step(
    state,
    ae_state,
    batch,
    mean,
    std,
    rng,
    recon_weight,
    latent_weight,
    xy_weight,
    xy_tolerance,
    gramian_diag_eps,
    gramian_warmup_steps: int,
    differentiate_gramian: bool,
    latent_loss_mode: str,
):
    """Run one latent dynamics update."""
    alpha = compute_gramian_alpha(state.step, gramian_warmup_steps)

    def loss_fn(params):
        metrics = dynamics_loss(
            state,
            ae_state,
            params,
            batch,
            mean,
            std,
            alpha,
            recon_weight,
            latent_weight,
            xy_weight,
            xy_tolerance,
            gramian_diag_eps,
            differentiate_gramian,
            latent_loss_mode,
            deterministic=False,
            rng=rng,
        )
        return metrics['loss'], metrics

    new_state, metrics = state.apply_loss_fn(loss_fn)
    param_norms = jax.tree_util.tree_map(jnp.linalg.norm, new_state.params)
    metrics['param/norm'] = jnp.linalg.norm(jnp.array(jax.tree_util.tree_leaves(param_norms)))
    return new_state, metrics


@partial(jax.jit, static_argnames=('gramian_warmup_steps', 'differentiate_gramian', 'latent_loss_mode'))
def eval_step(
    state,
    ae_state,
    batch,
    mean,
    std,
    recon_weight,
    latent_weight,
    xy_weight,
    xy_tolerance,
    gramian_diag_eps,
    gramian_warmup_steps: int,
    differentiate_gramian: bool,
    latent_loss_mode: str,
):
    """Evaluate latent dynamics without updating parameters."""
    alpha = compute_gramian_alpha(state.step, gramian_warmup_steps)
    return dynamics_loss(
        state,
        ae_state,
        state.params,
        batch,
        mean,
        std,
        alpha,
        recon_weight,
        latent_weight,
        xy_weight,
        xy_tolerance,
        gramian_diag_eps,
        differentiate_gramian,
        latent_loss_mode,
        deterministic=True,
    )


def save_checkpoint(
    state,
    save_dir,
    step,
    training_config,
    model_config,
    loss_config,
    ae_checkpoint_path,
    ae_model_config,
    ae_training_config,
    normalization_stats,
    flag_dict,
):
    """Save dynamics train state plus AE metadata needed for downstream use."""
    checkpoint = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'step': step,
        'flags': flag_dict,
        'training_config': config_to_dict(training_config),
        'model_config': config_to_dict(model_config),
        'loss_config': config_to_dict(loss_config),
        'optimizer_config': {
            'name': 'adam',
            'learning_rate': training_config.lr,
        },
        'ae_checkpoint_path': str(ae_checkpoint_path),
        'ae_model_config': config_to_dict(ae_model_config),
        'ae_training_config': config_to_dict(ae_training_config),
        'normalization': normalization_stats.to_dict(),
        'latent_dim': model_config.latent_dim,
        'action_dim': model_config.action_dim,
        'dynamics': flax.serialization.to_state_dict(state),
        'params': flax.serialization.to_state_dict(state.params),
    }
    save_path = os.path.join(save_dir, f'params_{step}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f'Saved to {save_path}')


def resolve_checkpoint_path(restore_path, restore_step):
    """Resolve a dynamics checkpoint file from either a file or checkpoint directory."""
    path = Path(restore_path).expanduser()
    if path.is_file():
        return path
    if restore_step is None:
        raise ValueError('restore_step is required when restore_path is a directory.')
    return path / f'params_{restore_step}.pkl'


def restore_checkpoint(
    state,
    restore_path,
    restore_step,
    training_config=None,
    model_config=None,
    loss_config=None,
):
    """Restore a dynamics state and reject incompatible objective/curriculum settings."""
    checkpoint_path = resolve_checkpoint_path(restore_path, restore_step)
    with checkpoint_path.open('rb') as f:
        checkpoint = pickle.load(f)
    if 'dynamics' not in checkpoint:
        raise ValueError(f'Checkpoint {checkpoint_path} does not contain a dynamics train state.')

    expected_sections = {
        'training_config': (
            training_config,
            {
                'rollout_horizons': (1,),
                'rollout_horizon1_steps': 0,
                'rollout_horizon5_steps': 0,
            },
        ),
        'model_config': (model_config, {'prediction_type': 'absolute'}),
        'loss_config': (
            loss_config,
            {
                'latent_loss_mode': 'gramian',
                'xy_weight': 0.0,
                'xy_tolerance': 1.0,
            },
        ),
    }
    for section, (expected_config, defaults) in expected_sections.items():
        if expected_config is None:
            continue
        saved_config = checkpoint.get(section, {})
        for key, default in defaults.items():
            saved_value = saved_config.get(key, default)
            expected_value = getattr(expected_config, key)
            if key == 'rollout_horizons':
                saved_value = tuple(saved_value)
                expected_value = tuple(expected_value)
            if saved_value != expected_value:
                raise ValueError(
                    f'Restore configuration mismatch for {section}.{key}: '
                    f'checkpoint={saved_value!r}, requested={expected_value!r}.'
                )

    state = flax.serialization.from_state_dict(state, checkpoint['dynamics'])
    restored_step = checkpoint.get('step', restore_step)
    print(f'Restored from {checkpoint_path}')
    return state, restored_step


def prepare_save_dir(config):
    """Create the final wandb-style save directory and validate it is writable."""
    exp_name = get_exp_name(config.seed)
    setup_wandb(project='OGBench', group=config.run_group, name=exp_name, mode=config.wandb_mode)
    save_dir = os.path.join(config.save_dir, wandb.run.project, config.run_group, exp_name)
    os.makedirs(save_dir, exist_ok=True)
    test_path = os.path.join(save_dir, '.write_test')
    with open(test_path, 'w') as f:
        f.write('ok')
    os.remove(test_path)
    return save_dir


def load_autoencoder_artifacts(checkpoint_path):
    """Load the frozen AE state, configs, and normalization stats."""
    checkpoint, ae_model_config, ae_training_config, stats = load_autoencoder_checkpoint(checkpoint_path)
    ae_state = restore_autoencoder_state(checkpoint, ae_model_config, ae_training_config)
    return checkpoint, ae_state, ae_model_config, ae_training_config, stats


def run_training(training_config, model_config, loss_config):
    """Run the full latent dynamics training workflow."""
    validate_training_config(training_config)
    validate_loss_config(loss_config)
    if training_config.rollout_horizons != (1,) and loss_config.latent_loss_mode != 'l2':
        raise ValueError('Recursive rollout training currently supports latent_loss_mode=l2 only.')

    random.seed(training_config.seed)
    np.random.seed(training_config.seed)

    _, ae_state, ae_model_config, ae_training_config, normalization_stats = load_autoencoder_artifacts(
        training_config.ae_checkpoint_path
    )
    train_dataset, val_dataset = load_transition_datasets(training_config, normalization_stats)
    action_dim = train_dataset['actions'].shape[-1]
    model_config = replace(
        model_config,
        latent_dim=ae_model_config.latent_dim,
        action_dim=action_dim,
    )
    validate_model_config(model_config)

    print('Encoding training transitions with frozen AE...')
    train_dataset = encode_transition_dataset(train_dataset, ae_state, training_config.encoder_batch_size)
    if val_dataset is not None:
        print('Encoding validation transitions with frozen AE...')
        val_dataset = encode_transition_dataset(val_dataset, ae_state, training_config.encoder_batch_size)

    save_dir = prepare_save_dir(training_config)
    training_config = replace(training_config, save_dir=save_dir)

    flag_dict = get_flag_dict()
    metadata = {
        'flags': flag_dict,
        'training_config': config_to_dict(training_config),
        'model_config': config_to_dict(model_config),
        'loss_config': config_to_dict(loss_config),
        'ae_checkpoint_path': training_config.ae_checkpoint_path,
        'ae_model_config': config_to_dict(ae_model_config),
        'normalization': normalization_stats.to_dict(),
    }
    write_json(os.path.join(training_config.save_dir, 'flags.json'), metadata)

    state = create_train_state(training_config.seed, model_config, training_config)
    start_step = 1
    if training_config.restore_path is not None:
        state, restored_step = restore_checkpoint(
            state,
            training_config.restore_path,
            training_config.restore_step,
            training_config=training_config,
            model_config=model_config,
            loss_config=loss_config,
        )
        start_step = int(restored_step) + 1

    rollout_training = training_config.rollout_horizons != (1,)
    mean = jnp.asarray(normalization_stats.mean)
    std = jnp.asarray(normalization_stats.std)
    prefetchers = []

    if rollout_training:
        train_rollouts = {
            horizon: RolloutDataset(train_dataset, horizon) for horizon in training_config.rollout_horizons
        }
        val_rollouts = (
            {horizon: RolloutDataset(val_dataset, horizon) for horizon in training_config.rollout_horizons}
            if val_dataset is not None
            else {}
        )
        sample_train_batches = {}
        for horizon, rollout_dataset in train_rollouts.items():
            prefetcher, sampler = create_batch_sampler(
                rollout_dataset,
                training_config.batch_size,
                training_config.prefetch_batches,
            )
            if prefetcher is not None:
                prefetchers.append(prefetcher)
            sample_train_batches[horizon] = sampler
        val_batches = {
            horizon: create_validation_batches(
                rollout_dataset,
                training_config.validation_batch_size,
                training_config.validation_batches,
                training_config.seed + 1,
            )
            for horizon, rollout_dataset in val_rollouts.items()
        }
    else:
        val_batches = create_validation_batches(
            val_dataset,
            training_config.validation_batch_size,
            training_config.validation_batches,
            training_config.seed + 1,
        )
        prefetcher, sample_train_batch = create_batch_sampler(
            train_dataset,
            training_config.batch_size,
            training_config.prefetch_batches,
        )
        if prefetcher is not None:
            prefetchers.append(prefetcher)
    train_logger = CsvLogger(os.path.join(training_config.save_dir, 'train.csv'))
    rng = jax.random.PRNGKey(training_config.seed + 1)
    first_time = time.time()
    last_time = time.time()

    try:
        for i in tqdm.tqdm(range(start_step, training_config.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
            rng, step_rng = jax.random.split(rng)
            if rollout_training:
                rollout_horizon = rollout_horizon_for_step(i, training_config)
                batch = sample_train_batches[rollout_horizon]()
            else:
                batch = sample_train_batch()
            state, update_info = train_step(
                state,
                ae_state,
                batch,
                mean,
                std,
                step_rng,
                loss_config.recon_weight,
                loss_config.latent_weight,
                loss_config.xy_weight,
                loss_config.xy_tolerance,
                loss_config.gramian_diag_eps,
                loss_config.gramian_warmup_steps,
                loss_config.differentiate_gramian,
                loss_config.latent_loss_mode,
            )

            if i % training_config.log_interval == 0:
                metrics = {f'training/{k}': v for k, v in to_float_dict(update_info).items()}
                if rollout_training:
                    for horizon, horizon_batches in val_batches.items():
                        val_metric_dicts = [
                            to_float_dict(
                                eval_step(
                                    state,
                                    ae_state,
                                    val_batch,
                                    mean,
                                    std,
                                    loss_config.recon_weight,
                                    loss_config.latent_weight,
                                    loss_config.xy_weight,
                                    loss_config.xy_tolerance,
                                    loss_config.gramian_diag_eps,
                                    loss_config.gramian_warmup_steps,
                                    loss_config.differentiate_gramian,
                                    loss_config.latent_loss_mode,
                                )
                            )
                            for val_batch in horizon_batches
                        ]
                        metrics.update(
                            {
                                f'validation/h{horizon}/{key}': value
                                for key, value in average_metrics(val_metric_dicts).items()
                            }
                        )
                elif len(val_batches) > 0:
                    val_metric_dicts = [
                        to_float_dict(
                            eval_step(
                                state,
                                ae_state,
                                val_batch,
                                mean,
                                std,
                                loss_config.recon_weight,
                                loss_config.latent_weight,
                                loss_config.xy_weight,
                                loss_config.xy_tolerance,
                                loss_config.gramian_diag_eps,
                                loss_config.gramian_warmup_steps,
                                loss_config.differentiate_gramian,
                                loss_config.latent_loss_mode,
                            )
                        )
                        for val_batch in val_batches
                    ]
                    metrics.update(
                        {f'validation/{key}': value for key, value in average_metrics(val_metric_dicts).items()}
                    )
                metrics['optimizer/lr'] = training_config.lr
                metrics['time/epoch_time'] = (time.time() - last_time) / training_config.log_interval
                metrics['time/total_time'] = time.time() - first_time
                last_time = time.time()
                wandb.log(metrics, step=i)
                train_logger.log(metrics, step=i)

            if i % training_config.save_interval == 0:
                save_checkpoint(
                    state,
                    training_config.save_dir,
                    i,
                    training_config,
                    model_config,
                    loss_config,
                    training_config.ae_checkpoint_path,
                    ae_model_config,
                    ae_training_config,
                    normalization_stats,
                    flag_dict,
                )

        if training_config.train_steps % training_config.save_interval != 0:
            save_checkpoint(
                state,
                training_config.save_dir,
                training_config.train_steps,
                training_config,
                model_config,
                loss_config,
                training_config.ae_checkpoint_path,
                ae_model_config,
                ae_training_config,
                normalization_stats,
                flag_dict,
            )
    finally:
        train_logger.close()
        for prefetcher in prefetchers:
            prefetcher.close()


def main(_):
    training_config, model_config, loss_config = create_configs()
    run_training(training_config, model_config, loss_config)


if __name__ == '__main__':
    app.run(main)
