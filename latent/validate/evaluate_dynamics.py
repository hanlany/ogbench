from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import flax
import jax
import jax.numpy as jnp
import numpy as np

from latent.train.dynamics import LatentDynamics
from latent.train.train_autoencoder import NormalizationStats, normalize_observations
from latent.train.train_dynamics import DynamicsModelConfig, load_autoencoder_artifacts, predict_latents
from ogbench import make_env_and_datasets


DEFAULT_HORIZONS = (1, 5, 10, 25)


@dataclass(frozen=True)
class DynamicsArtifacts:
    checkpoint: dict[str, Any]
    checkpoint_path: Path
    model: LatentDynamics
    params: Any
    model_config: DynamicsModelConfig
    ae_state: Any
    normalization: NormalizationStats
    ae_checkpoint_path: Path
    env_name: str
    dataset_dir: str
    dataset_path: str | None
    seed: int


def parse_horizons(value: str | list[int] | tuple[int, ...]) -> tuple[int, ...]:
    """Parse, validate, and sort requested rollout horizons."""
    if isinstance(value, str):
        try:
            horizons = tuple(int(item.strip()) for item in value.split(',') if item.strip())
        except ValueError as exc:
            raise ValueError(f'Horizons must be comma-separated integers, got {value!r}.') from exc
    else:
        horizons = tuple(int(item) for item in value)
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError(f'Horizons must contain positive integers, got {horizons}.')
    return tuple(sorted(set(horizons)))


def _model_config_from_dict(data: dict[str, Any]) -> DynamicsModelConfig:
    required = {'hidden_dims', 'latent_dim', 'action_dim', 'activation', 'layer_norm', 'dropout_rate'}
    missing = sorted(required - data.keys())
    if missing:
        raise ValueError(f'Dynamics model config is missing keys: {missing}.')
    config = DynamicsModelConfig(
        hidden_dims=tuple(int(dim) for dim in data['hidden_dims']),
        latent_dim=int(data['latent_dim']),
        action_dim=int(data['action_dim']),
        activation=str(data['activation']),
        layer_norm=bool(data['layer_norm']),
        dropout_rate=float(data['dropout_rate']),
        prediction_type=str(data.get('prediction_type', 'absolute')),
    )
    if config.latent_dim <= 0 or config.action_dim <= 0 or not config.hidden_dims:
        raise ValueError(f'Invalid dynamics model dimensions: {config}.')
    if config.prediction_type not in ('absolute', 'residual'):
        raise ValueError(f'Invalid dynamics prediction_type: {config.prediction_type!r}.')
    return config


def load_dynamics_artifacts(checkpoint_path: str | Path) -> DynamicsArtifacts:
    """Restore evaluation-only dynamics and linked autoencoder artifacts."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    with checkpoint_path.open('rb') as file:
        checkpoint = pickle.load(file)

    required = {'model_config', 'params', 'ae_checkpoint_path', 'normalization'}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f'Dynamics checkpoint {checkpoint_path} is missing keys: {missing}.')

    model_config = _model_config_from_dict(checkpoint['model_config'])
    model = LatentDynamics(
        hidden_dims=model_config.hidden_dims,
        latent_dim=model_config.latent_dim,
        activation=model_config.activation,
        layer_norm=model_config.layer_norm,
        dropout_rate=model_config.dropout_rate,
        prediction_type=model_config.prediction_type,
    )
    template = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, model_config.latent_dim), dtype=jnp.float32),
        jnp.zeros((1, model_config.action_dim), dtype=jnp.float32),
    )['params']
    try:
        params = flax.serialization.from_state_dict(template, checkpoint['params'])
    except ValueError as exc:
        raise ValueError(f'Dynamics parameters in {checkpoint_path} do not match model_config.') from exc

    ae_checkpoint_path = Path(checkpoint['ae_checkpoint_path']).expanduser().resolve()
    _, ae_state, ae_model_config, _, ae_stats = load_autoencoder_artifacts(ae_checkpoint_path)
    if ae_model_config.latent_dim != model_config.latent_dim:
        raise ValueError(
            f'Latent dimension mismatch: dynamics={model_config.latent_dim}, AE={ae_model_config.latent_dim}.'
        )

    saved_stats = NormalizationStats(
        mean=np.asarray(checkpoint['normalization']['mean'], dtype=np.float32),
        std=np.asarray(checkpoint['normalization']['std'], dtype=np.float32),
    )
    if saved_stats.mean.shape != (ae_model_config.obs_dim,) or saved_stats.std.shape != (ae_model_config.obs_dim,):
        raise ValueError('Dynamics checkpoint normalization shape does not match the linked AE observation dimension.')
    if not np.allclose(saved_stats.mean, ae_stats.mean) or not np.allclose(saved_stats.std, ae_stats.std):
        raise ValueError('Dynamics checkpoint normalization does not match the linked AE checkpoint.')

    training_config = checkpoint.get('training_config', checkpoint.get('flags', {}))
    for key in ('env_name', 'dataset_dir'):
        if key not in training_config:
            raise ValueError(f'Dynamics checkpoint training metadata is missing {key!r}.')
    return DynamicsArtifacts(
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        model=model,
        params=params,
        model_config=model_config,
        ae_state=ae_state,
        normalization=saved_stats,
        ae_checkpoint_path=ae_checkpoint_path,
        env_name=str(training_config['env_name']),
        dataset_dir=str(training_config['dataset_dir']),
        dataset_path=training_config.get('dataset_path'),
        seed=int(training_config.get('seed', checkpoint.get('flags', {}).get('seed', 0))),
    )


def episode_slices(terminals: np.ndarray, valids: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Return half-open episode state ranges, preferring compact-dataset `valids`."""
    terminals = np.asarray(terminals).reshape(-1)
    if valids is not None:
        valids = np.asarray(valids).reshape(-1)
        if valids.shape != terminals.shape:
            raise ValueError('valids and terminals must have the same shape.')
        ends = np.flatnonzero(valids <= 0)
    else:
        ends = np.flatnonzero(terminals > 0)
    if len(ends) == 0 or ends[-1] != len(terminals) - 1:
        raise ValueError('Dataset must mark the final state as an episode boundary.')
    starts = np.concatenate([[0], ends[:-1] + 1])
    slices = [(int(start), int(end + 1)) for start, end in zip(starts, ends)]
    if any(end <= start for start, end in slices):
        raise ValueError('Dataset contains an empty episode.')
    return slices


def build_rollout_windows(
    raw_dataset: dict[str, np.ndarray],
    max_horizon: int,
    rollouts_per_episode: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Build balanced, deterministic windows that stay inside recorded episodes."""
    if max_horizon <= 0:
        raise ValueError('max_horizon must be positive.')
    if rollouts_per_episode < 0:
        raise ValueError('rollouts_per_episode must be non-negative.')
    for key in ('observations', 'actions', 'terminals'):
        if key not in raw_dataset:
            raise ValueError(f'Validation dataset is missing required key {key!r}.')

    observations = np.asarray(raw_dataset['observations'], dtype=np.float32)
    actions = np.asarray(raw_dataset['actions'], dtype=np.float32)
    terminals = np.asarray(raw_dataset['terminals'])
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError('Only vector observations and actions are supported.')
    if len(observations) != len(actions) or len(observations) != len(terminals):
        raise ValueError('Observations, actions, and terminals must be aligned.')

    slices = episode_slices(terminals, raw_dataset.get('valids'))
    rng = np.random.default_rng(seed)
    selected_starts = []
    episode_ids = []
    for episode_id, (start, end) in enumerate(slices):
        candidates = np.arange(start, end - max_horizon, dtype=np.int64)
        if len(candidates) == 0:
            continue
        if rollouts_per_episode and len(candidates) > rollouts_per_episode:
            candidates = np.sort(rng.choice(candidates, size=rollouts_per_episode, replace=False))
        selected_starts.extend(candidates.tolist())
        episode_ids.extend([episode_id] * len(candidates))
    if not selected_starts:
        raise ValueError(f'No episodes are long enough for max_horizon={max_horizon}.')

    starts = np.asarray(selected_starts, dtype=np.int64)
    action_offsets = np.arange(max_horizon, dtype=np.int64)
    target_offsets = np.arange(1, max_horizon + 1, dtype=np.int64)
    return {
        'initial_observations': observations[starts],
        'actions': actions[starts[:, None] + action_offsets[None, :]],
        'target_observations': observations[starts[:, None] + target_offsets[None, :]],
        'start_indices': starts,
        'episode_ids': np.asarray(episode_ids, dtype=np.int64),
    }


def recursive_rollout(
    dynamics_fn: Callable[[jax.Array, jax.Array], jax.Array],
    initial_latents: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    """Recursively propagate predicted latents for a batch of action sequences."""
    time_major_actions = jnp.swapaxes(actions, 0, 1)

    def step(latents, step_actions):
        next_latents = dynamics_fn(latents, step_actions)
        return next_latents, next_latents

    _, predictions = jax.lax.scan(step, initial_latents, time_major_actions)
    return jnp.swapaxes(predictions, 0, 1)


def teacher_forced_predictions(
    dynamics_fn: Callable[[jax.Array, jax.Array], jax.Array],
    source_latents: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    """Predict every step independently from its ground-truth source latent."""
    batch_size, horizon, latent_dim = source_latents.shape
    flat = dynamics_fn(source_latents.reshape((-1, latent_dim)), actions.reshape((-1, actions.shape[-1])))
    return flat.reshape((batch_size, horizon, latent_dim))


def _predict_dynamics(
    artifacts: DynamicsArtifacts,
    initial_latents: np.ndarray,
    source_latents: np.ndarray,
    actions: np.ndarray,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError('batch_size must be positive.')

    def dynamics_fn(latents, step_actions):
        return artifacts.model.apply({'params': artifacts.params}, latents, step_actions, deterministic=True)

    rollout_fn = jax.jit(lambda z, u: recursive_rollout(dynamics_fn, z, u))
    teacher_fn = jax.jit(lambda z, u: teacher_forced_predictions(dynamics_fn, z, u))
    open_loop = []
    teacher = []
    for start in range(0, len(initial_latents), batch_size):
        end = start + batch_size
        open_loop.append(np.asarray(jax.device_get(rollout_fn(initial_latents[start:end], actions[start:end]))))
        teacher.append(np.asarray(jax.device_get(teacher_fn(source_latents[start:end], actions[start:end]))))
    return np.concatenate(open_loop), np.concatenate(teacher)


def _decode_latents(ae_state, latents: np.ndarray, batch_size: int) -> np.ndarray:
    flat = latents.reshape((-1, latents.shape[-1]))
    decoded = []
    for start in range(0, len(flat), batch_size):
        batch = jnp.asarray(flat[start : start + batch_size], dtype=jnp.float32)
        decoded.append(np.asarray(jax.device_get(ae_state(batch, method='decode', deterministic=True))))
    return np.concatenate(decoded).reshape((*latents.shape[:-1], -1)).astype(np.float32)


def _finite_rows(prediction: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(prediction), axis=-1)


def metric_summary(
    pred_latent: np.ndarray,
    pred_normalized: np.ndarray,
    pred_raw: np.ndarray,
    target_latent: np.ndarray,
    target_normalized: np.ndarray,
    target_raw: np.ndarray,
) -> dict[str, Any]:
    """Compute scalar, per-dimension, and Ant XY endpoint errors."""
    finite = _finite_rows(pred_latent) & _finite_rows(pred_normalized) & _finite_rows(pred_raw)
    result: dict[str, Any] = {'finite_rate': float(np.mean(finite))}
    if not np.any(finite):
        return {
            **result,
            'latent': {'mse': None, 'rmse': None, 'mean_l2': None},
            'normalized': {'mse': None, 'rmse': None, 'mae': None},
            'raw': {'mse': None, 'rmse': None, 'mae': None, 'per_dim_rmse': []},
            'xy': {'mean': None, 'median': None, 'p95': None},
        }
    latent_error = pred_latent[finite] - target_latent[finite]
    normalized_error = pred_normalized[finite] - target_normalized[finite]
    raw_error = pred_raw[finite] - target_raw[finite]
    latent_mse = float(np.mean(latent_error**2))
    normalized_mse = float(np.mean(normalized_error**2))
    raw_mse = float(np.mean(raw_error**2))
    xy_error = np.linalg.norm(raw_error[:, :2], axis=-1)
    result.update(
        {
            'latent': {
                'mse': latent_mse,
                'rmse': float(np.sqrt(latent_mse)),
                'mean_l2': float(np.mean(np.linalg.norm(latent_error, axis=-1))),
            },
            'normalized': {
                'mse': normalized_mse,
                'rmse': float(np.sqrt(normalized_mse)),
                'mae': float(np.mean(np.abs(normalized_error))),
            },
            'raw': {
                'mse': raw_mse,
                'rmse': float(np.sqrt(raw_mse)),
                'mae': float(np.mean(np.abs(raw_error))),
                'per_dim_rmse': np.sqrt(np.mean(raw_error**2, axis=0)).astype(float).tolist(),
            },
            'xy': {
                'mean': float(np.mean(xy_error)),
                'median': float(np.median(xy_error)),
                'p95': float(np.percentile(xy_error, 95)),
            },
        }
    )
    return result


def aggregate_metrics(
    predictions: dict[str, dict[str, np.ndarray]],
    targets: dict[str, np.ndarray],
    horizons: tuple[int, ...],
) -> dict[str, dict[str, Any]]:
    """Aggregate each prediction method at the requested horizons."""
    result = {}
    for method, values in predictions.items():
        result[method] = {}
        for horizon in horizons:
            index = horizon - 1
            result[method][str(horizon)] = metric_summary(
                values['latent'][:, index],
                values['normalized'][:, index],
                values['raw'][:, index],
                targets['latent'][:, index],
                targets['normalized'][:, index],
                targets['raw'][:, index],
            )
    return result


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def _write_summary(path: Path, metrics: dict[str, Any]):
    methods = ('open_loop', 'teacher_forced', 'ae_floor', 'persistence')
    lines = ['Multi-step latent dynamics evaluation', '']
    lines.append(f'checkpoint: {metrics["checkpoint_path"]}')
    lines.append(f'validation windows: {metrics["settings"]["num_rollouts"]}')
    lines.append('')
    lines.append('Raw observation RMSE by horizon')
    lines.append('horizon | ' + ' | '.join(methods))
    lines.append('--- | ' + ' | '.join(['---'] * len(methods)))
    for horizon in metrics['settings']['horizons']:
        values = []
        for method in methods:
            value = metrics['methods'][method][str(horizon)]['raw']['rmse']
            values.append('n/a' if value is None else f'{value:.6g}')
        lines.append(f'{horizon} | ' + ' | '.join(values))
    lines.append('')
    lines.append('Open-loop Ant XY endpoint error')
    lines.append('horizon | mean | median | p95 | finite rate')
    lines.append('--- | --- | --- | --- | ---')
    for horizon in metrics['settings']['horizons']:
        values = metrics['methods']['open_loop'][str(horizon)]
        xy = values['xy']
        lines.append(
            f'{horizon} | {xy["mean"]:.6g} | {xy["median"]:.6g} | {xy["p95"]:.6g} | {values["finite_rate"]:.6g}'
        )
    path.write_text('\n'.join(lines) + '\n')


def _select_example_indices(final_errors: np.ndarray, count: int) -> np.ndarray:
    count = min(max(count, 0), len(final_errors))
    if count == 0:
        return np.empty(0, dtype=np.int64)
    order = np.argsort(final_errors, kind='stable')
    positions = np.linspace(0, len(order) - 1, count).round().astype(int)
    return order[positions].astype(np.int64)


def _save_plots(
    output_dir: Path,
    metrics: dict[str, Any],
    raw_initial: np.ndarray,
    raw_targets: np.ndarray,
    raw_open_loop: np.ndarray,
    num_plot_rollouts: int,
) -> list[str]:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    horizons = metrics['settings']['horizons']
    methods = ('open_loop', 'teacher_forced', 'ae_floor', 'persistence')
    labels = {
        'open_loop': 'open loop',
        'teacher_forced': 'teacher forced',
        'ae_floor': 'AE floor',
        'persistence': 'persistence',
    }
    written = []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in methods:
        axes[0].plot(
            horizons,
            [metrics['methods'][method][str(h)]['raw']['rmse'] for h in horizons],
            marker='o',
            label=labels[method],
        )
        axes[1].plot(
            horizons,
            [metrics['methods'][method][str(h)]['latent']['rmse'] for h in horizons],
            marker='o',
            label=labels[method],
        )
    axes[0].set(title='Raw observation error', xlabel='horizon', ylabel='RMSE')
    axes[1].set(title='Latent error', xlabel='horizon', ylabel='RMSE')
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    path = plot_dir / 'error_by_horizon.png'
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(6, 4))
    for method in methods:
        ax.plot(
            horizons,
            [metrics['methods'][method][str(h)]['xy']['mean'] for h in horizons],
            marker='o',
            label=labels[method],
        )
    ax.set(title='Ant XY endpoint error', xlabel='horizon', ylabel='mean Euclidean error')
    ax.grid(alpha=0.25)
    ax.legend()
    path = plot_dir / 'xy_endpoint_error.png'
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    per_dim = np.asarray([metrics['methods']['open_loop'][str(h)]['raw']['per_dim_rmse'] for h in horizons])
    fig, ax = plt.subplots(figsize=(12, 4))
    image = ax.imshow(per_dim, aspect='auto', interpolation='nearest')
    ax.set(title='Open-loop raw per-dimension RMSE', xlabel='observation dimension', ylabel='horizon')
    ax.set_yticks(np.arange(len(horizons)), labels=horizons)
    fig.colorbar(image, ax=ax, label='RMSE')
    path = plot_dir / 'per_dimension_rmse.png'
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    final_errors = np.linalg.norm(raw_open_loop[:, -1, :2] - raw_targets[:, -1, :2], axis=-1)
    example_indices = _select_example_indices(final_errors, num_plot_rollouts)
    if len(example_indices):
        cols = min(4, len(example_indices))
        rows = int(np.ceil(len(example_indices) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
        for axis, index in zip(axes.ravel(), example_indices):
            truth = np.concatenate([raw_initial[index, None, :2], raw_targets[index, :, :2]], axis=0)
            prediction = np.concatenate([raw_initial[index, None, :2], raw_open_loop[index, :, :2]], axis=0)
            axis.plot(truth[:, 0], truth[:, 1], marker='.', label='ground truth')
            axis.plot(prediction[:, 0], prediction[:, 1], marker='.', label='open loop')
            axis.set_title(f'window {index}, final error={final_errors[index]:.3g}')
            axis.set_aspect('equal', adjustable='datalim')
            axis.legend()
        for axis in axes.ravel()[len(example_indices) :]:
            axis.axis('off')
        path = plot_dir / 'example_xy_rollouts.png'
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))
    return written


def evaluate_rollout_dataset(
    checkpoint_path: str | Path,
    raw_dataset: dict[str, np.ndarray],
    output_dir: str | Path,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    rollouts_per_episode: int = 100,
    batch_size: int = 1024,
    seed: int | None = None,
    num_plot_rollouts: int = 8,
    write_plots: bool = True,
) -> dict[str, Any]:
    """Evaluate a dynamics checkpoint on an already-loaded compact validation dataset."""
    horizons = parse_horizons(horizons)
    artifacts = load_dynamics_artifacts(checkpoint_path)
    seed = artifacts.seed if seed is None else seed
    windows = build_rollout_windows(raw_dataset, max(horizons), rollouts_per_episode, seed)
    if windows['actions'].shape[-1] != artifacts.model_config.action_dim:
        raise ValueError(
            f'Action dimension mismatch: dataset={windows["actions"].shape[-1]}, '
            f'dynamics={artifacts.model_config.action_dim}.'
        )

    raw_initial = windows['initial_observations']
    raw_targets = windows['target_observations']
    normalized_initial = normalize_observations(raw_initial, artifacts.normalization)
    normalized_targets = normalize_observations(raw_targets, artifacts.normalization)
    all_normalized = np.concatenate([normalized_initial[:, None], normalized_targets], axis=1)
    flat_latents = predict_latents(
        artifacts.ae_state,
        all_normalized.reshape((-1, all_normalized.shape[-1])),
        batch_size,
    )
    all_latents = flat_latents.reshape((*all_normalized.shape[:-1], -1))
    initial_latents = all_latents[:, 0]
    target_latents = all_latents[:, 1:]
    source_latents = all_latents[:, :-1]

    open_latents, teacher_latents = _predict_dynamics(
        artifacts,
        initial_latents,
        source_latents,
        windows['actions'],
        batch_size,
    )
    open_normalized = _decode_latents(artifacts.ae_state, open_latents, batch_size)
    teacher_normalized = _decode_latents(artifacts.ae_state, teacher_latents, batch_size)
    floor_normalized = _decode_latents(artifacts.ae_state, target_latents, batch_size)
    mean = artifacts.normalization.mean
    std = artifacts.normalization.std

    persistence_latents = np.repeat(initial_latents[:, None], max(horizons), axis=1)
    persistence_normalized = np.repeat(normalized_initial[:, None], max(horizons), axis=1)
    persistence_raw = np.repeat(raw_initial[:, None], max(horizons), axis=1)
    predictions = {
        'open_loop': {'latent': open_latents, 'normalized': open_normalized, 'raw': open_normalized * std + mean},
        'teacher_forced': {
            'latent': teacher_latents,
            'normalized': teacher_normalized,
            'raw': teacher_normalized * std + mean,
        },
        'ae_floor': {'latent': target_latents, 'normalized': floor_normalized, 'raw': floor_normalized * std + mean},
        'persistence': {'latent': persistence_latents, 'normalized': persistence_normalized, 'raw': persistence_raw},
    }
    targets = {'latent': target_latents, 'normalized': normalized_targets, 'raw': raw_targets}
    methods = aggregate_metrics(predictions, targets, horizons)

    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        'checkpoint_path': str(artifacts.checkpoint_path),
        'checkpoint_step': int(artifacts.checkpoint.get('step', -1)),
        'ae_checkpoint_path': str(artifacts.ae_checkpoint_path),
        'environment': artifacts.env_name,
        'model_config': artifacts.checkpoint['model_config'],
        'settings': {
            'horizons': list(horizons),
            'max_horizon': max(horizons),
            'rollouts_per_episode': rollouts_per_episode,
            'num_rollouts': len(raw_initial),
            'batch_size': batch_size,
            'seed': seed,
            'start_indices': windows['start_indices'].tolist(),
            'episode_ids': windows['episode_ids'].tolist(),
        },
        'methods': methods,
        'plots': [],
    }
    if write_plots:
        metrics['plots'] = _save_plots(
            output_dir,
            metrics,
            raw_initial,
            raw_targets,
            predictions['open_loop']['raw'],
            num_plot_rollouts,
        )
    metrics = _json_ready(metrics)
    with (output_dir / 'metrics.json').open('w') as file:
        json.dump(metrics, file, indent=2, sort_keys=True)
    _write_summary(output_dir / 'summary.txt', metrics)
    return metrics


def load_validation_dataset(
    artifacts: DynamicsArtifacts,
    env_name: str | None = None,
    dataset_dir: str | None = None,
    dataset_path: str | None = None,
) -> dict[str, np.ndarray]:
    """Load the compact held-out split while preserving episode boundary fields."""
    _, validation = make_env_and_datasets(
        env_name or artifacts.env_name,
        dataset_dir=dataset_dir or artifacts.dataset_dir,
        dataset_path=dataset_path if dataset_path is not None else artifacts.dataset_path,
        compact_dataset=True,
        dataset_only=True,
    )
    if validation is None:
        raise ValueError('A held-out validation dataset is required for rollout evaluation.')
    return validation


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate multi-step open-loop latent dynamics rollouts.')
    parser.add_argument('--checkpoint_path', required=True, help='Path to a dynamics params_<step>.pkl checkpoint.')
    parser.add_argument('--horizons', default='1,5,10,25', help='Comma-separated positive rollout horizons.')
    parser.add_argument('--rollouts_per_episode', type=int, default=100, help='Starts per episode; 0 uses all starts.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Inference batch size.')
    parser.add_argument('--seed', type=int, default=None, help='Sampling seed; defaults to the checkpoint seed.')
    parser.add_argument('--env_name', default=None, help='Override the checkpoint environment name.')
    parser.add_argument('--dataset_dir', default=None, help='Override the checkpoint dataset directory.')
    parser.add_argument('--dataset_path', default=None, help='Override the checkpoint training dataset path.')
    parser.add_argument('--output_dir', default=None, help='Output directory; defaults beside the checkpoint.')
    parser.add_argument('--num_plot_rollouts', type=int, default=8, help='Number of example XY rollouts to plot.')
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    artifacts = load_dynamics_artifacts(args.checkpoint_path)
    validation = load_validation_dataset(
        artifacts,
        env_name=args.env_name,
        dataset_dir=args.dataset_dir,
        dataset_path=args.dataset_path,
    )
    output_dir = args.output_dir or str(artifacts.checkpoint_path.parent / 'rollout_eval')
    evaluate_rollout_dataset(
        args.checkpoint_path,
        validation,
        output_dir,
        horizons=parse_horizons(args.horizons),
        rollouts_per_episode=args.rollouts_per_episode,
        batch_size=args.batch_size,
        seed=args.seed,
        num_plot_rollouts=args.num_plot_rollouts,
    )
    print(f'Wrote dynamics rollout evaluation to {output_dir}')


if __name__ == '__main__':
    main()
