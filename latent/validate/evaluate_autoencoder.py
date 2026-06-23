from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np

_LOCAL_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _LOCAL_DIR.parents[1]
_IMPLS_DIR = _REPO_ROOT / 'impls'

for _path in (_REPO_ROOT, _IMPLS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from impls.utils.datasets import Dataset  # noqa: E402
from latent.train.train_autoencoder import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    ModelConfig,
    NormalizationStats,
    TrainingConfig,
    create_train_state,
    normalize_observations,
    validate_model_config,
    validate_observations,
)
from ogbench import make_env_and_datasets  # noqa: E402


def _as_tuple(value):
    if value is None:
        return None
    return tuple(value)


def model_config_from_dict(data: dict[str, Any]) -> ModelConfig:
    """Recreate a ModelConfig from JSON/pickle-friendly checkpoint data."""
    return ModelConfig(
        hidden_dims=_as_tuple(data['hidden_dims']),
        latent_dim=int(data['latent_dim']),
        obs_dim=int(data['obs_dim']),
        decoder_hidden_dims=_as_tuple(data.get('decoder_hidden_dims')),
        activation=data.get('activation', 'gelu'),
        layer_norm=bool(data.get('layer_norm', False)),
        decoder_activate_final=bool(data.get('decoder_activate_final', False)),
        dropout_rate=float(data.get('dropout_rate', 0.0)),
    )


def training_config_from_dict(data: dict[str, Any]) -> TrainingConfig:
    """Recreate a TrainingConfig from checkpoint data."""
    return TrainingConfig(
        run_group=data.get('run_group', 'LatentAutoEncoder'),
        seed=int(data.get('seed', 0)),
        env_name=data.get('env_name', 'antmaze-large-navigate-v0'),
        dataset_dir=data.get('dataset_dir', DEFAULT_DATASET_DIR),
        dataset_path=data.get('dataset_path'),
        save_dir=data.get('save_dir', 'exp/'),
        restore_path=data.get('restore_path'),
        restore_step=data.get('restore_step'),
        wandb_mode=data.get('wandb_mode', 'disabled'),
        lr=float(data.get('lr', 3e-4)),
        min_lr=float(data.get('min_lr', 1e-6)),
        lr_plateau_patience=int(data.get('lr_plateau_patience', 5)),
        lr_plateau_factor=float(data.get('lr_plateau_factor', 0.5)),
        lr_plateau_min_delta=float(data.get('lr_plateau_min_delta', 1e-4)),
        batch_size=int(data.get('batch_size', 1024)),
        train_steps=int(data.get('train_steps', 1)),
        log_interval=int(data.get('log_interval', 1)),
        save_interval=int(data.get('save_interval', 1)),
        prefetch_batches=int(data.get('prefetch_batches', 0)),
        normalize_observations=bool(data.get('normalize_observations', True)),
        normalization_eps=float(data.get('normalization_eps', 1e-6)),
        validation_batches=int(data.get('validation_batches', 0)),
        validation_batch_size=int(data.get('validation_batch_size', data.get('batch_size', 1024))),
    )


def normalization_stats_from_dict(data: dict[str, Any]) -> NormalizationStats:
    """Recreate saved observation normalization statistics."""
    return NormalizationStats(
        mean=np.asarray(data['mean'], dtype=np.float32),
        std=np.asarray(data['std'], dtype=np.float32),
    )


def load_checkpoint(checkpoint_path: str | Path):
    """Load a checkpoint and recreate its config objects."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    with checkpoint_path.open('rb') as f:
        checkpoint = pickle.load(f)

    if 'model_config' not in checkpoint or 'normalization' not in checkpoint or 'autoencoder' not in checkpoint:
        raise ValueError(f'{checkpoint_path} is not a full autoencoder checkpoint.')

    model_config = model_config_from_dict(checkpoint['model_config'])
    training_config = training_config_from_dict(checkpoint.get('training_config', checkpoint.get('flags', {})))
    stats = normalization_stats_from_dict(checkpoint['normalization'])
    validate_model_config(model_config)
    return checkpoint, model_config, training_config, stats


def restore_autoencoder_state(checkpoint, model_config: ModelConfig, training_config: TrainingConfig):
    """Create a TrainState with checkpoint weights.

    Older checkpoints may contain optimizer state from a different Optax wrapper.
    Evaluation only needs params, so fall back to params/step restore if full
    TrainState deserialization cannot match the current optimizer state shape.
    """
    state = create_train_state(training_config.seed, model_config, training_config)
    try:
        return flax.serialization.from_state_dict(state, checkpoint['autoencoder'])
    except ValueError as exc:
        if 'params' not in checkpoint:
            raise exc
        params = flax.serialization.from_state_dict(state.params, checkpoint['params'])
        step = checkpoint.get('step', checkpoint.get('autoencoder', {}).get('step', state.step))
        return state.replace(step=step, params=params)


def make_dataset_from_raw(raw_observations, stats: NormalizationStats, split_name: str):
    """Create the compact Dataset shape used by AE training, with checkpoint normalization."""
    raw_observations = validate_observations(raw_observations, split_name)
    return Dataset.create(
        observations=normalize_observations(raw_observations, stats),
        raw_observations=raw_observations,
    )


def load_datasets_for_checkpoint(
    training_config: TrainingConfig,
    stats: NormalizationStats,
    env_name: str | None = None,
    dataset_dir: str | None = None,
    dataset_path: str | None = None,
):
    """Load train/validation raw observations and normalize with checkpoint stats."""
    train_dataset, val_dataset = make_env_and_datasets(
        env_name or training_config.env_name,
        dataset_dir=dataset_dir or training_config.dataset_dir,
        dataset_path=dataset_path if dataset_path is not None else training_config.dataset_path,
        compact_dataset=True,
        dataset_only=True,
    )
    datasets = {
        'train': make_dataset_from_raw(train_dataset['observations'], stats, 'training'),
    }
    if val_dataset is not None:
        datasets['validation'] = make_dataset_from_raw(val_dataset['observations'], stats, 'validation')
    return datasets


def sample_dataset(dataset, max_examples: int | None, seed: int):
    """Return a deterministic subset of a Dataset as plain NumPy arrays."""
    if max_examples is None or max_examples <= 0 or max_examples >= dataset.size:
        idxs = np.arange(dataset.size)
    else:
        rng = np.random.default_rng(seed)
        idxs = np.sort(rng.choice(dataset.size, size=max_examples, replace=False))
    subset = dataset.get_subset(idxs)
    return {
        'observations': np.asarray(subset['observations'], dtype=np.float32),
        'raw_observations': np.asarray(subset['raw_observations'], dtype=np.float32),
        'indices': idxs.astype(np.int64),
    }


def predict_autoencoder(state, observations, batch_size: int):
    """Run deterministic AE inference in batches."""
    reconstructions = []
    latents = []
    for start in range(0, len(observations), batch_size):
        batch = jnp.asarray(observations[start : start + batch_size], dtype=jnp.float32)
        batch_reconstructions, batch_latents = state(batch, deterministic=True)
        reconstructions.append(np.asarray(jax.device_get(batch_reconstructions), dtype=np.float32))
        latents.append(np.asarray(jax.device_get(batch_latents), dtype=np.float32))
    return np.concatenate(reconstructions, axis=0), np.concatenate(latents, axis=0)


def reconstruction_summary(observations, reconstructions, raw_observations, raw_reconstructions):
    """Compute scalar and per-dimension reconstruction diagnostics."""
    errors = reconstructions - observations
    raw_errors = raw_reconstructions - raw_observations
    per_dim_mse = np.mean(errors**2, axis=0)
    raw_per_dim_mse = np.mean(raw_errors**2, axis=0)
    example_mse = np.mean(raw_errors**2, axis=1)
    worst_dim_order = np.argsort(raw_per_dim_mse)[::-1]
    worst_example_order = np.argsort(example_mse)[::-1]

    mse = float(np.mean(errors**2))
    raw_mse = float(np.mean(raw_errors**2))
    return {
        'mse': mse,
        'rmse': float(np.sqrt(mse)),
        'mae': float(np.mean(np.abs(errors))),
        'max_abs_error': float(np.max(np.abs(errors))),
        'original/mse': raw_mse,
        'original/rmse': float(np.sqrt(raw_mse)),
        'original/mae': float(np.mean(np.abs(raw_errors))),
        'original/max_abs_error': float(np.max(np.abs(raw_errors))),
        'per_dim_mse': per_dim_mse.astype(float).tolist(),
        'original/per_dim_mse': raw_per_dim_mse.astype(float).tolist(),
        'worst_dims': [
            {'dim': int(dim), 'original_mse': float(raw_per_dim_mse[dim])}
            for dim in worst_dim_order[: min(10, len(worst_dim_order))]
        ],
        'worst_examples': [
            {'index': int(idx), 'original_mse': float(example_mse[idx])}
            for idx in worst_example_order[: min(10, len(worst_example_order))]
        ],
    }


def fit_pca(observations, latent_dim: int):
    """Fit a NumPy PCA baseline on normalized observations."""
    mean = np.mean(observations, axis=0, keepdims=True)
    centered = observations - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:latent_dim]
    return {'mean': mean.astype(np.float32), 'components': components.astype(np.float32)}


def apply_pca(pca, observations):
    """Encode and reconstruct observations with a fitted PCA baseline."""
    centered = observations - pca['mean']
    latents = centered @ pca['components'].T
    reconstructions = latents @ pca['components'] + pca['mean']
    return reconstructions.astype(np.float32), latents.astype(np.float32)


def _safe_corr(x, y):
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _rankdata(values):
    order = np.argsort(values, kind='mergesort')
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def distance_correlations(observations, latents, pair_count: int, seed: int):
    """Estimate observation-vs-latent distance correlation from random pairs."""
    if len(observations) < 2:
        return {'distance/pearson': float('nan'), 'distance/spearman': float('nan')}
    rng = np.random.default_rng(seed)
    first = rng.integers(len(observations), size=pair_count)
    second = rng.integers(len(observations), size=pair_count)
    keep = first != second
    first = first[keep]
    second = second[keep]
    obs_distances = np.linalg.norm(observations[first] - observations[second], axis=1)
    latent_distances = np.linalg.norm(latents[first] - latents[second], axis=1)
    return {
        'distance/pearson': _safe_corr(obs_distances, latent_distances),
        'distance/spearman': _safe_corr(_rankdata(obs_distances), _rankdata(latent_distances)),
    }


def knn_overlap(observations, latents, k: int, num_anchors: int, seed: int):
    """Estimate local neighborhood preservation from observation space to latent space."""
    if len(observations) < 2:
        return float('nan')
    k = min(k, len(observations) - 1)
    rng = np.random.default_rng(seed)
    anchor_count = min(num_anchors, len(observations))
    anchors = np.sort(rng.choice(len(observations), size=anchor_count, replace=False))

    obs_distances = np.linalg.norm(observations[anchors, None, :] - observations[None, :, :], axis=-1)
    latent_distances = np.linalg.norm(latents[anchors, None, :] - latents[None, :, :], axis=-1)
    overlaps = []
    for row, anchor in enumerate(anchors):
        obs_distances[row, anchor] = np.inf
        latent_distances[row, anchor] = np.inf
        obs_neighbors = set(np.argpartition(obs_distances[row], kth=k)[:k].tolist())
        latent_neighbors = set(np.argpartition(latent_distances[row], kth=k)[:k].tolist())
        overlaps.append(len(obs_neighbors & latent_neighbors) / k)
    return float(np.mean(overlaps))


def latent_utilization(latents, dead_std_threshold: float):
    """Summarize latent dimension use and covariance spectrum."""
    per_dim_std = np.std(latents, axis=0)
    if len(latents) > 1:
        covariance = np.cov(latents, rowvar=False)
        covariance = np.atleast_2d(covariance)
        eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    else:
        eigenvalues = np.zeros(latents.shape[-1], dtype=np.float64)
    positive = np.maximum(eigenvalues, 0.0)
    total = float(np.sum(positive))
    if total <= 1e-12:
        effective_rank = 0.0
    else:
        probs = positive / total
        effective_rank = float(np.exp(-np.sum(probs * np.log(probs + 1e-12))))

    return {
        'latent/mean': np.mean(latents, axis=0).astype(float).tolist(),
        'latent/std': per_dim_std.astype(float).tolist(),
        'latent/min': np.min(latents, axis=0).astype(float).tolist(),
        'latent/max': np.max(latents, axis=0).astype(float).tolist(),
        'latent/cov_eigenvalues': eigenvalues.astype(float).tolist(),
        'latent/effective_rank': effective_rank,
        'latent/dead_dims': int(np.sum(per_dim_std < dead_std_threshold)),
    }


def representation_summary(
    observations,
    latents,
    seed: int,
    pair_count: int,
    knn_k: int,
    knn_anchors: int,
    dead_std_threshold: float,
):
    """Compute latent representation diagnostics."""
    metrics = latent_utilization(latents, dead_std_threshold)
    metrics.update(distance_correlations(observations, latents, pair_count, seed))
    metrics['knn/overlap'] = knn_overlap(observations, latents, knn_k, knn_anchors, seed + 1)
    return metrics


def parse_dims(value: str | None):
    if value is None or value == '':
        return []
    return [int(item) for item in value.split(',') if item != '']


def valid_dims(dims, obs_dim: int):
    return [dim for dim in dims if 0 <= dim < obs_dim]


def fit_probe(latents, raw_observations, probe_dims, ridge: float):
    """Fit a ridge probe from latents to selected raw observation dimensions."""
    probe_dims = valid_dims(probe_dims, raw_observations.shape[-1])
    if len(probe_dims) == 0:
        return None
    design = np.concatenate([latents, np.ones((len(latents), 1), dtype=latents.dtype)], axis=1)
    targets = raw_observations[:, probe_dims]
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets)
    return {'weights': weights, 'probe_dims': probe_dims}


def evaluate_probe(probe, latents, raw_observations):
    """Evaluate a fitted probe on one split."""
    if probe is None:
        return {}
    design = np.concatenate([latents, np.ones((len(latents), 1), dtype=latents.dtype)], axis=1)
    targets = raw_observations[:, probe['probe_dims']]
    predictions = design @ probe['weights']
    errors = predictions - targets
    target_var = np.var(targets, axis=0)
    mse = np.mean(errors**2, axis=0)
    r2 = 1.0 - mse / np.maximum(target_var, 1e-12)
    return {
        'probe/dims': [int(dim) for dim in probe['probe_dims']],
        'probe/mse': mse.astype(float).tolist(),
        'probe/r2': r2.astype(float).tolist(),
        'probe/r2_mean': float(np.mean(r2)),
    }


def compare_to_pca(autoencoder_metrics, pca_metrics):
    """Return AE-minus-PCA deltas for core representation and reconstruction metrics."""
    deltas = {}
    for key in ('original/mse', 'mse', 'distance/pearson', 'distance/spearman', 'knn/overlap'):
        if key in autoencoder_metrics and key in pca_metrics:
            deltas[f'delta/{key}'] = float(autoencoder_metrics[key] - pca_metrics[key])
    return deltas


def save_json(path: Path, data):
    with path.open('w') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=True)


def _subplot_grid(num_items: int):
    cols = int(np.ceil(np.sqrt(num_items)))
    rows = int(np.ceil(num_items / cols))
    return rows, cols


def save_plots(output_dir: Path, split: str, raw, raw_recon, latents, color_dims, max_points: int, seed: int):
    """Write reconstruction and latent plots when matplotlib is available."""
    try:
        import matplotlib

        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    if len(raw) > max_points:
        idxs = np.sort(rng.choice(len(raw), size=max_points, replace=False))
    else:
        idxs = np.arange(len(raw))

    written = []
    dims = valid_dims(color_dims, raw.shape[-1])
    if not dims:
        return written

    rows, cols = _subplot_grid(len(dims))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
    flat_axes = axes.ravel()
    for ax, dim in zip(flat_axes, dims):
        ax.scatter(raw[idxs, dim], raw_recon[idxs, dim], s=4, alpha=0.35)
        lo = float(min(np.min(raw[idxs, dim]), np.min(raw_recon[idxs, dim])))
        hi = float(max(np.max(raw[idxs, dim]), np.max(raw_recon[idxs, dim])))
        ax.plot([lo, hi], [lo, hi], color='black', linewidth=1)
        ax.set_xlabel(f'raw dim {dim}')
        ax.set_ylabel('reconstructed')
        ax.set_title(f'dim {dim}')
    for ax in flat_axes[len(dims) :]:
        ax.axis('off')
    fig.suptitle(f'{split} reconstruction by raw dimension')
    path = plot_dir / f'{split}_reconstruction_dims.png'
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(str(path))

    if latents.shape[-1] >= 2:
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows), squeeze=False)
        flat_axes = axes.ravel()
        for ax, dim in zip(flat_axes, dims):
            points = ax.scatter(latents[idxs, 0], latents[idxs, 1], c=raw[idxs, dim], s=5, alpha=0.5, cmap='viridis')
            fig.colorbar(points, ax=ax, label=f'raw dim {dim}')
            ax.set_xlabel('latent dim 0')
            ax.set_ylabel('latent dim 1')
            ax.set_title(f'raw dim {dim}')
        for ax in flat_axes[len(dims) :]:
            ax.axis('off')
        fig.suptitle(f'{split} latent space colored by raw dimension')
        path = plot_dir / f'{split}_latent_by_dims.png'
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(str(path))
    return written


def write_summary(path: Path, metrics: dict[str, Any]):
    """Write a compact interpretation of the scalar metrics."""
    lines = ['AE checkpoint evaluation summary', '']
    for split in ('train', 'validation'):
        if split not in metrics['splits']:
            continue
        split_metrics = metrics['splits'][split]
        ae = split_metrics['autoencoder']
        pca = split_metrics['pca']
        lines.append(f'{split}:')
        lines.append(f'- AE original MSE: {ae["original/mse"]:.6g}')
        lines.append(f'- PCA original MSE: {pca["original/mse"]:.6g}')
        lines.append(f'- AE distance Spearman: {ae["distance/spearman"]:.4g}')
        lines.append(f'- PCA distance Spearman: {pca["distance/spearman"]:.4g}')
        lines.append(f'- AE kNN overlap: {ae["knn/overlap"]:.4g}')
        lines.append(f'- PCA kNN overlap: {pca["knn/overlap"]:.4g}')
        lines.append(f'- Dead latent dims: {ae["latent/dead_dims"]}')
        if 'probe/r2_mean' in ae:
            lines.append(f'- Probe R2 mean: {ae["probe/r2_mean"]:.4g}')
        reconstruction_ok = ae['original/mse'] <= 1.1 * pca['original/mse']
        geometry_ok = ae['distance/spearman'] >= pca['distance/spearman'] or ae['knn/overlap'] >= pca['knn/overlap']
        dead_ok = ae['latent/dead_dims'] == 0
        verdict = 'good' if reconstruction_ok and geometry_ok and dead_ok else 'needs inspection'
        lines.append(f'- Verdict: {verdict}')
        lines.append('')

    if 'train' in metrics['splits'] and 'validation' in metrics['splits']:
        train_mse = metrics['splits']['train']['autoencoder']['original/mse']
        val_mse = metrics['splits']['validation']['autoencoder']['original/mse']
        ratio = val_mse / max(train_mse, 1e-12)
        lines.append(f'validation/train original MSE ratio: {ratio:.4g}')
        lines.append('generalization verdict: good' if ratio <= 2.0 else 'generalization verdict: needs inspection')

    with path.open('w') as f:
        f.write('\n'.join(lines))


def evaluate_checkpoint_on_splits(
    checkpoint_path: str | Path,
    split_datasets: dict[str, Dataset],
    output_dir: str | Path,
    max_examples: int | None = 10000,
    batch_size: int = 4096,
    seed: int | None = None,
    probe_dims: list[int] | None = None,
    pair_count: int = 20000,
    knn_k: int = 10,
    knn_anchors: int = 512,
    dead_std_threshold: float = 1e-4,
    ridge: float = 1e-3,
    include_random_baseline: bool = False,
    write_plots: bool = True,
    plot_dims: list[int] | None = None,
    plot_max_points: int = 5000,
):
    """Evaluate one checkpoint on already-loaded train/validation Datasets."""
    checkpoint, model_config, training_config, stats = load_checkpoint(checkpoint_path)
    seed = training_config.seed if seed is None else seed
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    state = restore_autoencoder_state(checkpoint, model_config, training_config)
    random_state = create_train_state(seed + 997, model_config, training_config) if include_random_baseline else None

    samples = {
        split: sample_dataset(dataset, max_examples=max_examples, seed=seed + offset)
        for offset, (split, dataset) in enumerate(split_datasets.items())
    }
    if 'train' not in samples:
        raise ValueError('A train split is required to fit PCA and probe baselines.')

    pca = fit_pca(samples['train']['observations'], model_config.latent_dim)
    probe_dims = list(range(model_config.obs_dim)) if probe_dims is None else probe_dims
    split_predictions = {}

    for split, sample in samples.items():
        reconstructions, latents = predict_autoencoder(state, sample['observations'], batch_size)
        raw_reconstructions = reconstructions * stats.std + stats.mean
        pca_reconstructions, pca_latents = apply_pca(pca, sample['observations'])
        pca_raw_reconstructions = pca_reconstructions * stats.std + stats.mean
        split_predictions[split] = {
            'raw': sample['raw_observations'],
            'autoencoder_raw_reconstructions': raw_reconstructions,
            'autoencoder_latents': latents,
            'pca_latents': pca_latents,
            'autoencoder_reconstructions': reconstructions,
            'pca_reconstructions': pca_reconstructions,
            'pca_raw_reconstructions': pca_raw_reconstructions,
            'observations': sample['observations'],
        }

    probe = fit_probe(
        split_predictions['train']['autoencoder_latents'],
        split_predictions['train']['raw'],
        probe_dims,
        ridge,
    )

    metrics = {
        'checkpoint_path': str(Path(checkpoint_path).expanduser()),
        'checkpoint_step': int(checkpoint.get('step', -1)),
        'model_config': checkpoint['model_config'],
        'settings': {
            'max_examples': max_examples,
            'batch_size': batch_size,
            'seed': seed,
            'probe_dims': valid_dims(probe_dims, model_config.obs_dim),
            'pair_count': pair_count,
            'knn_k': knn_k,
            'knn_anchors': knn_anchors,
            'dead_std_threshold': dead_std_threshold,
            'include_random_baseline': include_random_baseline,
        },
        'splits': {},
        'plots': [],
    }

    for offset, (split, prediction) in enumerate(split_predictions.items()):
        observations = prediction['observations']
        raw = prediction['raw']
        ae = reconstruction_summary(
            observations,
            prediction['autoencoder_reconstructions'],
            raw,
            prediction['autoencoder_raw_reconstructions'],
        )
        ae.update(
            representation_summary(
                observations,
                prediction['autoencoder_latents'],
                seed + 100 * (offset + 1),
                pair_count,
                knn_k,
                knn_anchors,
                dead_std_threshold,
            )
        )
        ae.update(evaluate_probe(probe, prediction['autoencoder_latents'], raw))

        pca_metrics = reconstruction_summary(
            observations,
            prediction['pca_reconstructions'],
            raw,
            prediction['pca_raw_reconstructions'],
        )
        pca_metrics.update(
            representation_summary(
                observations,
                prediction['pca_latents'],
                seed + 100 * (offset + 1),
                pair_count,
                knn_k,
                knn_anchors,
                dead_std_threshold,
            )
        )

        metrics['splits'][split] = {
            'autoencoder': ae,
            'pca': pca_metrics,
            'ae_minus_pca': compare_to_pca(ae, pca_metrics),
        }

        if random_state is not None:
            random_reconstructions, random_latents = predict_autoencoder(random_state, observations, batch_size)
            random_raw_reconstructions = random_reconstructions * stats.std + stats.mean
            random_metrics = reconstruction_summary(
                observations, random_reconstructions, raw, random_raw_reconstructions
            )
            random_metrics.update(
                representation_summary(
                    observations,
                    random_latents,
                    seed + 1000 + 100 * (offset + 1),
                    pair_count,
                    knn_k,
                    knn_anchors,
                    dead_std_threshold,
                )
            )
            metrics['splits'][split]['random_autoencoder'] = random_metrics

        if write_plots:
            color_dims = plot_dims if plot_dims is not None else list(range(model_config.obs_dim))
            metrics['plots'].extend(
                save_plots(
                    output_dir,
                    split,
                    raw,
                    prediction['autoencoder_raw_reconstructions'],
                    prediction['autoencoder_latents'],
                    color_dims,
                    plot_max_points,
                    seed + offset,
                )
            )

    save_json(output_dir / 'metrics.json', metrics)
    write_summary(output_dir / 'summary.txt', metrics)
    return metrics


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Evaluate a trained latent autoencoder checkpoint.')
    parser.add_argument('--checkpoint_path', required=True, help='Path to params_<step>.pkl.')
    parser.add_argument('--output_dir', required=True, help='Directory for metrics.json, summary.txt, and plots.')
    parser.add_argument('--env_name', default=None, help='Override checkpoint training_config env_name.')
    parser.add_argument('--dataset_dir', default=None, help='Override checkpoint training_config dataset_dir.')
    parser.add_argument('--dataset_path', default=None, help='Override checkpoint training_config dataset_path.')
    parser.add_argument('--max_examples', type=int, default=10000, help='Examples per split; <=0 means full sweep.')
    parser.add_argument('--batch_size', type=int, default=4096, help='Inference batch size.')
    parser.add_argument('--seed', type=int, default=None, help='Evaluation sampling seed. Defaults to checkpoint seed.')
    parser.add_argument('--probe_dims', default=None, help='Comma-separated raw observation dims for probe. Defaults to all dims.')
    parser.add_argument(
        '--plot_dims', default=None, help='Comma-separated raw observation dims to plot. Defaults to all dims.'
    )
    parser.add_argument('--pair_count', type=int, default=20000, help='Random pairs for distance preservation.')
    parser.add_argument('--knn_k', type=int, default=10, help='Neighborhood size for kNN overlap.')
    parser.add_argument('--knn_anchors', type=int, default=512, help='Number of anchor points for kNN overlap.')
    parser.add_argument('--dead_std_threshold', type=float, default=1e-4, help='Latent std below this counts as dead.')
    parser.add_argument('--ridge', type=float, default=1e-3, help='Ridge penalty for linear probe.')
    parser.add_argument('--include_random_baseline', action='store_true', help='Also evaluate an untrained random AE.')
    parser.add_argument('--no_plots', action='store_true', help='Skip plot generation.')
    parser.add_argument('--plot_max_points', type=int, default=5000, help='Maximum points per plot.')
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    _, _, training_config, stats = load_checkpoint(args.checkpoint_path)
    training_config = replace(
        training_config,
        env_name=args.env_name or training_config.env_name,
        dataset_dir=args.dataset_dir or training_config.dataset_dir,
        dataset_path=args.dataset_path if args.dataset_path is not None else training_config.dataset_path,
    )
    datasets = load_datasets_for_checkpoint(
        training_config,
        stats,
        env_name=training_config.env_name,
        dataset_dir=training_config.dataset_dir,
        dataset_path=training_config.dataset_path,
    )
    evaluate_checkpoint_on_splits(
        checkpoint_path=args.checkpoint_path,
        split_datasets=datasets,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
        batch_size=args.batch_size,
        seed=args.seed,
        probe_dims=parse_dims(args.probe_dims) if args.probe_dims is not None else None,
        pair_count=args.pair_count,
        knn_k=args.knn_k,
        knn_anchors=args.knn_anchors,
        dead_std_threshold=args.dead_std_threshold,
        ridge=args.ridge,
        include_random_baseline=args.include_random_baseline,
        write_plots=not args.no_plots,
        plot_dims=parse_dims(args.plot_dims) if args.plot_dims is not None else None,
        plot_max_points=args.plot_max_points,
    )


if __name__ == '__main__':
    main()
