"""Frozen-model inference and deterministic demonstrated-control caches."""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from latent.validate.evaluate_dynamics import episode_slices, load_dynamics_artifacts


SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class PlannerCache:
    """Arrays and provenance for one deterministic planner sample pool."""

    state_latents: np.ndarray
    state_observations: np.ndarray
    state_dataset_indices: np.ndarray
    action_snippets: np.ndarray
    action_lengths: np.ndarray
    action_start_indices: np.ndarray
    metadata: dict[str, Any]
    calibration_latents: np.ndarray | None = None
    calibration_observations: np.ndarray | None = None
    calibration_dataset_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        arrays = (
            self.state_latents,
            self.state_observations,
            self.state_dataset_indices,
            self.action_snippets,
            self.action_lengths,
            self.action_start_indices,
        )
        for array in arrays:
            if not isinstance(array, np.ndarray):
                raise TypeError('Planner cache arrays must be NumPy arrays.')
        if self.state_latents.ndim != 2 or self.state_observations.ndim != 2:
            raise ValueError('State cache arrays must have shape (N, dimension).')
        if len(self.state_latents) != len(self.state_observations) != len(self.state_dataset_indices):
            raise ValueError('State cache arrays have inconsistent lengths.')
        if self.action_snippets.ndim != 3:
            raise ValueError('action_snippets must have shape (M, Tmax, action_dim).')
        if len(self.action_snippets) != len(self.action_lengths) != len(self.action_start_indices):
            raise ValueError('Action cache arrays have inconsistent lengths.')
        if np.any(self.action_lengths < 1) or np.any(self.action_lengths > self.action_snippets.shape[1]):
            raise ValueError('Action lengths must be in [1, Tmax].')
        if np.any(~np.isfinite(self.state_latents)) or np.any(~np.isfinite(self.action_snippets)):
            raise ValueError('Planner cache contains non-finite latent or action values.')
        if np.any(~np.isfinite(self.state_observations)):
            raise ValueError('Planner cache contains non-finite state observations.')
        if np.any(self.state_dataset_indices < 0) or np.any(self.action_start_indices < 0):
            raise ValueError('Planner cache dataset indices must be non-negative.')
        if self.calibration_latents is not None:
            if self.calibration_observations is None or self.calibration_dataset_indices is None:
                raise ValueError('Calibration latents, observations, and indices must be provided together.')
            if (
                len(self.calibration_latents)
                != len(self.calibration_observations)
                != len(self.calibration_dataset_indices)
            ):
                raise ValueError('Calibration arrays have inconsistent lengths.')
            if not np.all(np.isfinite(self.calibration_latents)) or not np.all(
                np.isfinite(self.calibration_observations)
            ):
                raise ValueError('Planner cache contains non-finite calibration values.')

    @property
    def max_snippet_length(self) -> int:
        return int(self.action_snippets.shape[1])

    @property
    def latent_dim(self) -> int:
        return int(self.state_latents.shape[-1])

    @property
    def observation_dim(self) -> int:
        return int(self.state_observations.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.action_snippets.shape[-1])

    def save(self, path: str | Path) -> tuple[Path, Path]:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            state_latents=self.state_latents,
            state_observations=self.state_observations,
            state_dataset_indices=self.state_dataset_indices,
            action_snippets=self.action_snippets,
            action_lengths=self.action_lengths,
            action_start_indices=self.action_start_indices,
            calibration_latents=self.calibration_latents
            if self.calibration_latents is not None
            else np.empty((0, self.latent_dim), dtype=np.float32),
            calibration_observations=self.calibration_observations
            if self.calibration_observations is not None
            else np.empty((0, self.observation_dim), dtype=np.float32),
            calibration_dataset_indices=self.calibration_dataset_indices
            if self.calibration_dataset_indices is not None
            else np.empty((0,), dtype=np.int64),
        )
        metadata_path = path.with_suffix(path.suffix + '.json')
        metadata_path.write_text(json.dumps(_jsonable(self.metadata), indent=2, sort_keys=True) + '\n')
        return path, metadata_path


def load_planner_cache(path: str | Path, expected: dict[str, Any] | None = None) -> PlannerCache:
    """Restore and strictly validate a cache, including requested provenance."""
    path = Path(path).expanduser().resolve()
    metadata_path = path.with_suffix(path.suffix + '.json')
    if not path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f'Planner cache and metadata are both required: {path}, {metadata_path}.')
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'Unsupported planner cache schema: {metadata.get("schema_version")!r}.')
    if expected:
        for key, value in expected.items():
            if metadata.get(key) != _jsonable(value):
                raise ValueError(f'Planner cache provenance mismatch for {key!r}: {metadata.get(key)!r} != {value!r}.')
    required_metadata = (
        'source_dataset_path',
        'source_dataset_sha256',
        'checkpoint_path',
        'checkpoint_sha256',
        'checkpoint_step',
        'ae_checkpoint_path',
        'ae_checkpoint_sha256',
        'normalization_sha256',
        'observation_dim',
        'latent_dim',
        'action_dim',
        'max_snippet_length',
        'sampling_seed',
    )
    missing = [key for key in required_metadata if key not in metadata]
    if missing:
        raise ValueError(f'Planner cache metadata is missing required provenance keys: {missing}.')
    with np.load(path, allow_pickle=False) as arrays:
        optional = arrays['calibration_latents']
        calibration_latents = optional if len(optional) else None
        optional = arrays['calibration_observations']
        calibration_observations = optional if len(optional) else None
        optional = arrays['calibration_dataset_indices']
        calibration_indices = optional if len(optional) else None
        cache = PlannerCache(
            state_latents=np.asarray(arrays['state_latents'], dtype=np.float32),
            state_observations=np.asarray(arrays['state_observations'], dtype=np.float32),
            state_dataset_indices=np.asarray(arrays['state_dataset_indices'], dtype=np.int64),
            action_snippets=np.asarray(arrays['action_snippets'], dtype=np.float32),
            action_lengths=np.asarray(arrays['action_lengths'], dtype=np.int64),
            action_start_indices=np.asarray(arrays['action_start_indices'], dtype=np.int64),
            metadata=metadata,
            calibration_latents=None
            if calibration_latents is None
            else np.asarray(calibration_latents, dtype=np.float32),
            calibration_observations=None
            if calibration_observations is None
            else np.asarray(calibration_observations, dtype=np.float32),
            calibration_dataset_indices=None
            if calibration_indices is None
            else np.asarray(calibration_indices, dtype=np.int64),
        )
    dimensions = {
        'observation_dim': cache.observation_dim,
        'latent_dim': cache.latent_dim,
        'action_dim': cache.action_dim,
        'max_snippet_length': cache.max_snippet_length,
    }
    for key, value in dimensions.items():
        if metadata[key] != value:
            raise ValueError(f'Planner cache metadata does not match its arrays for {key!r}.')
    return cache


class LatentModelAdapter:
    """Inference-only adapter around the Plan 04 dynamics and linked AE."""

    def __init__(self, artifacts: Any, max_horizon: int = 10):
        if max_horizon <= 0:
            raise ValueError('max_horizon must be positive.')
        self.artifacts = artifacts
        self.max_horizon = int(max_horizon)
        self.latent_dim = int(artifacts.model_config.latent_dim)
        self.action_dim = int(artifacts.model_config.action_dim)
        self.observation_dim = int(artifacts.normalization.mean.shape[0])
        if artifacts.ae_state is None or artifacts.model is None:
            raise ValueError('Dynamics artifacts must include both the AE and dynamics model.')
        normalization_mean = np.asarray(artifacts.normalization.mean, dtype=np.float32)
        normalization_std = np.asarray(artifacts.normalization.std, dtype=np.float32)
        if normalization_mean.shape != (self.observation_dim,) or normalization_std.shape != (self.observation_dim,):
            raise ValueError('Checkpoint normalization must be one vector matching observation_dim.')
        if not np.all(np.isfinite(normalization_mean)) or not np.all(np.isfinite(normalization_std)):
            raise ValueError('Checkpoint normalization must be finite.')
        if np.any(normalization_std <= 0):
            raise ValueError('Checkpoint normalization standard deviations must be positive.')
        self._normalization_mean = jnp.asarray(normalization_mean, dtype=jnp.float32)
        self._normalization_std = jnp.asarray(normalization_std, dtype=jnp.float32)

        def encode_fn(observations):
            normalized = (observations - self._normalization_mean) / self._normalization_std
            return artifacts.ae_state(normalized, method='encode', deterministic=True)

        def decode_fn(latents):
            normalized = artifacts.ae_state(latents, method='decode', deterministic=True)
            return normalized * self._normalization_std + self._normalization_mean

        def dynamics_fn(latents, actions):
            return artifacts.model.apply({'params': artifacts.params}, latents, actions, deterministic=True)

        def rollout_fn(initial_latents, actions, lengths):
            time_major = jnp.swapaxes(actions, 0, 1)
            active = jnp.swapaxes(jnp.arange(actions.shape[1])[None, :] < lengths[:, None], 0, 1)

            def step(latents, values):
                step_actions, step_active = values
                next_latents = dynamics_fn(latents, step_actions)
                # Carrying the previous state after the declared length makes the
                # padded suffix inert; callers must still use the returned mask.
                next_latents = jnp.where(step_active[:, None], next_latents, latents)
                return next_latents, next_latents

            _, predictions = jax.lax.scan(step, initial_latents, (time_major, active))
            return jnp.swapaxes(predictions, 0, 1)

        self._encode = jax.jit(encode_fn)
        self._decode = jax.jit(decode_fn)
        self._rollout = jax.jit(rollout_fn)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, max_horizon: int = 10) -> 'LatentModelAdapter':
        return cls(load_dynamics_artifacts(checkpoint_path), max_horizon=max_horizon)

    def encode(self, observations: np.ndarray) -> np.ndarray:
        observations = np.asarray(observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[-1] != self.observation_dim:
            raise ValueError(f'Expected observations with shape (N, {self.observation_dim}), got {observations.shape}.')
        if not np.all(np.isfinite(observations)):
            raise ValueError('Cannot encode non-finite observations.')
        result = np.asarray(jax.device_get(self._encode(jnp.asarray(observations))), dtype=np.float32)
        if result.shape != (len(observations), self.latent_dim):
            raise ValueError(
                f'The encoder returned shape {result.shape}, expected {(len(observations), self.latent_dim)}.'
            )
        if not np.all(np.isfinite(result)):
            raise ValueError('The encoder returned non-finite latents.')
        return result

    def decode(self, latents: np.ndarray) -> np.ndarray:
        latents = np.asarray(latents, dtype=np.float32)
        if latents.ndim != 2 or latents.shape[-1] != self.latent_dim:
            raise ValueError(f'Expected latents with shape (N, {self.latent_dim}), got {latents.shape}.')
        if not np.all(np.isfinite(latents)):
            raise ValueError('Cannot decode non-finite latents.')
        result = np.asarray(jax.device_get(self._decode(jnp.asarray(latents))), dtype=np.float32)
        if result.shape != (len(latents), self.observation_dim):
            raise ValueError(
                f'The decoder returned shape {result.shape}, expected {(len(latents), self.observation_dim)}.'
            )
        if not np.all(np.isfinite(result)):
            raise ValueError('The decoder returned non-finite observations.')
        return result

    def propagate(
        self, initial_latents: np.ndarray, actions: np.ndarray, lengths: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        initial_latents = np.asarray(initial_latents, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        lengths = np.asarray(lengths, dtype=np.int32).reshape(-1)
        if actions.ndim != 3 or actions.shape[-1] != self.action_dim or actions.shape[1] != self.max_horizon:
            raise ValueError(
                f'Expected actions with shape (N, {self.max_horizon}, {self.action_dim}), got {actions.shape}.'
            )
        if initial_latents.shape != (len(actions), self.latent_dim):
            raise ValueError('Initial latent and action batch shapes do not match.')
        if np.any(lengths < 1) or np.any(lengths > self.max_horizon):
            raise ValueError('Propagation lengths must be in [1, max_horizon].')
        if not np.all(np.isfinite(initial_latents)) or not np.all(np.isfinite(actions)):
            raise ValueError('Cannot propagate non-finite latents or actions.')
        predicted = np.asarray(
            jax.device_get(self._rollout(jnp.asarray(initial_latents), jnp.asarray(actions), jnp.asarray(lengths))),
            dtype=np.float32,
        )
        if predicted.shape != (len(actions), self.max_horizon, self.latent_dim):
            raise ValueError(
                f'The dynamics model returned shape {predicted.shape}, which does not match the fixed rollout shape.'
            )
        mask = np.arange(self.max_horizon)[None, :] < lengths[:, None]
        if not np.all(np.isfinite(predicted[mask])):
            raise ValueError('The dynamics model returned non-finite predicted latents.')
        return predicted, mask

    def decode_rollout(self, latents: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        latents = np.asarray(latents, dtype=np.float32)
        if latents.ndim < 2 or latents.shape[-1] != self.latent_dim:
            raise ValueError(f'Expected rollout latents ending in {self.latent_dim}, got {latents.shape}.')
        shape = latents.shape
        flat = latents.reshape(-1, self.latent_dim)
        decoded = np.asarray(jax.device_get(self._decode(jnp.asarray(flat))), dtype=np.float32)
        if decoded.shape != (len(flat), self.observation_dim):
            raise ValueError(
                f'The decoder returned rollout shape {decoded.shape}, expected {(len(flat), self.observation_dim)}.'
            )
        decoded = decoded.reshape(*shape[:-1], self.observation_dim)
        if mask is not None and not np.all(np.isfinite(decoded[np.asarray(mask, dtype=bool)])):
            raise ValueError('The decoder returned non-finite predicted observations.')
        if mask is None and not np.all(np.isfinite(decoded)):
            raise ValueError('The decoder returned non-finite predicted observations.')
        return decoded


def cache_identity(adapter: LatentModelAdapter, dataset_path: str | Path, max_snippet_length: int) -> dict[str, Any]:
    """Return immutable provenance fields used to validate a cache."""
    dataset_path = Path(dataset_path).expanduser().resolve()
    artifacts = adapter.artifacts
    checkpoint_path = Path(artifacts.checkpoint_path).expanduser().resolve()
    ae_checkpoint_path = Path(artifacts.ae_checkpoint_path).expanduser().resolve()
    normalization_mean = np.asarray(artifacts.normalization.mean, dtype=np.float32)
    normalization_std = np.asarray(artifacts.normalization.std, dtype=np.float32)
    return {
        'source_dataset_path': str(dataset_path),
        'source_dataset_sha256': _sha256(dataset_path),
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_sha256': _sha256(checkpoint_path),
        'checkpoint_step': int(artifacts.checkpoint.get('step', 0)),
        'ae_checkpoint_path': str(ae_checkpoint_path),
        'ae_checkpoint_sha256': _sha256(ae_checkpoint_path),
        'normalization_sha256': hashlib.sha256(normalization_mean.tobytes() + normalization_std.tobytes()).hexdigest(),
        'observation_dim': adapter.observation_dim,
        'latent_dim': adapter.latent_dim,
        'action_dim': adapter.action_dim,
        'max_snippet_length': int(max_snippet_length),
    }


def _cache_metadata(
    adapter: LatentModelAdapter,
    dataset_path: Path,
    max_snippet_length: int,
    state_samples: int,
    action_snippet_starts: int,
    seed: int,
    creation_command: str | None,
) -> dict[str, Any]:
    artifacts = adapter.artifacts
    metadata = {
        'schema_version': SCHEMA_VERSION,
        **cache_identity(adapter, dataset_path, max_snippet_length),
        'normalization_mean': np.asarray(artifacts.normalization.mean, dtype=np.float32).tolist(),
        'normalization_std': np.asarray(artifacts.normalization.std, dtype=np.float32).tolist(),
        'state_samples': int(state_samples),
        'action_snippet_starts': int(action_snippet_starts),
        'sampling_seed': int(seed),
        'padding_value': 0.0,
        'creation_command': creation_command or ' '.join(shlex.quote(item) for item in __import__('sys').argv),
    }
    return metadata


def build_planner_cache(
    adapter: LatentModelAdapter,
    dataset_path: str | Path,
    cache_path: str | Path,
    *,
    state_samples: int = 20_000,
    action_snippet_starts: int = 50_000,
    max_snippet_length: int = 10,
    seed: int = 0,
    creation_command: str | None = None,
) -> PlannerCache:
    """Build a deterministic state/snippet cache without crossing terminals."""
    dataset_path = Path(dataset_path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if state_samples <= 0 or action_snippet_starts <= 0 or max_snippet_length <= 0:
        raise ValueError('Cache sample counts and max_snippet_length must be positive.')
    with np.load(dataset_path, allow_pickle=False) as data:
        raw = {key: np.asarray(data[key]) for key in data.files}
    for key in ('observations', 'actions', 'terminals'):
        if key not in raw:
            raise ValueError(f'Dataset is missing {key!r}.')
    observations = np.asarray(raw['observations'], dtype=np.float32)
    actions = np.asarray(raw['actions'], dtype=np.float32)
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError('Cache builder supports vector observations/actions only.')
    if observations.shape[0] != actions.shape[0] or observations.shape[1] != adapter.observation_dim:
        raise ValueError('Dataset observation shape does not match the selected checkpoint.')
    if actions.shape[1] != adapter.action_dim:
        raise ValueError('Dataset action shape does not match the selected checkpoint.')
    slices = episode_slices(raw['terminals'], raw.get('valids'))
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(observations), dtype=np.int64)
    total_needed = min(len(all_indices), state_samples + max(1000, state_samples // 5))
    selected = np.sort(rng.choice(all_indices, size=total_needed, replace=False))
    state_indices = np.sort(selected[: min(state_samples, len(selected))])
    calibration_indices = np.sort(selected[len(state_indices) :])
    state_observations = observations[state_indices]
    calibration_observations = observations[calibration_indices]
    state_latents = adapter.encode(state_observations)
    calibration_latents = adapter.encode(calibration_observations) if len(calibration_indices) else None

    starts_by_episode = []
    for start, end in slices:
        # Leave one final transition unused: snippets then always have a real
        # post-action observation and cannot include a terminal boundary.
        starts_by_episode.extend(np.arange(start, end - 1, dtype=np.int64).tolist())
    starts_by_episode = np.asarray(starts_by_episode, dtype=np.int64)
    if not len(starts_by_episode):
        raise ValueError('No episode-safe action snippet starts are available.')
    chosen_positions = np.sort(
        rng.choice(len(starts_by_episode), size=min(action_snippet_starts, len(starts_by_episode)), replace=False)
    )
    starts = starts_by_episode[chosen_positions]
    lengths = np.empty(len(starts), dtype=np.int64)
    for index, start in enumerate(starts):
        episode_end = next(end for begin, end in slices if begin <= start < end)
        max_length = min(max_snippet_length, episode_end - int(start) - 1)
        lengths[index] = int(rng.integers(1, max_length + 1))
    snippets = np.zeros((len(starts), max_snippet_length, adapter.action_dim), dtype=np.float32)
    for index, (start, length) in enumerate(zip(starts, lengths)):
        snippets[index, :length] = actions[start : start + length]
    metadata = _cache_metadata(
        adapter,
        dataset_path,
        max_snippet_length,
        len(state_indices),
        len(starts),
        seed,
        creation_command,
    )
    metadata.update(
        {
            'state_dataset_indices_sha256': hashlib.sha256(state_indices.tobytes()).hexdigest(),
            'calibration_dataset_indices_sha256': hashlib.sha256(calibration_indices.tobytes()).hexdigest(),
            'action_start_indices_sha256': hashlib.sha256(starts.tobytes()).hexdigest(),
            'latent_mean': np.mean(state_latents, axis=0, dtype=np.float64).astype(np.float32).tolist(),
            'latent_std': np.maximum(np.std(state_latents, axis=0, dtype=np.float64), 1e-6).astype(np.float32).tolist(),
            'xy_scale': np.maximum(np.std(state_observations[:, :2], axis=0, dtype=np.float64), 1e-6)
            .astype(np.float32)
            .tolist(),
        }
    )
    cache = PlannerCache(
        state_latents=state_latents,
        state_observations=state_observations,
        state_dataset_indices=state_indices,
        action_snippets=snippets,
        action_lengths=lengths,
        action_start_indices=starts,
        metadata=metadata,
        calibration_latents=calibration_latents,
        calibration_observations=calibration_observations,
        calibration_dataset_indices=calibration_indices,
    )
    cache.save(cache_path)
    return cache


def calibrate_metrics(cache: PlannerCache, latent_weight: float = 1.0) -> dict[str, Any]:
    """Measure held-out nearest-anchor distances for the three planner metrics."""
    if cache.calibration_latents is None or not len(cache.calibration_latents):
        raise ValueError('Cache has no held-out calibration queries.')
    if latent_weight < 0:
        raise ValueError('latent_weight must be non-negative.')
    latent_mean = np.asarray(cache.metadata['latent_mean'], dtype=np.float32)
    latent_std = np.maximum(np.asarray(cache.metadata['latent_std'], dtype=np.float32), 1e-6)
    xy_scale = np.maximum(np.asarray(cache.metadata['xy_scale'], dtype=np.float32), 1e-6)

    def nearest_distances(name: str) -> np.ndarray:
        nearest = np.full(len(cache.calibration_latents), np.inf, dtype=np.float32)
        if name == 'latent':
            queries = (cache.calibration_latents - latent_mean) / latent_std
            anchors = (cache.state_latents - latent_mean) / latent_std
        elif name == 'xy':
            queries = cache.calibration_observations[:, :2]
            anchors = cache.state_observations[:, :2]
        else:
            queries = (
                (cache.calibration_latents - latent_mean) / latent_std,
                cache.calibration_observations[:, :2] / xy_scale,
            )
            anchors = (
                (cache.state_latents - latent_mean) / latent_std,
                cache.state_observations[:, :2] / xy_scale,
            )
        for start in range(0, len(nearest), 256):
            end = min(start + 256, len(nearest))
            if name == 'hybrid':
                latent_distance = np.mean(
                    (queries[0][start:end, None, :] - anchors[0][None, :, :]) ** 2,
                    axis=-1,
                )
                xy_distance = np.sum(
                    (queries[1][start:end, None, :] - anchors[1][None, :, :]) ** 2,
                    axis=-1,
                )
                distances = np.sqrt(xy_distance + latent_weight * latent_distance)
            else:
                distances = np.linalg.norm(queries[start:end, None, :] - anchors[None, :, :], axis=-1)
            nearest[start:end] = np.min(distances, axis=1)
        return nearest

    results = {}
    for name in ('latent', 'xy', 'hybrid'):
        nearest = nearest_distances(name)
        results[name] = {
            'p50': float(np.quantile(nearest, 0.50)),
            'p90': float(np.quantile(nearest, 0.90)),
            'p99': float(np.quantile(nearest, 0.99)),
            'count': int(len(nearest)),
        }
    return {'latent_weight': float(latent_weight), 'metrics': results}


def calibrate_cycle_error(adapter: LatentModelAdapter, cache: PlannerCache) -> dict[str, Any]:
    """Measure encode/decode cycle error on the cache's held-out queries."""
    if cache.calibration_latents is None or not len(cache.calibration_latents):
        raise ValueError('Cache has no held-out calibration queries.')
    decoded = adapter.decode(cache.calibration_latents)
    recoded = adapter.encode(decoded)
    latent_std = np.maximum(np.asarray(cache.metadata['latent_std'], dtype=np.float32), 1e-6)
    errors = np.linalg.norm((recoded - cache.calibration_latents) / latent_std, axis=-1)
    return {
        'count': int(len(errors)),
        'p50': float(np.quantile(errors, 0.50)),
        'p90': float(np.quantile(errors, 0.90)),
        'p99': float(np.quantile(errors, 0.99)),
        'max': float(np.max(errors)),
        'threshold_quantile': 'p99',
        'threshold': float(np.quantile(errors, 0.99)),
    }
