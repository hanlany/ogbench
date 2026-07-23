from pathlib import Path

import numpy as np

import jax.numpy as jnp

from planner.artifacts import (
    LatentModelAdapter,
    PlannerCache,
    build_planner_cache,
    cache_identity,
    calibrate_cycle_error,
    calibrate_metrics,
    load_planner_cache,
)


class _FakeArtifacts:
    checkpoint_path = Path('/tmp/fake-dynamics.pkl')
    ae_checkpoint_path = Path('/tmp/fake-ae.pkl')
    checkpoint = {'step': 7}
    normalization = type('Stats', (), {'mean': np.zeros(2, dtype=np.float32), 'std': np.ones(2, dtype=np.float32)})()


class _FakeAdapter:
    observation_dim = 2
    action_dim = 1
    latent_dim = 2
    artifacts = _FakeArtifacts()

    def encode(self, observations):
        return np.asarray(observations, dtype=np.float32)


class _AdapterAE:
    def __call__(self, values, *, method, deterministic):
        del deterministic
        if method in ('encode', 'decode'):
            return values
        raise ValueError(method)


class _AdapterDynamics:
    def apply(self, variables, latents, actions, deterministic):
        del variables, deterministic
        return latents + actions


class _AdapterArtifacts:
    model_config = type('Config', (), {'latent_dim': 2, 'action_dim': 2})()
    normalization = type(
        'Stats',
        (),
        {'mean': np.array([10.0, 20.0], dtype=np.float32), 'std': np.array([2.0, 4.0], dtype=np.float32)},
    )()
    ae_state = _AdapterAE()
    model = _AdapterDynamics()
    params = {}


class _NonFiniteDynamics(_AdapterDynamics):
    def apply(self, variables, latents, actions, deterministic):
        del variables, actions, deterministic
        return jnp.full_like(latents, jnp.nan)


def _dataset(path):
    observations = np.arange(20, dtype=np.float32).reshape(10, 2)
    actions = np.arange(10, dtype=np.float32).reshape(10, 1)
    terminals = np.array([False, False, False, False, True, False, False, False, False, True])
    np.savez(path, observations=observations, actions=actions, terminals=terminals)


def test_cache_is_deterministic_and_episode_safe(tmp_path):
    checkpoint = tmp_path / 'fake-dynamics.pkl'
    ae = tmp_path / 'fake-ae.pkl'
    _FakeArtifacts.checkpoint_path = checkpoint
    _FakeArtifacts.ae_checkpoint_path = ae
    checkpoint.write_bytes(b'dynamics')
    ae.write_bytes(b'ae')
    dataset = tmp_path / 'data.npz'
    _dataset(dataset)
    first = build_planner_cache(
        _FakeAdapter(),
        dataset,
        tmp_path / 'one.npz',
        state_samples=4,
        action_snippet_starts=4,
        max_snippet_length=3,
        seed=3,
    )
    second = build_planner_cache(
        _FakeAdapter(),
        dataset,
        tmp_path / 'two.npz',
        state_samples=4,
        action_snippet_starts=4,
        max_snippet_length=3,
        seed=3,
    )
    assert np.array_equal(first.state_dataset_indices, second.state_dataset_indices)
    assert np.array_equal(first.action_start_indices, second.action_start_indices)
    assert np.array_equal(first.action_lengths, second.action_lengths)
    for start, length in zip(first.action_start_indices, first.action_lengths):
        assert not (start < 5 <= start + length)
        assert not (start < 10 <= start + length)
    restored = load_planner_cache(tmp_path / 'one.npz', expected={'max_snippet_length': 3, 'sampling_seed': 3})
    assert np.array_equal(restored.action_snippets, first.action_snippets)
    checkpoint.write_bytes(b'changed checkpoint')
    changed_identity = cache_identity(_FakeAdapter(), dataset, 3)
    try:
        load_planner_cache(tmp_path / 'one.npz', expected=changed_identity)
    except ValueError as exc:
        assert 'provenance mismatch' in str(exc)
    else:
        raise AssertionError('Expected changed checkpoint identity to be rejected.')


def test_cache_rejects_provenance_mismatch(tmp_path):
    metadata = {'schema_version': 1}
    cache = PlannerCache(
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.int64),
        np.zeros((1, 2, 1), dtype=np.float32),
        np.ones(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        metadata,
    )
    path = tmp_path / 'cache.npz'
    cache.save(path)
    try:
        load_planner_cache(path, expected={'sampling_seed': 1})
    except ValueError as exc:
        assert 'provenance mismatch' in str(exc)
    else:
        raise AssertionError('Expected provenance mismatch.')


def test_cache_rejects_metadata_dimension_mismatch(tmp_path):
    cache = PlannerCache(
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.int64),
        np.zeros((1, 2, 1), dtype=np.float32),
        np.ones(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        {
            'schema_version': 1,
            'source_dataset_path': 'dataset.npz',
            'source_dataset_sha256': 'dataset-hash',
            'checkpoint_path': 'dynamics.pkl',
            'checkpoint_sha256': 'dynamics-hash',
            'checkpoint_step': 1,
            'ae_checkpoint_path': 'ae.pkl',
            'ae_checkpoint_sha256': 'ae-hash',
            'normalization_sha256': 'normalization-hash',
            'observation_dim': 2,
            'latent_dim': 2,
            'action_dim': 1,
            'max_snippet_length': 2,
            'sampling_seed': 0,
        },
    )
    path = tmp_path / 'cache.npz'
    cache.save(path)
    metadata_path = path.with_suffix('.npz.json')
    metadata = __import__('json').loads(metadata_path.read_text())
    metadata['latent_dim'] = 99
    metadata_path.write_text(__import__('json').dumps(metadata))
    try:
        load_planner_cache(path)
    except ValueError as exc:
        assert 'does not match its arrays' in str(exc)
    else:
        raise AssertionError('Expected cache array/metadata mismatch.')


def test_adapter_uses_checkpoint_normalization_and_masks_padded_rollout():
    adapter = LatentModelAdapter(_AdapterArtifacts(), max_horizon=3)
    observations = np.array([[12.0, 24.0]], dtype=np.float32)
    latents = adapter.encode(observations)
    assert np.array_equal(latents, np.array([[1.0, 1.0]], dtype=np.float32))
    assert np.array_equal(adapter.decode(latents), observations)
    actions = np.array([[[1.0, 2.0], [99.0, 99.0], [99.0, 99.0]], [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]])
    predicted, mask = adapter.propagate(np.zeros((2, 2), dtype=np.float32), actions, np.array([1, 3]))
    assert np.array_equal(mask, np.array([[True, False, False], [True, True, True]]))
    assert np.array_equal(predicted[0, 0], np.array([1.0, 2.0], dtype=np.float32))
    assert np.array_equal(predicted[0, 1], predicted[0, 0])
    assert np.array_equal(predicted[1, -1], np.array([6.0, 6.0], dtype=np.float32))


def test_adapter_rejects_nonfinite_model_output():
    artifacts = _AdapterArtifacts()
    artifacts.model = _NonFiniteDynamics()
    adapter = LatentModelAdapter(artifacts, max_horizon=1)
    try:
        adapter.propagate(np.zeros((1, 2), dtype=np.float32), np.zeros((1, 1, 2), dtype=np.float32), np.array([1]))
    except ValueError as exc:
        assert 'non-finite' in str(exc)
    else:
        raise AssertionError('Expected non-finite dynamics output to be rejected.')


def test_calibration_reports_nearest_quantiles_and_cycle_threshold():
    cache = PlannerCache(
        state_latents=np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        state_observations=np.array([[10.0, 20.0], [14.0, 20.0]], dtype=np.float32),
        state_dataset_indices=np.array([0, 1], dtype=np.int64),
        action_snippets=np.zeros((1, 1, 2), dtype=np.float32),
        action_lengths=np.ones(1, dtype=np.int64),
        action_start_indices=np.zeros(1, dtype=np.int64),
        metadata={'latent_mean': [1.0, 0.0], 'latent_std': [1.0, 1.0], 'xy_scale': [2.0, 1.0]},
        calibration_latents=np.array([[1.0, 0.0]], dtype=np.float32),
        calibration_observations=np.array([[12.0, 20.0]], dtype=np.float32),
        calibration_dataset_indices=np.array([2], dtype=np.int64),
    )
    metrics = calibrate_metrics(cache)
    assert set(metrics['metrics']) == {'latent', 'xy', 'hybrid'}
    assert metrics['metrics']['latent']['p50'] == 1.0
    cycle = calibrate_cycle_error(LatentModelAdapter(_AdapterArtifacts(), max_horizon=1), cache)
    assert cycle['threshold'] == 0.0
