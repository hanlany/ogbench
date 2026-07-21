import pickle
import tempfile
import unittest
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from latent.train.train_autoencoder import (
    ModelConfig as AutoEncoderModelConfig,
    NormalizationStats,
    TrainingConfig as AutoEncoderTrainingConfig,
    create_train_state as create_autoencoder_train_state,
    save_checkpoint as save_autoencoder_checkpoint,
)
from latent.train.train_dynamics import (
    DynamicsLossConfig,
    DynamicsModelConfig,
    DynamicsTrainingConfig,
    create_train_state as create_dynamics_train_state,
    save_checkpoint as save_dynamics_checkpoint,
)
from latent.validate.evaluate_dynamics import (
    aggregate_metrics,
    build_rollout_windows,
    episode_slices,
    evaluate_rollout_dataset,
    load_dynamics_artifacts,
    metric_summary,
    parse_horizons,
    recursive_rollout,
    teacher_forced_predictions,
)


def make_ae_model_config(obs_dim=4, latent_dim=2):
    return AutoEncoderModelConfig(
        hidden_dims=(4,),
        latent_dim=latent_dim,
        obs_dim=obs_dim,
        decoder_hidden_dims=(4,),
        activation='gelu',
        layer_norm=False,
        decoder_activate_final=False,
        dropout_rate=0.0,
    )


def make_ae_training_config(save_dir):
    return AutoEncoderTrainingConfig(
        run_group='test-ae',
        seed=0,
        env_name='dummy-v0',
        dataset_dir='unused',
        dataset_path=None,
        save_dir=str(save_dir),
        restore_path=None,
        restore_step=None,
        wandb_mode='disabled',
        lr=1e-3,
        min_lr=1e-6,
        lr_plateau_patience=5,
        lr_plateau_factor=0.5,
        lr_plateau_min_delta=1e-4,
        batch_size=4,
        train_steps=2,
        log_interval=1,
        save_interval=2,
        prefetch_batches=0,
        normalize_observations=True,
        normalization_eps=1e-6,
        validation_batches=1,
        validation_batch_size=4,
    )


def make_dynamics_model_config(latent_dim=2, action_dim=1, prediction_type='absolute'):
    return DynamicsModelConfig(
        hidden_dims=(4,),
        latent_dim=latent_dim,
        action_dim=action_dim,
        activation='gelu',
        layer_norm=False,
        dropout_rate=0.0,
        prediction_type=prediction_type,
    )


def make_dynamics_training_config(save_dir, ae_checkpoint_path):
    return DynamicsTrainingConfig(
        run_group='test-dynamics',
        seed=0,
        env_name='dummy-v0',
        dataset_dir='unused',
        dataset_path=None,
        save_dir=str(save_dir),
        restore_path=None,
        restore_step=None,
        wandb_mode='disabled',
        ae_checkpoint_path=str(ae_checkpoint_path),
        lr=1e-3,
        batch_size=2,
        train_steps=2,
        log_interval=1,
        save_interval=2,
        prefetch_batches=0,
        validation_batches=1,
        validation_batch_size=2,
        encoder_batch_size=2,
    )


def make_loss_config():
    return DynamicsLossConfig(
        recon_weight=1.0,
        latent_weight=1.0,
        gramian_warmup_steps=1,
        gramian_diag_eps=1e-3,
        differentiate_gramian=False,
        latent_loss_mode='l2',
    )


def create_test_checkpoints(root: Path, prediction_type='absolute'):
    ae_dir = root / 'ae'
    dynamics_dir = root / 'dynamics'
    ae_dir.mkdir()
    dynamics_dir.mkdir()
    ae_model_config = make_ae_model_config()
    ae_training_config = make_ae_training_config(ae_dir)
    ae_state = create_autoencoder_train_state(0, ae_model_config, ae_training_config)
    stats = NormalizationStats(mean=np.zeros(4, dtype=np.float32), std=np.ones(4, dtype=np.float32))
    save_autoencoder_checkpoint(ae_state, ae_dir, 2, ae_training_config, ae_model_config, stats, {'seed': 0})
    ae_path = ae_dir / 'params_2.pkl'

    dynamics_model_config = make_dynamics_model_config(prediction_type=prediction_type)
    dynamics_training_config = make_dynamics_training_config(dynamics_dir, ae_path)
    dynamics_state = create_dynamics_train_state(0, dynamics_model_config, dynamics_training_config)
    save_dynamics_checkpoint(
        dynamics_state,
        dynamics_dir,
        2,
        dynamics_training_config,
        dynamics_model_config,
        make_loss_config(),
        ae_path,
        ae_model_config,
        ae_training_config,
        stats,
        {'seed': 0},
    )
    return dynamics_dir / 'params_2.pkl'


class DynamicsEvaluationTest(unittest.TestCase):
    def test_parse_horizons(self):
        self.assertEqual(parse_horizons('25,1,5,5'), (1, 5, 25))
        with self.assertRaises(ValueError):
            parse_horizons('0,5')
        with self.assertRaises(ValueError):
            parse_horizons('')

    def test_episode_windows_use_valids_and_preserve_alignment(self):
        observations = np.arange(12, dtype=np.float32).reshape(6, 2)
        actions = np.arange(6, dtype=np.float32)[:, None]
        dataset = {
            'observations': observations,
            'actions': actions,
            # Compact OGBench terminals mark both the final transition and final state.
            'terminals': np.array([0, 1, 1, 0, 1, 1], dtype=np.float32),
            'valids': np.array([1, 1, 0, 1, 1, 0], dtype=np.float32),
        }

        self.assertEqual(episode_slices(dataset['terminals'], dataset['valids']), [(0, 3), (3, 6)])
        windows = build_rollout_windows(dataset, max_horizon=2, rollouts_per_episode=0, seed=0)

        np.testing.assert_array_equal(windows['start_indices'], [0, 3])
        np.testing.assert_array_equal(windows['actions'][0, :, 0], [0, 1])
        np.testing.assert_array_equal(windows['target_observations'][0], observations[[1, 2]])
        np.testing.assert_array_equal(windows['episode_ids'], [0, 1])

    def test_window_sampling_is_deterministic_and_balanced(self):
        observations = np.arange(32, dtype=np.float32).reshape(16, 2)
        dataset = {
            'observations': observations,
            'actions': np.zeros((16, 1), dtype=np.float32),
            'terminals': np.array([0] * 7 + [1] + [0] * 7 + [1], dtype=np.float32),
        }
        first = build_rollout_windows(dataset, max_horizon=2, rollouts_per_episode=2, seed=7)
        second = build_rollout_windows(dataset, max_horizon=2, rollouts_per_episode=2, seed=7)
        np.testing.assert_array_equal(first['start_indices'], second['start_indices'])
        np.testing.assert_array_equal(np.bincount(first['episode_ids']), [2, 2])

    def test_window_sampling_applies_deterministic_total_cap(self):
        observations = np.arange(60, dtype=np.float32).reshape(30, 2)
        dataset = {
            'observations': observations,
            'actions': np.zeros((30, 1), dtype=np.float32),
            'terminals': np.array(([0] * 9 + [1]) * 3, dtype=np.float32),
        }
        first = build_rollout_windows(
            dataset,
            max_horizon=2,
            rollouts_per_episode=0,
            seed=7,
            max_rollouts=5,
        )
        second = build_rollout_windows(
            dataset,
            max_horizon=2,
            rollouts_per_episode=0,
            seed=7,
            max_rollouts=5,
        )
        self.assertEqual(len(first['start_indices']), 5)
        np.testing.assert_array_equal(first['start_indices'], second['start_indices'])
        with self.assertRaisesRegex(ValueError, 'max_rollouts must be positive'):
            build_rollout_windows(dataset, max_horizon=2, rollouts_per_episode=0, seed=7, max_rollouts=0)

    def test_recursive_and_teacher_forced_rollouts(self):
        initial = jnp.array([[0.0]], dtype=jnp.float32)
        actions = jnp.ones((1, 3, 1), dtype=jnp.float32)
        exact = recursive_rollout(lambda z, u: z + u, initial, actions)
        np.testing.assert_allclose(exact, [[[1.0], [2.0], [3.0]]])

        target_sources = jnp.array([[[0.0], [1.0], [2.0]]], dtype=jnp.float32)

        def biased_fn(z, u):
            return z + u + 0.1

        open_loop = recursive_rollout(biased_fn, initial, actions)
        teacher = teacher_forced_predictions(biased_fn, target_sources, actions)
        target = np.array([[[1.0], [2.0], [3.0]]], dtype=np.float32)
        np.testing.assert_allclose(np.asarray(open_loop) - target, [[[0.1], [0.2], [0.3]]], atol=1e-6)
        np.testing.assert_allclose(np.asarray(teacher) - target, 0.1, atol=1e-6)

    def test_metric_aggregation_and_finite_rate(self):
        target = np.zeros((2, 2), dtype=np.float32)
        prediction = np.array([[1.0, 2.0], [np.nan, 0.0]], dtype=np.float32)
        summary = metric_summary(prediction, prediction, prediction, target, target, target)

        self.assertEqual(summary['finite_rate'], 0.5)
        self.assertAlmostEqual(summary['raw']['rmse'], np.sqrt(2.5))
        np.testing.assert_allclose(summary['raw']['per_dim_rmse'], [1.0, 2.0])
        self.assertAlmostEqual(summary['xy']['mean'], np.sqrt(5.0))

        sequence = np.ones((2, 2, 2), dtype=np.float32)
        metrics = aggregate_metrics(
            {'method': {'latent': sequence, 'normalized': sequence, 'raw': sequence}},
            {'latent': np.zeros_like(sequence), 'normalized': np.zeros_like(sequence), 'raw': np.zeros_like(sequence)},
            (1, 2),
        )
        self.assertEqual(set(metrics['method']), {'1', '2'})
        self.assertEqual(metrics['method']['2']['raw']['rmse'], 1.0)

    def test_prediction_type_checkpoint_metadata_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_path = create_test_checkpoints(root, prediction_type='residual')
            residual_artifacts = load_dynamics_artifacts(checkpoint_path)
            self.assertEqual(residual_artifacts.model_config.prediction_type, 'residual')

            with checkpoint_path.open('rb') as file:
                legacy_checkpoint = pickle.load(file)
            del legacy_checkpoint['model_config']['prediction_type']
            legacy_path = root / 'legacy_absolute.pkl'
            with legacy_path.open('wb') as file:
                pickle.dump(legacy_checkpoint, file)

            legacy_artifacts = load_dynamics_artifacts(legacy_path)
            self.assertEqual(legacy_artifacts.model_config.prediction_type, 'absolute')

    def test_checkpoint_validation_and_end_to_end_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_path = create_test_checkpoints(root)
            artifacts = load_dynamics_artifacts(checkpoint_path)
            self.assertEqual(artifacts.model_config.latent_dim, 2)
            self.assertEqual(artifacts.model_config.action_dim, 1)

            observations = np.arange(40, dtype=np.float32).reshape(10, 4) / 10.0
            dataset = {
                'observations': observations,
                'actions': np.ones((10, 1), dtype=np.float32) * 0.1,
                'terminals': np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 1], dtype=np.float32),
                'valids': np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0], dtype=np.float32),
            }
            output_dir = root / 'rollout_eval'
            metrics = evaluate_rollout_dataset(
                checkpoint_path,
                dataset,
                output_dir,
                horizons=(1, 2),
                rollouts_per_episode=1,
                max_rollouts=1,
                batch_size=2,
                seed=3,
                num_plot_rollouts=2,
                evaluation_env_name='dummy-explore-v0',
            )

            self.assertEqual(metrics['settings']['num_rollouts'], 1)
            self.assertEqual(metrics['settings']['max_rollouts'], 1)
            self.assertEqual(metrics['environment'], 'dummy-explore-v0')
            self.assertEqual(metrics['training_environment'], 'dummy-v0')
            self.assertEqual(set(metrics['methods']), {'open_loop', 'teacher_forced', 'ae_floor', 'persistence'})
            self.assertTrue((output_dir / 'metrics.json').is_file())
            self.assertTrue((output_dir / 'summary.txt').is_file())
            for name in (
                'error_by_horizon.png',
                'xy_endpoint_error.png',
                'per_dimension_rmse.png',
                'example_xy_rollouts.png',
            ):
                self.assertTrue((output_dir / 'plots' / name).is_file(), name)

            with checkpoint_path.open('rb') as file:
                broken = pickle.load(file)
            del broken['params']
            broken_path = root / 'broken.pkl'
            with broken_path.open('wb') as file:
                pickle.dump(broken, file)
            with self.assertRaisesRegex(ValueError, 'missing keys'):
                load_dynamics_artifacts(broken_path)


if __name__ == '__main__':
    unittest.main()
