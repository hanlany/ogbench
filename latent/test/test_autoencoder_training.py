import pickle
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from impls.utils.datasets import Dataset
from latent.train.train_autoencoder import (
    ModelConfig,
    TrainingConfig,
    compute_normalization_stats,
    create_train_state,
    create_validation_batches,
    eval_step,
    normalize_observations,
    maybe_reduce_lr,
    restore_checkpoint,
    save_checkpoint,
    split_autoencoder_params,
    train_step,
    validate_model_config,
    validate_observations,
    validate_training_config,
)


def make_training_config(save_dir):
    return TrainingConfig(
        run_group='test',
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
        validation_batches=2,
        validation_batch_size=4,
    )


def make_model_config(obs_dim=3):
    return ModelConfig(
        hidden_dims=(8, 8),
        latent_dim=2,
        obs_dim=obs_dim,
        decoder_hidden_dims=None,
        activation='gelu',
        layer_norm=False,
        decoder_activate_final=False,
        dropout_rate=0.0,
    )


class AutoencoderTrainingTest(unittest.TestCase):
    def test_validate_observations_accepts_vector_observations(self):
        observations = validate_observations(np.ones((5, 3), dtype=np.float64), 'training')
        self.assertEqual(observations.shape, (5, 3))
        self.assertEqual(observations.dtype, np.float32)

    def test_validate_observations_rejects_non_vector_observations(self):
        for shape in [(), (3,), (2, 3, 4)]:
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    validate_observations(np.ones(shape), 'training')

    def test_normalization_uses_train_statistics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config = make_training_config(tmpdir)
            observations = np.array([[1.0, 2.0], [3.0, 2.0], [5.0, 2.0]], dtype=np.float32)

            stats = compute_normalization_stats(observations, config)
            normalized = normalize_observations(observations, stats)

            np.testing.assert_allclose(stats.mean, np.array([3.0, 2.0], dtype=np.float32))
            self.assertTrue(np.all(stats.std >= config.normalization_eps))
            np.testing.assert_allclose(np.mean(normalized[:, 0]), 0.0, atol=1e-6)
            np.testing.assert_allclose(normalized[:, 1], 0.0, atol=1e-6)

    def test_config_validation_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            training_config = make_training_config(tmpdir)
            validate_training_config(training_config)
            validate_model_config(make_model_config())

            with self.assertRaises(ValueError):
                validate_training_config(replace(training_config, lr=0.0))
            with self.assertRaises(ValueError):
                validate_model_config(make_model_config(obs_dim=2))

    def test_plateau_reduction_updates_optimizer_lr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            training_config = replace(
                make_training_config(tmpdir),
                lr=1e-3,
                min_lr=1e-5,
                lr_plateau_patience=1,
                lr_plateau_factor=0.5,
                lr_plateau_min_delta=0.0,
            )
            state = create_train_state(0, make_model_config(obs_dim=3), training_config)

            state, current_lr, best_metric, plateau_count, lr_reduced = maybe_reduce_lr(
                state,
                training_config.lr,
                metric=1.0,
                best_metric=1.0,
                plateau_count=0,
                config=training_config,
            )

            self.assertTrue(lr_reduced)
            self.assertEqual(plateau_count, 0)
            self.assertEqual(best_metric, 1.0)
            self.assertAlmostEqual(current_lr, 5e-4)
            opt_lr = float(np.asarray(state.opt_state.hyperparams['learning_rate']))
            self.assertAlmostEqual(opt_lr, 5e-4)

    def test_train_eval_and_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            training_config = make_training_config(tmpdir)
            model_config = make_model_config(obs_dim=3)
            state = create_train_state(0, model_config, training_config)

            raw_observations = np.arange(12, dtype=np.float32).reshape(4, 3)
            stats = compute_normalization_stats(raw_observations, training_config)
            observations = normalize_observations(raw_observations, stats)
            batch = {
                'observations': observations,
                'raw_observations': raw_observations,
            }

            new_state, train_metrics = train_step(state, batch, jax.random.PRNGKey(1))
            eval_metrics = eval_step(new_state, batch, jnp.asarray(stats.mean), jnp.asarray(stats.std))

            self.assertEqual(new_state.step, state.step + 1)
            self.assertIn('mse', train_metrics)
            self.assertIn('original/mse', eval_metrics)
            self.assertEqual(set(split_autoencoder_params(new_state.params)), {'encoder', 'decoder'})

            save_checkpoint(new_state, tmpdir, 2, training_config, model_config, stats, {'seed': 0})
            checkpoint_path = Path(tmpdir) / 'params_2.pkl'
            with checkpoint_path.open('rb') as f:
                checkpoint = pickle.load(f)

            self.assertEqual(checkpoint['schema_version'], 1)
            self.assertEqual(checkpoint['normalization']['mean'], stats.mean.tolist())
            self.assertEqual(set(checkpoint['components']), {'encoder', 'decoder'})

            restored_state, restored_step = restore_checkpoint(state, str(checkpoint_path), None)
            self.assertEqual(restored_step, 2)
            self.assertEqual(restored_state.step, new_state.step)

    def test_fixed_validation_batches_are_deterministic(self):
        observations = np.arange(30, dtype=np.float32).reshape(10, 3)
        dataset = Dataset.create(observations=observations, raw_observations=observations)

        batches_a = create_validation_batches(dataset, batch_size=4, num_batches=2, seed=7)
        batches_b = create_validation_batches(dataset, batch_size=4, num_batches=2, seed=7)

        for batch_a, batch_b in zip(batches_a, batches_b):
            np.testing.assert_array_equal(batch_a['observations'], batch_b['observations'])


if __name__ == '__main__':
    unittest.main()
