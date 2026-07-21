import pickle
import tempfile
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from latent.train.dynamics import LatentDynamics
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
    create_train_state,
    dynamics_loss,
    encode_transition_dataset,
    eval_step,
    load_autoencoder_artifacts,
    make_transition_dataset,
    restore_checkpoint,
    save_checkpoint,
    train_step,
    validate_loss_config,
    validate_model_config,
    validate_training_config,
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


def make_dynamics_training_config(save_dir, ae_checkpoint_path='ae.pkl'):
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
        ae_checkpoint_path=ae_checkpoint_path,
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


def make_loss_config(
    gramian_warmup_steps=100000,
    latent_loss_mode='l2',
    xy_weight=0.0,
    xy_tolerance=1.0,
):
    return DynamicsLossConfig(
        recon_weight=1.0,
        latent_weight=1.0,
        gramian_warmup_steps=gramian_warmup_steps,
        gramian_diag_eps=1e-3,
        differentiate_gramian=False,
        latent_loss_mode=latent_loss_mode,
        xy_weight=xy_weight,
        xy_tolerance=xy_tolerance,
    )


def make_batch():
    return {
        'latents': np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float32),
        'actions': np.array([[0.5], [-0.5]], dtype=np.float32),
        'next_latents': np.array([[0.2, -0.1], [0.4, 0.2]], dtype=np.float32),
        'next_observations': np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.0, -0.1, 0.2, -0.2]],
            dtype=np.float32,
        ),
        'raw_next_observations': np.array(
            [[0.1, 0.2, 0.3, 0.4], [0.0, -0.1, 0.2, -0.2]],
            dtype=np.float32,
        ),
    }


class DynamicsTrainingTest(unittest.TestCase):
    def test_latent_dynamics_shape(self):
        model = LatentDynamics(hidden_dims=(8,), latent_dim=3)
        latents = jnp.ones((5, 3), dtype=jnp.float32)
        actions = jnp.ones((5, 2), dtype=jnp.float32)

        params = model.init(jax.random.PRNGKey(0), latents, actions)['params']
        predictions = model.apply({'params': params}, latents, actions)

        self.assertEqual(predictions.shape, (5, 3))

    def test_residual_prediction_adds_network_delta(self):
        latents = jnp.array([[0.1, -0.2, 0.3], [0.4, 0.5, -0.6]], dtype=jnp.float32)
        actions = jnp.array([[0.2, -0.1], [0.3, 0.4]], dtype=jnp.float32)
        absolute = LatentDynamics(hidden_dims=(4,), latent_dim=3, prediction_type='absolute')
        params = absolute.init(jax.random.PRNGKey(0), latents, actions)['params']
        absolute_predictions = absolute.apply({'params': params}, latents, actions)
        residual = LatentDynamics(hidden_dims=(4,), latent_dim=3, prediction_type='residual')
        residual_predictions = residual.apply({'params': params}, latents, actions)

        self.assertEqual(residual_predictions.shape, latents.shape)
        np.testing.assert_allclose(residual_predictions, latents + absolute_predictions, rtol=1e-6)

    def test_transition_dataset_validation_and_alignment(self):
        stats = NormalizationStats(
            mean=np.array([1.0, 2.0], dtype=np.float32),
            std=np.array([2.0, 4.0], dtype=np.float32),
        )
        raw_dataset = {
            'observations': np.array([[1.0, 2.0], [3.0, 6.0]], dtype=np.float32),
            'actions': np.array([[0.25], [-0.25]], dtype=np.float32),
            'next_observations': np.array([[5.0, 10.0], [7.0, 14.0]], dtype=np.float32),
        }

        dataset = make_transition_dataset(raw_dataset, stats, 'test')

        np.testing.assert_allclose(dataset['observations'], np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
        np.testing.assert_allclose(dataset['next_observations'], np.array([[2.0, 2.0], [3.0, 3.0]], dtype=np.float32))
        np.testing.assert_array_equal(dataset['actions'], raw_dataset['actions'])

        with self.assertRaises(ValueError):
            make_transition_dataset({**raw_dataset, 'actions': np.array([0.0, 1.0], dtype=np.float32)}, stats, 'bad')
        with self.assertRaises(ValueError):
            make_transition_dataset(
                {**raw_dataset, 'next_observations': np.ones((3, 2), dtype=np.float32)},
                stats,
                'bad',
            )

    def test_config_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            validate_training_config(make_dynamics_training_config(tmpdir))
            validate_model_config(make_dynamics_model_config())
            validate_loss_config(make_loss_config())

            with self.assertRaises(ValueError):
                validate_training_config(make_dynamics_training_config(tmpdir, ae_checkpoint_path=None))
            with self.assertRaises(ValueError):
                validate_model_config(make_dynamics_model_config(latent_dim=0))
            with self.assertRaises(ValueError):
                validate_model_config(make_dynamics_model_config(prediction_type='invalid'))
            with self.assertRaises(ValueError):
                validate_loss_config(DynamicsLossConfig(0.0, 0.0, 1, 1e-3, False))
            with self.assertRaises(ValueError):
                validate_loss_config(DynamicsLossConfig(1.0, 1.0, 1, 1e-3, False, 'invalid'))
            with self.assertRaises(ValueError):
                validate_loss_config(make_loss_config(xy_weight=-1.0))
            with self.assertRaises(ValueError):
                validate_loss_config(make_loss_config(xy_tolerance=0.0))
            validate_loss_config(DynamicsLossConfig(0.0, 0.0, 1, 1e-3, False, xy_weight=1.0))

    def test_pure_l2_mode_exactly_uses_l2_and_bypasses_gramian(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ae_model_config = make_ae_model_config()
            ae_training_config = make_ae_training_config(tmpdir)
            ae_state = create_autoencoder_train_state(0, ae_model_config, ae_training_config)
            dynamics_state = create_train_state(0, make_dynamics_model_config(), make_dynamics_training_config(tmpdir))

            metrics = dynamics_loss(
                dynamics_state,
                ae_state,
                dynamics_state.params,
                make_batch(),
                jnp.zeros((4,), dtype=jnp.float32),
                jnp.ones((4,), dtype=jnp.float32),
                alpha=1.0,
                recon_weight=1.0,
                latent_weight=1.0,
                xy_weight=0.0,
                xy_tolerance=1.0,
                gramian_diag_eps=1e-3,
                differentiate_gramian=False,
                latent_loss_mode='l2',
                deterministic=True,
            )

            np.testing.assert_array_equal(metrics['latent/loss'], metrics['latent/l2'])
            np.testing.assert_array_equal(metrics['gramian/alpha'], 0.0)
            np.testing.assert_array_equal(metrics['gramian/energy'], 0.0)

    def test_xy_loss_uses_original_units_and_zero_weight_preserves_loss(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ae_model_config = make_ae_model_config()
            ae_training_config = make_ae_training_config(tmpdir)
            ae_state = create_autoencoder_train_state(0, ae_model_config, ae_training_config)
            dynamics_state = create_train_state(0, make_dynamics_model_config(), make_dynamics_training_config(tmpdir))
            batch = make_batch()
            mean = jnp.array([10.0, -5.0, 2.0, 3.0], dtype=jnp.float32)
            std = jnp.array([2.0, 4.0, 0.5, 0.25], dtype=jnp.float32)

            def metrics(xy_weight, xy_tolerance):
                return dynamics_loss(
                    dynamics_state,
                    ae_state,
                    dynamics_state.params,
                    batch,
                    mean,
                    std,
                    alpha=1.0,
                    recon_weight=1.0,
                    latent_weight=1.0,
                    xy_weight=xy_weight,
                    xy_tolerance=xy_tolerance,
                    gramian_diag_eps=1e-3,
                    differentiate_gramian=False,
                    latent_loss_mode='l2',
                    deterministic=True,
                )

            base = metrics(xy_weight=0.0, xy_tolerance=1.0)
            tolerance_two = metrics(xy_weight=0.0, xy_tolerance=2.0)
            weighted = metrics(xy_weight=5.0, xy_tolerance=2.0)

            np.testing.assert_array_equal(base['loss'], base['recon/mse'] + base['latent/loss'])
            np.testing.assert_allclose(base['xy/loss'], 4.0 * tolerance_two['xy/loss'], rtol=1e-6)
            np.testing.assert_allclose(
                weighted['loss'],
                tolerance_two['loss'] + 5.0 * tolerance_two['xy/loss'],
                rtol=1e-6,
            )

    def test_gramian_loss_is_finite_and_alpha_zero_uses_l2(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ae_model_config = make_ae_model_config()
            ae_training_config = make_ae_training_config(tmpdir)
            ae_state = create_autoencoder_train_state(0, ae_model_config, ae_training_config)
            dynamics_state = create_train_state(0, make_dynamics_model_config(), make_dynamics_training_config(tmpdir))
            mean = jnp.zeros((4,), dtype=jnp.float32)
            std = jnp.ones((4,), dtype=jnp.float32)

            metrics = dynamics_loss(
                dynamics_state,
                ae_state,
                dynamics_state.params,
                make_batch(),
                mean,
                std,
                alpha=0.0,
                recon_weight=1.0,
                latent_weight=1.0,
                xy_weight=0.0,
                xy_tolerance=1.0,
                gramian_diag_eps=1e-3,
                differentiate_gramian=False,
                latent_loss_mode='gramian',
                deterministic=True,
            )

            for key in ['loss', 'latent/l2', 'latent/loss', 'gramian/energy', 'gramian/eig_min', 'gramian/eig_max']:
                self.assertTrue(np.isfinite(float(np.asarray(metrics[key]))), key)
            np.testing.assert_allclose(metrics['latent/loss'], metrics['latent/l2'], rtol=1e-6)

    def test_train_eval_and_checkpoint_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ae_model_config = make_ae_model_config()
            ae_training_config = make_ae_training_config(tmpdir)
            ae_state = create_autoencoder_train_state(0, ae_model_config, ae_training_config)
            dynamics_model_config = make_dynamics_model_config()
            dynamics_training_config = make_dynamics_training_config(tmpdir)
            loss_config = make_loss_config(gramian_warmup_steps=0)
            dynamics_state = create_train_state(0, dynamics_model_config, dynamics_training_config)
            mean = jnp.zeros((4,), dtype=jnp.float32)
            std = jnp.ones((4,), dtype=jnp.float32)

            new_state, train_metrics = train_step(
                dynamics_state,
                ae_state,
                make_batch(),
                mean,
                std,
                jax.random.PRNGKey(1),
                loss_config.recon_weight,
                loss_config.latent_weight,
                loss_config.xy_weight,
                loss_config.xy_tolerance,
                loss_config.gramian_diag_eps,
                loss_config.gramian_warmup_steps,
                loss_config.differentiate_gramian,
                loss_config.latent_loss_mode,
            )
            eval_metrics = eval_step(
                new_state,
                ae_state,
                make_batch(),
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

            self.assertEqual(new_state.step, dynamics_state.step + 1)
            self.assertIn('original/mse', train_metrics)
            self.assertIn('gramian/condition', eval_metrics)

            stats = NormalizationStats(mean=np.zeros(4, dtype=np.float32), std=np.ones(4, dtype=np.float32))
            save_checkpoint(
                new_state,
                tmpdir,
                2,
                dynamics_training_config,
                dynamics_model_config,
                loss_config,
                'ae.pkl',
                ae_model_config,
                ae_training_config,
                stats,
                {'seed': 0},
            )
            checkpoint_path = Path(tmpdir) / 'params_2.pkl'
            with checkpoint_path.open('rb') as f:
                checkpoint = pickle.load(f)

            self.assertEqual(checkpoint['schema_version'], 1)
            self.assertEqual(checkpoint['latent_dim'], dynamics_model_config.latent_dim)
            self.assertEqual(checkpoint['action_dim'], dynamics_model_config.action_dim)
            self.assertFalse(checkpoint['loss_config']['differentiate_gramian'])
            self.assertEqual(checkpoint['loss_config']['latent_loss_mode'], 'l2')
            self.assertEqual(checkpoint['loss_config']['xy_weight'], 0.0)
            self.assertEqual(checkpoint['loss_config']['xy_tolerance'], 1.0)

            restored_state, restored_step = restore_checkpoint(dynamics_state, str(checkpoint_path), None)
            self.assertEqual(restored_step, 2)
            self.assertEqual(restored_state.step, new_state.step)

    def test_autoencoder_checkpoint_load_and_latent_encoding(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ae_model_config = make_ae_model_config()
            ae_training_config = make_ae_training_config(tmpdir)
            ae_state = create_autoencoder_train_state(0, ae_model_config, ae_training_config)
            stats = NormalizationStats(mean=np.zeros(4, dtype=np.float32), std=np.ones(4, dtype=np.float32))
            save_autoencoder_checkpoint(ae_state, tmpdir, 2, ae_training_config, ae_model_config, stats, {'seed': 0})
            checkpoint_path = Path(tmpdir) / 'params_2.pkl'

            _, loaded_ae_state, _, _, loaded_stats = load_autoencoder_artifacts(checkpoint_path)
            raw_dataset = {
                'observations': np.ones((3, 4), dtype=np.float32),
                'actions': np.ones((3, 1), dtype=np.float32),
                'next_observations': np.ones((3, 4), dtype=np.float32) * 2.0,
            }
            transition_dataset = make_transition_dataset(raw_dataset, loaded_stats, 'test')
            latent_dataset = encode_transition_dataset(transition_dataset, loaded_ae_state, batch_size=2)

            self.assertEqual(latent_dataset['latents'].shape, (3, ae_model_config.latent_dim))
            self.assertEqual(latent_dataset['next_latents'].shape, (3, ae_model_config.latent_dim))
            np.testing.assert_array_equal(latent_dataset['actions'], raw_dataset['actions'])


if __name__ == '__main__':
    unittest.main()
