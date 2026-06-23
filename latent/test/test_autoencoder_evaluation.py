import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from impls.utils.datasets import Dataset
from latent.train.train_autoencoder import (
    ModelConfig,
    TrainingConfig,
    compute_normalization_stats,
    create_train_state,
    save_checkpoint,
)
from latent.validate.evaluate_autoencoder import (
    apply_pca,
    evaluate_checkpoint_on_splits,
    fit_pca,
    load_checkpoint,
    make_dataset_from_raw,
    training_config_from_dict,
    reconstruction_summary,
    restore_autoencoder_state,
    sample_dataset,
    save_plots,
)


def make_training_config(save_dir):
    return TrainingConfig(
        run_group='test',
        seed=3,
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
        validation_batches=0,
        validation_batch_size=4,
    )


def make_model_config(obs_dim=4):
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


def make_checkpoint(tmpdir):
    training_config = make_training_config(tmpdir)
    model_config = make_model_config()
    state = create_train_state(training_config.seed, model_config, training_config)
    raw_observations = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.5],
            [1.0, 0.0, 0.8, 0.3],
            [0.0, 1.0, 0.5, 0.1],
            [1.0, 1.0, 0.3, -0.2],
            [2.0, 1.0, -0.2, -0.4],
            [2.0, 2.0, -0.4, -0.6],
        ],
        dtype=np.float32,
    )
    stats = compute_normalization_stats(raw_observations, training_config)
    save_checkpoint(state, tmpdir, 2, training_config, model_config, stats, {'seed': training_config.seed})
    return Path(tmpdir) / 'params_2.pkl', raw_observations, stats


class AutoencoderEvaluationTest(unittest.TestCase):
    def test_training_config_from_old_checkpoint_data_uses_lr_defaults(self):
        config = training_config_from_dict({'lr': 1e-3})

        self.assertEqual(config.min_lr, 1e-6)
        self.assertEqual(config.lr_plateau_patience, 5)
        self.assertEqual(config.lr_plateau_factor, 0.5)
        self.assertEqual(config.lr_plateau_min_delta, 1e-4)

    def test_restore_falls_back_when_optimizer_state_schema_changed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path, _, _ = make_checkpoint(tmpdir)
            checkpoint, model_config, training_config, _ = load_checkpoint(checkpoint_path)
            checkpoint['autoencoder']['opt_state'] = {'0': {}, '1': {}}

            state = restore_autoencoder_state(checkpoint, model_config, training_config)

            self.assertEqual(state.step, checkpoint['step'])
            self.assertEqual(state.params.keys(), checkpoint['params'].keys())

    def test_checkpoint_load_and_restore(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path, _, _ = make_checkpoint(tmpdir)

            checkpoint, model_config, training_config, stats = load_checkpoint(checkpoint_path)
            state = restore_autoencoder_state(checkpoint, model_config, training_config)

            self.assertEqual(checkpoint['step'], 2)
            self.assertEqual(model_config.latent_dim, 2)
            self.assertEqual(stats.mean.shape, (4,))
            self.assertEqual(state.step, checkpoint['autoencoder']['step'])

    def test_reconstruction_summary_tracks_worst_dimensions(self):
        observations = np.zeros((3, 2), dtype=np.float32)
        reconstructions = np.asarray([[0.0, 1.0], [0.0, 2.0], [0.0, 3.0]], dtype=np.float32)

        metrics = reconstruction_summary(observations, reconstructions, observations, reconstructions)

        self.assertAlmostEqual(metrics['mse'], 14.0 / 6.0, places=6)
        self.assertEqual(metrics['worst_dims'][0]['dim'], 1)
        self.assertEqual(metrics['worst_examples'][0]['index'], 2)

    def test_pca_baseline_shapes(self):
        observations = np.arange(24, dtype=np.float32).reshape(6, 4)
        pca = fit_pca(observations, latent_dim=2)
        reconstructions, latents = apply_pca(pca, observations)

        self.assertEqual(reconstructions.shape, observations.shape)
        self.assertEqual(latents.shape, (6, 2))

    def test_deterministic_sampling(self):
        observations = np.arange(40, dtype=np.float32).reshape(10, 4)
        dataset = Dataset.create(observations=observations, raw_observations=observations)

        sample_a = sample_dataset(dataset, max_examples=4, seed=11)
        sample_b = sample_dataset(dataset, max_examples=4, seed=11)

        np.testing.assert_array_equal(sample_a['indices'], sample_b['indices'])
        np.testing.assert_array_equal(sample_a['observations'], sample_b['observations'])

    def test_save_plots_writes_combined_dimension_grids(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            raw = np.arange(40, dtype=np.float32).reshape(10, 4)
            raw_recon = raw + 0.1
            latents = np.column_stack([np.linspace(0, 1, 10), np.linspace(1, 0, 10)]).astype(np.float32)

            paths = save_plots(Path(tmpdir), 'train', raw, raw_recon, latents, [0, 1, 2, 3], 10, 0)

            self.assertEqual(
                [Path(path).name for path in paths],
                ['train_reconstruction_dims.png', 'train_latent_by_dims.png'],
            )
            for path in paths:
                self.assertTrue(Path(path).exists())

    def test_evaluate_checkpoint_writes_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path, raw_observations, stats = make_checkpoint(tmpdir)
            train = make_dataset_from_raw(raw_observations, stats, 'training')
            validation_raw = raw_observations + np.asarray([0.1, -0.1, 0.05, 0.0], dtype=np.float32)
            validation = make_dataset_from_raw(validation_raw, stats, 'validation')
            output_dir = Path(tmpdir) / 'eval'

            metrics = evaluate_checkpoint_on_splits(
                checkpoint_path,
                {'train': train, 'validation': validation},
                output_dir,
                max_examples=5,
                batch_size=3,
                seed=5,
                probe_dims=[0, 1],
                pair_count=32,
                knn_k=1,
                knn_anchors=3,
                write_plots=False,
            )

            self.assertTrue((output_dir / 'metrics.json').exists())
            self.assertTrue((output_dir / 'summary.txt').exists())
            self.assertIn('train', metrics['splits'])
            self.assertIn('pca', metrics['splits']['validation'])
            self.assertIn('ae_minus_pca', metrics['splits']['train'])

            with (output_dir / 'metrics.json').open('r') as f:
                written = json.load(f)
            self.assertEqual(written['checkpoint_step'], 2)


if __name__ == '__main__':
    unittest.main()
