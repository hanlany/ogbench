import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb
from absl import app, flags

REPO_ROOT = Path(__file__).resolve().parents[2]
IMPLS_DIR = REPO_ROOT / 'impls'
for path in (REPO_ROOT, IMPLS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ogbench import make_env_and_datasets  # noqa: E402
from utils.datasets import Dataset  # noqa: E402
from utils.flax_utils import TrainState  # noqa: E402
from utils.log_utils import CsvLogger, get_exp_name, get_flag_dict, setup_wandb  # noqa: E402

from latent.train.autoencoder import AutoEncoder, reconstruction_metrics  # noqa: E402

FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'LatentAutoEncoder', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_dir', '~/.ogbench/data', 'Directory to save/load OGBench datasets.')
flags.DEFINE_string('dataset_path', None, 'Optional path to the training dataset file.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_enum('wandb_mode', 'online', ['online', 'offline', 'disabled'], 'Weights & Biases mode.')

flags.DEFINE_integer('latent_dim', 2, 'Latent dimension.')
flags.DEFINE_list('hidden_dims', ['512', '512'], 'Comma-separated hidden dimensions for the encoder MLP.')
flags.DEFINE_float('lr', 3e-4, 'Learning rate.')
flags.DEFINE_integer('batch_size', 1024, 'Batch size.')
flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')


def parse_hidden_dims():
    """Parse hidden dimensions from flags."""
    hidden_dims = tuple(int(dim) for dim in FLAGS.hidden_dims)
    if len(hidden_dims) == 0 or any(dim <= 0 for dim in hidden_dims):
        raise ValueError(f'Expected positive hidden dimensions, got {hidden_dims}.')
    return hidden_dims


def validate_observations(observations, split_name):
    """Validate and cast vector observations."""
    observations = np.asarray(observations)
    if observations.ndim != 2:
        raise ValueError(
            f'Only vector observations are supported for now. '
            f'Expected {split_name} observations with shape (N, obs_dim), got {observations.shape}.'
        )
    return observations.astype(np.float32, copy=False)


def load_vector_datasets():
    """Load OGBench datasets and keep only vector observations."""
    train_dataset, val_dataset = make_env_and_datasets(
        FLAGS.env_name,
        dataset_dir=FLAGS.dataset_dir,
        dataset_path=FLAGS.dataset_path,
        compact_dataset=True,
        dataset_only=True,
    )
    train_observations = validate_observations(train_dataset['observations'], 'training')
    train_dataset = Dataset.create(observations=train_observations)

    if val_dataset is None:
        return train_dataset, None

    val_observations = validate_observations(val_dataset['observations'], 'validation')
    val_dataset = Dataset.create(observations=val_observations)
    return train_dataset, val_dataset


def create_train_state(seed, obs_dim, hidden_dims):
    """Initialize autoencoder train state."""
    model_def = AutoEncoder(hidden_dims=hidden_dims, latent_dim=FLAGS.latent_dim, obs_dim=obs_dim)
    rng = jax.random.PRNGKey(seed)
    params = model_def.init(rng, jnp.zeros((1, obs_dim), dtype=jnp.float32))['params']
    tx = optax.adam(learning_rate=FLAGS.lr)
    return TrainState.create(model_def, params, tx=tx)


@jax.jit
def train_step(state, batch):
    """Run one autoencoder update."""

    def loss_fn(params):
        reconstructions, latents = state(batch['observations'], params=params)
        metrics = reconstruction_metrics(batch['observations'], reconstructions)
        metrics['latent/norm'] = jnp.mean(jnp.linalg.norm(latents, axis=-1))
        return metrics['mse'], metrics

    return state.apply_loss_fn(loss_fn)


@jax.jit
def eval_step(state, batch):
    """Evaluate reconstruction metrics without updating parameters."""
    reconstructions, latents = state(batch['observations'])
    metrics = reconstruction_metrics(batch['observations'], reconstructions)
    metrics['latent/norm'] = jnp.mean(jnp.linalg.norm(latents, axis=-1))
    return metrics


def to_float_dict(metrics):
    """Convert JAX scalar metrics to Python floats for loggers."""
    return {k: float(np.asarray(v)) for k, v in metrics.items()}


def split_autoencoder_params(params):
    """Expose encoder and decoder parameter collections from the full autoencoder params."""
    return {
        'encoder': params['encoder'],
        'decoder': params['decoder'],
    }


def save_checkpoint(state, save_dir, step, config):
    """Save full autoencoder params plus separately usable encoder/decoder params."""
    checkpoint = {
        'step': step,
        'config': config,
        'autoencoder': flax.serialization.to_state_dict(state),
        'params': flax.serialization.to_state_dict(state.params),
        'components': flax.serialization.to_state_dict(split_autoencoder_params(state.params)),
    }
    save_path = os.path.join(save_dir, f'params_{step}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f'Saved to {save_path}')


def main(_):
    hidden_dims = parse_hidden_dims()

    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    train_dataset, val_dataset = load_vector_datasets()
    obs_dim = train_dataset['observations'].shape[-1]
    if not 0 < FLAGS.latent_dim < obs_dim:
        raise ValueError(f'Expected 0 < latent_dim < obs_dim, got latent_dim={FLAGS.latent_dim}, obs_dim={obs_dim}.')

    exp_name = get_exp_name(FLAGS.seed)
    setup_wandb(project='OGBench', group=FLAGS.run_group, name=exp_name, mode=FLAGS.wandb_mode)
    FLAGS.save_dir = os.path.join(FLAGS.save_dir, wandb.run.project, FLAGS.run_group, exp_name)
    os.makedirs(FLAGS.save_dir, exist_ok=True)

    flag_dict = get_flag_dict()
    flag_dict['hidden_dims'] = hidden_dims
    flag_dict['obs_dim'] = obs_dim
    with open(os.path.join(FLAGS.save_dir, 'flags.json'), 'w') as f:
        json.dump(flag_dict, f)

    state = create_train_state(FLAGS.seed, obs_dim, hidden_dims)
    train_logger = CsvLogger(os.path.join(FLAGS.save_dir, 'train.csv'))
    first_time = time.time()
    last_time = time.time()

    for i in tqdm.tqdm(range(1, FLAGS.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
        batch = train_dataset.sample(FLAGS.batch_size)
        state, update_info = train_step(state, batch)

        if i % FLAGS.log_interval == 0:
            metrics = {f'training/{k}': v for k, v in to_float_dict(update_info).items()}
            if val_dataset is not None:
                val_batch = val_dataset.sample(FLAGS.batch_size)
                val_info = eval_step(state, val_batch)
                metrics.update({f'validation/{k}': v for k, v in to_float_dict(val_info).items()})
            metrics['time/epoch_time'] = (time.time() - last_time) / FLAGS.log_interval
            metrics['time/total_time'] = time.time() - first_time
            last_time = time.time()
            wandb.log(metrics, step=i)
            train_logger.log(metrics, step=i)

        if i % FLAGS.save_interval == 0:
            save_checkpoint(state, FLAGS.save_dir, i, flag_dict)

    if FLAGS.train_steps % FLAGS.save_interval != 0:
        save_checkpoint(state, FLAGS.save_dir, FLAGS.train_steps, flag_dict)

    train_logger.close()


if __name__ == '__main__':
    app.run(main)
