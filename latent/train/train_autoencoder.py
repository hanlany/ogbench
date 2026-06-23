from __future__ import annotations

import json
import os
import pickle
import queue
import random
import threading
import time
from dataclasses import asdict, dataclass, replace
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
    from latent.train.autoencoder import AutoEncoder, reconstruction_metrics
    from ogbench import make_env_and_datasets
except ImportError as e:  # pragma: no cover - exercised by direct script misuse.
    raise ImportError(
        'Could not import OGBench training dependencies. Run from the repository root with '
        '`PYTHONPATH=. python latent/train/train_autoencoder.py ...`, or install the repo in editable mode.'
    ) from e


CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_DATASET_DIR = str(Path(__file__).resolve().parents[2] / 'data_gen_scripts' / 'data')
FLAGS = flags.FLAGS

flags.DEFINE_string('run_group', 'LatentAutoEncoder', 'Run group.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string('env_name', 'antmaze-large-navigate-v0', 'Environment (dataset) name.')
flags.DEFINE_string('dataset_dir', DEFAULT_DATASET_DIR, 'Directory to save/load OGBench datasets.')
flags.DEFINE_string('dataset_path', None, 'Optional path to the training dataset file.')
flags.DEFINE_string('save_dir', 'exp/', 'Save directory.')
flags.DEFINE_string('restore_path', None, 'Path to a checkpoint file or directory containing params_<step>.pkl.')
flags.DEFINE_integer('restore_step', None, 'Checkpoint step to restore when restore_path is a directory.')
flags.DEFINE_enum('wandb_mode', 'online', ['online', 'offline', 'disabled'], 'Weights & Biases mode.')

flags.DEFINE_integer('latent_dim', 12, 'Latent dimension.')
flags.DEFINE_list('hidden_dims', ['512', '512'], 'Comma-separated hidden dimensions for the encoder MLP.')
flags.DEFINE_list('decoder_hidden_dims', None, 'Optional comma-separated hidden dimensions for the decoder MLP.')
flags.DEFINE_enum('activation', 'gelu', ['elu', 'gelu', 'relu', 'swish', 'tanh'], 'MLP activation.')
flags.DEFINE_bool('layer_norm', False, 'Whether to use layer normalization after hidden dense layers.')
flags.DEFINE_bool('decoder_activate_final', False, 'Whether to apply the activation to decoder outputs.')
flags.DEFINE_float('dropout_rate', 0.0, 'Dropout rate for hidden layers.')
flags.DEFINE_float('lr', 3e-4, 'Learning rate.')
flags.DEFINE_float('min_lr', 1e-6, 'Minimum learning rate after plateau reductions.')
flags.DEFINE_integer('lr_plateau_patience', 5, 'Number of logging intervals without improvement before reducing LR; 0 disables reductions.')
flags.DEFINE_float('lr_plateau_factor', 0.5, 'Multiplier applied to LR when the monitored metric plateaus.')
flags.DEFINE_float('lr_plateau_min_delta', 1e-4, 'Minimum metric improvement required to reset LR plateau patience.')
flags.DEFINE_integer('batch_size', 1024, 'Batch size.')
flags.DEFINE_integer('train_steps', 1000000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 5000, 'Logging interval.')
flags.DEFINE_integer('save_interval', 1000000, 'Saving interval.')
flags.DEFINE_integer('prefetch_batches', 2, 'Number of host batches to prefetch; 0 disables prefetching.')

flags.DEFINE_bool('normalize_observations', True, 'Whether to normalize observations with train-set statistics.')
flags.DEFINE_float('normalization_eps', 1e-6, 'Minimum observation standard deviation used for normalization.')
flags.DEFINE_integer(
    'validation_batches',
    16,
    'Number of fixed validation batches to evaluate. Set to 0 for a deterministic full validation sweep.',
)
flags.DEFINE_integer('validation_batch_size', None, 'Validation batch size. Defaults to batch_size.')


@dataclass(frozen=True)
class ModelConfig:
    hidden_dims: tuple[int, ...]
    latent_dim: int
    obs_dim: int
    decoder_hidden_dims: tuple[int, ...] | None
    activation: str
    layer_norm: bool
    decoder_activate_final: bool
    dropout_rate: float


@dataclass(frozen=True)
class TrainingConfig:
    run_group: str
    seed: int
    env_name: str
    dataset_dir: str
    dataset_path: str | None
    save_dir: str
    restore_path: str | None
    restore_step: int | None
    wandb_mode: str
    lr: float
    min_lr: float
    lr_plateau_patience: int
    lr_plateau_factor: float
    lr_plateau_min_delta: float
    batch_size: int
    train_steps: int
    log_interval: int
    save_interval: int
    prefetch_batches: int
    normalize_observations: bool
    normalization_eps: float
    validation_batches: int
    validation_batch_size: int


@dataclass(frozen=True)
class NormalizationStats:
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self):
        return {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
        }


def parse_dims(values, name, allow_none=False):
    """Parse a flag list into positive integer dimensions."""
    if values is None:
        if allow_none:
            return None
        raise ValueError(f'{name} must not be None.')
    dims = tuple(int(dim) for dim in values)
    if len(dims) == 0 or any(dim <= 0 for dim in dims):
        raise ValueError(f'Expected positive {name}, got {dims}.')
    return dims


def parse_hidden_dims():
    """Parse encoder hidden dimensions from flags."""
    return parse_dims(FLAGS.hidden_dims, 'hidden_dims')


def create_configs(obs_dim=None):
    """Create explicit configs from CLI flags."""
    validation_batch_size = FLAGS.validation_batch_size or FLAGS.batch_size
    training_config = TrainingConfig(
        run_group=FLAGS.run_group,
        seed=FLAGS.seed,
        env_name=FLAGS.env_name,
        dataset_dir=FLAGS.dataset_dir,
        dataset_path=FLAGS.dataset_path,
        save_dir=FLAGS.save_dir,
        restore_path=FLAGS.restore_path,
        restore_step=FLAGS.restore_step,
        wandb_mode=FLAGS.wandb_mode,
        lr=FLAGS.lr,
        min_lr=FLAGS.min_lr,
        lr_plateau_patience=FLAGS.lr_plateau_patience,
        lr_plateau_factor=FLAGS.lr_plateau_factor,
        lr_plateau_min_delta=FLAGS.lr_plateau_min_delta,
        batch_size=FLAGS.batch_size,
        train_steps=FLAGS.train_steps,
        log_interval=FLAGS.log_interval,
        save_interval=FLAGS.save_interval,
        prefetch_batches=FLAGS.prefetch_batches,
        normalize_observations=FLAGS.normalize_observations,
        normalization_eps=FLAGS.normalization_eps,
        validation_batches=FLAGS.validation_batches,
        validation_batch_size=validation_batch_size,
    )
    model_config = ModelConfig(
        hidden_dims=parse_hidden_dims(),
        latent_dim=FLAGS.latent_dim,
        obs_dim=-1 if obs_dim is None else obs_dim,
        decoder_hidden_dims=parse_dims(FLAGS.decoder_hidden_dims, 'decoder_hidden_dims', allow_none=True),
        activation=FLAGS.activation,
        layer_norm=FLAGS.layer_norm,
        decoder_activate_final=FLAGS.decoder_activate_final,
        dropout_rate=FLAGS.dropout_rate,
    )
    return training_config, model_config


def validate_training_config(config):
    """Fail fast on invalid training settings."""
    positive_ints = {
        'batch_size': config.batch_size,
        'train_steps': config.train_steps,
        'log_interval': config.log_interval,
        'save_interval': config.save_interval,
        'validation_batch_size': config.validation_batch_size,
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
    if config.min_lr <= 0:
        raise ValueError(f'min_lr must be positive, got {config.min_lr}.')
    if config.min_lr > config.lr:
        raise ValueError(f'min_lr must be <= lr, got min_lr={config.min_lr}, lr={config.lr}.')
    if config.lr_plateau_patience < 0:
        raise ValueError(f'lr_plateau_patience must be non-negative, got {config.lr_plateau_patience}.')
    if not 0 < config.lr_plateau_factor < 1:
        raise ValueError(f'lr_plateau_factor must be in (0, 1), got {config.lr_plateau_factor}.')
    if config.lr_plateau_min_delta < 0:
        raise ValueError(f'lr_plateau_min_delta must be non-negative, got {config.lr_plateau_min_delta}.')
    if config.normalization_eps <= 0:
        raise ValueError(f'normalization_eps must be positive, got {config.normalization_eps}.')


def validate_model_config(config):
    """Fail fast on invalid model settings."""
    if config.obs_dim <= 0:
        raise ValueError(f'obs_dim must be positive, got {config.obs_dim}.')
    if not 0 < config.latent_dim < config.obs_dim:
        raise ValueError(f'Expected 0 < latent_dim < obs_dim, got latent_dim={config.latent_dim}, obs_dim={config.obs_dim}.')
    if config.dropout_rate < 0 or config.dropout_rate >= 1:
        raise ValueError(f'dropout_rate must be in [0, 1), got {config.dropout_rate}.')


def validate_observations(observations, split_name):
    """Validate and cast vector observations."""
    observations = np.asarray(observations)
    if observations.ndim != 2:
        raise ValueError(
            f'Only vector observations are supported for now. '
            f'Expected {split_name} observations with shape (N, obs_dim), got {observations.shape}.'
        )
    if len(observations) == 0:
        raise ValueError(f'Expected non-empty {split_name} observations.')
    return observations.astype(np.float32, copy=False)


def compute_normalization_stats(observations, config):
    """Compute train-set observation normalization statistics."""
    if not config.normalize_observations:
        return NormalizationStats(
            mean=np.zeros(observations.shape[-1], dtype=np.float32),
            std=np.ones(observations.shape[-1], dtype=np.float32),
        )
    mean = np.mean(observations, axis=0, dtype=np.float64).astype(np.float32)
    std = np.std(observations, axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(std, config.normalization_eps).astype(np.float32)
    return NormalizationStats(mean=mean, std=std)


def normalize_observations(observations, stats):
    """Normalize observations with precomputed train-set statistics."""
    return ((observations - stats.mean) / stats.std).astype(np.float32)


def load_vector_datasets(config):
    """Load OGBench datasets and keep only normalized vector observations."""
    train_dataset, val_dataset = make_env_and_datasets(
        config.env_name,
        dataset_dir=config.dataset_dir,
        dataset_path=config.dataset_path,
        compact_dataset=True,
        dataset_only=True,
    )
    train_raw = validate_observations(train_dataset['observations'], 'training')
    stats = compute_normalization_stats(train_raw, config)
    train_dataset = Dataset.create(
        observations=normalize_observations(train_raw, stats),
        raw_observations=train_raw,
    )

    if val_dataset is None:
        return train_dataset, None, stats

    val_raw = validate_observations(val_dataset['observations'], 'validation')
    val_dataset = Dataset.create(
        observations=normalize_observations(val_raw, stats),
        raw_observations=val_raw,
    )
    return train_dataset, val_dataset, stats


def create_train_state(seed, model_config, training_config):
    """Initialize autoencoder train state."""
    model_def = AutoEncoder(
        hidden_dims=model_config.hidden_dims,
        latent_dim=model_config.latent_dim,
        obs_dim=model_config.obs_dim,
        decoder_hidden_dims=model_config.decoder_hidden_dims,
        activation=model_config.activation,
        layer_norm=model_config.layer_norm,
        decoder_activate_final=model_config.decoder_activate_final,
        dropout_rate=model_config.dropout_rate,
    )
    rng = jax.random.PRNGKey(seed)
    params = model_def.init(rng, jnp.zeros((1, model_config.obs_dim), dtype=jnp.float32))['params']
    tx = optax.inject_hyperparams(optax.adam)(learning_rate=training_config.lr)
    return TrainState.create(model_def, params, tx=tx)


@jax.jit
def train_step(state, batch, rng):
    """Run one autoencoder update."""

    def loss_fn(params):
        reconstructions, latents = state(
            batch['observations'],
            params=params,
            deterministic=False,
            rngs={'dropout': rng},
        )
        metrics = reconstruction_metrics(batch['observations'], reconstructions)
        metrics['latent/norm'] = jnp.mean(jnp.linalg.norm(latents, axis=-1))
        metrics['latent/mean'] = jnp.mean(latents)
        metrics['latent/std'] = jnp.std(latents)
        return metrics['mse'], metrics

    new_state, metrics = state.apply_loss_fn(loss_fn)
    param_norms = jax.tree_util.tree_map(jnp.linalg.norm, new_state.params)
    metrics['param/norm'] = jnp.linalg.norm(jnp.array(jax.tree_util.tree_leaves(param_norms)))
    return new_state, metrics


@jax.jit
def eval_step(state, batch, mean, std):
    """Evaluate reconstruction metrics without updating parameters."""
    reconstructions, latents = state(batch['observations'], deterministic=True)
    metrics = reconstruction_metrics(batch['observations'], reconstructions)
    metrics['latent/norm'] = jnp.mean(jnp.linalg.norm(latents, axis=-1))
    metrics['latent/mean'] = jnp.mean(latents)
    metrics['latent/std'] = jnp.std(latents)

    raw_reconstructions = reconstructions * std + mean
    raw_metrics = reconstruction_metrics(batch['raw_observations'], raw_reconstructions)
    metrics.update({f'original/{k}': v for k, v in raw_metrics.items()})
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


def config_to_dict(config):
    """Convert dataclass configs to JSON/checkpoint-friendly dictionaries."""
    data = asdict(config)
    return data


def save_checkpoint(state, save_dir, step, training_config, model_config, normalization_stats, flag_dict):
    """Save full autoencoder params plus separately usable encoder/decoder params."""
    checkpoint = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'step': step,
        'flags': flag_dict,
        'training_config': config_to_dict(training_config),
        'model_config': config_to_dict(model_config),
        'optimizer_config': {
            'name': 'adam',
            'learning_rate': training_config.lr,
            'min_learning_rate': training_config.min_lr,
            'plateau_patience': training_config.lr_plateau_patience,
            'plateau_factor': training_config.lr_plateau_factor,
            'plateau_min_delta': training_config.lr_plateau_min_delta,
        },
        'normalization': normalization_stats.to_dict(),
        'autoencoder': flax.serialization.to_state_dict(state),
        'params': flax.serialization.to_state_dict(state.params),
        'components': flax.serialization.to_state_dict(split_autoencoder_params(state.params)),
    }
    save_path = os.path.join(save_dir, f'params_{step}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f'Saved to {save_path}')


def resolve_checkpoint_path(restore_path, restore_step):
    """Resolve a checkpoint file from either a direct file path or checkpoint directory."""
    path = Path(restore_path).expanduser()
    if path.is_file():
        return path
    if restore_step is None:
        raise ValueError('restore_step is required when restore_path is a directory.')
    return path / f'params_{restore_step}.pkl'


def restore_checkpoint(state, restore_path, restore_step):
    """Restore a train state from a checkpoint."""
    checkpoint_path = resolve_checkpoint_path(restore_path, restore_step)
    with checkpoint_path.open('rb') as f:
        checkpoint = pickle.load(f)
    if 'autoencoder' not in checkpoint:
        raise ValueError(f'Checkpoint {checkpoint_path} does not contain a full autoencoder train state.')
    state = flax.serialization.from_state_dict(state, checkpoint['autoencoder'])
    restored_step = checkpoint.get('step', restore_step)
    print(f'Restored from {checkpoint_path}')
    return state, restored_step


def create_validation_batches(dataset, batch_size, num_batches, seed):
    """Create deterministic validation batches with stable shapes."""
    if dataset is None:
        return []
    if num_batches == 0:
        num_examples = dataset.size
        idx_groups = [np.arange(start, min(start + batch_size, num_examples)) for start in range(0, num_examples, batch_size)]
    else:
        rng = np.random.default_rng(seed)
        idx_groups = [rng.integers(dataset.size, size=batch_size) for _ in range(num_batches)]
    return [dataset.get_subset(idxs) for idxs in idx_groups]


def average_metrics(metric_dicts):
    """Average a list of metric dictionaries."""
    if len(metric_dicts) == 0:
        return {}
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([metrics[key] for metrics in metric_dicts])) for key in keys}


def set_optimizer_lr(state, learning_rate):
    """Update the injected Optax learning rate without resetting optimizer moments."""
    opt_state = state.opt_state
    if not hasattr(opt_state, 'hyperparams') or 'learning_rate' not in opt_state.hyperparams:
        raise ValueError('Optimizer state does not expose an injectable learning_rate hyperparameter.')
    hyperparams = dict(opt_state.hyperparams)
    hyperparams['learning_rate'] = jnp.asarray(learning_rate, dtype=hyperparams['learning_rate'].dtype)
    return state.replace(opt_state=opt_state._replace(hyperparams=hyperparams))


def maybe_reduce_lr(state, current_lr, metric, best_metric, plateau_count, config):
    """Reduce LR when the monitored metric stops improving."""
    if metric < best_metric - config.lr_plateau_min_delta:
        return state, current_lr, metric, 0, False

    plateau_count += 1
    if config.lr_plateau_patience == 0 or plateau_count < config.lr_plateau_patience or current_lr <= config.min_lr:
        return state, current_lr, best_metric, plateau_count, False

    new_lr = max(current_lr * config.lr_plateau_factor, config.min_lr)
    if new_lr >= current_lr:
        return state, current_lr, best_metric, plateau_count, False
    state = set_optimizer_lr(state, new_lr)
    return state, new_lr, best_metric, 0, True


class BatchPrefetcher:
    """Small host-side prefetcher for Dataset.sample batches."""

    def __init__(self, dataset, batch_size, num_prefetch):
        self.dataset = dataset
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=num_prefetch)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                self.queue.put(self.dataset.sample(self.batch_size), timeout=0.1)
            except queue.Full:
                continue

    def sample(self):
        return self.queue.get()

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=1)


def create_batch_sampler(dataset, batch_size, prefetch_batches):
    """Create either a prefetching or direct batch sampler."""
    if prefetch_batches == 0:
        return None, lambda: dataset.sample(batch_size)
    prefetcher = BatchPrefetcher(dataset, batch_size, prefetch_batches)
    return prefetcher, prefetcher.sample


def write_json(path, data):
    """Write a JSON file with stable formatting."""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, sort_keys=True)


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


def run_training(training_config, model_config):
    """Run the full autoencoder training workflow."""
    validate_training_config(training_config)

    random.seed(training_config.seed)
    np.random.seed(training_config.seed)

    train_dataset, val_dataset, normalization_stats = load_vector_datasets(training_config)
    obs_dim = train_dataset['observations'].shape[-1]
    model_config = replace(model_config, obs_dim=obs_dim)
    validate_model_config(model_config)

    save_dir = prepare_save_dir(training_config)
    training_config = replace(training_config, save_dir=save_dir)

    flag_dict = get_flag_dict()
    metadata = {
        'flags': flag_dict,
        'training_config': config_to_dict(training_config),
        'model_config': config_to_dict(model_config),
        'normalization': normalization_stats.to_dict(),
    }
    write_json(os.path.join(training_config.save_dir, 'flags.json'), metadata)

    state = create_train_state(training_config.seed, model_config, training_config)
    start_step = 1
    if training_config.restore_path is not None:
        state, restored_step = restore_checkpoint(state, training_config.restore_path, training_config.restore_step)
        start_step = int(restored_step) + 1

    val_batches = create_validation_batches(
        val_dataset,
        training_config.validation_batch_size,
        training_config.validation_batches,
        training_config.seed + 1,
    )
    mean = jnp.asarray(normalization_stats.mean)
    std = jnp.asarray(normalization_stats.std)

    prefetcher, sample_train_batch = create_batch_sampler(
        train_dataset,
        training_config.batch_size,
        training_config.prefetch_batches,
    )
    train_logger = CsvLogger(os.path.join(training_config.save_dir, 'train.csv'))
    current_lr = training_config.lr
    best_plateau_metric = float('inf')
    plateau_count = 0
    rng = jax.random.PRNGKey(training_config.seed + 1)
    first_time = time.time()
    last_time = time.time()

    try:
        for i in tqdm.tqdm(range(start_step, training_config.train_steps + 1), smoothing=0.1, dynamic_ncols=True):
            rng, step_rng = jax.random.split(rng)
            batch = sample_train_batch()
            state, update_info = train_step(state, batch, step_rng)

            if i % training_config.log_interval == 0:
                metrics = {f'training/{k}': v for k, v in to_float_dict(update_info).items()}
                plateau_metric_name = 'training/mse'
                if len(val_batches) > 0:
                    val_metric_dicts = [to_float_dict(eval_step(state, val_batch, mean, std)) for val_batch in val_batches]
                    metrics.update({f'validation/{k}': v for k, v in average_metrics(val_metric_dicts).items()})
                    plateau_metric_name = 'validation/mse'
                state, current_lr, best_plateau_metric, plateau_count, lr_reduced = maybe_reduce_lr(
                    state,
                    current_lr,
                    metrics[plateau_metric_name],
                    best_plateau_metric,
                    plateau_count,
                    training_config,
                )
                training_config = replace(training_config, lr=current_lr)
                metrics['optimizer/lr'] = current_lr
                metrics['optimizer/lr_plateau_metric'] = metrics[plateau_metric_name]
                metrics['optimizer/lr_plateau_count'] = plateau_count
                metrics['optimizer/lr_reduced'] = float(lr_reduced)
                metrics['time/epoch_time'] = (time.time() - last_time) / training_config.log_interval
                metrics['time/total_time'] = time.time() - first_time
                last_time = time.time()
                wandb.log(metrics, step=i)
                train_logger.log(metrics, step=i)

            if i % training_config.save_interval == 0:
                save_checkpoint(state, training_config.save_dir, i, training_config, model_config, normalization_stats, flag_dict)

        if training_config.train_steps % training_config.save_interval != 0:
            save_checkpoint(
                state,
                training_config.save_dir,
                training_config.train_steps,
                training_config,
                model_config,
                normalization_stats,
                flag_dict,
            )
    finally:
        train_logger.close()
        if prefetcher is not None:
            prefetcher.close()


def main(_):
    training_config, model_config = create_configs()
    run_training(training_config, model_config)


if __name__ == '__main__':
    app.run(main)
