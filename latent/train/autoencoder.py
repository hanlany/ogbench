from typing import Optional, Sequence

import flax.linen as nn
import jax.numpy as jnp


def get_activation(name):
    """Return an activation function by name."""
    activations = {
        'elu': nn.elu,
        'gelu': nn.gelu,
        'relu': nn.relu,
        'swish': nn.swish,
        'tanh': nn.tanh,
    }
    if name not in activations:
        raise ValueError(f'Unsupported activation {name}. Expected one of {sorted(activations)}.')
    return activations[name]


class MLP(nn.Module):
    """Simple MLP used by the latent encoder and decoder."""

    hidden_dims: Sequence[int]
    output_dim: int
    activation: str = 'gelu'
    layer_norm: bool = False
    activate_final: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic=True):
        activation = get_activation(self.activation)
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            if self.layer_norm:
                x = nn.LayerNorm()(x)
            x = activation(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = nn.Dense(self.output_dim)(x)
        if self.activate_final:
            x = activation(x)
        return x


class MLPEncoder(nn.Module):
    """Map vector observations to a lower-dimensional latent state."""

    hidden_dims: Sequence[int]
    latent_dim: int
    activation: str = 'gelu'
    layer_norm: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, observations, deterministic=True):
        return MLP(
            self.hidden_dims,
            self.latent_dim,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout_rate=self.dropout_rate,
        )(observations, deterministic=deterministic)


class MLPDecoder(nn.Module):
    """Map latent states back to vector observations."""

    hidden_dims: Sequence[int]
    output_dim: int
    activation: str = 'gelu'
    layer_norm: bool = False
    activate_final: bool = False
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, latents, deterministic=True):
        return MLP(
            self.hidden_dims,
            self.output_dim,
            activation=self.activation,
            layer_norm=self.layer_norm,
            activate_final=self.activate_final,
            dropout_rate=self.dropout_rate,
        )(latents, deterministic=deterministic)


class AutoEncoder(nn.Module):
    """Joint MLP autoencoder for vector observations."""

    hidden_dims: Sequence[int]
    latent_dim: int
    obs_dim: int
    decoder_hidden_dims: Optional[Sequence[int]] = None
    activation: str = 'gelu'
    layer_norm: bool = False
    decoder_activate_final: bool = False
    dropout_rate: float = 0.0

    def setup(self):
        decoder_hidden_dims = self.decoder_hidden_dims
        if decoder_hidden_dims is None:
            decoder_hidden_dims = tuple(reversed(self.hidden_dims))

        self.encoder = MLPEncoder(
            self.hidden_dims,
            self.latent_dim,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout_rate=self.dropout_rate,
        )
        self.decoder = MLPDecoder(
            decoder_hidden_dims,
            self.obs_dim,
            activation=self.activation,
            layer_norm=self.layer_norm,
            activate_final=self.decoder_activate_final,
            dropout_rate=self.dropout_rate,
        )

    def encode(self, observations, deterministic=True):
        return self.encoder(observations, deterministic=deterministic)

    def decode(self, latents, deterministic=True):
        return self.decoder(latents, deterministic=deterministic)

    def __call__(self, observations, deterministic=True):
        latents = self.encode(observations, deterministic=deterministic)
        reconstructions = self.decode(latents, deterministic=deterministic)
        return reconstructions, latents


def reconstruction_metrics(observations, reconstructions):
    """Return reconstruction metrics."""
    errors = reconstructions - observations
    abs_errors = jnp.abs(errors)
    mse = jnp.mean(errors**2)
    per_dim_mse = jnp.mean(errors**2, axis=0)
    return {
        'mae': jnp.mean(abs_errors),
        'max_abs_error': jnp.max(abs_errors),
        'mse': mse,
        'per_dim_mse_mean': jnp.mean(per_dim_mse),
        'per_dim_mse_max': jnp.max(per_dim_mse),
        'rmse': jnp.sqrt(mse),
    }
