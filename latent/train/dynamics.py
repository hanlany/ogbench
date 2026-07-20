from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp

from latent.train.autoencoder import MLP


class LatentDynamics(nn.Module):
    """Predict the next latent state from the current latent and action."""

    hidden_dims: Sequence[int]
    latent_dim: int
    activation: str = 'gelu'
    layer_norm: bool = False
    dropout_rate: float = 0.0
    prediction_type: str = 'absolute'

    @nn.compact
    def __call__(self, latents, actions, deterministic=True):
        inputs = jnp.concatenate([latents, actions], axis=-1)
        predictions = MLP(
            self.hidden_dims,
            self.latent_dim,
            activation=self.activation,
            layer_norm=self.layer_norm,
            dropout_rate=self.dropout_rate,
        )(inputs, deterministic=deterministic)
        if self.prediction_type == 'absolute':
            return predictions
        if self.prediction_type == 'residual':
            return latents + predictions
        raise ValueError(f'Unsupported prediction_type {self.prediction_type!r}.')
