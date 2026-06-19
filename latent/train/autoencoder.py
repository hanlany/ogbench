from typing import Any, Sequence

import flax.linen as nn
import jax.numpy as jnp


class MLP(nn.Module):
    """Simple MLP used by the latent encoder and decoder."""

    hidden_dims: Sequence[int]
    output_dim: int
    activation: Any = nn.gelu

    @nn.compact
    def __call__(self, x):
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = self.activation(x)
        return nn.Dense(self.output_dim)(x)


class MLPEncoder(nn.Module):
    """Map vector observations to a lower-dimensional latent state."""

    hidden_dims: Sequence[int]
    latent_dim: int

    @nn.compact
    def __call__(self, observations):
        return MLP(self.hidden_dims, self.latent_dim)(observations)


class MLPDecoder(nn.Module):
    """Map latent states back to vector observations."""

    hidden_dims: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, latents):
        return MLP(self.hidden_dims, self.output_dim)(latents)


class AutoEncoder(nn.Module):
    """Joint MLP autoencoder for vector observations."""

    hidden_dims: Sequence[int]
    latent_dim: int
    obs_dim: int

    def setup(self):
        self.encoder = MLPEncoder(self.hidden_dims, self.latent_dim)
        self.decoder = MLPDecoder(tuple(reversed(self.hidden_dims)), self.obs_dim)

    def encode(self, observations):
        return self.encoder(observations)

    def decode(self, latents):
        return self.decoder(latents)

    def __call__(self, observations):
        latents = self.encode(observations)
        reconstructions = self.decode(latents)
        return reconstructions, latents


def reconstruction_metrics(observations, reconstructions):
    """Return MSE and RMSE reconstruction metrics."""
    mse = jnp.mean((reconstructions - observations) ** 2)
    return {
        'mse': mse,
        'rmse': jnp.sqrt(mse),
    }

