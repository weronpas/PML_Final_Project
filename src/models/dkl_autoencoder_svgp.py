import sys
from pathlib import Path

import gpytorch
import torch
import torch.nn as nn
from gpytorch.means import ConstantMean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    NaturalVariationalDistribution,   # sostituisce CholeskyVariationalDistribution
    VariationalStrategy,
)


# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.kernels import DegradationKernel, SpaceTimeKernel


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 8,
                 hidden_dims: tuple[int, int] = (128, 64),
                 use_time_input: bool = True):
        super().__init__()
        self.use_time_input = use_time_input
        # +1 for the normalized cycle if enabled
        effective_input = input_dim + (1 if use_time_input else 0)
        layer_dims = (effective_input, *hidden_dims, latent_dim)
        layers = []
        for index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if index < len(layer_dims) - 2:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor,
                normalized_cycle: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_time_input and normalized_cycle is not None:
            t = normalized_cycle.view(-1, 1)
            x = torch.cat([x, t], dim=-1)
        return self.network(x)


class Decoder(nn.Module):
    """Symmetric decoder that reconstructs the original telemetry space."""

    def __init__(self, output_dim: int, latent_dim: int = 8, hidden_dims: tuple[int, int] = (64, 128)):
        super().__init__()
        layer_dims = (latent_dim, *hidden_dims, output_dim)
        layers = []
        for index, (in_dim, out_dim) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            layers.append(nn.Linear(in_dim, out_dim))
            if index < len(layer_dims) - 2:
                layers.append(nn.LayerNorm(out_dim))
                layers.append(nn.GELU())
        self.network = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.network(z)


class AutoencoderFeatureExtractor(nn.Module):
    """Compatibility wrapper used by existing loaders and checkpoints."""

    def __init__(self, input_dim: int, latent_dim: int = 8):
        super().__init__()
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim)
        self.decoder = Decoder(output_dim=input_dim, latent_dim=latent_dim)

    def forward(self, x: torch.Tensor, normalized_cycle: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(x, normalized_cycle)

    def reconstruct(self, x: torch.Tensor, normalized_cycle: torch.Tensor | None = None) -> torch.Tensor:
        latent = self.encoder(x, normalized_cycle)
        return self.decoder(latent)


class LatentSpaceSVGP(ApproximateGP):
    """Variational GP operating in the augmented space (latent embedding ‖ time).

    Input layout expected by this GP:
        x[..., :latent_dim]  →  z  (encoder output, scaled to [-1, 1])
        x[..., latent_dim:]  →  t  (normalised cycle, scalar in [0, 1])

    The inducing points must therefore have shape (M, latent_dim + 1).
    SpaceTimeKernel factorises the covariance as k_space(z,z') * k_time(t,t'),
    so the GP can distinguish engines with similar sensor profiles but at
    different life stages — the key information missing from a purely spatial GP.
    """

    def __init__(self, inducing_points: torch.Tensor, latent_dim: int):
        # inducing_points: (M, latent_dim + 1)  — already augmented with t
        variational_distribution = NaturalVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)

        self.mean_module = ConstantMean()
        # SpaceTimeKernel splits the input internally; it only needs latent_dim
        # to know where the z/t boundary lies.
        self.covar_module = SpaceTimeKernel(latent_dim=latent_dim)

    def forward(self, latent_x: torch.Tensor):
        mean_x = self.mean_module(latent_x)
        covar_x = self.covar_module(latent_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKLAutoencoderSVGP(nn.Module):
    """Autoencoder + latent GP composite used throughout the DKL pipeline."""

    def __init__(self, feature_extractor: nn.Module, inducing_points: torch.Tensor, latent_dim: int):
        super().__init__()

        if hasattr(feature_extractor, 'encoder') and hasattr(feature_extractor, 'decoder'):
            self.encoder = feature_extractor.encoder
            self.decoder = feature_extractor.decoder
            self.input_dim = getattr(feature_extractor, 'input_dim', None)
        else:
            self.encoder = feature_extractor
            self.decoder = None
            self.input_dim = None

        self.latent_dim = int(latent_dim)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)
        self.gp_layer = LatentSpaceSVGP(inducing_points=inducing_points, latent_dim=self.latent_dim)

    def encode(self, x: torch.Tensor, normalized_cycle: torch.Tensor | None = None) -> torch.Tensor:
        """Return the augmented GP input: [z_scaled ‖ t].

        The GP kernel (SpaceTimeKernel) expects the last dimension to be laid
        out as [z_0, …, z_{D-1}, t] so it can slice at self.latent_dim.

        If normalized_cycle is None (e.g. during inference without time info)
        we fall back to a zero time column, which keeps the call signature
        backward-compatible but removes temporal information.
        """
        latent = self.encoder(x, normalized_cycle)
        z_scaled = self.scale_to_bounds(latent)   # maps z to [-1, 1]

        if normalized_cycle is not None:
            t = normalized_cycle.view(-1, 1)       # (B, 1)  already in [0, 1]
        else:
            t = torch.zeros(x.size(0), 1, dtype=x.dtype, device=x.device)

        return torch.cat([z_scaled, t], dim=-1)    # (B, latent_dim + 1)


    def reconstruct(self, x: torch.Tensor, normalized_cycle: torch.Tensor | None = None) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError('Decoder is unavailable on this DKLAutoencoderSVGP instance.')
        latent = self.encoder(x, normalized_cycle)
        return self.decoder(latent)

    def forward(self, x: torch.Tensor, normalized_cycle: torch.Tensor | None = None):
        latent_x = self.encode(x, normalized_cycle)
        return self.gp_layer(latent_x)
        
    def predict_with_uncertainty(
        self,
        x: torch.Tensor,
        normalized_cycle: torch.Tensor | None = None,
        likelihood=None,
        confidence: float = 0.95,
    ):
        with torch.no_grad(), gpytorch.settings.fast_pred_var(False):  # ← disabilita l'approssimazione
            posterior = self.forward(x, normalized_cycle=normalized_cycle)
            predictive = likelihood(posterior) if likelihood is not None else posterior
            mean = predictive.mean
            variance = predictive.variance
            std = torch.sqrt(torch.clamp(variance, min=1e-12))

            from scipy import stats
            z_value = float(stats.norm.ppf((1 + confidence) / 2))
            lower = mean - z_value * std
            upper = mean + z_value * std

        return {
            'distribution': predictive,
            'mean': mean,
            'variance': variance,
            'lower': lower,
            'upper': upper,
        }