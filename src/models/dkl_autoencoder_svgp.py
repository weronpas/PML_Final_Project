import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean


import sys
import warnings
from pathlib import Path

# Resolve project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Importiamo il kernel custom fornito
from src.models.kernels import DegradationKernel

class AutoencoderFeatureExtractor(nn.Module):
    """
    Rete neurale Autoencoder. L'encoder estrae le feature per il GP (DKL), 
    mentre il decoder serve a mantenere l'integrità delle feature ricostruendo l'input.
    """
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 64):
        super().__init__()
        
        # Encoder: mappa i dati originali nello spazio latente
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Decoder: ricostruisce i dati originali (usato durante il training)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # Il forward primario restituisce lo spazio latente per il GP
        return self.encoder(x)

    def reconstruct(self, x):
        # Utilizzato per calcolare la reconstruction loss
        latent = self.encoder(x)
        return self.decoder(latent)

class DKLAutoencoderSVGP(ApproximateGP):
    """
    Modello Deep Kernel Learning SVGP. 
    Usa l'encoder dell'Autoencoder come feature extractor prima di applicare il DegradationKernel.
    """
    def __init__(self, feature_extractor: nn.Module, inducing_points: torch.Tensor, latent_dim: int):
        # NOTA: gli inducing_points ora vivono nello SPAZIO LATENTE
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        
        variational_strategy = VariationalStrategy(
            self, 
            inducing_points, 
            variational_distribution, 
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        
        self.feature_extractor = feature_extractor
        
        # Scala l'output della rete neurale per la stabilità del GP (comune nel DKL)
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)
        
        # Componenti Core GP che usano la dimensionalità latente
        self.mean_module = ConstantMean()
        # Usa il tuo Kernel informato dalla fisica
        self.covar_module = DegradationKernel(num_dimensions=latent_dim)

    def forward(self, x):
        # 1. Passa i dati crudi attraverso l'Encoder
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)
        
        # 2. GP Prior distribution sulle feature latenti
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)