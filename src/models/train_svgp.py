import sys
from pathlib import Path
import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.svgp import RotatingMachinerySVGP

def train_svgp(train_loader: DataLoader, num_features: int, num_inducing: int = 100, epochs: int = 10, lr: float = 0.01):
    """
    Initializes and trains the SVGP model maximizing the Evidence Lower Bound (ELBO).
    """
    # Initialize inducing points using the first batch of training data
    x_init_batch, _ = next(iter(train_loader))
    inducing_points = x_init_batch[:num_inducing].clone()
    
    # Fallback if the first batch size is smaller than the requested inducing points
    if inducing_points.size(0) < num_inducing:
        inducing_points = torch.randn(num_inducing, num_features)

    model = RotatingMachinerySVGP(inducing_points=inducing_points, num_dimensions=num_features)
    likelihood = GaussianLikelihood()
    likelihood.noise = torch.tensor([0.05])

    model.train()
    likelihood.train()

    # Joint optimizer for GP hyperparameters, variational parameters, and likelihood noise
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr)

    # Variational ELBO objective function scaling by total dataset size
    mll = VariationalELBO(likelihood, model, num_data=len(train_loader.dataset))

    print(f"--- Starting SVGP Training ({epochs} Epochs) ---")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass through the variational posterior q(f)
            output = model(x)
            
            # Maximize ELBO by minimizing the Negative ELBO Loss
            loss = -mll(output, y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Loss (Negative ELBO): {epoch_loss / len(train_loader):.4f}")
        
    print("-----------------------------------------\n")
    return model, likelihood