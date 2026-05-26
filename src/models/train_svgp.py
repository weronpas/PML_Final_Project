import sys
from pathlib import Path
import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from torch.utils.data import DataLoader

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.svgp import RotatingMachinerySVGP
from src.utils.metrics import negative_log_likelihood, evaluate_model_on_loader

def train_svgp(train_loader: DataLoader, num_features: int, num_inducing: int = 100, epochs: int = 10, lr: float = 0.01):
    """
    Initializes and trains the SVGP model using a probabilistic loss.
    """
    # Initialize inducing points using the first batch of training data
    first_batch = next(iter(train_loader))
    x_init_batch = first_batch[0]
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

    print(f"--- Starting SVGP Training ({epochs} Epochs) ---")
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            optimizer.zero_grad()
            
            # Forward pass through the variational posterior q(f)
            output = model(x)
            
            # Minimize the negative log-likelihood when the model returns a distribution
            loss = negative_log_likelihood(output, y, likelihood=likelihood)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - Loss (Negative Log Likelihood): {epoch_loss / len(train_loader):.4f}")

    final_report = evaluate_model_on_loader(model, train_loader, likelihood=likelihood)
    print("SVGP Training Evaluation Summary:")
    for metric_name, metric_value in final_report.items():
        print(f"  {metric_name}: {metric_value:.4f}" if isinstance(metric_value, (int, float)) else f"  {metric_name}: {metric_value}")
        
    print("-----------------------------------------\n")
    return model, likelihood