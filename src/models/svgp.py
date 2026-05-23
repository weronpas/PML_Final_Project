import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean

from src.models.kernels import DegradationKernel

class RotatingMachinerySVGP(ApproximateGP):
    """
    Stochastic Variational Gaussian Process for RUL prediction.
    Utilizes inducing points for scalability (O(M^3) instead of O(N^3)).
    """
    def __init__(self, inducing_points: torch.Tensor, num_dimensions: int):
        # Define the variational distribution q(u) using Cholesky for stability
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        
        # Strategy maps the inducing points to the full dataset
        # learn_inducing_locations=True allows Adam to optimize their positions
        variational_strategy = VariationalStrategy(
            self, 
            inducing_points, 
            variational_distribution, 
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        
        # Core GP components
        self.mean_module = ConstantMean()
        self.covar_module = DegradationKernel(num_dimensions=num_dimensions)

    def forward(self, x):
        """
        Computes the prior predictive distribution for the input x.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)