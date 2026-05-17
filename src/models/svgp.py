import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class RotatingMachinerySVGP(ApproximateGP):
    """
    Sparse Variational Gaussian Process model for rotating machinery degradation.
    """
    def __init__(self, inducing_points: torch.Tensor, custom_kernel: gpytorch.kernels.Kernel):
        """
        Initializes the SVGP model.
        
        Args:
            inducing_points (torch.Tensor): Initial locations for inducing points.
            custom_kernel (gpytorch.kernels.Kernel): Physics-informed kernel.
        """
        # Define the variational distribution and strategy
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self, 
            inducing_points, 
            variational_distribution, 
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        
        # Mean and Covariance (Kernel)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = custom_kernel

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        """
        Forward pass for the GP.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)