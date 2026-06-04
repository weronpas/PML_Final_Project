import torch
import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel, MaternKernel


class DegradationKernel(gpytorch.kernels.Kernel):
    """
    Physics-informed kernel for rotating machinery degradation.
    Linear (global monotonic wear trend) + RBF (local sensor fluctuations).
    """
    def __init__(self, num_dimensions: int, **kwargs):
        super().__init__(**kwargs)
        self.rbf_module    = ScaleKernel(RBFKernel(ard_num_dims=num_dimensions))
        self.linear_module = ScaleKernel(LinearKernel(num_dimensions=num_dimensions))

    def forward(self, x1, x2, diag=False, **params):
        return (
            self.linear_module(x1, x2, diag=diag, **params)
            + self.rbf_module(x1, x2, diag=diag, **params)
        )


class SpaceTimeKernel(gpytorch.kernels.Kernel):
    """
    Product kernel for spatio-temporal GP on degradation data.

    Input layout  →  x[..., :latent_dim] = z   (latent embedding)
                     x[..., latent_dim:] = t   (normalised time, [0,1])

    k((z,t),(z',t')) = k_space(z,z') * k_time(t,t')

    Product (not sum) ensures similarity requires both feature AND temporal
    proximity — engines with identical telemetry but different ages have
    different RUL and should not be conflated.

    k_space = DegradationKernel  (Linear + RBF-ARD)
    k_time  = Matérn(nu)         (default nu=1.5 for once-differentiable paths)
    """

    def __init__(self, latent_dim: int, matern_nu: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = int(latent_dim)
        self.k_space    = DegradationKernel(num_dimensions=latent_dim)
        self.k_time     = ScaleKernel(MaternKernel(nu=matern_nu, ard_num_dims=1))

    def forward(self, x1, x2, diag=False, **params):
        z1, t1 = x1[..., :self.latent_dim], x1[..., self.latent_dim:]
        z2, t2 = x2[..., :self.latent_dim], x2[..., self.latent_dim:]

        K_space = self.k_space(z1, z2, diag=diag, **params)
        K_time  = self.k_time(t1, t2, diag=diag, **params)

        return K_space * K_time