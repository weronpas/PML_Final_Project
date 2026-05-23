import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel

class DegradationKernel(gpytorch.kernels.Kernel):
    """
    Physics-informed kernel for rotating machinery degradation.
    Linear (global monotonic wear trend) + RBF (local sensor fluctuations).
    """
    def __init__(self, num_dimensions: int, **kwargs):
        super().__init__(**kwargs)
        
        # ARD (ard_num_dims) enables automatic feature selection per sensor
        # It will automatically ignore constant sensors identified in EDA
        self.rbf_module = ScaleKernel(RBFKernel(ard_num_dims=num_dimensions))
        
        # Captures the overall downward trend towards failure
        self.linear_module = ScaleKernel(LinearKernel(num_dimensions=num_dimensions))

    def forward(self, x1, x2, diag=False, **params):
        # K(x, x') = K_linear(x, x') + K_rbf(x, x')
        return self.linear_module(x1, x2, diag=diag, **params) + \
               self.rbf_module(x1, x2, diag=diag, **params)