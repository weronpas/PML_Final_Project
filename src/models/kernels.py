import gpytorch
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel, MaternKernel


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


class SpaceTimeKernel(gpytorch.kernels.Kernel):
    """
    Product kernel for spatio-temporal GP on degradation data.

    Input layout (last dimension):
        x[..., :latent_dim]  →  latent embedding z  (sensor feature space)
        x[..., latent_dim:]  →  normalised time t   (scalar, already in [0,1])

    Kernel factorisation:
        k((z,t), (z',t')) = k_space(z, z') * k_time(t, t')

    Rationale for the product (vs sum):
      - A *sum* would allow high similarity even when z≈z' but t is very
        different (or vice-versa), conflating engines at different life stages.
      - A *product* requires both feature AND temporal proximity for two points
        to be considered similar, which correctly encodes that engines with
        similar telemetry but different ages have different RUL.

    k_space  = DegradationKernel (Linear + RBF with ARD)
               Linear term: global monotonic wear trend over the manifold.
               RBF term:    local sensor fluctuations.

    k_time   = Matérn-3/2 (single lengthscale)
               Chosen over RBF because degradation curves are only once
               mean-square differentiable — Matérn-3/2 matches that smoothness
               without over-smoothing near end-of-life discontinuities.
               A ScaleKernel wrapper lets the model control the relative
               contribution of the temporal axis.
    """

    def __init__(self, latent_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = int(latent_dim)

        # Spatial component: same physics-informed structure as before
        self.k_space = DegradationKernel(num_dimensions=latent_dim)

        # Temporal component: Matérn-3/2 on the scalar time axis
        # ScaleKernel gives an independent output-scale hyperparameter so the
        # optimiser can weight temporal vs spatial contributions automatically.
        self.k_time = ScaleKernel(MaternKernel(nu=1.5, ard_num_dims=1))

    # ------------------------------------------------------------------
    # GPyTorch Kernel contract: implement forward(), not __call__().
    # forward() receives *already-expanded* tensors; diag and last_dim_is_batch
    # are forwarded transparently to sub-kernels via their __call__.
    # ------------------------------------------------------------------
    def forward(self, x1: "torch.Tensor", x2: "torch.Tensor", diag: bool = False, **params):
        # Split input along the feature axis
        z1, t1 = x1[..., :self.latent_dim], x1[..., self.latent_dim:]
        z2, t2 = x2[..., :self.latent_dim], x2[..., self.latent_dim:]

        # Each sub-kernel returns a LazyEvaluatedKernelTensor; the * operator
        # is overloaded in gpytorch to produce a MulLinearOperator, so no
        # dense materialisation happens until the solver needs it.
        K_space = self.k_space(z1, z2, diag=diag, **params)
        K_time  = self.k_time(t1, t2, diag=diag, **params)

        return K_space * K_time