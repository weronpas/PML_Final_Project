import torch
import torch.nn.functional as F
from gpytorch.likelihoods import GaussianLikelihood
from src.utils.metrics import negative_log_likelihood, evaluate_model_on_loader


def set_reproducible_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

def train_dkl_autoencoder(train_loader, input_dim, latent_dim=4, num_inducing=256, epochs=30, lr=0.001, encoder_lr=None, gp_lr=None, weight_decay=1e-5, lambda_recon=0.5, learn_likelihood_noise=True, init_likelihood_noise=0.01, label_noise_std=0.01, seed: int | None = None):
    from src.models.dkl_autoencoder_svgp import AutoencoderFeatureExtractor, DKLAutoencoderSVGP

    if seed is not None:
        set_reproducible_seed(int(seed))

    # 1. Inizializza il feature extractor (Autoencoder)
    feature_extractor = AutoencoderFeatureExtractor(input_dim=input_dim, latent_dim=latent_dim)
    
    # 2. Inizializza gli inducing points nello SPAZIO INPUT (compatibile con la VariationalStrategy)
    first_batch = next(iter(train_loader))
    x_init_batch = first_batch[0]
    inducing_points = x_init_batch[:num_inducing].clone()

    if inducing_points.size(0) < num_inducing:
        inducing_points = torch.randn(num_inducing, input_dim)

    # 3. Setup del Modello e Likelihood
    model = DKLAutoencoderSVGP(feature_extractor, inducing_points, latent_dim)
    likelihood = GaussianLikelihood()
    # Initialize observation noise variance to a reasonable value; allow learning if enabled.
    try:
        likelihood.noise = torch.tensor(float(init_likelihood_noise))
    except Exception:
        pass
    # Optionally allow the observation noise to be learned
    if not learn_likelihood_noise:
        for p in likelihood.parameters():
            p.requires_grad = False
    
    model.train()
    likelihood.train()

    # 4. Ottimizzatore congiunto con learning-rate groups
    # Backwards-compatible lr handling: if gp_lr/encoder_lr not provided, derive from lr
    if gp_lr is None:
        gp_lr = lr
    if encoder_lr is None:
        encoder_lr = lr * 0.1

    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': model.covar_module.parameters(), 'lr': gp_lr},
        {'params': model.mean_module.parameters(), 'lr': gp_lr},
        {'params': model.variational_parameters(), 'lr': gp_lr},
        {'params': likelihood.parameters(), 'lr': gp_lr},
    ])

    print(f"--- Starting DKL Autoencoder SVGP Training ({epochs} Epochs) ---")
    for epoch in range(epochs):
        epoch_mll_loss = 0.0
        epoch_recon_loss = 0.0
        
        for batch in train_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch

            # Add small Gaussian label noise as an aleatoric regularizer
            if label_noise_std and label_noise_std > 0.0:
                y_noisy = y + (torch.randn_like(y) * float(label_noise_std))
            else:
                y_noisy = y
            optimizer.zero_grad()
            
            # --- TASK 1: Regressione RUL (SVGP) ---
            output = model(x)
            loss_mll = negative_log_likelihood(output, y_noisy, likelihood=likelihood)
            
            # --- TASK 2: Ricostruzione (Autoencoder) ---
            x_reconstructed = model.feature_extractor.reconstruct(x)
            loss_recon = F.mse_loss(x_reconstructed, x)
            
            # --- Loss Totale Combinata ---
            # lambda_recon bilancia l'importanza della ricostruzione rispetto alla prediction del GP
            total_loss = loss_mll + (lambda_recon * loss_recon)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_mll_loss += loss_mll.item()
            epoch_recon_loss += loss_recon.item()
            
        print(f"Epoch {epoch+1:02d} | MLL Loss: {epoch_mll_loss/len(train_loader):.4f} | Recon Loss: {epoch_recon_loss/len(train_loader):.4f}")

    final_report = evaluate_model_on_loader(model, train_loader, likelihood=likelihood)
    print("DKL Autoencoder SVGP Training Evaluation Summary:")
    for metric_name, metric_value in final_report.items():
        print(f"  {metric_name}: {metric_value:.4f}" if isinstance(metric_value, (int, float)) else f"  {metric_name}: {metric_value}")

    return model, likelihood