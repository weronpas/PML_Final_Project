import torch
import torch.nn.functional as F
import gpytorch
from sklearn.cluster import KMeans
from gpytorch.likelihoods import GaussianLikelihood
from src.utils.metrics import evaluate_model_on_loader


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

def _collect_latent_embeddings(feature_extractor, train_loader, device):
    feature_extractor.eval()
    latents = []
    with torch.no_grad():
        for batch in train_loader:
            x = batch[0].to(device)
            latents.append(feature_extractor.encoder(x).detach().cpu())
    feature_extractor.train()
    if not latents:
        raise ValueError('Unable to collect latent embeddings from an empty training loader.')
    return torch.cat(latents, dim=0)


def train_dkl_autoencoder(
    train_loader,
    input_dim,
    latent_dim=8,
    num_inducing=500,
    epochs=30,
    lr=0.001,
    encoder_lr=None,
    decoder_lr=None,
    gp_lr=None,
    weight_decay=1e-5,
    lambda_recon=0.1,
    learn_likelihood_noise=True,
    init_likelihood_noise=0.01,
    label_noise_std=0.0,
    seed: int | None = None,
):
    from src.models.dkl_autoencoder_svgp import AutoencoderFeatureExtractor, DKLAutoencoderSVGP

    if seed is not None:
        set_reproducible_seed(int(seed))

    # 1. Inizializza il feature extractor (Autoencoder)
    feature_extractor = AutoencoderFeatureExtractor(input_dim=input_dim, latent_dim=latent_dim)
    
    # 2. Initialize inducing points in the latent space using K-Means centroids.
    device = next(feature_extractor.parameters()).device
    latent_embeddings = _collect_latent_embeddings(feature_extractor, train_loader, device=device)
    latent_np = latent_embeddings.numpy()
    n_clusters = min(int(num_inducing), int(latent_np.shape[0]))
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed if seed is not None else 42, n_init=10)
    kmeans.fit(latent_np)
    inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
    if inducing_points.size(0) < num_inducing:
        repeat_idx = torch.randint(0, inducing_points.size(0), (num_inducing - inducing_points.size(0),), device=device)
        inducing_points = torch.cat([inducing_points, inducing_points[repeat_idx]], dim=0)

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

    # 4. Split optimization between representation learning and GP fitting.
    if encoder_lr is None:
        encoder_lr = 1e-4
    if decoder_lr is None:
        decoder_lr = 1e-4
    if gp_lr is None:
        gp_lr = 1e-2

    optimizer = torch.optim.Adam([
        {'params': feature_extractor.encoder.parameters(), 'lr': encoder_lr, 'weight_decay': weight_decay},
        {'params': feature_extractor.decoder.parameters(), 'lr': decoder_lr, 'weight_decay': weight_decay},
        {'params': model.gp_layer.parameters(), 'lr': gp_lr},
        {'params': likelihood.parameters(), 'lr': gp_lr * 0.1},
    ])

    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))

    print(f"--- Starting DKL Autoencoder SVGP Training ({epochs} Epochs) ---")
    for epoch in range(epochs):
        epoch_mll_loss = 0.0
        epoch_recon_loss = 0.0
        
        for batch in train_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch

            # Add small Gaussian label noise only when explicitly requested.
            if label_noise_std and label_noise_std > 0.0:
                y_noisy = y + (torch.randn_like(y) * float(label_noise_std))
            else:
                y_noisy = y
            optimizer.zero_grad()

            with gpytorch.settings.cholesky_jitter(1e-4):
                output = model(x)
                variational_elbo = mll(output, y_noisy.squeeze(-1))
                x_reconstructed = feature_extractor.reconstruct(x)
                loss_recon = F.mse_loss(x_reconstructed, x)
                total_loss = (-1.0 * variational_elbo) + (lambda_recon * loss_recon)

            total_loss.backward()
            optimizer.step()
            
            epoch_mll_loss += float((-1.0 * variational_elbo).item())
            epoch_recon_loss += loss_recon.item()
            
        print(f"Epoch {epoch+1:02d} | ELBO Loss: {epoch_mll_loss/len(train_loader):.4f} | Recon Loss: {epoch_recon_loss/len(train_loader):.4f}")

    final_report = evaluate_model_on_loader(model, train_loader, likelihood=likelihood)
    print("DKL Autoencoder SVGP Training Evaluation Summary:")
    for metric_name, metric_value in final_report.items():
        print(f"  {metric_name}: {metric_value:.4f}" if isinstance(metric_value, (int, float)) else f"  {metric_name}: {metric_value}")

    return model, likelihood