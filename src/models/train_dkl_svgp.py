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
            # FIXED: Catch time element
            t = batch[3].to(device) if len(batch) == 4 else None
            
            # FIXED: Pass normalized_cycle
            latents.append(feature_extractor.encoder(x, normalized_cycle=t).detach().cpu())
            
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
    lambda_recon=1.0,       # raised from 0.1 — forces a structured latent space
    lambda_asymm=0.5,       # weight for PHM asymmetric penalty (late preds cost more)
    asymm_late_penalty=1.5, # multiplier applied to late prediction errors (ŷ > y)
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

    # 1b. Warm-up: pre-train the autoencoder on reconstruction only so that
    #     the encoder produces a meaningful latent space before we run K-Means
    #     for inducing point initialization.  Without this the inducing points
    #     are seeded from a random projection and the GP never recovers.
    device = next(feature_extractor.parameters()).device
    _warmup_epochs = 10
    _warmup_lr = 5e-4
    _warmup_optimizer = torch.optim.Adam(
        feature_extractor.parameters(), lr=_warmup_lr, weight_decay=weight_decay
    )
    print(f"--- Encoder Warm-up ({_warmup_epochs} epochs, reconstruction only) ---")
    feature_extractor.train()
    for _ep in range(_warmup_epochs):
        _ep_loss = 0.0
        for batch in train_loader:
            x_w = batch[0].to(device)
            # FIXED: Catch time element
            t_w = batch[3].to(device) if len(batch) == 4 else None
            
            _warmup_optimizer.zero_grad()
            # FIXED: Pass normalized_cycle to reconstruct()
            _loss = F.mse_loss(feature_extractor.reconstruct(x_w, normalized_cycle=t_w), x_w)
            
            _loss.backward()
            _warmup_optimizer.step()
            _ep_loss += _loss.item()
        print(f"  Warm-up epoch {_ep+1:02d} | Recon Loss: {_ep_loss/len(train_loader):.4f}")


    # 2. Initialize inducing points in the latent space using K-Means centroids.
    #    Now collected from a trained encoder → meaningful clusters.
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
    # FIX: Gabbia Matematica per l'Incertezza (Varianza)
    # Impediamo all'ottimizzatore di far esplodere il rumore verso l'infinito.
    # Obblighiamo il modello a calibrare l'incertezza vera basandosi sui dati.
    noise_constraint = gpytorch.constraints.Interval(1e-4, 2.0)
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint)
    
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
    #    Default LR ratio is kept at ~10x (not 100x as before) so the encoder
    #    gradient is not drowned out by fast-moving GP hyperparameters.
    if encoder_lr is None:
        encoder_lr = 5e-4
    if decoder_lr is None:
        decoder_lr = 5e-4
    if gp_lr is None:
        gp_lr = 5e-3

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
        epoch_asymm_loss = 0.0
        
        for batch in train_loader:
            # FIXED: Unpack 4-tuple safely
            if len(batch) == 4:
                x, y, _, t = batch
            elif len(batch) == 3:
                x, y, _ = batch
                t = None
            else:
                x, y = batch
                t = None

            # Add small Gaussian label noise only when explicitly requested.
            if label_noise_std and label_noise_std > 0.0:
                y_noisy = y + (torch.randn_like(y) * float(label_noise_std))
            else:
                y_noisy = y
                
            x = x.to(device)
            y_noisy = y_noisy.to(device)
            if t is not None:
                t = t.to(device)
                
            optimizer.zero_grad()

            with gpytorch.settings.cholesky_jitter(1e-4):
                # FIXED: Pass normalized_cycle to forward()
                output = model(x, normalized_cycle=t)
                variational_elbo = mll(output, y_noisy.squeeze(-1))
                
                # FIXED: Pass normalized_cycle to reconstruct()
                x_reconstructed = feature_extractor.reconstruct(x, normalized_cycle=t)
                loss_recon = F.mse_loss(x_reconstructed, x)
                

                # PHM asymmetric penalty: late predictions (mean > true RUL)
                # are penalized more heavily than early ones, aligning training
                # with the C-MAPSS evaluation metric.
                pred_errors = output.mean - y_noisy.squeeze(-1)
                asym_weights = torch.where(
                    pred_errors > 0,
                    torch.full_like(pred_errors, float(asymm_late_penalty)),
                    torch.ones_like(pred_errors),
                )
                loss_asymm = (asym_weights * pred_errors.pow(2)).mean()

                total_loss = (
                    (-1.0 * variational_elbo)
                    + (lambda_recon * loss_recon)
                    + (lambda_asymm * loss_asymm)
                )

            total_loss.backward()
            optimizer.step()
            
            epoch_mll_loss += float((-1.0 * variational_elbo).item())
            epoch_recon_loss += loss_recon.item()
            epoch_asymm_loss += loss_asymm.item()
            
        print(
            f"Epoch {epoch+1:02d} | ELBO Loss: {epoch_mll_loss/len(train_loader):.4f}"
            f" | Recon Loss: {epoch_recon_loss/len(train_loader):.4f}"
            f" | Asymm Loss: {epoch_asymm_loss/len(train_loader):.4f}"
        )

    final_report = evaluate_model_on_loader(model, train_loader, likelihood=likelihood)
    print("DKL Autoencoder SVGP Training Evaluation Summary:")
    for metric_name, metric_value in final_report.items():
        print(f"  {metric_name}: {metric_value:.4f}" if isinstance(metric_value, (int, float)) else f"  {metric_name}: {metric_value}")

    return model, likelihood