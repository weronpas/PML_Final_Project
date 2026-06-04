import torch
import torch.nn.functional as F
import gpytorch
from sklearn.cluster import KMeans
from gpytorch.likelihoods import GaussianLikelihood
from src.utils.metrics import evaluate_model_on_loader
from gpytorch.optim import NGD  # Natural Gradient Descent


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

def _interval_score_loss(
    mean: torch.Tensor,
    std: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.05,
    alpha_lo_scale: float = 0.5,
) -> torch.Tensor:
    """Asymmetric Interval Score (Gneiting & Raftery, 2007).

    Jointly minimises interval width and miscoverage, with an asymmetric
    penalty that makes the lower bound more conservative (wider downward
    coverage) to bias predictions toward early RUL estimates and reduce
    the C-MAPSS penalty.

    IS = (upper - lower)
         + (2/α_lo) * max(lower - y, 0)   ← heavy penalty for late preds
         + (2/α_hi) * max(y - upper, 0)   ← lighter penalty for early preds

    Args:
        mean:          GP predictive mean  (batch,)
        std:           GP predictive std   (batch,)
        y:             targets             (batch,)
        alpha:         total miscoverage budget (0.05 → 95% CI)
        alpha_lo_scale: fraction of alpha assigned to the lower tail.
                        < 0.5  → lower tail gets less budget → lower bound
                        is pushed further down → more conservative toward
                        early predictions.  Default 0.5 = symmetric.
                        Use ~0.2 to strongly bias toward early predictions.
    """
    alpha_lo = alpha * alpha_lo_scale          # e.g. 0.05 * 0.2 = 0.010
    alpha_hi = alpha * (1.0 - alpha_lo_scale)  # e.g. 0.05 * 0.8 = 0.040

    from torch.distributions import Normal
    dist = Normal(mean, std.clamp(min=1e-6))
    lower = dist.icdf(torch.tensor(alpha_lo / 2.0, device=mean.device))
    upper = dist.icdf(torch.tensor(1.0 - alpha_hi / 2.0, device=mean.device))

    width    = (upper - lower).mean()
    miss_lo  = F.relu(lower - y).mean() * (2.0 / alpha_lo)   # y below lower → late pred
    miss_hi  = F.relu(y - upper).mean() * (2.0 / alpha_hi)   # y above upper → early pred

    return width + miss_lo + miss_hi


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
    lambda_recon=1.0,            # forces a structured latent space
    lambda_asymm=0.5,            # lowered: asymm loss can explode — keep it gentle
    lambda_interval=1.0,         # weight for asymmetric interval score loss
    interval_alpha: float = 0.05,          # target miscoverage (0.05 → 95% CI)
    interval_alpha_lo_scale: float = 0.2,  # <0.5 → bias lower bound downward (early preds)
    label_noise_std=0.0,
    beta_kl=0.5,                 # regularises GP variational distribution
    grad_clip: float = 1.0,      # max gradient norm — prevents asymm loss explosions
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

    # Deriva rul_scale automaticamente dal dataset del loader
    # (= std dei target RUL in spazio originale, usato per PHM08 e per calibrare il noise)
    _dataset = getattr(train_loader, "dataset", None)
    _tt = getattr(_dataset, "target_transform", None) if _dataset is not None else None
    _scaler = getattr(_tt, "scaler", None) if _tt is not None else None
    if _scaler is not None and hasattr(_scaler, "scale_"):
        rul_scale = float(_scaler.scale_[0])
    else:
        rul_scale = 1.0
        print("[WARNING] rul_scale non derivabile dal loader — PHM08 opera in spazio z.")

    # Noise auto-calibrato dal dataset: target_noise_std_cycles è l'unico
    # iperparametro interpretabile (std aleatoria in cicli fisici), poi viene
    # convertito in spazio z dividendo per rul_scale.
    # Questo garantisce lo stesso significato fisico su tutti i dataset
    # (FD001/FD002/FD003/FD004) indipendentemente dalla scala del target.
    _target_noise_std_cycles = 10.0          # ~10 cicli std aleatoria: ragionevole per CMAPSS
    _noise_z = (_target_noise_std_cycles / rul_scale) ** 2

    likelihood = GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Positive()
    )
    likelihood.noise = torch.tensor(_noise_z)
    for p in likelihood.parameters():
        p.requires_grad = False              # noise fisso, non appreso

    print(f"RUL scale derivato dal dataset: {rul_scale:.4f}")
    print(f"Likelihood noise auto-calibrato: {_noise_z:.4f}  (={_target_noise_std_cycles:.1f} cicli std in spazio fisico)")

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

    # Adam per encoder, decoder, kernel hyperparams
    optimizer = torch.optim.Adam([
        {'params': feature_extractor.encoder.parameters(), 'lr': encoder_lr},
        {'params': feature_extractor.decoder.parameters(), 'lr': decoder_lr},
        {'params': model.gp_layer.hyperparameters(), 'lr': gp_lr},
    ])


    # NGD per i soli parametri variazionali q(u) — aggiornamento Bayesiano
    ngd_optimizer = NGD(
        model.gp_layer.variational_parameters(),
        num_data=len(train_loader.dataset),
        lr=0.1,
    )

    mll = gpytorch.mlls.VariationalELBO(
        likelihood,
        model.gp_layer,
        num_data=len(train_loader.dataset),
        beta=beta_kl,
    )
    
    print(f"--- Starting DKL Autoencoder SVGP Training ({epochs} Epochs) ---")
    for epoch in range(epochs):
        epoch_mll_loss      = 0.0
        epoch_recon_loss    = 0.0
        epoch_asymm_loss    = 0.0
        epoch_interval_loss = 0.0

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

            x       = x.to(device)
            y_noisy = y_noisy.to(device)
            if t is not None:
                t = t.to(device)

            optimizer.zero_grad()
            ngd_optimizer.zero_grad()

            with gpytorch.settings.cholesky_jitter(1e-4):
                output = model(x, normalized_cycle=t)
                variational_elbo = mll(output, y_noisy.squeeze(-1))

                x_reconstructed = feature_extractor.reconstruct(x, normalized_cycle=t)
                loss_recon = F.mse_loss(x_reconstructed, x)

                # PHM08 asymmetric loss in spazio fisico (cicli originali)
                pred_errors = (output.mean - y_noisy.squeeze(-1)) * float(rul_scale)
                loss_asymm = torch.where(
                    pred_errors < 0,
                    torch.exp(torch.clamp(-pred_errors / 13.0, max=10.0)) - 1.0,
                    torch.exp(torch.clamp( pred_errors / 10.0, max=10.0)) - 1.0,
                ).mean()

                # Asymmetric Interval Score — ottimizza direttamente PICP e
                # larghezza degli intervalli con bias verso early predictions.
                # Viene calcolata in spazio z (stessa scala dell'output del GP)
                # per compatibilità con i gradienti dell'ELBO.
                pred_std = output.variance.clamp(min=1e-6).sqrt()
                loss_interval = _interval_score_loss(
                    mean=output.mean,
                    std=pred_std,
                    y=y_noisy.squeeze(-1),
                    alpha=interval_alpha,
                    alpha_lo_scale=interval_alpha_lo_scale,
                )

            elbo_loss = -variational_elbo

            # STEP 1: NGD riceve il gradiente dell'ELBO puro (parametri variazionali).
            # retain_graph=True perché il grafo serve ancora per aux_loss.
            elbo_loss.backward(retain_graph=True)
            ngd_optimizer.step()

            # STEP 2: Adam aggiorna encoder/decoder/kernel hyperparams su:
            #   - recon:    mantiene latent space strutturato
            #   - asymm:    guida le predizioni medie verso early
            #   - interval: calibra larghezza e copertura degli intervalli
            optimizer.zero_grad()
            aux_loss = (
                lambda_recon      * loss_recon
                + lambda_asymm    * loss_asymm
                + lambda_interval * loss_interval
            )
            aux_loss.backward()
            # Gradient clipping: impedisce che l'asymm loss esploda su batch
            # con errori grandi (es. engine in fase di saturazione RUL).
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group["params"]],
                max_norm=grad_clip,
            )
            optimizer.step()

            epoch_mll_loss      += float(elbo_loss.item())
            epoch_recon_loss    += loss_recon.item()
            epoch_asymm_loss    += loss_asymm.item()
            epoch_interval_loss += loss_interval.item()

        n = len(train_loader)
        print(
            f"Epoch {epoch+1:02d}"
            f" | ELBO: {epoch_mll_loss/n:.4f}"
            f" | Recon: {epoch_recon_loss/n:.4f}"
            f" | Asymm: {epoch_asymm_loss/n:.4f}"
            f" | Interval: {epoch_interval_loss/n:.4f}"
            f" | Noise: {likelihood.noise.item():.4f}"
        )

    final_report = evaluate_model_on_loader(model, train_loader, likelihood=likelihood)
    print("DKL Autoencoder SVGP Training Evaluation Summary:")
    for metric_name, metric_value in final_report.items():
        print(f"  {metric_name}: {metric_value:.4f}" if isinstance(metric_value, (int, float)) else f"  {metric_name}: {metric_value}")

    return model, likelihood