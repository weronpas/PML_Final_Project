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
    lambda_asymm=2.0,     # weight for PHM08 asymmetric loss
    label_noise_std=0.0,
    beta_kl=0.1,  
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
    
    likelihood = GaussianLikelihood()
    likelihood.noise = torch.tensor(0.1)   # ~std 7 cicli in spazio z → ~290 cicli² originale
    for p in likelihood.parameters():
        p.requires_grad = False            # noise fisso, non appreso


    # Deriva rul_scale automaticamente dal dataset del loader
    # (= std dei target RUL in spazio originale, usato per PHM08)
    _dataset = getattr(train_loader, "dataset", None)
    _tt = getattr(_dataset, "target_transform", None) if _dataset is not None else None
    _scaler = getattr(_tt, "scaler", None) if _tt is not None else None
    if _scaler is not None and hasattr(_scaler, "scale_"):
        rul_scale = float(_scaler.scale_[0])
    else:
        rul_scale = 1.0
        print("[WARNING] rul_scale non derivabile dal loader — PHM08 opera in spazio z.")
    print(f"Likelihood noise (fisso): {likelihood.noise.item():.4f}")
    print(f"RUL scale derivato dal dataset: {rul_scale:.4f}")

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
            ngd_optimizer.zero_grad()

            with gpytorch.settings.cholesky_jitter(1e-4):
                output = model(x, normalized_cycle=t)
                variational_elbo = mll(output, y_noisy.squeeze(-1))

                x_reconstructed = feature_extractor.reconstruct(x, normalized_cycle=t)
                loss_recon = F.mse_loss(x_reconstructed, x)

                # pred_errors in cicli originali → PHM08 opera sulla scala corretta
                pred_errors = (output.mean - y_noisy.squeeze(-1)) * float(rul_scale)
                # PHM08 asymmetric loss: minimo naturale a d ≈ -3 cicli (early)
                # early (d<0): exp(-d/13)-1  → cresce lentamente
                # late  (d>0): exp( d/10)-1  → cresce più velocemente
                loss_asymm = torch.where(
                    pred_errors < 0,
                    torch.exp(torch.clamp(-pred_errors / 13.0, max=10.0)) - 1.0,
                    torch.exp(torch.clamp( pred_errors / 10.0, max=10.0)) - 1.0,
                ).mean()

            elbo_loss = -variational_elbo

            # STEP 1: NGD + likelihood ricevono il gradiente dell'ELBO puro.
            # retain_graph=True perche' il grafo serve ancora per total_loss.
            elbo_loss.backward(retain_graph=True)
            ngd_optimizer.step()

            # STEP 2: Adam aggiorna encoder/decoder/kernel su recon + asymm SOLTANTO
            # elbo_loss è già stato usato al STEP 1 — non ricalcolarlo qui
            optimizer.zero_grad()
            aux_loss = lambda_recon * loss_recon + lambda_asymm * loss_asymm
            aux_loss.backward()
            optimizer.step()

            epoch_mll_loss   += float(elbo_loss.item())
            epoch_recon_loss += loss_recon.item()
            epoch_asymm_loss += loss_asymm.item()
            
        print(
            f"Epoch {epoch+1:02d} | ELBO Loss: {epoch_mll_loss/len(train_loader):.4f}"
            f" | Recon Loss: {epoch_recon_loss/len(train_loader):.4f}"
            f" | Asymm Loss: {epoch_asymm_loss/len(train_loader):.4f}"
            f" | Noise: {likelihood.noise.item():.4f}"
        )

    final_report = evaluate_model_on_loader(model, train_loader, likelihood=likelihood)
    print("DKL Autoencoder SVGP Training Evaluation Summary:")
    for metric_name, metric_value in final_report.items():
        print(f"  {metric_name}: {metric_value:.4f}" if isinstance(metric_value, (int, float)) else f"  {metric_name}: {metric_value}")

    return model, likelihood