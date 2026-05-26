import torch
import torch.nn.functional as F
from gpytorch.likelihoods import GaussianLikelihood
from src.utils.metrics import negative_log_likelihood, evaluate_model_on_loader

def train_dkl_autoencoder(train_loader, input_dim, latent_dim=4, num_inducing=100, epochs=10, lr=0.01, lambda_recon=0.5):
    from src.models.dkl_autoencoder_svgp import AutoencoderFeatureExtractor, DKLAutoencoderSVGP

    # 1. Inizializza il feature extractor (Autoencoder)
    feature_extractor = AutoencoderFeatureExtractor(input_dim=input_dim, latent_dim=latent_dim)
    
    # 2. Inizializza gli inducing points nello SPAZIO INPUT
    first_batch = next(iter(train_loader))
    x_init_batch = first_batch[0]
    inducing_points = x_init_batch[:num_inducing].clone()

    if inducing_points.size(0) < num_inducing:
        inducing_points = torch.randn(num_inducing, input_dim)

    # 3. Setup del Modello e Likelihood
    model = DKLAutoencoderSVGP(feature_extractor, inducing_points, latent_dim)
    likelihood = GaussianLikelihood()
    
    model.train()
    likelihood.train()

    # 4. Ottimizzatore congiunto
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4}, # Pesi NN
        {'params': model.covar_module.parameters()},                            # Pesi Kernel
        {'params': model.mean_module.parameters()},                             # GP Mean
        {'params': model.variational_parameters()},                             # Inducing points & varianza
        {'params': likelihood.parameters()},                                    # Rumore
    ], lr=lr)

    print(f"--- Starting DKL Autoencoder SVGP Training ({epochs} Epochs) ---")
    for epoch in range(epochs):
        epoch_mll_loss = 0.0
        epoch_recon_loss = 0.0
        
        for batch in train_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            optimizer.zero_grad()
            
            # --- TASK 1: Regressione RUL (SVGP) ---
            output = model(x)
            loss_mll = negative_log_likelihood(output, y, likelihood=likelihood)
            
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