import sys
import warnings
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import gpytorch
from torch.utils.data import DataLoader
import joblib
import json
import random

# Resolve project root pathing
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.utils.metrics import (
    negative_log_likelihood,
    prediction_interval_coverage,
    average_interval_width,
    gaussian_nll_from_mean_var,
    crps_gaussian,
    r2_score,
    evaluate_model_on_loader,
    evaluate_prognostics_model,
)
from src.data.data_loader import StreamingCMAPSSDataset, load_cmapss_data
from src.models.svgp import RotatingMachinerySVGP
from src.models.dkl_autoencoder_svgp import DKLAutoencoderSVGP


def set_reproducible_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass



def simulate_online_stream_and_update(
    model, 
    likelihood, 
    stream_loader: DataLoader, 
    update_every_x_cycles: int = 10, 
    fine_tune_epochs: int = 3, 
    lr: float = 0.005,
    lambda_recon: float = 0.5,
    weight_decay: float = 1e-4,
    tune_feature_extractor: bool = False,
    grad_clip_norm: float = 1.0,
    device: str = "cpu",
    collect_metrics: bool = False,
    rul_scale_factor: float = 1.0  # <--- NEW: Denormalizes predictions to raw cycles
):
    """
    Simulates a real-time sensor stream from C-MAPSS data, updating the model 
    online every X cycles and tracking the behavior of confidence intervals.
    
    Dynamically supports both standard SVGP and DKLAutoencoderSVGP architectures.
    """
    is_dkl = isinstance(model, DKLAutoencoderSVGP)
    
    print(f"=== Starting Online Bayesian Updating Simulation ===")
    print(f"Detected Model Type: {'Deep Kernel Learning (DKL) SVGP' if is_dkl else 'Standard SVGP'}")
    print(f"Updating variational parameters every {update_every_x_cycles} cycles.\n")
    
    model.to(device)
    likelihood.to(device)
    
    # Storage buffers for streaming window data
    collected_x = []
    collected_y = []
    cycle_counter = 0
    global_cycles = 0
    
    model.eval()
    likelihood.eval()

    for batch_idx, batch in enumerate(stream_loader):
        # Unpack loader stream data
        if len(batch) == 3:
            x_step, y_step, _ = batch
        else:
            x_step, y_step = batch
            
        x_step = x_step.to(device)
        y_step = y_step.to(device)
        
        # 1. STREAMING PREDICTION & UNCERTAINTY TRACKING (Debugging Goal)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # Get posterior distribution through the strategy mapping
            posterior_dist = model(x_step)
            # Incorporate observation noise using the Gaussian likelihood
            predictive_dist = likelihood(posterior_dist)
            
            # --- FIX APPLIED HERE: Denormalize Mean and Variance ---
            # Multiply mean by the scale factor, and variance by the scale factor SQUARED
            pred_mean = predictive_dist.mean.cpu().numpy() * rul_scale_factor
            pred_variance = predictive_dist.variance.cpu().numpy() * (rul_scale_factor ** 2)
            
            # Track 95% Confidence Bounds (± 1.96 * standard deviation) using the true scaled variance
            pred_std = np.sqrt(pred_variance)
            lower_bound = pred_mean - (1.96 * pred_std)
            upper_bound = pred_mean + (1.96 * pred_std)
            
            true_val = y_step.cpu().numpy().flatten()
            covered = (true_val >= lower_bound) & (true_val <= upper_bound)
            picp_step = np.mean(covered)

        # Optionally collect per-sample predictions for later metric aggregation
        if collect_metrics:
            if 'metrics_preds' not in locals():
                metrics_preds = []
                metrics_trues = []
                metrics_vars = []
                metrics_lowers = []
                metrics_uppers = []
            # flatten and extend
            metrics_preds.extend(np.atleast_1d(pred_mean).flatten().tolist())
            metrics_vars.extend(np.atleast_1d(pred_variance).flatten().tolist())
            metrics_trues.extend(true_val.flatten().tolist())
            metrics_lowers.extend(np.atleast_1d(lower_bound).flatten().tolist())
            metrics_uppers.extend(np.atleast_1d(upper_bound).flatten().tolist())

        # Log telemetry details to monitor interval tracking over time
        if global_cycles % update_every_x_cycles == 0:
            print(f"[Cycle {global_cycles:03d}] Mean Predicted RUL: {pred_mean.mean():.2f} | Avg Variance (σ²): {pred_variance.mean():.4f} | Window PICP: {picp_step*100:.1f}%")

        # Accumulate data for the upcoming streaming fine-tuning optimization step
        collected_x.append(x_step)
        collected_y.append(y_step)
        batch_size_now = len(x_step)
        cycle_counter += batch_size_now
        global_cycles += batch_size_now
        
        # 2. ONLINE BAYESIAN VARIATIONAL UPDATE (Streaming Fine-Tuning)
        if cycle_counter >= update_every_x_cycles and fine_tune_epochs > 0:
            X_update = torch.cat(collected_x, dim=0)
            y_update = torch.cat(collected_y, dim=0)
            
            model.train()
            likelihood.train()
            
            # Build parameter optimize lists based on the active architecture model
            if is_dkl:
                param_groups = [
                    {'params': model.variational_strategy.parameters(), 'lr': lr},
                    {'params': model.covar_module.parameters(), 'lr': lr * 0.25, 'weight_decay': weight_decay},
                    {'params': model.mean_module.parameters(), 'lr': lr * 0.25}
                ]
                if tune_feature_extractor:
                    param_groups.append(
                        {'params': model.feature_extractor.parameters(), 'lr': lr * 0.1, 'weight_decay': weight_decay}
                    )
                # Keep likelihood noise calibrated during online adaptation to avoid overconfident collapse.
                param_groups.append({'params': likelihood.parameters(), 'lr': lr * 0.1})
            else:
                param_groups = [
                    {'params': model.variational_strategy.parameters(), 'lr': lr},
                    {'params': likelihood.parameters(), 'lr': lr * 0.1}
                ]
                
            online_optimizer = torch.optim.Adam(param_groups)
            
            # Fast streaming local optimization epochs
            for ft_epoch in range(fine_tune_epochs):
                online_optimizer.zero_grad()
                output = model(X_update)
                
                # --- FIX APPLIED HERE: Scale targets back down to [0, 1] for model training ---
                scaled_y_update = y_update / rul_scale_factor
                
                if hasattr(output, 'log_prob'):
                    loss_mll = -likelihood(output).log_prob(scaled_y_update.squeeze(-1)).mean()
                else:
                    loss_mll = negative_log_likelihood(output, scaled_y_update, likelihood=likelihood)
                
                # Calculate reconstruction preservation penalty if using DKL
                if is_dkl:
                    x_reconstructed = model.feature_extractor.reconstruct(X_update)
                    loss_recon = F.mse_loss(x_reconstructed, X_update)
                    total_loss = loss_mll + (lambda_recon * loss_recon)
                else:
                    total_loss = loss_mll
                    
                total_loss.backward()
                if grad_clip_norm and grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
                online_optimizer.step()
                
            # Reset back to operational evaluation mode
            model.eval()
            likelihood.eval()
            
            # Flush streaming memory buffers
            collected_x = []
            collected_y = []
            cycle_counter = 0

    print("\n=== Online Bayesian Updating Simulation Completed Successfully ===")
    results = {'model': model, 'likelihood': likelihood}
    if collect_metrics:
        # compute aggregated metrics
        y_pred = np.array(metrics_preds)
        y_var = np.array(metrics_vars)
        y_true = np.array(metrics_trues)
        lower = np.array(metrics_lowers)
        upper = np.array(metrics_uppers)

        # Use centralized metric utilities from src.utils.metrics
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        mae = float(np.mean(np.abs(y_pred - y_true)))
        picp = prediction_interval_coverage(lower, upper, y_true)
        avg_pred_var = float(np.mean(y_var))
        nll = gaussian_nll_from_mean_var(y_pred, y_var, y_true)
        crps = crps_gaussian(y_pred, y_var, y_true)
        r2 = r2_score(y_pred, y_true)
        avg_interval_w = average_interval_width(lower, upper)

        metrics = {
            'rmse': rmse,
            'mae': mae,
            'nll': nll,
            'crps': crps,
            'r2': r2,
            'picp_95': picp,
            'avg_interval_width_95': avg_interval_w,
            'avg_pred_variance': avg_pred_var,
            'num_samples': int(len(y_true))
        }

        preds = {
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'y_var': y_var.tolist(),
            'lower_95': lower.tolist(),
            'upper_95': upper.tolist()
        }

        results['metrics'] = metrics
        results['predictions'] = preds

    return results

if __name__ == "__main__":
    from sklearn.preprocessing import StandardScaler
    from src.models.dkl_autoencoder_svgp import AutoencoderFeatureExtractor
    from src.models.train_dkl_svgp import train_dkl_autoencoder
    
    print("Verifying pipeline safety mechanics for DKL compatibility...")

    SEED = 42
    set_reproducible_seed(SEED)
    
    # Load the real FD001 training stream from the repository data folder.
    features = [f's_{i}' for i in range(1, 22)]
    train_path = str(project_root / 'data' / 'raw' / 'train_FD001.txt')
    df_fd001 = load_cmapss_data(train_path)
    df_fd001 = df_fd001.sort_values(['unit_nr', 'time_cycles']).reset_index(drop=True)

    input_dim = len(features)
    latent_dim = 4

    # Use consistent RUL clipping/scale for CMAPSS
    MAX_RUL_SCALE = 125.0
    df_fd001['RUL'] = df_fd001['RUL'].clip(upper=MAX_RUL_SCALE)

    scaler = StandardScaler()
    # Training dataset: scale RUL targets to [0,1] by dividing by MAX_RUL_SCALE
    fd001_dataset = StreamingCMAPSSDataset(
        df=df_fd001,
        features=features,
        scaler=scaler,
        fit_scaler=True,
        target_scale=MAX_RUL_SCALE
    )
    fd001_stream_loader = DataLoader(fd001_dataset, batch_size=1, shuffle=False)
    # Train the full DKL-SVGP pipeline using the dedicated trainer instead of manual pretraining.
    train_loader = DataLoader(fd001_dataset, batch_size=256, shuffle=True)
    
    
    dkl_model, dkl_likelihood = train_dkl_autoencoder(
        train_loader=train_loader,
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_inducing=256,
        epochs=70,
        lr=0.0001,
        lambda_recon=0.5,
        seed=SEED,
    )
    
    # Execute full operational pipeline safety check 
    train_results = simulate_online_stream_and_update(
        model=dkl_model,
        likelihood=dkl_likelihood,
        stream_loader=fd001_stream_loader,
        update_every_x_cycles=10,
        fine_tune_epochs=1,
        lr=0.002,
        lambda_recon=0.5,
        weight_decay=1e-4,
        tune_feature_extractor=False,
        grad_clip_norm=1.0,
        rul_scale_factor=1.0 
    )
    dkl_model = train_results['model']
    dkl_likelihood = train_results['likelihood']

    # --- Persist training evaluation metrics (full report + config) ---
    try:
        train_eval_dir = project_root / 'artifacts' / 'evaluation'
        train_eval_dir.mkdir(parents=True, exist_ok=True)
        train_metrics_path = train_eval_dir / 'dkl_fd001_train_metrics.json'

        train_metrics = evaluate_model_on_loader(dkl_model, train_loader, likelihood=dkl_likelihood)
        # Enrich with reproducibility / config metadata
        train_metrics['seed'] = int(SEED)
        train_metrics['train_config'] = {
            'epochs': 70,
            'lr': 0.0001,
            'num_inducing': 256,
            'lambda_recon': 0.5,
            'input_dim': input_dim,
            'latent_dim': latent_dim,
            'max_rul_scale': float(MAX_RUL_SCALE)
        }

        with open(train_metrics_path, 'w') as fh:
            json.dump(train_metrics, fh, indent=2)
        print(f"Saved training metrics to: {train_metrics_path}")
    except Exception as e:
        print(f"Failed to persist training metrics: {e}")

    # Save trained model + likelihood + scaler for later evaluation
    ckpt_dir = project_root / 'artifacts' / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / 'dkl_fd001_checkpoint.pth'
    scaler_path = ckpt_dir / 'scaler_fd001.pkl'

    try:
        torch.save({
            'model_state_dict': dkl_model.state_dict(),
            'likelihood_state_dict': dkl_likelihood.state_dict()
        }, str(ckpt_path))
        joblib.dump(scaler, str(scaler_path))
        print(f"Saved checkpoint to: {ckpt_path}")
    except Exception as e:
        print(f"Warning: failed to save checkpoint/scaler: {e}")
    # Overwrite with richer metadata for reproducibility
    try:
        import time
        torch.save({
            'model_state_dict': dkl_model.state_dict(),
            'likelihood_state_dict': dkl_likelihood.state_dict(),
            'meta': {
                'saved_at': time.time(),
                'seed': int(SEED),
                'max_rul_scale': float(MAX_RUL_SCALE),
                'num_inducing': int(dkl_model.variational_strategy.inducing_points.size(0))
            }
        }, str(ckpt_path))
    except Exception as e:
        print(f"Warning: failed to save enriched checkpoint: {e}")

    # --- Run the simulator over the FD001 TEST split with true RUL labels ---
    print("\nRunning FD001 TEST stream (with RUL labels) for evaluation only...")
    test_path = str(project_root / 'data' / 'raw' / 'test_FD001.txt')
    rul_path = str(project_root / 'data' / 'raw' / 'RUL_FD001.txt')
    df_fd001_test = load_cmapss_data(test_path, rul_path)
    df_fd001_test = df_fd001_test.sort_values(['unit_nr', 'time_cycles']).reset_index(drop=True)

    df_fd001_test['RUL'] = df_fd001_test['RUL'].clip(upper=MAX_RUL_SCALE)

    if scaler_path.exists():
        try:
            scaler_test = joblib.load(str(scaler_path))
            fit_scaler_test = False
            print(f"Loaded scaler from {scaler_path}")
        except Exception:
            scaler_test = StandardScaler()
            fit_scaler_test = True
    else:
        scaler_test = StandardScaler()
        fit_scaler_test = True

    # Test dataset: keep raw RUL values in [0, MAX_RUL_SCALE] for metric calculation
    fd001_test_dataset = StreamingCMAPSSDataset(
        df=df_fd001_test,
        features=features,
        scaler=scaler_test,
        fit_scaler=fit_scaler_test,
        target_scale=1.0
    )
    fd001_test_loader = DataLoader(fd001_test_dataset, batch_size=1, shuffle=False)

    dkl_model_test = None
    dkl_likelihood_test = gpytorch.likelihoods.GaussianLikelihood()
    try:
        with torch.no_grad():
            # Use a modest initial noise (on normalized scale) if checkpoint loading fails.
            dkl_likelihood_test.noise = torch.tensor(0.01)
    except Exception:
        pass

    if ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location='cpu')
            fe_test = AutoencoderFeatureExtractor(input_dim=input_dim, latent_dim=latent_dim)
            # Rebuild model with the same inducing-point count used during training.
            if isinstance(ckpt, dict) and 'meta' in ckpt and isinstance(ckpt['meta'], dict) and 'num_inducing' in ckpt['meta']:
                num_inducing_test = int(ckpt['meta']['num_inducing'])
            else:
                inducing_from_state = ckpt.get('model_state_dict', {}).get('variational_strategy.inducing_points', None)
                num_inducing_test = int(inducing_from_state.size(0)) if inducing_from_state is not None else 256

            inducing_pts_latent_test = torch.randn(num_inducing_test, input_dim)
            dkl_model_test = DKLAutoencoderSVGP(
                feature_extractor=fe_test,
                inducing_points=inducing_pts_latent_test,
                latent_dim=latent_dim
            )
            dkl_model_test.load_state_dict(ckpt['model_state_dict'], strict=True)
            dkl_likelihood_test.load_state_dict(ckpt['likelihood_state_dict'], strict=True)
            print(f"Loaded trained model + likelihood from {ckpt_path}")
        except Exception as e:
            print(f"Failed to load checkpoint, falling back to fresh model: {e}")

    if dkl_model_test is None:
        fe_test = AutoencoderFeatureExtractor(input_dim=input_dim, latent_dim=latent_dim)
        inducing_pts_latent_test = torch.randn(256, input_dim)
        dkl_model_test = DKLAutoencoderSVGP(
            feature_extractor=fe_test,
            inducing_points=inducing_pts_latent_test,
            latent_dim=latent_dim
        )

    # Evaluation only: no fine-tuning. Collect metrics and persist them.
    # Evaluation: simulator will remultiply model's normalized predictions by `MAX_RUL_SCALE`
    eval_results = simulate_online_stream_and_update(
        model=dkl_model_test,
        likelihood=dkl_likelihood_test,
        stream_loader=fd001_test_loader,
        update_every_x_cycles=10,
        fine_tune_epochs=0, 
        lr=0.001,
        lambda_recon=0.0,
        collect_metrics=True,
        rul_scale_factor=MAX_RUL_SCALE 
    )

    # Persist evaluation metrics and predictions
    eval_dir = project_root / 'artifacts' / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = eval_dir / 'dkl_fd001_test_metrics.json'
    preds_path = eval_dir / 'dkl_fd001_test_predictions.json'

    try:
        if isinstance(eval_results, dict) and 'metrics' in eval_results:
            eval_results['metrics']['seed'] = int(SEED)
            # Add prognostics-style report (PHM08, fleet correlation, counts, etc.)
            # Build prognostics-style report from the already-scaled predictions
            try:
                preds = eval_results.get('predictions', {})
                if preds and 'y_pred' in preds and 'y_true' in preds:
                    prognostics_report = evaluate_prognostics_model(
                        preds['y_pred'],
                        preds['y_true'],
                        predictive_vars=preds.get('y_var', None),
                        lower_bounds=preds.get('lower_95', None),
                        upper_bounds=preds.get('upper_95', None),
                    )
                else:
                    prognostics_report = None
            except Exception:
                prognostics_report = None
            if prognostics_report is not None:
                eval_results['metrics']['prognostics_report'] = prognostics_report

            with open(metrics_path, 'w') as fh:
                json.dump(eval_results['metrics'], fh, indent=2)
            with open(preds_path, 'w') as fh:
                json.dump(eval_results['predictions'], fh)
            print(f"Saved evaluation metrics to: {metrics_path}")
            print(f"Saved evaluation predictions to: {preds_path}")
        else:
            print("No metrics were returned from evaluation.")
    except Exception as e:
        print(f"Failed to persist evaluation artifacts: {e}")