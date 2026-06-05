import sys
import warnings
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn.functional as F
import numpy as np
import gpytorch
from torch.utils.data import DataLoader
import joblib
import json
import random
import csv

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
from src.data.data_loader import StreamingCMAPSSDataset, load_cmapss_data, get_sensor_feature_columns
from src.models.dkl_autoencoder_svgp import DKLAutoencoderSVGP, AutoencoderFeatureExtractor


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
    rul_scale_factor: float = 1.0,
    use_natural_gradient: bool = True,   # NEW: enable true Bayesian online learning
    ngd_lr: float = 0.1,                 # NEW: lr for NGD (separate from Adam)
):
    """
    Simulates a real-time sensor stream from C-MAPSS data, updating the model 
    online every X cycles and tracking the behavior of confidence intervals.
    
    Dynamically supports both standard SVGP and DKLAutoencoderSVGP architectures.
    """
    is_dkl = isinstance(model, DKLAutoencoderSVGP)
    target_gp = model.gp_layer if is_dkl else model

    dataset = getattr(stream_loader, 'dataset', None)
    num_data_total = len(dataset) if dataset is not None else 10_000
    target_inverse = getattr(dataset, 'inverse_transform', None) if dataset is not None else None
    
    print(f"=== Starting Online Bayesian Updating Simulation ===")
    print(f"Detected Model Type: {'Deep Kernel Learning (DKL) SVGP' if is_dkl else 'Standard SVGP'}")
    print(f"Updating variational parameters every {update_every_x_cycles} cycles.\n")
    
    
    # ------------------------------------------------------------------ #
    # Setup NGD optimizer once, outside the loop                       #
    # ------------------------------------------------------------------ #
    if use_natural_gradient:
        from gpytorch.optim import NGD
        ngd_optimizer = NGD(
            target_gp.variational_parameters(),
            num_data=num_data_total,
            lr=ngd_lr,
        )
        mll = gpytorch.mlls.VariationalELBO(
            likelihood, target_gp, num_data=num_data_total
        )

    model.eval()
    likelihood.eval()

    for batch_idx, batch in enumerate(stream_loader):
        # --- unpack batch (unchanged) ---
        if len(batch) == 4:
            x_step, y_step, unit_step, t_step = batch
        elif len(batch) == 3:
            x_step, y_step, unit_step = batch
            t_step = None
        else:
            x_step, y_step = batch
            t_step = None

        x_step = x_step.to(device)
        y_step = y_step.to(device)
        if t_step is not None:
            t_step = t_step.to(device)

        # ------------------------------------------------------------------ #
        # 1. PREDICTION (unchanged)                                         #
        # ------------------------------------------------------------------ #
        with torch.no_grad():
            prediction   = model.predict_with_uncertainty(x_step, normalized_cycle=t_step, likelihood=likelihood)
            pred_mean     = prediction['mean'].cpu().numpy()
            pred_variance = prediction['variance'].cpu().numpy()
            lower_bound   = prediction['lower'].cpu().numpy()
            upper_bound   = prediction['upper'].cpu().numpy()
            true_val      = y_step.cpu().numpy().flatten()

            if callable(target_inverse):
                pred_mean, pred_variance = target_inverse(pred_mean.flatten(), variance=pred_variance.flatten())
                true_val    = target_inverse(true_val)
                lower_bound = target_inverse(lower_bound.flatten())
                upper_bound = target_inverse(upper_bound.flatten())


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

        # ------------------------------------------------------------------ #
        # 2. UPDATE — NGD applies here                                      #
        # ------------------------------------------------------------------ #
        if use_natural_gradient:
            # Freeze all except the variational parameters
            # (encoder, decoder, kernel remain fixed during streaming)
            model.train()
            if is_dkl:
                model.encoder.eval()
                model.decoder.eval()
                # Freeze the kernel — only update q(u)
                for p in target_gp.covar_module.parameters():
                    p.requires_grad_(False)

            ngd_optimizer.zero_grad()

            with gpytorch.settings.cholesky_jitter(1e-3):
                output = (
                    model(x_step, normalized_cycle=t_step) if is_dkl
                    else model(x_step)
                )
                loss = -mll(output, y_step.squeeze(-1))

            loss.backward()
            ngd_optimizer.step()   # Bayesian update in closed form

            # Restore kernel
            for p in target_gp.covar_module.parameters():
                p.requires_grad_(True)

            model.eval()
            likelihood.eval()



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

    print("Verifying pipeline safety mechanics for DKL compatibility across FD001/FD002/FD003/FD004...")

    SEED = 42
    FD_SUBSETS = ['FD001']    #, 'FD002', 'FD003', 'FD004']
    MAX_RUL_SCALE = 125.0

    def persist_aggregate_summary(run_summaries: list) -> None:
        eval_dir = project_root / 'artifacts' / 'evaluation'
        eval_dir.mkdir(parents=True, exist_ok=True)

        summary_json_path = eval_dir / 'dkl_all_subsets_summary.json'
        summary_csv_path = eval_dir / 'dkl_all_subsets_summary.csv'

        rows = []
        for summary in run_summaries:
            row = {
                'subset': summary.get('subset'),
                'num_sensors': summary.get('num_sensors'),
            }
            metrics_path = summary.get('test_metrics_path')
            if metrics_path and Path(metrics_path).exists():
                try:
                    with open(metrics_path, 'r') as fh:
                        metrics = json.load(fh)
                    for key in [
                        'rmse',
                        'mae',
                        'nll',
                        'crps',
                        'r2',
                        'picp_95',
                        'avg_interval_width_95',
                        'avg_pred_variance',
                        'num_samples',
                    ]:
                        if key in metrics:
                            row[key] = metrics[key]
                except Exception:
                    row['metrics_read_error'] = True
            else:
                row['metrics_read_error'] = True
            rows.append(row)

        with open(summary_json_path, 'w') as fh:
            json.dump(rows, fh, indent=2)

        fieldnames = [
            'subset',
            'num_sensors',
            'rmse',
            'mae',
            'nll',
            'crps',
            'r2',
            'picp_95',
            'avg_interval_width_95',
            'avg_pred_variance',
            'num_samples',
            'metrics_read_error',
        ]
        with open(summary_csv_path, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        print(f"Saved aggregate summary JSON to: {summary_json_path}")
        print(f"Saved aggregate summary CSV to: {summary_csv_path}")

    def run_fd_subset_pipeline(fd_subset: str, seed: int) -> dict:
        set_reproducible_seed(seed)
        print(f"\n===================== {fd_subset}: PIPELINE START =====================")

        train_path = str(project_root / 'data' / 'raw' / f'train_{fd_subset}.txt')
        test_path = str(project_root / 'data' / 'raw' / f'test_{fd_subset}.txt')
        rul_path = str(project_root / 'data' / 'raw' / f'RUL_{fd_subset}.txt')

        df_train = load_cmapss_data(train_path)
        df_train = df_train.sort_values(['unit_nr', 'time_cycles']).reset_index(drop=True)

        features = get_sensor_feature_columns(df_train)
        if not features:
            raise ValueError(f"No sensor columns found for {fd_subset}")

        input_dim = len(features)
        latent_dim = 8

        feature_scaler = StandardScaler()
        time_scaler = MinMaxScaler()

        train_dataset = StreamingCMAPSSDataset(
            df=df_train,
            features=features,
            scaler=feature_scaler,
            fit_scaler=True,
            time_scaler=time_scaler,          # Inject time scaler
            fit_time_scaler=True,             # Learn train min/max
            fit_target_transform=True
        )
        train_stream_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)

        dkl_model, dkl_likelihood = train_dkl_autoencoder(
            train_loader=train_loader,
            input_dim=len(features),
            latent_dim=latent_dim,
            num_inducing=500,
            epochs=5,
            encoder_lr=1e-4,
            decoder_lr=1e-4,
            gp_lr=1e-4,
            lambda_recon=0.1,
            seed=seed,
        )

        # Streaming-style adaptation on training stream.
        train_results = simulate_online_stream_and_update(
            model=dkl_model,
            likelihood=dkl_likelihood,
            stream_loader=train_stream_loader,
            update_every_x_cycles=10,
            fine_tune_epochs=1,
            lr=0.002,
            lambda_recon=0.1,
            weight_decay=1e-4,
            tune_feature_extractor=False,
            grad_clip_norm=1.0,
        )
        dkl_model = train_results['model']
        dkl_likelihood = train_results['likelihood']

        # Persist train metrics.
        train_eval_dir = project_root / 'artifacts' / 'evaluation'
        train_eval_dir.mkdir(parents=True, exist_ok=True)
        train_metrics_path = train_eval_dir / f'dkl_{fd_subset.lower()}_train_metrics.json'

        train_metrics = evaluate_model_on_loader(dkl_model, train_loader, likelihood=dkl_likelihood)
        train_metrics['subset'] = fd_subset
        train_metrics['seed'] = int(seed)
        train_metrics['train_config'] = {
            'epochs': 50,
            'lr': 0.0001,
            'num_inducing': 256,
            'lambda_recon': 0.5,
            'input_dim': input_dim,
            'latent_dim': latent_dim,
            'num_sensors': len(features),
            'max_rul_scale': float(MAX_RUL_SCALE),
        }
        with open(train_metrics_path, 'w') as fh:
            json.dump(train_metrics, fh, indent=2)
        print(f"Saved training metrics to: {train_metrics_path}")

        # Save trained model + likelihood + scaler for subset-specific evaluation.
        ckpt_dir = project_root / 'artifacts' / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / f'dkl_{fd_subset.lower()}_checkpoint.pth'
        scaler_path = ckpt_dir / f'scaler_{fd_subset.lower()}.pkl'

        import time
        torch.save(
            {
                'model_state_dict': dkl_model.state_dict(),
                'likelihood_state_dict': dkl_likelihood.state_dict(),
                'meta': {
                    'saved_at': time.time(),
                    'subset': fd_subset,
                    'seed': int(seed),
                    'max_rul_scale': float(MAX_RUL_SCALE),
                    'num_inducing': int(dkl_model.gp_layer.variational_strategy.inducing_points.size(0)),
                    'input_dim': int(input_dim),
                    'latent_dim': int(latent_dim),
                    'features': list(features),
                },
            },
            str(ckpt_path),
        )
        joblib.dump({'feature': feature_scaler, 'time': time_scaler}, str(scaler_path))
        print(f"Saved checkpoint to: {ckpt_path}")

        # --- Run TEST stream (with true RUL labels) for evaluation only ---
        print(f"\nRunning {fd_subset} TEST stream (with RUL labels) for evaluation only...")
        df_test = load_cmapss_data(test_path, rul_path)
        df_test = df_test.sort_values(['unit_nr', 'time_cycles']).reset_index(drop=True)

        saved_scalers = joblib.load(str(scaler_path))
        
        # Apply to Test/Streaming Data
        test_dataset = StreamingCMAPSSDataset(
            df=df_test,                                # FIXED
            features=features,                         # FIXED
            scaler=saved_scalers['feature'],           # FIXED: Use loaded scaler
            fit_scaler=False,
            time_scaler=saved_scalers['time'],         # FIXED: Use loaded scaler
            fit_time_scaler=False,                     
            target_transform=train_dataset.target_transform,
            fit_target_transform=False
        )

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        dkl_model_test = None
        dkl_likelihood_test = gpytorch.likelihoods.GaussianLikelihood()
        try:
            with torch.no_grad():
                dkl_likelihood_test.noise = torch.tensor(0.01)
        except Exception:
            pass

        if ckpt_path.exists():
            ckpt = torch.load(str(ckpt_path), map_location='cpu')
            ckpt_meta = ckpt.get('meta', {}) if isinstance(ckpt, dict) else {}
            num_inducing_test = int(ckpt_meta.get('num_inducing', 256))
            input_dim_test = int(ckpt_meta.get('input_dim', input_dim))
            latent_dim_test = int(ckpt_meta.get('latent_dim', latent_dim))

            fe_test = AutoencoderFeatureExtractor(input_dim=input_dim_test, latent_dim=latent_dim_test)
            inducing_pts_input_test = torch.randn(num_inducing_test, latent_dim_test)
            dkl_model_test = DKLAutoencoderSVGP(
                feature_extractor=fe_test,
                inducing_points=inducing_pts_input_test,
                latent_dim=latent_dim_test,
            )
            dkl_model_test.load_state_dict(ckpt['model_state_dict'], strict=True)
            dkl_likelihood_test.load_state_dict(ckpt['likelihood_state_dict'], strict=True)
            print(f"Loaded trained model + likelihood from {ckpt_path}")

        if dkl_model_test is None:
            raise RuntimeError(f"Unable to initialize evaluation model for {fd_subset}")

        eval_results = simulate_online_stream_and_update(
            model=dkl_model_test,
            likelihood=dkl_likelihood_test,
            stream_loader=test_loader,
            update_every_x_cycles=10,
            fine_tune_epochs=0,
            lr=0.001,
            lambda_recon=0.0,
            collect_metrics=True,
        )

        eval_dir = project_root / 'artifacts' / 'evaluation'
        eval_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = eval_dir / f'dkl_{fd_subset.lower()}_test_metrics.json'
        preds_path = eval_dir / f'dkl_{fd_subset.lower()}_test_predictions.json'

        if isinstance(eval_results, dict) and 'metrics' in eval_results:
            eval_results['metrics']['subset'] = fd_subset
            eval_results['metrics']['seed'] = int(seed)
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
            print(f"No metrics were returned from evaluation for {fd_subset}.")

        return {
            'subset': fd_subset,
            'num_sensors': len(features),
            'train_metrics_path': str(train_metrics_path),
            'test_metrics_path': str(metrics_path),
            'test_predictions_path': str(preds_path),
        }

    run_summaries = []
    for i, fd_subset in enumerate(FD_SUBSETS):
        try:
            summary = run_fd_subset_pipeline(fd_subset=fd_subset, seed=SEED + i)
            run_summaries.append(summary)
        except Exception as e:
            print(f"[{fd_subset}] Pipeline failed: {e}")

    print("\n===================== ALL SUBSETS SUMMARY =====================")
    for summary in run_summaries:
        print(
            f"{summary['subset']}: sensors={summary['num_sensors']} | "
            f"test_metrics={summary['test_metrics_path']} | "
            f"test_predictions={summary['test_predictions_path']}"
        )

    persist_aggregate_summary(run_summaries)