import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Distribution


def negative_log_likelihood(prediction, true_rul, likelihood=None):
    """
    Computes a training loss for probabilistic models.

    If the model returns a probability distribution, this uses the negative
    log-likelihood of the target under that distribution. If a likelihood is
    provided, it is applied before scoring the target. For point estimates, it
    falls back to MSE.
    """
    target = true_rul.squeeze(-1) if true_rul.ndim > 1 else true_rul

    if isinstance(prediction, Distribution):
        predictive_distribution = likelihood(prediction) if likelihood is not None else prediction
        return -predictive_distribution.log_prob(target)

    return F.mse_loss(prediction.squeeze(-1), target)


def evaluate_model_on_loader(model, data_loader, likelihood=None, device=None):
    """
    Runs a trained model over a dataloader and evaluates the resulting point
    predictions with the full prognostics report.

    For probabilistic models, the posterior mean is used as the estimated RUL.
    If batch metadata is present, it is ignored here because these metrics are
    defined on predicted and true RUL values.
    """
    if device is None:
        device = next(model.parameters()).device

    target_inverse = None
    dataset = getattr(data_loader, 'dataset', None)
    if dataset is not None:
        target_inverse = getattr(dataset, 'inverse_transform', None)

    was_training = model.training
    model.eval()
    if likelihood is not None:
        likelihood.eval()

    estimated_rul = []
    true_rul = []
    predictive_vars = []
    lower_bounds = []
    upper_bounds = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 4:
                x, y, _, t = batch
            elif len(batch) == 3:
                x, y, _ = batch
                t = None
            else:
                x, y = batch
                t = None

            x = x.to(device)
            y = y.to(device)

            if hasattr(model, 'predict_with_uncertainty'):
                # FIXED: Pass the normalized time context here
                prediction = model.predict_with_uncertainty(x, normalized_cycle=t, likelihood=likelihood)
            else:
                # FIXED: Fallback branch also attempts to pass time if supported
                try:
                    prediction = model(x, normalized_cycle=t)
                except TypeError:
                    prediction = model(x)

            if isinstance(prediction, dict):
                batch_estimated_rul = prediction.get('mean')
                batch_pred_var = prediction.get('variance')
                batch_lower = prediction.get('lower')
                batch_upper = prediction.get('upper')
            elif isinstance(prediction, Distribution):
                predictive_distribution = likelihood(prediction) if likelihood is not None else prediction
                batch_estimated_rul = predictive_distribution.mean
                batch_pred_var = predictive_distribution.variance
                batch_std = torch.sqrt(torch.clamp(batch_pred_var, min=1e-12))
                batch_lower = batch_estimated_rul - (1.96 * batch_std)
                batch_upper = batch_estimated_rul + (1.96 * batch_std)
            else:
                batch_estimated_rul = prediction
                batch_pred_var = None
                batch_lower = None
                batch_upper = None

            batch_estimated_rul = batch_estimated_rul.detach().cpu().view(-1).numpy()
            batch_true = y.detach().cpu().view(-1).numpy()
            if batch_pred_var is not None:
                batch_pred_var = batch_pred_var.detach().cpu().view(-1).numpy()
            if batch_lower is not None:
                batch_lower = batch_lower.detach().cpu().view(-1).numpy()
            if batch_upper is not None:
                batch_upper = batch_upper.detach().cpu().view(-1).numpy()

            if callable(target_inverse):
                if batch_pred_var is not None:
                    batch_estimated_rul, batch_pred_var = target_inverse(batch_estimated_rul, variance=batch_pred_var)
                else:
                    batch_estimated_rul = target_inverse(batch_estimated_rul)
                batch_true = target_inverse(batch_true)
                if batch_lower is not None:
                    batch_lower = target_inverse(batch_lower)
                if batch_upper is not None:
                    batch_upper = target_inverse(batch_upper)

            estimated_rul.extend(np.asarray(batch_estimated_rul).reshape(-1).tolist())
            true_rul.extend(np.asarray(batch_true).reshape(-1).tolist())
            if batch_pred_var is not None:
                predictive_vars.extend(np.asarray(batch_pred_var).reshape(-1).tolist())
                lower_bounds.extend(np.asarray(batch_lower).reshape(-1).tolist())
                upper_bounds.extend(np.asarray(batch_upper).reshape(-1).tolist())

    if was_training:
        model.train()
        if likelihood is not None:
            likelihood.train()

    return evaluate_prognostics_model(
        estimated_rul,
        true_rul,
        predictive_vars=predictive_vars if predictive_vars else None,
        lower_bounds=lower_bounds if lower_bounds else None,
        upper_bounds=upper_bounds if upper_bounds else None,
    )

def calculate_prediction_error(estimated_rul, true_rul):
    """
    Calculates the prediction error 'd' as defined in the paper:
    d = Estimated RUL - True RUL
    
    A value of d < 0 indicates an early prediction.
    A value of d >= 0 indicates a late prediction.
    """
    return np.array(estimated_rul) - np.array(true_rul)

def phm_scoring_function(estimated_rul, true_rul):
    """
    Implements the official asymmetric scoring function from PHM'08 (Eq. 11 in the paper).
    It penalizes late predictions (d >= 0) more severely than early predictions (d < 0).
    
    Parameters:
    a1 = 10 (penalty factor for early predictions)
    a2 = 13 (penalty factor for late predictions)
    """
    d = calculate_prediction_error(estimated_rul, true_rul)
    scores = np.zeros_like(d, dtype=float)
    
    mask_early = d < 0
    mask_late  = d >= 0

    # Clamp prima di exp per evitare overflow
    scores[mask_early] = np.exp(np.clip(-d[mask_early] / 10.0, -500, 500)) - 1.0
    scores[mask_late]  = np.exp(np.clip( d[mask_late]  / 13.0, -500, 500)) - 1.0

    return float(np.sum(scores))

def fleet_correlation_metric(estimated_rul, true_rul):
    """
    Calculates the correlation metric (e.g., Pearson Correlation Coefficient)
    between the estimated RULs and true RULs across the different UUTs in the fleet.
    As suggested in the paper, this helps identify the consistency of the algorithm.
    """
    if len(estimated_rul) < 2:
        return 0.0
    correlation = np.corrcoef(estimated_rul, true_rul)[0, 1]
    return correlation

def evaluate_prognostics_model(estimated_rul, true_rul, predictive_vars=None, lower_bounds=None, upper_bounds=None):
    """
    Generates a comprehensive report combining mean errors, the asymmetric PHM score,
    and fleet consistency via the correlation index.
    """
    errors = calculate_prediction_error(estimated_rul, true_rul)
    
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    phm_score = float(phm_scoring_function(estimated_rul, true_rul))
    fleet_corr = float(fleet_correlation_metric(estimated_rul, true_rul))
    r2 = r2_score(estimated_rul, true_rul)

    if predictive_vars is not None and len(predictive_vars) > 0:
        avg_pred_var = float(np.mean(np.array(predictive_vars).flatten()))
    else:
        avg_pred_var = None

    if lower_bounds is not None and upper_bounds is not None and len(lower_bounds) > 0 and len(upper_bounds) > 0:
        picp_95 = prediction_interval_coverage(lower_bounds, upper_bounds, true_rul)
        avg_interval_width_95 = average_interval_width(lower_bounds, upper_bounds)
    else:
        picp_95 = None
        avg_interval_width_95 = None
    
    # Count the distribution of predictions
    late_predictions = int(np.sum(errors > 0))
    early_predictions = int(np.sum(errors < 0))
    perfect_predictions = int(np.sum(errors == 0))
    
    report = {
        "PHM08 Asymmetric Score (Lower is better)": phm_score,
        "Fleet Correlation Metric (Higher is better)": fleet_corr,
        "R2 (Higher is better)": r2,
        "MAE (Mean Absolute Error)": mae,
        "RMSE (Root Mean Squared Error)": rmse,
        "Perfect Predictions (d = 0)": perfect_predictions,
        "Early Predictions (d < 0, Safer)": early_predictions,
        "Late Predictions (d > 0, Dangerous)": late_predictions
    }

    if picp_95 is not None:
        report["picp_95"] = picp_95
    if avg_interval_width_95 is not None:
        report["avg_interval_width_95"] = avg_interval_width_95
    if avg_pred_var is not None:
        report["avg_pred_variance"] = avg_pred_var
    
    return report


def prediction_interval_coverage(lower_bounds, upper_bounds, true_values):
    lb = np.array(lower_bounds).flatten()
    ub = np.array(upper_bounds).flatten()
    y  = np.array(true_values).flatten()
    
    # Exclude samples with invalid intervals
    valid = np.isfinite(lb) & np.isfinite(ub)
    if not np.any(valid):
        return float('nan')
    covered = (y[valid] >= lb[valid]) & (y[valid] <= ub[valid])
    return float(np.mean(covered))


def average_interval_width(lower_bounds, upper_bounds):
    """Average width of prediction intervals (upper - lower)."""
    lb = np.array(lower_bounds).flatten()
    ub = np.array(upper_bounds).flatten()
    width = ub - lb
    # Ignore NaN/inf produced by saturation in inverse_transform
    valid = np.isfinite(width)
    if not np.any(valid):
        return float('nan')
    return float(np.mean(width[valid]))


def gaussian_nll_from_mean_var(mean_preds, var_preds, true_values, eps: float = 1e-6):
    """
    Compute Gaussian negative log-likelihood given predictive means and variances.
    Uses closed-form per-sample NLL and returns the mean NLL over samples.
    """
    mu = np.array(mean_preds).flatten()
    var = np.array(var_preds).flatten()
    y = np.array(true_values).flatten()
    var = np.maximum(var, eps)
    nll_terms = 0.5 * (np.log(2 * np.pi * var) + ((y - mu) ** 2) / var)
    return float(np.mean(nll_terms))


def crps_gaussian(mean_preds, var_preds, true_values):
    """
    Continuous Ranked Probability Score (CRPS) for Gaussian predictive distributions.
    Uses the closed-form expression for a Normal predictive distribution.
    CRPS = sigma * [ z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ]
    where z = (x - mu)/sigma, phi/pdf and Phi/cdf are standard normal.
    Returns the mean CRPS over the samples.
    """
    mu = np.array(mean_preds).flatten()
    var = np.array(var_preds).flatten()
    y = np.array(true_values).flatten()
    sigma = np.sqrt(np.maximum(var, 1e-12))
    z = (y - mu) / sigma

    # vectorized standard normal pdf and cdf using numpy and math.erf
    from math import erf, sqrt, pi
    # pdf: phi(z) = exp(-0.5 z^2) / sqrt(2*pi)
    phi = np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    # cdf: Phi(z) = 0.5 * (1 + erf(z / sqrt(2)))
    Phi = 0.5 * (1.0 + np.vectorize(erf)(z / np.sqrt(2.0)))

    crps = sigma * (z * (2 * Phi - 1) + 2 * phi - 1.0 / np.sqrt(pi))
    return float(np.mean(crps))


def r2_score(estimated_rul, true_rul):
    """Coefficient of determination (R^2) between estimates and targets."""
    y = np.array(true_rul).flatten()
    y_hat = np.array(estimated_rul).flatten()
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)
