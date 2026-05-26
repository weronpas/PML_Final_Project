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

    was_training = model.training
    model.eval()
    if likelihood is not None:
        likelihood.eval()

    estimated_rul = []
    true_rul = []

    with torch.no_grad():
        for batch in data_loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch

            x = x.to(device)
            y = y.to(device)

            prediction = model(x)
            if isinstance(prediction, Distribution):
                predictive_distribution = likelihood(prediction) if likelihood is not None else prediction
                batch_estimated_rul = predictive_distribution.mean
            else:
                batch_estimated_rul = prediction

            estimated_rul.extend(batch_estimated_rul.detach().cpu().view(-1).tolist())
            true_rul.extend(y.detach().cpu().view(-1).tolist())

    if was_training:
        model.train()
        if likelihood is not None:
            likelihood.train()

    return evaluate_prognostics_model(estimated_rul, true_rul)

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
    
    # Initialize the scores array
    scores = np.zeros_like(d, dtype=float)
    
    # Case d < 0 (Early prediction)
    mask_early = d < 0
    scores[mask_early] = np.exp(-d[mask_early] / 10.0) - 1.0
    
    # Case d >= 0 (Late prediction)
    mask_late = d >= 0
    scores[mask_late] = np.exp(d[mask_late] / 13.0) - 1.0
    
    # The total score is the sum of the scores of all UUTs (Units Under Test)
    total_score = np.sum(scores)
    return total_score

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

def evaluate_prognostics_model(estimated_rul, true_rul):
    """
    Generates a comprehensive report combining mean errors, the asymmetric PHM score,
    and fleet consistency via the correlation index.
    """
    errors = calculate_prediction_error(estimated_rul, true_rul)
    
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    phm_score = float(phm_scoring_function(estimated_rul, true_rul))
    fleet_corr = float(fleet_correlation_metric(estimated_rul, true_rul))
    
    # Count the distribution of predictions
    late_predictions = int(np.sum(errors > 0))
    early_predictions = int(np.sum(errors < 0))
    perfect_predictions = int(np.sum(errors == 0))
    
    report = {
        "PHM08 Asymmetric Score (Lower is better)": phm_score,
        "Fleet Correlation Metric (Higher is better)": fleet_corr,
        "MAE (Mean Absolute Error)": mae,
        "RMSE (Root Mean Squared Error)": rmse,
        "Perfect Predictions (d = 0)": perfect_predictions,
        "Early Predictions (d < 0, Safer)": early_predictions,
        "Late Predictions (d > 0, Dangerous)": late_predictions
    }
    
    return report
