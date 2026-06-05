import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler


BASE_CMAPSS_COLUMNS = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']


def smooth_clip_rul(values: Union[np.ndarray, torch.Tensor], max_rul: float) -> Union[np.ndarray, torch.Tensor]:
    """Apply a smooth approximation of min(values, max_rul)."""
    if torch.is_tensor(values):
        max_rul_tensor = torch.as_tensor(max_rul, dtype=values.dtype, device=values.device)
        return max_rul_tensor - F.softplus(max_rul_tensor - values)

    values_array = np.asarray(values, dtype=np.float64)
    return max_rul - np.logaddexp(0.0, max_rul - values_array)


def inverse_smooth_clip_rul(values, max_rul):
    eps = 1e-8
    values_array = np.asarray(values, dtype=np.float64)
    clipped = np.minimum(values_array, max_rul - eps)
    arg = max_rul - clipped                        # almost 0 for values near max_rul
    safe_arg = np.maximum(arg, 1e-7)               # clamp before expm1
    return max_rul - np.log(np.expm1(safe_arg))


@dataclass
class SmoothRULTargetTransform:
    """Standardize smooth-clipped RUL targets and invert them back to cycles."""

    max_rul: float = 125.0
    scaler: StandardScaler = field(default_factory=StandardScaler)
    fitted: bool = False

    # Margin below max_rul where the inverse becomes numerically unstable.
    # Values within 2 cycles of max_rul are physically indistinguishable ("saturated").
    CLAMP_MARGIN: float = field(default=2.0, init=False, repr=False)
    MAX_DERIVATIVE: float = field(default=50.0, init=False, repr=False)

    def fit(self, targets: Union[np.ndarray, torch.Tensor]) -> "SmoothRULTargetTransform":
        targets_array = np.asarray(targets, dtype=np.float64).reshape(-1, 1)
        self.scaler.fit(targets_array)
        self.fitted = True
        return self

    def transform(self, targets: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if not self.fitted:
            raise RuntimeError("SmoothRULTargetTransform must be fit before calling transform().")

        if torch.is_tensor(targets):
            transformed = (targets - float(self.scaler.mean_[0])) / float(self.scaler.scale_[0])
            return transformed

        targets_array = np.asarray(targets, dtype=np.float64).reshape(-1, 1)
        transformed = self.scaler.transform(targets_array).reshape(-1)
        return transformed

    def inverse_transform(
        self,
        values: Union[np.ndarray, torch.Tensor],
        variance: Optional[Union[np.ndarray, torch.Tensor]] = None,
        is_upper_bound: bool = False,
    ):
        if not self.fitted:
            raise RuntimeError("SmoothRULTargetTransform must be fit before calling inverse_transform().")

        # For upper confidence bounds we use a much smaller margin
        # (0.1 cycles) so the cap does not artificially truncate upper coverage.
        # For the mean and lower bound we use CLAMP_MARGIN=2.0
        # to maintain derivative stability.
        safe_max = self.max_rul - (0.1 if is_upper_bound else self.CLAMP_MARGIN)

        if torch.is_tensor(values):
            smooth_values = values * float(self.scaler.scale_[0]) + float(self.scaler.mean_[0])
            smooth_values = torch.clamp(smooth_values, max=safe_max)
            raw_values = inverse_smooth_clip_rul(smooth_values, self.max_rul)
            if variance is None:
                return raw_values
            variance_tensor = (
                variance if torch.is_tensor(variance)
                else torch.as_tensor(variance, dtype=values.dtype, device=values.device)
            )
            derivative = self._inverse_derivative_torch(smooth_values)
            raw_variance = variance_tensor * (derivative ** 2)
            return raw_values, raw_variance

        values_array = np.asarray(values, dtype=np.float64).reshape(-1)
        smooth_values = values_array * float(self.scaler.scale_[0]) + float(self.scaler.mean_[0])
        smooth_values = np.clip(smooth_values, -np.inf, safe_max)
        raw_values = inverse_smooth_clip_rul(smooth_values, self.max_rul)
        if variance is None:
            return raw_values

        variance_array = np.asarray(variance, dtype=np.float64).reshape(-1)
        # Consistent derivative: evaluated on the same clamped smooth_values
        derivative = self._inverse_derivative_numpy(smooth_values)
        raw_variance = variance_array * (derivative ** 2)
        return raw_values, raw_variance


    def inverse_interval(self, lower, upper):
        # Apply inverse_transform independently to lower and upper
        # WITHOUT clamping lower to the same value as upper
        lower_raw = self.inverse_transform(lower)
        upper_raw = self.inverse_transform(upper)
        # If upper is saturated at the max, lower can still be lower
        return lower_raw, upper_raw

    def _inverse_derivative_numpy(self, smooth_values: np.ndarray) -> np.ndarray:
        clipped = np.minimum(np.asarray(smooth_values, dtype=np.float64), self.max_rul - self.CLAMP_MARGIN)
        exp_arg = np.clip(self.max_rul - clipped, 0.0, 50.0)
        exp_term = np.exp(exp_arg)
        deriv_clip = exp_term / np.maximum(exp_term - 1.0, 1e-6)
        deriv_clip = np.minimum(deriv_clip, self.MAX_DERIVATIVE)
        
        # Chain rule: include the scaler factor
        full_deriv = deriv_clip * float(self.scaler.scale_[0])
        return full_deriv

    def _inverse_derivative_torch(self, smooth_values: torch.Tensor) -> torch.Tensor:
        max_rul_t = torch.as_tensor(self.max_rul, dtype=smooth_values.dtype, device=smooth_values.device)
        clipped = torch.clamp(smooth_values, max=max_rul_t - self.CLAMP_MARGIN)
        exp_arg = torch.clamp(max_rul_t - clipped, min=0.0, max=50.0)
        exp_term = torch.exp(exp_arg)
        deriv_clip = exp_term / torch.clamp(exp_term - 1.0, min=1e-6)
        deriv_clip = torch.clamp(deriv_clip, max=self.MAX_DERIVATIVE)

        # Chain rule: include the scaler factor
        scale = torch.as_tensor(float(self.scaler.scale_[0]), dtype=smooth_values.dtype, device=smooth_values.device)
        full_deriv = deriv_clip * scale
        return full_deriv
    
def infer_cmapss_columns(file_path: str) -> List[str]:
    """Infer CMAPSS column names from a raw text file with whitespace-separated values."""
    with open(file_path, 'r') as fh:
        first_line = fh.readline().strip()

    num_cols = len(first_line.split())
    num_sensor_cols = max(0, num_cols - len(BASE_CMAPSS_COLUMNS))
    return BASE_CMAPSS_COLUMNS + [f's_{i}' for i in range(1, num_sensor_cols + 1)]


def get_sensor_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return sensor feature columns present in the dataframe, ordered by sensor index."""
    sensor_cols = [c for c in df.columns if c.startswith('s_')]
    sensor_cols.sort(key=lambda c: int(c.split('_')[1]) if c.split('_')[1].isdigit() else c)
    return sensor_cols

def load_cmapss_data(file_path: str, rul_file_path: Optional[str] = None, max_rul: Optional[int] = 125, smooth_targets: bool = False) -> pd.DataFrame:
    # Infer columns so the same code works across FD001/FD002/FD003/FD004 variants.
    col_names = infer_cmapss_columns(file_path)
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=col_names)
    
    if rul_file_path is None:
        # Train set logic (run-to-failure): RUL = max_cycles - current_cycle
        max_cycles = df.groupby('unit_nr')['time_cycles'].transform('max')
        df['RUL'] = max_cycles - df['time_cycles']
    else:
        # Test set logic: RUL requires true remaining cycles from separate file
        true_rul = pd.read_csv(rul_file_path, sep=r'\s+', header=None, names=['RUL_end'])
        true_rul['unit_nr'] = true_rul.index + 1
        
        max_cycles = df.groupby('unit_nr')['time_cycles'].max().reset_index()
        max_cycles = max_cycles.rename(columns={'time_cycles': 'max_test_cycles'})
        
        df = df.merge(max_cycles, on='unit_nr')
        df = df.merge(true_rul, on='unit_nr')
        

        # RUL = final RUL + max cycle in test file - current cycle
        df['RUL'] = df['RUL_end'] + df['max_test_cycles'] - df['time_cycles']
        df = df.drop(columns=['max_test_cycles', 'RUL_end'])
        
    # Apply a smooth saturation at max_rul so the training target remains differentiable.
    if max_rul is not None:
        if smooth_targets:
            df['RUL'] = smooth_clip_rul(df['RUL'].values, float(max_rul))
        else:
            df['RUL'] = df['RUL'].clip(upper=max_rul)

    return df

class StreamingCMAPSSDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features: List[str],
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        time_scaler: Optional[MinMaxScaler] = None,
        fit_time_scaler: bool = False,
        target_transform: Optional[SmoothRULTargetTransform] = None,
        fit_target_transform: bool = False,
        max_rul: float = 125.0,
    ):
        X_raw = df[features].values  
        self.unit_nrs = torch.tensor(df['unit_nr'].values, dtype=torch.int32)
        
        # Extract and normalize time_cycles using the dedicated scaler
        time_raw = df['time_cycles'].values.astype(np.float64).reshape(-1, 1)
        
        if fit_time_scaler and time_scaler is not None:
            time_scaled = time_scaler.fit_transform(time_raw)
        elif time_scaler is not None:
            time_scaled = time_scaler.transform(time_raw)
        else:
            # Fallback to unscaled if no scaler is provided
            time_scaled = time_raw

        self.time_cycles = torch.tensor(time_scaled.reshape(-1), dtype=torch.float32)
        self.time_scaler = time_scaler

        # Apply scaling to features (fit only on train set to avoid leakage)
        if fit_scaler and scaler is not None:
            X_scaled = scaler.fit_transform(X_raw)
        elif scaler is not None:
            X_scaled = scaler.transform(X_raw)
        else:
            X_scaled = X_raw

        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        
        # Target (RUL) processing
        y_vals = np.asarray(df['RUL'].values, dtype=np.float64).reshape(-1)
        if target_transform is None:
            target_transform = SmoothRULTargetTransform(max_rul=max_rul)

        if fit_target_transform:
            target_transform.fit(y_vals)
        elif not target_transform.fitted:
            raise ValueError("target_transform must be fitted or fit_target_transform=True for StreamingCMAPSSDataset.")

        self.target_transform = target_transform
        y_transformed = self.target_transform.transform(y_vals)
        y_transformed = np.asarray(y_transformed, dtype=np.float32).reshape(-1, 1)
        
        self.y = torch.tensor(y_transformed, dtype=torch.float32)
        self.y_raw = torch.tensor(y_vals, dtype=torch.float32).unsqueeze(1)
        self.scaler = scaler
        self.max_rul = float(max_rul)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Yield a 4-tuple: (features, target, unit, time)
        return self.X[idx], self.y[idx], self.unit_nrs[idx], self.time_cycles[idx]

    def inverse_transform(self, values, variance=None, is_upper_bound=False):
        return self.target_transform.inverse_transform(values, variance=variance, is_upper_bound=is_upper_bound)
    
class RegimeConditionedScaler:
    """
    Specialized scaler for multi-regime datasets (FD002, FD004).
    Applies KMeans on the operational setting columns (setting_1, setting_2, setting_3)
    to identify the 6 regimes. It scales sensor features independently for each regime,
    removing the shift caused by altitude/mach changes.
    """
    def __init__(self, n_regimes: int = 6, setting_cols_indices: tuple = (0, 1, 2)):
        self.n_regimes = n_regimes
        self.setting_cols_indices = setting_cols_indices
        # n_init=10 ensures correct convergence to the 6 expected centroids
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        self.scalers = {i: StandardScaler() for i in range(n_regimes)}
        self.is_fitted = False

    def fit(self, X: np.ndarray):
        X_arr = np.asarray(X)
        # Extract only the 3 operational settings
        settings = X_arr[:, self.setting_cols_indices]
        
        # Identify which of the 6 regimes each cycle belongs to
        clusters = self.kmeans.fit_predict(settings)
        
        # Fit an independent StandardScaler for each regime
        for i in range(self.n_regimes):
            mask = (clusters == i)
            if np.sum(mask) > 0:
                self.scalers[i].fit(X_arr[mask])
        
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("RegimeConditionedScaler must be fit before calling transform().")
            
        X_arr = np.asarray(X)
        settings = X_arr[:, self.setting_cols_indices]
        # Assign test cycles to regimes learned during training
        clusters = self.kmeans.predict(settings)
        
        X_scaled = np.zeros_like(X_arr, dtype=np.float64)
        
        # Scale each cycle with the scaler for its specific regime
        for i in range(self.n_regimes):
            mask = (clusters == i)
            if np.sum(mask) > 0:
                X_scaled[mask] = self.scalers[i].transform(X_arr[mask])
                
        return X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)