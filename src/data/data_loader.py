import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional


BASE_CMAPSS_COLUMNS = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']


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

def load_cmapss_data(file_path: str, rul_file_path: Optional[str] = None, max_rul: Optional[int] = 125) -> pd.DataFrame:
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
        
    # Limita la RUL massima a max_rul (es. 125) per modellare il degrado piecewise-linear
    if max_rul is not None:
        df['RUL'] = df['RUL'].clip(upper=max_rul)

    return df

class StreamingCMAPSSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features: List[str], scaler: Optional[StandardScaler] = None, fit_scaler: bool = False, target_scale: float = 1.0):
        X_raw = df[features].values  
        self.unit_nrs = torch.tensor(df['unit_nr'].values, dtype=torch.int32) # Aggiungi questa riga
    
        
        # Apply scaling if provided (fit only on train set to avoid leakage)
        if fit_scaler and scaler is not None:
            X_scaled = scaler.fit_transform(X_raw)
        elif scaler is not None:
            X_scaled = scaler.transform(X_raw)
        else:
            X_scaled = X_raw

            
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        # Targets: optionally scale RUL by `target_scale` so training can use 0-1 targets.
        # If target_scale == 1.0 this preserves raw RUL values (0..MAX_RUL).
        y_vals = df['RUL'].values.astype(float)
        if target_scale and float(target_scale) != 1.0:
            y_vals = y_vals / float(target_scale)
        # Target must be shape (N, 1) for GPyTorch/PyTorch loss functions
        self.y = torch.tensor(y_vals, dtype=torch.float32).unsqueeze(1)
        self.scaler = scaler
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            return self.X[idx], self.y[idx], self.unit_nrs[idx]