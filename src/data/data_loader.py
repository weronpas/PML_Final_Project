import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, List


CMAPSS_COLUMNS = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + \
                 [f's_{i}' for i in range(1, 22)]

def load_cmapss_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep='\s+', header=None, names=CMAPSS_COLUMNS)
    
    # Counting RUL (Remaining Useful Life)
    max_cycles = df.groupby('unit_nr')['time_cycles'].transform('max')
    df['RUL'] = max_cycles - df['time_cycles']
    return df

class StreamingCMAPSSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, features: List[str]):
        self.X = torch.tensor(df[features].values, dtype=torch.float32)
        self.y = torch.tensor(df['RUL'].values, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]