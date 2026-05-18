import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.data_loader import load_cmapss_data

def evaluate_baseline():
    features = [f's_{i}' for i in range(1, 22)]
    
    train_path = str(project_root / 'data' / 'raw' / 'train_FD001.txt')
    test_path = str(project_root / 'data' / 'raw' / 'test_FD001.txt')
    rul_path = str(project_root / 'data' / 'raw' / 'RUL_FD001.txt')
    
    df_train = load_cmapss_data(train_path)
    df_test = load_cmapss_data(test_path, rul_path)
    
    X_train_raw = df_train[features].values
    y_train = df_train['RUL'].values
    
    X_test_raw = df_test[features].values
    y_test = df_test['RUL'].values
    
    # Scale features (fit only on train set to prevent data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    # Train Random Forest baseline model
    rf_baseline = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_baseline.fit(X_train_scaled, y_train)
    
    # Evaluate on unseen test set
    test_predictions = rf_baseline.predict(X_test_scaled)
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
    mae_test = mean_absolute_error(y_test, test_predictions)
    
    print("\n--- BASELINE BENCHMARK RESULTS (TEST SET) ---")
    print(f"RMSE: {rmse_test:.2f} cycles")
    print(f"MAE:  {mae_test:.2f} cycles\n")
    
    return rf_baseline, scaler

if __name__ == "__main__":
    evaluate_baseline()