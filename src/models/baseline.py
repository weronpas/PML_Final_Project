import sys
from pathlib import Path
import numpy as np
import json
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.data.data_loader import load_cmapss_data, get_sensor_feature_columns


def persist_baseline_summary(results: dict) -> None:
    eval_dir = project_root / 'artifacts' / 'evaluation'
    eval_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = eval_dir / 'baseline_all_subsets_summary.json'
    summary_csv_path = eval_dir / 'baseline_all_subsets_summary.csv'

    rows = []
    for subset in sorted(results.keys()):
        item = results[subset]
        rows.append(
            {
                'subset': item.get('subset'),
                'num_sensors': item.get('num_sensors'),
                'rmse': item.get('rmse'),
                'mae': item.get('mae'),
            }
        )

    with open(summary_json_path, 'w') as fh:
        json.dump(rows, fh, indent=2)

    with open(summary_csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['subset', 'num_sensors', 'rmse', 'mae'])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Saved baseline aggregate summary JSON to: {summary_json_path}")
    print(f"Saved baseline aggregate summary CSV to: {summary_csv_path}")

def evaluate_baseline_for_subset(fd_subset: str):
    train_path = str(project_root / 'data' / 'raw' / f'train_{fd_subset}.txt')
    test_path = str(project_root / 'data' / 'raw' / f'test_{fd_subset}.txt')
    rul_path = str(project_root / 'data' / 'raw' / f'RUL_{fd_subset}.txt')

    df_train = load_cmapss_data(train_path)
    df_test = load_cmapss_data(test_path, rul_path)

    features = get_sensor_feature_columns(df_train)
    if not features:
        raise ValueError(f"No sensor features detected for {fd_subset}")
    
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

    print(f"\n--- BASELINE BENCHMARK RESULTS ({fd_subset} TEST SET) ---")
    print(f"Sensors used: {len(features)}")
    print(f"RMSE: {rmse_test:.2f} cycles")
    print(f"MAE:  {mae_test:.2f} cycles\n")
    
    return {
        'subset': fd_subset,
        'num_sensors': len(features),
        'rmse': float(rmse_test),
        'mae': float(mae_test),
        'model': rf_baseline,
        'scaler': scaler,
    }


def evaluate_baseline(fd_subsets=None):
    if fd_subsets is None:
        fd_subsets = ['FD001', 'FD002', 'FD003', 'FD004']

    results = {}
    for fd_subset in fd_subsets:
        subset_result = evaluate_baseline_for_subset(fd_subset)
        results[fd_subset] = subset_result

    persist_baseline_summary(results)

    return results


if __name__ == "__main__":
    evaluate_baseline()