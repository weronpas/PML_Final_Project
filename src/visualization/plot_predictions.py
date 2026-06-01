#!/usr/bin/env python3
"""
Plot predicted vs true RUL over cycles with 95% CI shaded.

Example:
  python src/visualization/plot_predictions.py --subset FD003 --unit 1

Saves PNG to `artifacts/evaluation/plots/` by default.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_predictions(subset: str, base_dir: Path) -> pd.DataFrame:
    csv_path = base_dir / f'dkl_{subset.lower()}_test_predictions_per_cycle.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def plot_unit(df: pd.DataFrame, unit_nr: int, out_path: Path, show: bool = False):
    df_unit = df[df['unit_nr'] == unit_nr].sort_values('time_cycles')
    if df_unit.empty:
        raise ValueError(f"No data for unit_nr={unit_nr}")

    x = df_unit['time_cycles'].values
    y_true = df_unit['RUL'].values
    y_pred = df_unit['y_pred'].values

    # Try to get uncertainty bounds
    lower = None
    upper = None
    if 'lower_95' in df_unit.columns and 'upper_95' in df_unit.columns and not df_unit['lower_95'].isnull().all():
        try:
            lower = df_unit['lower_95'].astype(float).values
            upper = df_unit['upper_95'].astype(float).values
        except Exception:
            lower = None
            upper = None
    elif 'y_var' in df_unit.columns and not df_unit['y_var'].isnull().all():
        try:
            std = np.sqrt(np.maximum(df_unit['y_var'].astype(float).values, 0.0))
            lower = y_pred - 1.96 * std
            upper = y_pred + 1.96 * std
        except Exception:
            lower = None
            upper = None

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_true, label='True RUL', color='black', linewidth=2)
    plt.plot(x, y_pred, label='Predicted RUL', color='C1', linewidth=2)
    if lower is not None and upper is not None:
        plt.fill_between(x, lower, upper, color='C1', alpha=0.2, label='95% CI')

    plt.xlabel('Time Cycles')
    plt.ylabel('Remaining Useful Life (cycles)')
    plt.title(f'Unit {unit_nr} — True vs Predicted RUL')
    plt.legend()
    plt.grid(alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot per-cycle RUL predictions for CMAPSS subsets')
    parser.add_argument('--subset', type=str, default='FD003', help='FD subset (FD001..FD004)')
    parser.add_argument('--unit', type=int, required=True, help='unit_nr to plot')
    parser.add_argument('--predictions-dir', type=str, default='artifacts/evaluation', help='Directory with per-cycle CSVs')
    parser.add_argument('--out', type=str, default=None, help='Output PNG path (defaults to artifacts/evaluation/plots/plot_{subset}_unit{unit}.png)')
    parser.add_argument('--show', action='store_true', help='Show plot interactively')

    args = parser.parse_args()
    base_dir = Path(args.predictions_dir)
    subset = args.subset

    df = load_predictions(subset, base_dir)

    out_path = Path(args.out) if args.out else base_dir / 'plots' / f'plot_{subset.lower()}_unit{args.unit}.png'

    plot_unit(df, args.unit, out_path, show=args.show)
    print(f"Saved plot to: {out_path}")


if __name__ == '__main__':
    main()
