"""
Main Training, Simulated Annealing Feature Selection, and Benchmarking Pipeline.
College ANN Assignment: Predictive-Maintenance Neural Model with Simulated Annealing (SA)
"""

import os
import json
import time
import argparse
import joblib
import numpy as np
import pandas as pd

from src.data_preprocessing import load_data, get_train_val_test_split
from src.simulated_annealing import SimulatedAnnealingFeatureSelector
from src.model_trainer import PredictiveMaintenanceANN
from src.evaluate import evaluate_model, compare_models

def run_pipeline(
    data_path='data/ai4i2020.csv',
    n_iterations=80,
    initial_temp=1.0,
    alpha=0.92,
    penalty_lambda=0.02,
    output_dir='models',
    random_state=42
):
    print("=" * 75)
    print("  PREDICTIVE MAINTENANCE ANN WITH SIMULATED ANNEALING FEATURE SELECTION  ")
    print("=" * 75)

    start_total_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    print(f"\n[1/5] Loading real AI4I 2020 dataset from '{data_path}'...")
    df = load_data(data_path)
    print(f"      Loaded {df.shape[0]} records and {df.shape[1]} columns successfully.")
    print(f"      Failure distribution: {df['Machine failure'].value_counts().to_dict()}")

    # 2. Preprocess, Engineer Features, and Split (Train/Val/Test)
    print("\n[2/5] Preprocessing and feature engineering...")
    split_data = get_train_val_test_split(df, test_size=0.20, val_size=0.15, random_state=random_state)
    X_train = split_data['X_train']
    y_train = split_data['y_train']
    X_val = split_data['X_val']
    y_val = split_data['y_val']
    X_test = split_data['X_test']
    y_test = split_data['y_test']
    feature_names = split_data['feature_names']
    scaler = split_data['scaler']

    print(f"      Feature set ({len(feature_names)} features): {feature_names}")
    print(f"      Train set: {X_train.shape[0]} samples (Failures: {int(np.sum(y_train))})")
    print(f"      Val set:   {X_val.shape[0]} samples (Failures: {int(np.sum(y_val))})")
    print(f"      Test set:  {X_test.shape[0]} samples (Failures: {int(np.sum(y_test))})")

    # 3. Train Baseline ANN Model (All Features)
    print("\n[3/5] Training Benchmark 1: Baseline ANN (All 11 Features)...")
    baseline_start = time.time()
    baseline_ann = PredictiveMaintenanceANN(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        random_state=random_state,
        name="Baseline ANN (All Features)"
    )
    baseline_ann.fit(X_train, y_train, X_val=X_val, y_val=y_val, feature_names=feature_names)
    baseline_time = time.time() - baseline_start
    print(f"      Baseline ANN trained in {baseline_time:.2f}s | Optimal Val Threshold: {baseline_ann.best_threshold_:.3f}")

    # 4. Run Simulated Annealing Feature Selection
    print(f"\n[4/5] Running Simulated Annealing Feature Selection ({n_iterations} iterations)...")
    sa_start = time.time()
    sa_selector = SimulatedAnnealingFeatureSelector(
        n_iterations=n_iterations,
        initial_temp=initial_temp,
        alpha=alpha,
        penalty_lambda=penalty_lambda,
        eval_hidden_layers=(32, 16),
        eval_max_iter=120,
        random_state=random_state
    )
    sa_selector.fit(X_train, y_train, X_val, y_val, feature_names=feature_names, verbose=True)
    sa_time = time.time() - sa_start

    selected_features = sa_selector.selected_features_
    selected_indices = sa_selector.selected_indices_
    print(f"\n      Simulated Annealing completed in {sa_time:.2f}s.")
    print(f"      Reduced features: {len(feature_names)} -> {len(selected_features)} features.")
    print(f"      Selected Subset: {selected_features}")

    # Train Benchmark 2: SA-Optimized ANN
    print("\n[5/5] Training Benchmark 2: SA-Optimized ANN (Selected Features Only)...")
    sa_model_start = time.time()
    sa_ann = PredictiveMaintenanceANN(
        hidden_layer_sizes=(64, 32),
        max_iter=300,
        random_state=random_state,
        name=f"SA-Optimized ANN ({len(selected_features)} Features)"
    )
    sa_ann.fit(
        X_train, y_train, X_val=X_val, y_val=y_val,
        feature_names=selected_features,
        selected_indices=selected_indices
    )
    sa_model_time = time.time() - sa_model_start
    print(f"      SA-Optimized ANN trained in {sa_model_time:.2f}s | Optimal Val Threshold: {sa_ann.best_threshold_:.3f}")

    # 5. Evaluate both models on the Unbiased Test Set
    print("\n" + "=" * 75)
    print("                     UNBIASED TEST SET EVALUATION                        ")
    print("=" * 75)
    baseline_metrics = evaluate_model(baseline_ann, X_test, y_test)
    baseline_metrics['Train Time (s)'] = round(baseline_time, 2)

    sa_metrics = evaluate_model(sa_ann, X_test, y_test)
    sa_metrics['Train Time (s)'] = round(sa_model_time, 2)

    comparison_df = compare_models([baseline_metrics, sa_metrics])
    print(comparison_df.to_string(index=False))

    # Feature Reduction percentage
    reduction_pct = ((len(feature_names) - len(selected_features)) / len(feature_names)) * 100
    print(f"\nFeature count reduced by: {reduction_pct:.1f}%")

    # Save artifacts for Streamlit app and persistence
    print(f"\nSaving artifacts to '{output_dir}/'...")
    baseline_ann.save(os.path.join(output_dir, 'baseline_ann.joblib'))
    sa_ann.save(os.path.join(output_dir, 'sa_optimized_ann.joblib'))
    joblib.dump(scaler, os.path.join(output_dir, 'scaler.joblib'))

    # Save test dataset and splits
    test_bundle = {
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'selected_features': selected_features,
        'selected_indices': selected_indices.tolist()
    }
    joblib.dump(test_bundle, os.path.join(output_dir, 'test_bundle.joblib'))

    # Save SA convergence history
    with open(os.path.join(output_dir, 'sa_history.json'), 'w') as f:
        json.dump(sa_selector.history_, f, indent=2)

    # Save final metrics
    metrics_summary = {
        'baseline': baseline_metrics,
        'sa_optimized': sa_metrics,
        'selected_features': selected_features,
        'all_features': feature_names,
        'feature_reduction_percent': round(reduction_pct, 2),
        'sa_runtime_seconds': round(sa_time, 2),
        'total_pipeline_time_seconds': round(time.time() - start_total_time, 2)
    }
    with open(os.path.join(output_dir, 'metrics_summary.json'), 'w') as f:
        json.dump(metrics_summary, f, indent=2)

    print(f"[Done] All models and metadata successfully stored in '{output_dir}'.")
    return metrics_summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predictive Maintenance ANN with Simulated Annealing")
    parser.add_argument('--data-path', type=str, default='data/ai4i2020.csv', help='Path to AI4I 2020 CSV')
    parser.add_argument('--n-iterations', type=int, default=80, help='SA iterations (manageable on laptop)')
    parser.add_argument('--t0', type=float, default=1.0, help='Initial temperature')
    parser.add_argument('--alpha', type=float, default=0.92, help='Cooling rate')
    parser.add_argument('--penalty', type=float, default=0.02, help='Feature penalty lambda')
    parser.add_argument('--output-dir', type=str, default='models', help='Directory to save models')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        n_iterations=args.n_iterations,
        initial_temp=args.t0,
        alpha=args.alpha,
        penalty_lambda=args.penalty,
        output_dir=args.output_dir,
        random_state=args.random_state
    )
