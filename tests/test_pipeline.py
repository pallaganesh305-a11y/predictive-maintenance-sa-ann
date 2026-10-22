"""
Unit and Integration Test Suite
Verifies data preprocessing, leakage prevention, simulated annealing mechanics,
model training, threshold tuning, and evaluation metrics.
"""

import os
import unittest
import numpy as np
import pandas as pd

from src.data_preprocessing import (
    load_data,
    engineer_features,
    prepare_features_and_target,
    get_train_val_test_split,
    FAILURE_MODES,
    IDENTIFIER_COLUMNS,
    TARGET_COLUMN
)
from src.simulated_annealing import SimulatedAnnealingFeatureSelector
from src.model_trainer import PredictiveMaintenanceANN
from src.evaluate import evaluate_model, compare_models

DATASET_PATH = 'data/ai4i2020.csv'

class TestDataPreprocessing(unittest.TestCase):
    def test_load_data(self):
        """Test dataset loading from disk."""
        self.assertTrue(os.path.exists(DATASET_PATH), f"Dataset file {DATASET_PATH} not found.")
        df = load_data(DATASET_PATH)
        self.assertEqual(len(df), 10000, "Dataset should have 10,000 rows.")
        self.assertIn(TARGET_COLUMN, df.columns, "Target column 'Machine failure' must be present.")

    def test_missing_data_error(self):
        """Test error raised when data file does not exist."""
        with self.assertRaises(FileNotFoundError):
            load_data('data/non_existent_dataset.csv')

    def test_feature_engineering_formulas(self):
        """Test thermodynamics and mechanics formulas."""
        df_sample = pd.DataFrame({
            'Air temperature [K]': [300.0],
            'Process temperature [K]': [310.0],
            'Rotational speed [rpm]': [1500],
            'Torque [Nm]': [40.0],
            'Tool wear [min]': [100]
        })
        eng = engineer_features(df_sample)

        # Temp diff = 310 - 300 = 10 K
        self.assertAlmostEqual(eng['Temp_Diff [K]'].iloc[0], 10.0)

        # Angular velocity = 1500 * (2 * pi / 60) = 50 * pi
        # Power = 40 * 50 * pi = 2000 * pi ~ 6283.185 W
        expected_power = 40.0 * 1500 * (2 * np.pi / 60.0)
        self.assertAlmostEqual(eng['Power [W]'].iloc[0], expected_power, places=3)

        # Overstrain = 100 * 40 = 4000
        self.assertEqual(eng['Overstrain'].iloc[0], 4000.0)

    def test_target_leakage_prevention(self):
        """Strict verification that failure modes and identifiers never enter feature matrix X."""
        df = load_data(DATASET_PATH)
        X_df, y, feature_names, failure_modes = prepare_features_and_target(df)

        for col in FAILURE_MODES:
            self.assertNotIn(col, feature_names, f"Target leakage detected! Column {col} is inside feature names.")
            self.assertNotIn(col, X_df.columns, f"Target leakage detected! Column {col} is inside X_df.")

        for col in IDENTIFIER_COLUMNS:
            self.assertNotIn(col, feature_names, f"Identifier {col} must not be in feature set.")

        self.assertNotIn(TARGET_COLUMN, feature_names, "Target column must not be in feature set.")

        # Ensure Type is one-hot encoded
        self.assertIn('Type_L', feature_names)
        self.assertIn('Type_M', feature_names)
        self.assertIn('Type_H', feature_names)

    def test_data_splits_and_scaling(self):
        """Verify train/val/test splits, stratification, and scaling."""
        df = load_data(DATASET_PATH)
        split = get_train_val_test_split(df, test_size=0.20, val_size=0.15, random_state=42)

        X_train, y_train = split['X_train'], split['y_train']
        X_val, y_val = split['X_val'], split['y_val']
        X_test, y_test = split['X_test'], split['y_test']

        # Total counts
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), 10000)
        self.assertEqual(len(X_test), 2000)
        self.assertEqual(len(X_val), 1500)
        self.assertEqual(len(X_train), 6500)

        # Stratified balance check: failure rate should be ~3.39% across all splits
        for split_y, name in [(y_train, 'train'), (y_val, 'val'), (y_test, 'test')]:
            fail_rate = np.mean(split_y)
            self.assertGreater(fail_rate, 0.02, f"{name} failure rate is too low.")
            self.assertLess(fail_rate, 0.05, f"{name} failure rate is too high.")

        # Scaler check: train mean should be close to 0
        means = np.mean(X_train, axis=0)
        for m in means:
            self.assertAlmostEqual(m, 0.0, places=1)


class TestSimulatedAnnealing(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.D = 8
        self.sa = SimulatedAnnealingFeatureSelector(
            n_iterations=10,
            initial_temp=1.0,
            alpha=0.9,
            penalty_lambda=0.02,
            eval_max_iter=30,
            random_state=42
        )

    def test_neighbor_generation(self):
        """Test neighbor bit-flip mutation logic."""
        state = np.ones(self.D, dtype=int)
        for _ in range(20):
            neighbor = self.sa._generate_neighbor(state)
            self.assertEqual(len(neighbor), self.D)
            # Must differ by at least 1 bit
            self.assertNotEqual(np.sum(np.abs(neighbor - state)), 0)
            # Must not be all zeros
            self.assertGreaterEqual(np.sum(neighbor), 1)

    def test_zero_state_guard(self):
        """Ensure neighbor generator recovers from all-zero state."""
        all_zeros = np.zeros(self.D, dtype=int)
        neighbor = self.sa._generate_neighbor(all_zeros)
        self.assertGreaterEqual(np.sum(neighbor), 1)

    def test_sa_execution_on_synthetic_data(self):
        """Test end-to-end SA fitting and history logging on small dataset."""
        X_tr = np.random.randn(200, 5)
        y_tr = np.random.choice([0, 1], size=200, p=[0.9, 0.1])
        X_va = np.random.randn(100, 5)
        y_va = np.random.choice([0, 1], size=100, p=[0.9, 0.1])

        sa = SimulatedAnnealingFeatureSelector(
            n_iterations=5,
            initial_temp=1.0,
            alpha=0.8,
            eval_max_iter=20,
            random_state=42
        )
        sa.fit(X_tr, y_tr, X_va, y_va, verbose=False)

        self.assertIsNotNone(sa.best_state_)
        self.assertGreaterEqual(len(sa.selected_features_), 1)
        self.assertEqual(len(sa.history_), 5)
        self.assertIn('temperature', sa.history_[0])
        self.assertIn('best_energy', sa.history_[0])

        X_transformed = sa.transform(X_va)
        self.assertEqual(X_transformed.shape[1], len(sa.selected_indices_))


class TestModelTrainerAndEvaluation(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X_train = np.random.randn(200, 6)
        self.y_train = np.random.choice([0, 1], size=200, p=[0.9, 0.1])
        self.X_val = np.random.randn(100, 6)
        self.y_val = np.random.choice([0, 1], size=100, p=[0.9, 0.1])
        self.X_test = np.random.randn(100, 6)
        self.y_test = np.random.choice([0, 1], size=100, p=[0.9, 0.1])

    def test_ann_model_training_and_threshold(self):
        """Test MLP model fit, threshold tuning, and predictions."""
        ann = PredictiveMaintenanceANN(
            hidden_layer_sizes=(16, 8),
            max_iter=40,
            random_state=42
        )
        ann.fit(self.X_train, self.y_train, self.X_val, self.y_val)

        # Threshold should be tuned between 0.1 and 0.9
        self.assertGreaterEqual(ann.best_threshold_, 0.1)
        self.assertLessEqual(ann.best_threshold_, 0.9)

        probs = ann.predict_proba(self.X_test)
        self.assertEqual(probs.shape, (100, 2))
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

        preds = ann.predict(self.X_test)
        self.assertEqual(len(preds), 100)
        self.assertTrue(set(np.unique(preds)).issubset({0, 1}))

    def test_evaluation_metrics_computation(self):
        """Test calculation of comprehensive evaluation metrics."""
        ann = PredictiveMaintenanceANN(
            hidden_layer_sizes=(16, 8),
            max_iter=40,
            random_state=42
        )
        ann.fit(self.X_train, self.y_train, self.X_val, self.y_val)
        metrics = evaluate_model(ann, self.X_test, self.y_test)

        required_keys = [
            'Model', 'Features Used', 'Threshold', 'Accuracy',
            'F1-Score', 'Recall (Sensitivity)', 'Precision',
            'Specificity', 'ROC-AUC', 'PR-AUC (Avg Precision)',
            'Confusion Matrix'
        ]
        for key in required_keys:
            self.assertIn(key, metrics, f"Metric key '{key}' is missing from evaluation.")

        comparison = compare_models([metrics])
        self.assertEqual(len(comparison), 1)
        self.assertIn('F1-Score', comparison.columns)


if __name__ == '__main__':
    unittest.main()
