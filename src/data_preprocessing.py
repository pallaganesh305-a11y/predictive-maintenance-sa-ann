"""
Data Preprocessing and Feature Engineering Module
Dataset: AI4I 2020 Predictive Maintenance Dataset

Prevents target leakage by completely separating failure mode indicators
('TWF', 'HDF', 'PWF', 'OSF', 'RNF') and machine identifiers ('UDI', 'Product ID')
from the predictive feature set.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

IDENTIFIER_COLUMNS = ['UDI', 'Product ID']
FAILURE_MODES = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
TARGET_COLUMN = 'Machine failure'
BASE_SENSOR_COLUMNS = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

def load_data(filepath='data/ai4i2020.csv'):
    """
    Loads the AI4I 2020 dataset from the given path.
    Raises FileNotFoundError if file is missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please place the genuine "
            f"AI4I 2020 predictive maintenance CSV at '{filepath}'."
        )
    df = pd.read_csv(filepath)
    return df

def engineer_features(df):
    """
    Applies domain-knowledge thermodynamic and electromechanical feature engineering:
    1. Temp_Diff [K]: Temperature differential between process and ambient air.
    2. Power [W]: Mechanical shaft power (Torque * angular velocity).
    3. Overstrain: Interaction of cumulative tool wear and cutting torque.
    """
    df_eng = df.copy()

    # Temperature differential
    df_eng['Temp_Diff [K]'] = df_eng['Process temperature [K]'] - df_eng['Air temperature [K]']

    # Mechanical rotational power (Watts): Power = Torque (N*m) * omega (rad/s)
    # omega = rpm * (2 * pi / 60)
    angular_velocity = df_eng['Rotational speed [rpm]'] * (2 * np.pi / 60.0)
    df_eng['Power [W]'] = df_eng['Torque [Nm]'] * angular_velocity

    # Overstrain product (min * Nm)
    df_eng['Overstrain'] = df_eng['Tool wear [min]'] * df_eng['Torque [Nm]']

    return df_eng

def prepare_features_and_target(df):
    """
    Transforms raw dataframe into feature matrix X and target y.
    - Prevents target leakage by dropping specific failure modes ('TWF', 'HDF', etc.).
    - Drops identifiers ('UDI', 'Product ID').
    - One-hot encodes the categorical 'Type' feature ('L', 'M', 'H').
    - Returns feature matrix X, target y, and optional failure_modes_df for diagnosis.
    """
    # Verify target column exists
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    y = df[TARGET_COLUMN].astype(int).values

    # Extract failure mode flags for post-hoc explanation (NOT to be used in X)
    existing_modes = [col for col in FAILURE_MODES if col in df.columns]
    failure_modes_df = df[existing_modes].copy() if existing_modes else None

    # Apply feature engineering
    df_feat = engineer_features(df)

    # Columns to drop to strictly prevent data leakage and index bias
    cols_to_drop = [TARGET_COLUMN] + IDENTIFIER_COLUMNS + FAILURE_MODES
    cols_to_drop = [c for c in cols_to_drop if c in df_feat.columns]

    X_df = df_feat.drop(columns=cols_to_drop)

    # One-hot encode product Type ('L', 'M', 'H')
    if 'Type' in X_df.columns:
        X_df = pd.get_dummies(X_df, columns=['Type'], prefix='Type', dtype=float)

    feature_names = list(X_df.columns)
    return X_df, y, feature_names, failure_modes_df

def get_train_val_test_split(df, test_size=0.20, val_size=0.15, random_state=42):
    """
    Performs stratified train/validation/test split and fits StandardScaler on train set only.
    Validation set is used strictly for Simulated Annealing optimization.
    Test set is held out strictly for final unbiased benchmark evaluation.
    """
    X_df, y, feature_names, failure_modes_df = prepare_features_and_target(df)

    # First split: Hold out test set (stratified by target)
    X_train_val, X_test_raw, y_train_val, y_test = train_test_split(
        X_df.values, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Second split: Train vs Validation (from train_val)
    # val_ratio relative to train_val
    relative_val_size = val_size / (1.0 - test_size)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=relative_val_size, stratify=y_train_val, random_state=random_state
    )

    # Fit scaler strictly on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)

    split_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'failure_modes': failure_modes_df
    }
    return split_data
