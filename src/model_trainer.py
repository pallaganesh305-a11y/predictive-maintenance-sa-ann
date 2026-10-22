"""
Artificial Neural Network (MLP) Model Trainer Module
Handles MLP training, class imbalance weighting, validation threshold tuning,
and model persistence.
"""

import os
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.utils.class_weight import compute_sample_weight

class PredictiveMaintenanceANN:
    """
    Wrapper for Scikit-Learn's MLPClassifier with:
    - Balanced class weighting for imbalanced failure detection
    - Threshold tuning on validation set for optimal F1
    - Easy serialization and prediction
    """

    def __init__(
        self,
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        alpha=1e-3,
        max_iter=300,
        random_state=42,
        name="ANN_Model"
    ):
        self.name = name
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
            early_stopping=True,
            n_iter_no_change=15,
            validation_fraction=0.1
        )
        self.best_threshold_ = 0.5
        self.feature_names_ = None
        self.selected_indices_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, selected_indices=None):
        """
        Trains the MLPClassifier using balanced sample weights.
        If validation data is provided, tunes the decision threshold to maximize validation F1.
        """
        self.feature_names_ = feature_names
        self.selected_indices_ = selected_indices

        # If feature subset is specified, slice columns
        if selected_indices is not None:
            X_tr = X_train[:, selected_indices]
            X_va = X_val[:, selected_indices] if X_val is not None else None
        else:
            X_tr = X_train
            X_va = X_val

        # Compute balanced weights to handle the 3.39% failure class
        sample_weights = compute_sample_weight('balanced', y_train)

        self.model.fit(X_tr, y_train, sample_weight=sample_weights)

        # Optimize classification threshold on validation set
        if X_va is not None and y_val is not None:
            val_probs = self.model.predict_proba(X_va)[:, 1]
            thresholds = np.linspace(0.1, 0.9, 81)
            best_f1 = -1.0
            best_thresh = 0.5

            for t in thresholds:
                preds = (val_probs >= t).astype(int)
                score = f1_score(y_val, preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_thresh = t

            self.best_threshold_ = float(best_thresh)
        else:
            self.best_threshold_ = 0.5

        return self

    def predict_proba(self, X):
        """Returns failure probabilities [P(normal), P(failure)]."""
        if self.selected_indices_ is not None:
            X_sub = X[:, self.selected_indices_]
        else:
            X_sub = X
        return self.model.predict_proba(X_sub)

    def predict(self, X, threshold=None):
        """
        Predicts binary failure labels using either the tuned threshold
        or a user-supplied threshold.
        """
        thresh = self.best_threshold_ if threshold is None else threshold
        probs = self.predict_proba(X)[:, 1]
        return (probs >= thresh).astype(int)

    def save(self, filepath):
        """Saves the fitted model and metadata to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        """Loads a saved model from disk."""
        return joblib.load(filepath)
