"""
Simulated Annealing Feature Selection Engine
Academic implementation with Metropolis acceptance criterion,
geometric cooling schedule, and parsimonious energy formulation.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight

class SimulatedAnnealingFeatureSelector:
    """
    Simulated Annealing Optimizer for Feature Subset Selection in Neural Networks.

    Energy Function:
        E(s) = - [ F1_val(s) - lambda * (k / D) ]
        where:
            s: binary feature selection vector in {0, 1}^D
            k: number of selected features (sum(s))
            D: total number of available candidate features
            lambda: penalty hyperparameter encouraging model parsimony / sparsity
            F1_val(s): validation F1 score achieved by the ANN using subset s

    Metropolis Criterion:
        Delta_E = E(cand) - E(curr)
        If Delta_E <= 0: Accept cand unconditionally (P = 1.0)
        If Delta_E > 0: Accept cand with Boltzmann probability:
            P(accept) = exp(-Delta_E / T)
    """

    def __init__(
        self,
        n_iterations=80,
        initial_temp=1.0,
        alpha=0.92,
        min_temp=1e-3,
        penalty_lambda=0.02,
        eval_hidden_layers=(32, 16),
        eval_max_iter=120,
        random_state=42
    ):
        self.n_iterations = n_iterations
        self.initial_temp = initial_temp
        self.alpha = alpha
        self.min_temp = min_temp
        self.penalty_lambda = penalty_lambda
        self.eval_hidden_layers = eval_hidden_layers
        self.eval_max_iter = eval_max_iter
        self.random_state = random_state

        self.rng = np.random.RandomState(random_state)
        self.best_state_ = None
        self.best_energy_ = float('inf')
        self.best_score_ = -1.0
        self.history_ = []
        self._cache = {}

    def _evaluate_state(self, state, X_train, y_train, X_val, y_val):
        """
        Trains a fast evaluation MLP on the selected feature mask and computes validation F1.
        Uses memoization cache to prevent redundant neural network retraining.
        """
        state_key = tuple(state)
        if state_key in self._cache:
            return self._cache[state_key]

        selected_indices = np.where(state == 1)[0]
        if len(selected_indices) == 0:
            # Degenerate state (no features selected): heavy penalty
            return 999.0, 0.0

        X_tr_sub = X_train[:, selected_indices]
        X_va_sub = X_val[:, selected_indices]

        # Handle class imbalance using balanced sample weights
        sample_weights = compute_sample_weight('balanced', y_train)

        # Lightweight evaluator ANN
        eval_mlp = MLPClassifier(
            hidden_layer_sizes=self.eval_hidden_layers,
            activation='relu',
            solver='adam',
            alpha=1e-3,
            max_iter=self.eval_max_iter,
            random_state=self.random_state,
            early_stopping=True,
            n_iter_no_change=10
        )

        try:
            eval_mlp.fit(X_tr_sub, y_train, sample_weight=sample_weights)
            y_pred = eval_mlp.predict(X_va_sub)
            val_f1 = f1_score(y_val, y_pred, zero_division=0)
        except Exception:
            val_f1 = 0.0

        # Parsimonious Energy formulation:
        # We wish to maximize F1 and minimize the feature ratio k/D
        k = len(selected_indices)
        D = len(state)
        energy = -(val_f1 - self.penalty_lambda * (k / float(D)))

        self._cache[state_key] = (energy, val_f1)
        return energy, val_f1

    def _generate_neighbor(self, state):
        """
        Generates a neighboring state by perturbing 1 or 2 feature bits.
        Guarantees that at least one feature remains selected.
        """
        neighbor = state.copy()
        D = len(state)

        # Decide whether to flip 1 or 2 bits
        n_flips = 1 if self.rng.rand() < 0.75 else 2
        flip_indices = self.rng.choice(D, size=min(n_flips, D), replace=False)

        for idx in flip_indices:
            neighbor[idx] = 1 - neighbor[idx]

        # Guard against zero-feature mask
        if np.sum(neighbor) == 0:
            forced_idx = self.rng.randint(0, D)
            neighbor[forced_idx] = 1

        return neighbor

    def fit(self, X_train, y_train, X_val, y_val, feature_names=None, verbose=True):
        """
        Executes Simulated Annealing feature selection.
        Strictly operates on X_train and X_val (no test set leakage).
        """
        D = X_train.shape[1]
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(D)]

        # Initialize: start with all features active
        current_state = np.ones(D, dtype=int)
        current_energy, current_f1 = self._evaluate_state(
            current_state, X_train, y_train, X_val, y_val
        )

        self.best_state_ = current_state.copy()
        self.best_energy_ = current_energy
        self.best_score_ = current_f1
        self.history_ = []

        temperature = self.initial_temp

        if verbose:
            print(f"--- Starting Simulated Annealing Feature Selection ---")
            print(f"Total features: {D} | Iterations: {self.n_iterations} | T0: {self.initial_temp} | alpha: {self.alpha}")
            print(f"Initial State: {np.sum(current_state)} features | Energy: {current_energy:.4f} | F1: {current_f1:.4f}\n")

        for iteration in range(1, self.n_iterations + 1):
            # 1. Perturb state to obtain neighbor
            candidate_state = self._generate_neighbor(current_state)
            candidate_energy, candidate_f1 = self._evaluate_state(
                candidate_state, X_train, y_train, X_val, y_val
            )

            # 2. Compute energy differential
            delta_e = candidate_energy - current_energy

            # 3. Metropolis acceptance check
            accepted = False
            acceptance_prob = 1.0

            if delta_e <= 0:
                # Direct improvement
                accepted = True
                acceptance_prob = 1.0
            else:
                # Candidate is worse: Boltzmann acceptance probability
                # Clamp delta_e / temperature to avoid overflow
                scaled_delta = max(-50.0, min(50.0, delta_e / max(temperature, 1e-8)))
                acceptance_prob = np.exp(-scaled_delta)
                if self.rng.rand() < acceptance_prob:
                    accepted = True

            if accepted:
                current_state = candidate_state.copy()
                current_energy = candidate_energy
                current_f1 = candidate_f1

                # Update global best if this is the lowest energy seen so far
                if current_energy < self.best_energy_:
                    self.best_energy_ = current_energy
                    self.best_score_ = current_f1
                    self.best_state_ = current_state.copy()

            # Record iteration telemetry
            best_selected_indices = np.where(self.best_state_ == 1)[0]
            best_feature_names = [feature_names[i] for i in best_selected_indices]

            record = {
                'iteration': iteration,
                'temperature': float(temperature),
                'current_energy': float(current_energy),
                'best_energy': float(self.best_energy_),
                'current_f1': float(current_f1),
                'best_f1': float(self.best_score_),
                'current_n_features': int(np.sum(current_state)),
                'best_n_features': int(np.sum(self.best_state_)),
                'delta_e': float(delta_e),
                'accepted': bool(accepted),
                'acceptance_prob': float(acceptance_prob),
                'best_features': best_feature_names
            }
            self.history_.append(record)

            if verbose and (iteration % 10 == 0 or iteration == self.n_iterations):
                print(
                    f"Iter {iteration:3d}/{self.n_iterations} | "
                    f"Temp: {temperature:6.4f} | "
                    f"Current Feats: {np.sum(current_state):2d} (F1: {current_f1:.4f}) | "
                    f"Best Feats: {np.sum(self.best_state_):2d} (Best F1: {self.best_score_:.4f}, E: {self.best_energy_:.4f})"
                )

            # 4. Geometric temperature decay
            temperature = max(self.min_temp, temperature * self.alpha)

        selected_indices = np.where(self.best_state_ == 1)[0]
        self.selected_features_ = [feature_names[i] for i in selected_indices]
        self.selected_indices_ = selected_indices

        if verbose:
            print(f"\n--- Simulated Annealing Completed ---")
            print(f"Optimal features ({len(self.selected_features_)} / {D}): {self.selected_features_}")
            print(f"Best Validation F1: {self.best_score_:.4f} | Final Energy: {self.best_energy_:.4f}")

        return self

    def transform(self, X):
        """Reduces feature matrix X to the SA-selected features."""
        if self.best_state_ is None:
            raise ValueError("Selector has not been fitted yet.")
        return X[:, self.selected_indices_]
