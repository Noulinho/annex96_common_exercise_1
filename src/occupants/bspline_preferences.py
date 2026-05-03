"""B-spline occupancy model used to gate dynamic comfort preferences."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize
from scipy.special import expit


class FixedBSplineIMCOccupancyPredictor:
    """Time-periodic B-spline Markov occupancy predictor.

    This is the cleaned-up home for the B-spline occupancy logic from
    `notebooks/my_mpc.ipynb`.
    """

    def __init__(
        self,
        period=24,
        spline_degree=3,
        n_internal_knots=4,
        l2_phi=1e-3,
        min_samples_per_state=25,
        maxiter=1000,
        random_state=42,
        verbose=False,
    ):
        self.period = period
        self.spline_degree = spline_degree
        self.n_internal_knots = n_internal_knots
        self.l2_phi = l2_phi
        self.min_samples_per_state = min_samples_per_state
        self.maxiter = maxiter
        self.random_state = random_state
        self.verbose = verbose
        self.states_ = None
        self.state_to_idx_ = None
        self.row_models_ = {}
        self.knot_vector_ = None
        self.interior_knots_ = None
        self.n_basis_ = None

    def _time_in_period(self, t_idx):
        return np.mod(np.asarray(t_idx, dtype=float), self.period)

    def _build_fixed_knot_vector(self):
        k = self.spline_degree
        interior = (
            np.linspace(0.0, float(self.period), self.n_internal_knots + 2)[1:-1]
            if self.n_internal_knots > 0
            else np.array([], dtype=float)
        )
        knot_vector = np.concatenate([np.repeat(0.0, k + 1), interior, np.repeat(float(self.period), k + 1)])
        return knot_vector, interior

    def _basis_matrix(self, t_values):
        t_values = np.asarray(t_values, dtype=float)
        k = self.spline_degree
        n_basis = len(self.knot_vector_) - k - 1
        basis = np.zeros((len(t_values), n_basis), dtype=float)

        for r in range(n_basis):
            coeff = np.zeros(n_basis, dtype=float)
            coeff[r] = 1.0
            vals = BSpline(self.knot_vector_, coeff, k, extrapolate=False)(t_values)
            basis[:, r] = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)

        basis[basis < 0] = 0.0
        row_sums = basis.sum(axis=1, keepdims=True)
        valid = row_sums[:, 0] > 0
        basis[valid] /= row_sums[valid]
        return basis

    def _row_probabilities(self, basis, phi):
        psi = basis @ phi
        q = expit(psi)
        probs = q / np.clip(q.sum(axis=1, keepdims=True), 1e-12, None)
        return probs, psi

    def _row_objective(self, flat_phi, basis, y_next_idx, state_count):
        phi = flat_phi.reshape(self.n_basis_, state_count)
        probs, _ = self._row_probabilities(basis, phi)
        selected = probs[np.arange(len(y_next_idx)), y_next_idx]
        nll = -np.sum(np.log(np.clip(selected, 1e-12, 1.0)))
        reg_phi = self.l2_phi * np.sum(phi**2)
        return float(nll + reg_phi)

    def fit(self, occupancy_series):
        y = np.asarray(occupancy_series, dtype=int)
        if len(y) < 2:
            raise ValueError("Need at least two occupancy observations.")

        self.states_ = np.sort(np.unique(y))
        self.state_to_idx_ = {state: i for i, state in enumerate(self.states_)}
        self.knot_vector_, self.interior_knots_ = self._build_fixed_knot_vector()
        self.n_basis_ = len(self.knot_vector_) - self.spline_degree - 1

        y_curr = y[:-1]
        y_next = y[1:]
        t_period = self._time_in_period(np.arange(len(y) - 1))
        state_count = len(self.states_)
        rng = np.random.default_rng(self.random_state)
        self.row_models_ = {}

        for state in self.states_:
            mask = y_curr == state
            y_next_idx = np.array([self.state_to_idx_[v] for v in y_next[mask]], dtype=int)
            counts = np.bincount(y_next_idx, minlength=state_count).astype(float)
            empirical = counts / counts.sum() if counts.sum() > 0 else np.ones(state_count) / state_count

            if mask.sum() < self.min_samples_per_state:
                self.row_models_[state] = {
                    "fitted": False,
                    "empirical_probs": empirical,
                    "Phi": None,
                }
                continue

            basis = self._basis_matrix(t_period[mask])
            eps = 1e-6
            target_q = np.clip(empirical, eps, 1 - eps)
            psi0 = np.log(target_q / (1 - target_q))
            phi0 = np.tile(psi0, (self.n_basis_, 1)) / self.n_basis_
            phi0 += 0.01 * rng.standard_normal(phi0.shape)

            result = minimize(
                fun=self._row_objective,
                x0=phi0.ravel(),
                args=(basis, y_next_idx, state_count),
                method="L-BFGS-B",
                options={"maxiter": self.maxiter},
            )
            self.row_models_[state] = {
                "fitted": True,
                "empirical_probs": empirical,
                "Phi": result.x.reshape(self.n_basis_, state_count),
            }

        return self

    def transition_matrix(self, t_idx: int):
        if self.states_ is None:
            raise RuntimeError("Fit the occupancy model first.")

        state_count = len(self.states_)
        matrix = np.zeros((state_count, state_count), dtype=float)
        basis = self._basis_matrix([self._time_in_period([t_idx])[0]])

        for row_idx, state in enumerate(self.states_):
            row_model = self.row_models_.get(state)
            if row_model is None:
                row = np.ones(state_count, dtype=float) / state_count
            elif not row_model["fitted"]:
                row = row_model["empirical_probs"].copy()
            else:
                row, _ = self._row_probabilities(basis, row_model["Phi"])
                row = row[0]

            row = np.clip(row, 1e-12, None)
            matrix[row_idx, :] = row / row.sum()

        return matrix

    def predict_distribution(self, current_state: int, start_t: int, horizon: int):
        if current_state not in self.state_to_idx_:
            raise ValueError(f"Current state {current_state} was not seen during fitting.")

        state_count = len(self.states_)
        pi = np.zeros(state_count, dtype=float)
        pi[self.state_to_idx_[current_state]] = 1.0

        distributions = []
        for h in range(int(horizon)):
            pi = pi @ self.transition_matrix(int(start_t) + h)
            distributions.append(pi.copy())

        return np.asarray(distributions)

    def predict_expected_occupancy(self, current_state: int, start_t: int, horizon: int):
        distributions = self.predict_distribution(current_state, start_t, horizon)
        expected = distributions @ self.states_
        return expected, distributions

    def predict_proba(self, current_state: int, t_idx: int) -> dict[int, float]:
        row_model = self.row_models_.get(current_state)
        if row_model is None:
            probs = np.ones(len(self.states_), dtype=float) / len(self.states_)
        elif not row_model["fitted"]:
            probs = row_model["empirical_probs"]
        else:
            basis = self._basis_matrix([self._time_in_period([t_idx])[0]])
            probs, _ = self._row_probabilities(basis, row_model["Phi"])
            probs = probs[0]

        return {int(state): float(prob) for state, prob in zip(self.states_, probs)}


class DynamicPreferenceModel:
    """Temperature preference model gated by predicted occupancy probability."""

    def __init__(self, occupied_temperature: float, unoccupied_temperature: float = 18.0):
        self.occupied_temperature = float(occupied_temperature)
        self.unoccupied_temperature = float(unoccupied_temperature)

    def predict(self, occupied_probability: float) -> float:
        p = float(np.clip(occupied_probability, 0.0, 1.0))
        return p * self.occupied_temperature + (1.0 - p) * self.unoccupied_temperature
