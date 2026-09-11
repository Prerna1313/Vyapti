# =============================================================================
# PS26055 — ADVANCED SCHEDULER INTEGRATION PROTOTYPE (ROBUST, HONEST NAMING)
# =============================================================================
#
# scheduler_core.py
# ------------------
# This is main.py's algorithm content (HMM / BOCPD / BCoR-inspired /
# Tsallis-inspired / AdvancedSchedulerPrototype / run_episode_safe),
# extracted verbatim into an importable module.
#
# WHAT CHANGED VS THE ORIGINAL main.py, AND WHY:
#   The original main.py called `snapshot_download(...)` and iterated the
#   entire TSRD corpus at MODULE IMPORT TIME (i.e. merely writing
#   `import main` triggered a HuggingFace network download). That is fine
#   for a one-shot CLI script, but fatal for a FastAPI server: the app
#   would try to download TSRD the instant the adapter module loaded,
#   before the process could even bind a port, and would hang/crash if
#   HuggingFace was unreachable or the dataset wasn't cached yet.
#
#   The dataset-loading code (originally "6. CONFIG + DATA LOADING") has
#   been moved out to dataset.py as `load_tsrd_dataset()`, which is a
#   plain function you call explicitly. Importing scheduler_core.py (or
#   dataset.py) now does nothing but define classes/functions — no
#   network access, no disk I/O, no side effects.
#
#   No scheduler class below (OnlineStickyHMM, GaussianBOCPD,
#   ContextualThompsonScheduler, TsallisInspiredExplorer,
#   AdvancedSchedulerPrototype, run_episode_safe) has been changed in
#   any way — this is a straight copy/paste of that code out of the
#   original main.py.
#
# See main.py in this same folder for the CLI entry point (unchanged
# behavior — `python main.py` still downloads TSRD and runs the original
# 20-episode evaluation, exactly as before) and backend/ for the FastAPI
# adapter that calls load_tsrd_dataset() lazily, on first use.

import json
import time
from datetime import datetime, timezone
from math import lgamma, log
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import scipy.stats as stats

import sys
from pathlib import Path
_vyapti_root = Path(__file__).resolve().parent.parent
if str(_vyapti_root) not in sys.path:
    sys.path.insert(0, str(_vyapti_root))

from vyapti_simulator.tsrd import TSRDEnvironment, DetectionConfig
from vyapti_simulator.core.environment import SimulationConfig

# =============================================================================
# 1. ONLINE STICKY HMM (FINITE-STATE, NOT HDP)
# =============================================================================

class OnlineStickyHMM:
    """
    Finite-state sticky Gaussian HMM with online updates.
    - Fixed number of states (no nonparametric state growth).
    - Sticky self-transition bias via kappa.
    - Online forward filtering + moment updates.
    """

    def __init__(self, n_states: int = 4, kappa: float = 10.0, prior_var: float = 0.1):
        self.n_states = n_states
        self.kappa = kappa
        self.prior_var = prior_var

        self.state_means = np.zeros(n_states)
        self.state_vars = np.ones(n_states) * prior_var
        self.transition_matrix = np.zeros((n_states, n_states))
        self.belief = None

        self.state_counts = np.zeros(n_states)
        self.transition_counts = np.zeros((n_states, n_states))
        self.state_sum = np.zeros(n_states)
        self.state_sq_sum = np.zeros(n_states)

    def reset(self):
        spread = np.linspace(0.1, 0.9, self.n_states)
        self.state_means[:] = spread
        self.state_vars[:] = self.prior_var
        self._init_transition_matrix()
        self.belief = np.ones(self.n_states) / self.n_states
        self.state_counts[:] = 0.0
        self.transition_counts[:] = 0.0
        self.state_sum[:] = 0.0
        self.state_sq_sum[:] = 0.0

    def _init_transition_matrix(self):
        sticky = self.kappa / (self.kappa + 1.0)
        cross = (1.0 - sticky) / (self.n_states - 1) if self.n_states > 1 else 0.0
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.transition_matrix[i, j] = sticky if i == j else cross

    def _forward_filter(self, obs: float) -> np.ndarray:
        if self.belief is None:
            self.reset()
        predicted = self.belief @ self.transition_matrix
        likelihoods = np.array([
            stats.norm.pdf(obs, self.state_means[s], np.sqrt(self.state_vars[s]))
            for s in range(self.n_states)
        ])
        updated = predicted * likelihoods
        updated = updated / (updated.sum() + 1e-10)
        return updated

    def _online_update(self, obs: float, new_belief: np.ndarray, prev_belief: np.ndarray):
        self.state_counts += new_belief
        self.state_sum += obs * new_belief
        self.state_sq_sum += obs**2 * new_belief

        for i in range(self.n_states):
            for j in range(self.n_states):
                self.transition_counts[i, j] += prev_belief[i] * new_belief[j]

        for s in range(self.n_states):
            if self.state_counts[s] > 1.0:
                self.state_means[s] = self.state_sum[s] / self.state_counts[s]
                var_est = (self.state_sq_sum[s] / self.state_counts[s]) - self.state_means[s]**2
                self.state_vars[s] = max(var_est, 0.001)

        sticky = self.kappa / (self.kappa + 1.0)
        for i in range(self.n_states):
            row_count = self.transition_counts[i, :].sum()
            if row_count > 1e-6:
                for j in range(self.n_states):
                    count = self.transition_counts[i, j] + (self.kappa if i == j else 0.0)
                    self.transition_matrix[i, j] = count / (row_count + self.kappa)
            else:
                for j in range(self.n_states):
                    self.transition_matrix[i, j] = sticky if i == j else (1.0 - sticky) / (self.n_states - 1)

    def update(self, obs: float) -> np.ndarray:
        prev_belief = self.belief.copy() if self.belief is not None else np.ones(self.n_states) / self.n_states
        new_belief = self._forward_filter(obs)
        self._online_update(obs, new_belief, prev_belief)
        self.belief = new_belief
        return self.belief

    def get_belief_state(self) -> np.ndarray:
        return self.belief if self.belief is not None else np.ones(self.n_states) / self.n_states

    def get_entropy(self) -> float:
        belief = self.get_belief_state()
        return -np.sum(belief * np.log(belief + 1e-10))

    def get_predicted_observation_probability(self) -> float:
        """
        Return a scalar predictive likelihood for the last observation.
        This is a simplified proxy for use with BOCPD.
        """
        if self.belief is None:
            return 1.0 / self.n_states
        # Approximate predictive as mixture of Gaussians evaluated at last obs
        # We don't store last obs here; instead, return a generic "uncertainty" proxy.
        # In a more rigorous version, you'd pass obs and compute p(obs | belief).
        return float(np.mean(self.belief))


# =============================================================================
# 2. GAUSSIAN BOCPD (ADAMS–MACKAY STYLE)
# =============================================================================

class GaussianBOCPD:
    """
    Bayesian Online Change-Point Detection for a scalar stream.
    - Normal-Inverse-Gamma conjugate model.
    - Student-t predictive likelihood.
    - Log-space recursion.
    - Bounded run-length memory.
    """

    def __init__(
        self,
        hazard: float = 0.05,
        max_run_length: int = 80,
        mu0: float = 0.5,
        kappa0: float = 1.0,
        alpha0: float = 1.0,
        beta0: float = 0.05,
    ):
        if not (0.0 < hazard < 1.0):
            raise ValueError("hazard must be strictly between 0 and 1")
        if not isinstance(max_run_length, int) or max_run_length < 1:
            raise ValueError("max_run_length must be an integer >= 1")
        if kappa0 <= 0.0:
            raise ValueError("kappa0 must be > 0")
        if alpha0 <= 0.0:
            raise ValueError("alpha0 must be > 0")
        if beta0 <= 0.0:
            raise ValueError("beta0 must be > 0")

        self.hazard = float(hazard)
        self.max_run_length = int(max_run_length)
        self.mu0 = float(mu0)
        self.kappa0 = float(kappa0)
        self.alpha0 = float(alpha0)
        self.beta0 = float(beta0)

        self.reset()

    def reset(self):
        size = self.max_run_length + 1
        self.log_run_length_posterior = np.full(size, -np.inf)
        self.log_run_length_posterior[0] = 0.0

        self.kappa = np.full(size, self.kappa0, dtype=float)
        self.mu = np.full(size, self.mu0, dtype=float)
        self.alpha = np.full(size, self.alpha0, dtype=float)
        self.beta = np.full(size, self.beta0, dtype=float)

        self.t = 0
        self.change_probability = 0.0
        self.last_observation = None
        self.last_predictive_log_prob = np.nan
        self.last_residual = np.nan

    @staticmethod
    def _logsumexp(values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            return -np.inf
        max_value = np.max(values[finite])
        return float(max_value + np.log(np.sum(np.exp(values[finite] - max_value))))

    @staticmethod
    def _log_student_t_pdf(
        x: float,
        mean: np.ndarray,
        scale2: np.ndarray,
        degrees_of_freedom: np.ndarray,
    ) -> np.ndarray:
        x = float(x)
        mean = np.asarray(mean, dtype=float)
        scale2 = np.maximum(np.asarray(scale2, dtype=float), 1e-12)
        nu = np.maximum(np.asarray(degrees_of_freedom, dtype=float), 1e-12)
        result = np.empty_like(mean, dtype=float)
        for i in range(len(mean)):
            scale2_i = scale2[i]
            nu_i = nu[i]
            result[i] = (
                lgamma((nu_i + 1.0) / 2.0)
                - lgamma(nu_i / 2.0)
                - 0.5 * (np.log(nu_i) + np.log(np.pi) + np.log(scale2_i))
                - ((nu_i + 1.0) / 2.0)
                * np.log(1.0 + ((x - mean[i]) ** 2) / (nu_i * scale2_i))
            )
        return result

    def _predictive_log_probability(self, x: float) -> np.ndarray:
        predictive_scale2 = (
            self.beta * (self.kappa + 1.0)
            / (self.alpha * self.kappa)
        )
        degrees_of_freedom = 2.0 * self.alpha
        return self._log_student_t_pdf(
            x=x,
            mean=self.mu,
            scale2=predictive_scale2,
            degrees_of_freedom=degrees_of_freedom,
        )

    def _update_posterior_parameters(self, x: float):
        size = self.max_run_length + 1
        old_kappa = self.kappa.copy()
        old_mu = self.mu.copy()
        old_alpha = self.alpha.copy()
        old_beta = self.beta.copy()

        new_kappa = np.full(size, self.kappa0, dtype=float)
        new_mu = np.full(size, self.mu0, dtype=float)
        new_alpha = np.full(size, self.alpha0, dtype=float)
        new_beta = np.full(size, self.beta0, dtype=float)

        # r = 0 restart
        new_kappa[0] = self.kappa0 + 1.0
        new_mu[0] = (self.kappa0 * self.mu0 + x) / new_kappa[0]
        new_alpha[0] = self.alpha0 + 0.5
        new_beta[0] = (
            self.beta0
            + 0.5 * self.kappa0 * (x - self.mu0) ** 2 / new_kappa[0]
        )

        # growth transitions
        for previous_r in range(self.max_run_length):
            new_r = previous_r + 1
            kappa_prev = old_kappa[previous_r]
            mu_prev = old_mu[previous_r]
            alpha_prev = old_alpha[previous_r]
            beta_prev = old_beta[previous_r]

            kappa_new = kappa_prev + 1.0
            mu_new = (kappa_prev * mu_prev + x) / kappa_new
            alpha_new = alpha_prev + 0.5
            beta_new = (
                beta_prev
                + 0.5 * kappa_prev * (x - mu_prev) ** 2 / kappa_new
            )

            new_kappa[new_r] = kappa_new
            new_mu[new_r] = mu_new
            new_alpha[new_r] = alpha_new
            new_beta[new_r] = beta_new

        self.kappa = new_kappa
        self.mu = new_mu
        self.alpha = new_alpha
        self.beta = new_beta

    def update(self, observation: float) -> float:
        x = float(observation)
        if not np.isfinite(x):
            raise ValueError("observation must be finite")

        log_predictive = self._predictive_log_probability(x)
        self.last_predictive_log_prob = float(
            np.sum(np.exp(self.log_run_length_posterior) * log_predictive)
        )

        old_log_posterior = self.log_run_length_posterior.copy()
        new_log_posterior = np.full_like(old_log_posterior, -np.inf)

        log_hazard = np.log(self.hazard)
        log_survival = np.log(1.0 - self.hazard)

        # changepoint transition (r_t = 0)
        new_log_posterior[0] = self._logsumexp(
            old_log_posterior + log_hazard + log_predictive
        )

        # growth transitions (r_t = r+1)
        growth_terms = (
            old_log_posterior[:-1]
            + log_survival
            + log_predictive[:-1]
        )
        new_log_posterior[1:] = growth_terms

        normalizer = self._logsumexp(new_log_posterior)
        if not np.isfinite(normalizer):
            raise FloatingPointError("BOCPD posterior normalization failed")

        new_log_posterior -= normalizer
        self.log_run_length_posterior = new_log_posterior

        posterior = np.exp(new_log_posterior)
        expected_nll = float(-np.sum(posterior * log_predictive))
        self.last_residual = expected_nll

        self._update_posterior_parameters(x)
        self.t += 1
        self.last_observation = x
        self.change_probability = float(posterior[0])

        return self.change_probability

    def get_change_probability(self) -> float:
        return float(self.change_probability)

    def get_run_length_posterior(self) -> np.ndarray:
        return np.exp(self.log_run_length_posterior.copy())

    def get_map_run_length(self) -> int:
        return int(np.argmax(self.log_run_length_posterior))

    def get_expected_run_length(self) -> float:
        posterior = self.get_run_length_posterior()
        run_lengths = np.arange(len(posterior))
        return float(np.sum(run_lengths * posterior))


# =============================================================================
# 3. CONTEXTUAL THOMPSON SCHEDULER (BCoR-INSPIRED)
# =============================================================================

class ContextualThompsonScheduler:
    """
    Per-arm contextual linear model with Thompson-style sampling.
    - reward ≈ context^T theta_b
    - Online gradient-style updates.
    - Gaussian sampling for exploration.
    """

    def __init__(self, n_arms: int = 36, context_dim: int = 4, learning_rate: float = 0.1):
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.learning_rate = learning_rate
        self.theta_mu = np.zeros((n_arms, context_dim))
        self.theta_cov = np.eye(context_dim) * 1.0
        self.rng = np.random.default_rng()

    def reset(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.theta_mu = np.zeros((self.n_arms, self.context_dim))
        self.theta_cov = np.eye(self.context_dim) * 1.0

    def predict(self, arm: int, context: np.ndarray) -> float:
        return float(context.dot(self.theta_mu[arm]))

    def sample_utility(self, arm: int, context: np.ndarray) -> float:
        theta_sample = self.rng.multivariate_normal(self.theta_mu[arm], self.theta_cov)
        return float(context.dot(theta_sample))

    def update(self, arm: int, context: np.ndarray, reward: float):
        pred = self.predict(arm, context)
        error = reward - pred
        self.theta_mu[arm] += self.learning_rate * error * context
        self.theta_cov = self.theta_cov * 0.99


# =============================================================================
# 4. TSALLIS-INSPIRED EXPLORER (HEURISTIC)
# =============================================================================

class TsallisInspiredExplorer:
    """
    Loss-based exploration distribution inspired by Tsallis-INF.
    - Maintains cumulative losses per arm.
    - Produces a distribution favoring under-performing arms.
    - Does NOT implement formal 1/2-Tsallis-INF.
    """

    def __init__(self, n_arms: int = 36, eta_init: float = 0.3, eta_decay: float = 0.9995, alpha: float = 0.5):
        self.n_arms = n_arms
        self.eta = eta_init
        self.eta_decay = eta_decay
        self.alpha = alpha
        self.cumulative_losses = np.zeros(n_arms)

    def get_exploration_distribution(self) -> np.ndarray:
        L_max = np.max(self.cumulative_losses)
        distribution = np.zeros(self.n_arms)
        for i in range(self.n_arms):
            diff = L_max - self.cumulative_losses[i]
            distribution[i] = (self.eta * max(diff, 0) + 1e-10) ** (1 / (1 - self.alpha))
        dist_sum = distribution.sum()
        if dist_sum > 1e-10:
            distribution = distribution / dist_sum
        else:
            distribution = np.ones(self.n_arms) / self.n_arms
        return distribution

    def update(self, arm: int, reward: float):
        loss = 1.0 - reward
        self.cumulative_losses[arm] += loss
        self.eta = self.eta * self.eta_decay


# =============================================================================
# 5. ADVANCED SCHEDULER (INTEGRATION PROTOTYPE)
# =============================================================================

class AdvancedSchedulerPrototype:
    """
    Advanced scheduler integration prototype:
      - OnlineStickyHMM per band
      - GaussianBOCPD on residuals
      - ContextualThompsonScheduler (BCoR-inspired)
      - TsallisInspiredExplorer
      - Per-band value Q, time/entropy bonuses
      - Adaptive dwell (not physically active in TSRD)
    """

    def __init__(
        self,
        band_count: int = 36,
        hmm_states: int = 4,
        hmm_kappa: float = 10.0,
        bocpd_hazard: float = 0.05,
        bcor_lr: float = 0.1,
        tsallis_eta: float = 0.3,
        tsallis_alpha: float = 0.5,
        dwell_options: List[float] = [20, 50, 100],
    ):
        self.band_count = band_count
        self.dwell_options = dwell_options

        self.hmm_models = [
            OnlineStickyHMM(n_states=hmm_states, kappa=hmm_kappa)
            for _ in range(band_count)
        ]
        self.bocpd_models = [
            GaussianBOCPD(hazard=bocpd_hazard, max_run_length=80)
            for _ in range(band_count)
        ]
        self.bcor = ContextualThompsonScheduler(n_arms=band_count, context_dim=4, learning_rate=bcor_lr)
        self.tsallis = TsallisInspiredExplorer(n_arms=band_count, eta_init=tsallis_eta, alpha=tsallis_alpha)

        self.last_visit = np.zeros(band_count)
        self.last_retune_time = np.full(band_count, -1000)
        self.time_step = 0
        self.obs_history = []

        self.visits = np.zeros(band_count, dtype=int)
        self.hits = np.zeros(band_count, dtype=int)
        self.snr_sum = np.zeros(band_count, dtype=float)
        self.snr_count = np.zeros(band_count, dtype=int)

        # INTEGRATION ADDITION (backend adapter): last_diagnostics is
        # populated at the end of select_action() with the per-band
        # arrays (Q, bcor_util, change_prob, time_bonus, entropy_bonus,
        # combined utility U, tsallis distribution, hybrid_scores) that
        # select_action() already computes internally every call. This
        # does NOT change what select_action() returns or how best_band
        # is chosen — it only exposes those already-computed numbers so
        # the FastAPI adapter can report real per-band scheduler state
        # (e.g. for the Scheduler page) instead of inventing or
        # re-deriving them. See select_action() below for exactly where
        # this is set.
        self.last_diagnostics: Optional[Dict[str, Any]] = None

    def reset(self, seed: int):
        np.random.seed(seed)
        self.last_visit = np.zeros(self.band_count)
        self.last_retune_time = np.full(self.band_count, -1000)
        self.time_step = 0
        self.obs_history = []
        for hmm in self.hmm_models:
            hmm.reset()
        for bocpd in self.bocpd_models:
            bocpd.reset()
        self.bcor.reset(seed=seed)
        self.tsallis = TsallisInspiredExplorer(n_arms=self.band_count, eta_init=self.tsallis.eta, alpha=self.tsallis.alpha)
        self.visits[:] = 0
        self.hits[:] = 0
        self.snr_sum[:] = 0.0
        self.snr_count[:] = 0

    def _build_context(self, band: int, t: int) -> np.ndarray:
        belief = self.hmm_models[band].get_belief_state()
        entropy = self.hmm_models[band].get_entropy()
        delta_t = t - self.last_visit[band]
        recent_visits = [o for o in self.obs_history[-30:] if o.get('band') == band]
        recent_hit_rate = np.mean([o.get('hit', False) for o in recent_visits]) if recent_visits else 0.0
        activity_proxy = belief[0] if len(belief) > 0 else 0.5
        context = np.array([activity_proxy, entropy, delta_t, recent_hit_rate])
        if context.std() > 1e-10:
            context = (context - context.mean()) / context.std()
        return context

    def _select_dwell(self, band: int) -> float:
        belief = self.hmm_models[band].get_belief_state()
        entropy = -np.sum(belief * np.log(belief + 1e-10))
        # Normalize entropy to [0,1] for 4 states
        H_max = np.log(4)
        H_norm = entropy / H_max if H_max > 0 else 0.0
        recent_visits = [o for o in self.obs_history[-10:] if o.get('band') == band]
        avg_snr = np.mean([o.get('snr', 10.0) for o in recent_visits]) if recent_visits else 10.0
        if H_norm > 0.7 or avg_snr < 5.0:
            dwell = 100
        elif H_norm > 0.4 or avg_snr < 10.0:
            dwell = 50
        else:
            dwell = 20
        return dwell

    def select_action(
        self,
        obs_history: List,
        t: int,
        legal_bands: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        self.obs_history = obs_history
        self.time_step = t

        if legal_bands is None:
            legal_bands = list(range(self.band_count))
        else:
            min_retune_time = 2
            legal_bands = [
                b for b in legal_bands
                if (t - self.last_retune_time[b]) >= min_retune_time
            ]

        if obs_history:
            last = obs_history[-1]
            band = last.get('band', 0)
            snr = last.get('snr', 10.0)
            norm_obs = snr / 20.0
            self.hmm_models[band].update(norm_obs)
            # Use negative log predictive likelihood as residual
            pred_prob = self.hmm_models[band].get_predicted_observation_probability()
            residual = -np.log(pred_prob + 1e-10)
            self.bocpd_models[band].update(residual)

        # Per-band value Q
        Q = np.zeros(self.band_count, dtype=float)
        visited_mask = self.visits > 0
        if np.any(visited_mask):
            mean_snr_visited = np.divide(
                self.snr_sum[visited_mask],
                np.maximum(self.snr_count[visited_mask], 1),
            )
            snr_conf = np.clip((mean_snr_visited + 10.0) / 20.0, 0.0, 1.0)
            Q[visited_mask] = (self.hits[visited_mask] / self.visits[visited_mask]) * snr_conf

        # BCoR-style utilities for legal bands only
        bcor_util = np.full(self.band_count, -np.inf, dtype=float)
        for b in legal_bands:
            ctx = self._build_context(b, t)
            bcor_util[b] = self.bcor.sample_utility(b, ctx)

        # BOCPD change probabilities
        change_prob = np.array([self.bocpd_models[i].get_change_probability() for i in range(self.band_count)])

        # Time and entropy bonuses
        time_bonus = np.zeros(self.band_count, dtype=float)
        for i in range(self.band_count):
            dt = t - self.last_visit[i]
            if dt > 10:
                time_bonus[i] = 0.02 * (dt - 10)

        entropy_bonus = np.zeros(self.band_count, dtype=float)
        for i in range(self.band_count):
            ent = self.hmm_models[i].get_entropy()
            H_max = np.log(4)
            H_norm = ent / H_max if H_max > 0 else 0.0
            entropy_bonus[i] = 0.05 * H_norm

        # Combined utility
        w_Q, w_bcor, w_cp, w_time, w_ent = 1.0, 0.5, 0.3, 0.2, 0.1
        U = w_Q * Q + w_bcor * bcor_util + w_cp * change_prob + w_time * time_bonus + w_ent * entropy_bonus

        # Tsallis exploration
        tsallis_dist = self.tsallis.get_exploration_distribution()
        u_max = np.max(np.abs(U)) + 1e-10
        alpha = 0.2
        hybrid_scores = (1 - alpha) * U + alpha * tsallis_dist * u_max

        legal_mask = np.zeros(self.band_count, dtype=bool)
        legal_mask[legal_bands] = True
        legal_scores = hybrid_scores.copy()
        legal_scores[~legal_mask] = -np.inf

        # INTEGRATION ADDITION (backend adapter): expose the per-band
        # arrays computed above, unchanged, for observability. This line
        # only assigns to self.last_diagnostics; it reads none of the
        # values it stores and writes none of the arrays used below, so
        # it cannot affect best_band or dwell selection.
        self.last_diagnostics = {
            'band': list(range(self.band_count)),
            'q_value': Q.tolist(),
            'bcor_utility': bcor_util.tolist(),
            'change_probability': change_prob.tolist(),
            'time_bonus': time_bonus.tolist(),
            'entropy_bonus': entropy_bonus.tolist(),
            'combined_utility': U.tolist(),
            'tsallis_exploration_weight': tsallis_dist.tolist(),
            'hybrid_score': hybrid_scores.tolist(),
            'legal_mask': legal_mask.tolist(),
        }

        best_band = int(np.argmax(legal_scores))
        dwell = self._select_dwell(best_band)

        self.last_visit[best_band] = t
        self.last_retune_time[best_band] = t

        return {'band': best_band, 'dwell_ms': dwell}

    def update(self, action: Dict[str, Any], obs: Dict[str, Any], system_b_metrics=None):
        band = action['band']
        snr = obs.get('snr', 10.0)
        hit = obs.get('hit', False)

        self.visits[band] += 1
        if hit:
            self.hits[band] += 1
            self.snr_sum[band] += snr
            self.snr_count[band] += 1

        confidence = float(np.clip((snr + 10.0) / 20.0, 0.0, 1.0))
        reward = float(hit) * confidence

        ctx = self._build_context(band, self.time_step)
        self.bcor.update(band, ctx, reward)
        self.tsallis.update(band, reward)

        self.obs_history.append({
            'band': band,
            'hit': hit,
            'snr': snr,
            'time': self.time_step,
        })

        if system_b_metrics is not None:
            self.last_system_b_metrics = system_b_metrics


# =============================================================================
# 6. CONFIG + DATA LOADING
# =============================================================================

SIM_CONFIG = SimulationConfig(
    receiver_ibw_mhz=500.0,
    total_spectrum_mhz=18000.0,
    band_count=36,
    dwell_time_ms=50.0,
    retune_time_ms=1.0,
    max_emitters=35,
    time_slots=600,
    detection_probability=0.9,
    false_alarm_probability=0.01,
)

BEST_DETECTION_CONFIG = DetectionConfig(
    detection_threshold_db=2.0,
    no_detection_threshold_db=-4.0,
    false_alarm_probability=0.10,
    agc_window_slots=0,
)

# NOTE: the original main.py had TSRD dataset download + corpus loading
# ("Preload valid episodes") right here, at module scope. That code has
# moved to dataset.py::load_tsrd_dataset(), which is NOT called by this
# module. Call it explicitly wherever you need VALID_EPISODES (see
# main.py's __main__ block, or backend/dataset_service.py for the
# FastAPI-side lazy loader with the same caching behavior).


# =============================================================================
# 7. ROBUST EVALUATION
# =============================================================================

def run_episode_safe(scheduler: AdvancedSchedulerPrototype, env: TSRDEnvironment, seed: int, n_steps: int = 600):
    env.reset(seed=seed)
    scheduler.reset(seed=seed)
    obs_hist, hits = [], []
    dwell_times = []

    for t in range(n_steps):
        action = scheduler.select_action(obs_hist, t)
        band = action['band']
        dwell = action.get('dwell_ms', 50)
        dwell_times.append(dwell)

        step_result = env.step(band)
        obs = step_result[0] if isinstance(step_result, tuple) else step_result
        obs['band'] = band
        obs['snr'] = obs.get('snr_db', 10.0)
        obs_hist.append(obs)

        scheduler.update(action, obs)
        hits.append(obs.get('hit', False))

    pd = float(np.mean(hits))
    avg_dwell = float(np.mean(dwell_times))
    return pd, {'avg_dwell': avg_dwell, 'n_hits': int(np.sum(hits))}

# NOTE: the original main.py's `if __name__ == "__main__":` CLI runner
# block was here. It has moved to main.py in this same folder, unchanged
# in logic — it now just imports AdvancedSchedulerPrototype, SIM_CONFIG,
# BEST_DETECTION_CONFIG, run_episode_safe from this module instead of
# defining them inline. scheduler_core.py itself has no __main__ block
# and produces no output or side effects on import.