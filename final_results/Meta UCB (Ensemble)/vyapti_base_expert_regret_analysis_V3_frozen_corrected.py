#!/usr/bin/env python3
"""
VYAPTI — BASE-EXPERT ORACLE-RELATIVE REGRET ANALYSIS V3 (FROZEN CORRECTED)

Computes empirical oracle-relative EXPECTED pseudo-regret.

For expert i:

    regret_i(n)
        = sum_{t=1}^n [
              r_oracle(t) - r_expert_i(t)
          ]

where r_oracle(t) and r_expert_i(t) are EXPECTED rewards,
not realized hit/miss samples.

For Vyapti:

    active band:
        E[hit] = Pd

    inactive band:
        E[hit] = Pfa

The oracle knows the hidden transmission truth and selects the
band with the largest expected immediate reward.

This produces empirical regret curves and empirical_C(alpha) values.

IMPORTANT:
------------
This does NOT establish the high-probability regret assumption
required by Cutkosky et al. (2020).

It is an empirical oracle-relative analysis of the base experts.

R_i is intentionally NOT derived here.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from huggingface_hub import snapshot_download

from vyapti_simulator.core.scheduler_interface import BaseScheduler
from vyapti_simulator.core.environment import SimulationConfig
from vyapti_simulator.tsrd.tsrd_environment import TSRDStareEnvironment


# =============================================================================
# CONFIGURATION
# =============================================================================

BAND_COUNT = 36
HORIZON = 600

PD = 0.90
PFA = 0.01

TRAIN_EPISODES = None

# These are reported as empirical envelope exponents.
ALPHA_CANDIDATES = [
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]

# Envelope statistic: maximum observed episode regret.
# This is an empirical worst-case, not a statistical confidence bound.
ENVELOPE_STATISTIC = "maximum_observed_episode_regret"
ENVELOPE_QUANTILE = 1.0

OUTPUT_FILE = Path(
    "/kaggle/working/base_expert_regret_analysis_V3.json"
)

EXPERT_NAMES = [
    "UCB1",
    "Whittle-Inspired",
    "RLessUCB",
    "ARP",
]


# =============================================================================
# YOUR FOUR BASE EXPERT CLASSES
# =============================================================================

class UCB1Scheduler(BaseScheduler):
    def __init__(self, band_count: int):
        super().__init__(band_count, provenance_note="UCB1")
        self.band_count = band_count
        self.counts = np.zeros(band_count, dtype=np.int64)
        self.rewards = np.zeros(band_count, dtype=float)
        self.local_t = 0
        self.rng = np.random.default_rng()

    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.counts.fill(0)
        self.rewards.fill(0.0)
        self.local_t = 0

    def select_action(self, observation_history, current_time_slot: int) -> int:
        if self.local_t < self.band_count:
            return self.local_t % self.band_count
        means = self.rewards / np.maximum(self.counts, 1)
        scores = np.full(self.band_count, -np.inf)
        visited = self.counts > 0
        scores[visited] = means[visited] + np.sqrt(
            2.0 * np.log(self.local_t + 1.0) / self.counts[visited]
        )
        scores += self.rng.uniform(0.0, 1e-12, self.band_count)
        return int(np.argmax(scores))

    def update(self, action: int, observation: dict):
        self.counts[action] += 1
        self.rewards[action] += float(bool(observation.get("hit", False)))
        self.local_t += 1


class WhittleInspiredScheduler(BaseScheduler):
    def __init__(self, band_count: int, lambda_param: float = 0.3):
        super().__init__(band_count, provenance_note="Whittle-Inspired")
        self.band_count = band_count
        self.lambda_param = lambda_param
        self.belief = np.full(band_count, 0.5, dtype=float)
        self.local_t = 0
        self.rng = np.random.default_rng()

    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.belief.fill(0.5)
        self.local_t = 0

    def select_action(self, observation_history, current_time_slot: int) -> int:
        if self.local_t < self.band_count:
            return self.local_t % self.band_count
        scores = self.belief * (1.0 - self.lambda_param)
        scores += self.rng.uniform(0.0, 1e-12, self.band_count)
        return int(np.argmax(scores))

    def update(self, action: int, observation: dict):
        hit = bool(observation.get("hit", False))
        prior = self.belief[action]
        # CORRECTED: Match simulator's Pd/Pfa
        pd_assumed, pfa_assumed = 0.9, 0.01
        if hit:
            lh1, lh0 = pd_assumed, pfa_assumed
        else:
            lh1, lh0 = 1.0 - pd_assumed, 1.0 - pfa_assumed
        numerator = lh1 * prior
        denominator = numerator + lh0 * (1.0 - prior)
        self.belief[action] = np.clip(
            numerator / denominator if denominator > 1e-12 else 0.5,
            0.01,
            0.99,
        )
        self.local_t += 1


class RLessUCBScheduler(BaseScheduler):
    def __init__(self, band_count: int, alpha: float = 1.5):
        super().__init__(band_count, provenance_note="RLessUCB")
        self.band_count = band_count
        self.alpha = alpha
        self.counts = np.zeros(band_count, dtype=np.int64)
        self.rewards = np.zeros(band_count, dtype=float)
        self.last_visit = np.full(band_count, -np.inf, dtype=float)
        self.local_t = 0
        self.rng = np.random.default_rng()

    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.counts.fill(0)
        self.rewards.fill(0.0)
        self.last_visit.fill(-np.inf)
        self.local_t = 0

    def select_action(self, observation_history, current_time_slot: int) -> int:
        if self.local_t < self.band_count:
            return self.local_t % self.band_count
        unvisited = np.flatnonzero(self.counts == 0)
        if len(unvisited):
            return int(unvisited[0])
        means = self.rewards / self.counts
        elapsed = np.maximum(0.0, current_time_slot - self.last_visit)
        scores = means + np.sqrt(self.alpha * np.log(elapsed + 1.0) / self.counts)
        scores += self.rng.uniform(0.0, 1e-12, self.band_count)
        return int(np.argmax(scores))

    def update(self, action: int, observation: dict):
        self.counts[action] += 1
        self.rewards[action] += float(bool(observation.get("hit", False)))
        self.last_visit[action] = float(observation.get("global_t", self.local_t))
        self.local_t += 1


class ARPScheduler(BaseScheduler):
    def __init__(self, band_count: int, wA: float = 0.6, wR: float = 0.25, wU: float = 0.15):
        super().__init__(band_count, provenance_note="ARP heuristic")
        self.band_count = band_count
        self.wA, self.wR, self.wU = wA, wR, wU
        self.hit_history = {b: [] for b in range(band_count)}
        self.ema_hit_rate = np.full(band_count, 0.5, dtype=float)
        self.recency = np.zeros(band_count, dtype=float)
        self.uncertainty = np.full(band_count, 0.5, dtype=float)
        self.local_t = 0
        self.rng = np.random.default_rng()

    def reset(self, seed: int, scenario_config=None):
        self.rng = np.random.default_rng(seed)
        self.hit_history = {b: [] for b in range(self.band_count)}
        self.ema_hit_rate.fill(0.5)
        self.recency.fill(0.0)
        self.uncertainty.fill(0.5)
        self.local_t = 0

    def select_action(self, observation_history, current_time_slot: int) -> int:
        if self.local_t < self.band_count:
            return self.local_t % self.band_count
        scores = self.wA * self.ema_hit_rate + self.wR * self.recency + self.wU * self.uncertainty
        scores += self.rng.uniform(0.0, 1e-12, self.band_count)
        return int(np.argmax(scores))

    def update(self, action: int, observation: dict):
        hit = float(bool(observation.get("hit", False)))
        self.hit_history[action].append(hit)
        if len(self.hit_history[action]) > 40:
            self.hit_history[action] = self.hit_history[action][-40:]
        self.ema_hit_rate[action] = 0.3 * hit + 0.7 * self.ema_hit_rate[action]
        self.recency[action] = 1.0 if hit else self.recency[action] * 0.98
        for band in range(self.band_count):
            history = self.hit_history[band][-20:]
            if len(history) > 5:
                self.uncertainty[band] = float(np.std(history))
        self.local_t += 1


EXPERT_CLASSES = {
    "UCB1": UCB1Scheduler,
    "Whittle-Inspired": WhittleInspiredScheduler,
    "RLessUCB": RLessUCBScheduler,
    "ARP": ARPScheduler,
}


# =============================================================================
# SIMULATOR
# =============================================================================

def make_sim_config():
    return SimulationConfig(
        band_count=BAND_COUNT,
        time_slots=HORIZON,
        dwell_time_ms=50.0,
        receiver_ibw_mhz=500.0,
        total_spectrum_mhz=18000.0,
    )


# =============================================================================
# TRUTH GRID EXTRACTION (CORRECTED)
# =============================================================================

def extract_truth_matrix(hidden_truth):
    """
    Extract truth from HiddenTruthGrid object.
    
    Returns:
        truth[t, band] - 2-D boolean array
    """
    # Handle HiddenTruthGrid object
    if hasattr(hidden_truth, 'grid'):
        truth = hidden_truth.grid
    else:
        truth = np.asarray(hidden_truth)
    
    # Handle different dimensionalities
    if truth.ndim == 3:
        # Shape might be [1, time, band] or [time, band, 1]
        if truth.shape[0] == 1:
            truth = truth[0]  # Remove singleton dimension
        elif truth.shape[-1] == 1:
            truth = truth[:, :, 0]  # Remove last singleton dimension
        else:
            # Take first slice along last dimension
            truth = truth[:, :, 0]
    
    if truth.ndim == 2:
        if truth.shape == (HORIZON, BAND_COUNT):
            return truth.astype(bool)
        if truth.shape == (BAND_COUNT, HORIZON):
            return truth.T.astype(bool)
    
    raise ValueError(
        f"Unsupported hidden_truth shape after extraction: {truth.shape}. "
        f"Expected (HORIZON={HORIZON}, BAND_COUNT={BAND_COUNT}) or "
        f"(BAND_COUNT={BAND_COUNT}, HORIZON={HORIZON})."
    )


# =============================================================================
# EXPECTED REWARD MODEL
# =============================================================================

def expected_reward_matrix(hidden_truth):
    """
    Construct:

        E[reward | time, band]

    Active band:
        Pd

    Inactive band:
        Pfa
    """

    truth = extract_truth_matrix(hidden_truth)

    return np.where(
        truth,
        PD,
        PFA,
    ).astype(np.float64)


def oracle_expected_reward_series(hidden_truth):
    """
    Truth-aware oracle.

    At each time t:

        r*_t = max_b E[reward | t, b]
    """

    reward_matrix = expected_reward_matrix(hidden_truth)

    return np.max(
        reward_matrix,
        axis=1,
    )


# =============================================================================
# SINGLE EXPERT EPISODE
# =============================================================================

def run_expert_episode(
    expert_name,
    episode_file,
    scan_file,
    seed,
):
    """
    Run one base expert and return EXPECTED reward trajectories.

    Returns:
        oracle_expected_rewards
        expert_expected_rewards
        selected_actions
        realized_rewards
    """

    env = TSRDStareEnvironment.from_stare_mode(
        stare_file=str(episode_file),
        scan_file=str(scan_file),
        sim_config=make_sim_config(),
    )

    env.reset(
        seed=seed
    )

    scheduler = EXPERT_CLASSES[
        expert_name
    ](
        band_count=BAND_COUNT
    )

    scheduler.reset(
        seed=seed
    )

    expected_rewards = (
        expected_reward_matrix(
            env.hidden_truth
        )
    )

    oracle_rewards = np.max(
        expected_rewards,
        axis=1,
    )

    expert_expected_rewards = []
    selected_actions = []
    realized_rewards = []

    obs_history = []

    for t in range(HORIZON):

        action = scheduler.select_action(
            obs_history,
            t
        )

        if not (
            0 <= action < BAND_COUNT
        ):
            raise RuntimeError(
                f"Expert {expert_name} produced "
                f"invalid action {action}."
            )

        step_result = env.step(
            action
        )

        obs = (
            step_result[0]
            if isinstance(step_result, tuple)
            else step_result
        )

        obs = dict(obs)

        obs["band"] = int(action)
        obs["global_t"] = int(t)

        # -------------------------------------------------------------
        # EXPECTED reward of the band actually selected by the expert.
        #
        # This is what pseudo-regret needs.
        # -------------------------------------------------------------

        expert_expected_rewards.append(
            expected_rewards[t, action]
        )

        # Realized reward retained only as a diagnostic.
        realized_rewards.append(
            float(
                bool(
                    obs.get(
                        "hit",
                        False
                    )
                )
            )
        )

        selected_actions.append(
            int(action)
        )

        obs_history.append(
            obs
        )

        scheduler.update(
            action,
            obs
        )

    return (
        np.asarray(
            oracle_rewards,
            dtype=np.float64
        ),

        np.asarray(
            expert_expected_rewards,
            dtype=np.float64
        ),

        np.asarray(
            selected_actions,
            dtype=np.int64
        ),

        np.asarray(
            realized_rewards,
            dtype=np.float64
        ),
    )


# =============================================================================
# EMPIRICAL REGRET ENVELOPES
# =============================================================================

def calculate_envelope(
    regret_curve,
    alpha,
):
    """
    For a fixed alpha construct the smallest empirical envelope:

        empirical_C(alpha) * n^alpha

    that covers the selected regret curve.

    Note:
    This DOES NOT select alpha.
    It only calculates empirical_C for a PRESELECTED alpha.
    """

    regret_curve = np.asarray(
        regret_curve,
        dtype=np.float64
    )

    # Defensive check: cumulative regret should be >= 0
    if np.any(regret_curve < -1e-12):
        raise RuntimeError(
            f"Negative cumulative regret detected: "
            f"min={regret_curve.min():.8e}"
        )

    regret_curve = np.maximum(regret_curve, 0.0)

    n = np.arange(
        1,
        len(regret_curve) + 1,
        dtype=np.float64,
    )

    ratios = (
        regret_curve
        / np.power(n, alpha)
    )

    empirical_C = float(
        np.max(ratios)
    )

    envelope = (
        empirical_C
        * np.power(
            n,
            alpha
        )
    )

    covered = bool(
        np.all(
            regret_curve
            <= envelope + 1e-12
        )
    )

    return {
        "alpha": float(alpha),
        "empirical_C": empirical_C,
        "covered": covered,
        "final_regret": float(
            regret_curve[-1]
        ),
        "final_envelope": float(
            envelope[-1]
        ),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 90)
    print(
        "VYAPTI — BASE-EXPERT ORACLE-RELATIVE "
        "REGRET ANALYSIS V3 (FROZEN CORRECTED)"
    )
    print("=" * 90)

    print()
    print(
        "Expected-reward pseudo-regret:"
    )

    print(
        "    oracle expected reward"
        " - expert expected reward"
    )

    print()
    print(
        f"Pd  = {PD}"
    )

    print(
        f"Pfa = {PFA}"
    )

    print(
        f"Horizon = {HORIZON}"
    )

    # -------------------------------------------------------------------------
    # DOWNLOAD TSRD
    # -------------------------------------------------------------------------

    snapshot_path = snapshot_download(
        repo_id=(
            "alan-turing-institute/"
            "turing-synthetic-radar-dataset"
        ),
        repo_type="dataset",
        revision=(
            "68a07b0e0189c5b4ec748c4b66dedfe26f8f1c51"
        ),
        cache_dir=(
            "/root/.cache/huggingface/hub"
        ),
    )

    root = Path(
        snapshot_path
    )

    train_stare = (
        root
        / "stare"
        / "train_stare"
    )

    train_scan = (
        root
        / "scan"
        / "train_scan"
    )

    stare_files = sorted(
        train_stare.glob(
            "config_*.h5"
        )
    )

    paired = [
        f
        for f in stare_files
        if (
            train_scan / f.name
        ).exists()
    ]

    if TRAIN_EPISODES is not None:
        paired = paired[
            :TRAIN_EPISODES
        ]

    if not paired:
        raise RuntimeError(
            "No paired training scenarios found."
        )

    print(
        f"\nTraining scenarios: {len(paired)}"
    )

    # -------------------------------------------------------------------------
    # ANALYSE EACH EXPERT
    # -------------------------------------------------------------------------

    results = {}

    for expert_name in EXPERT_NAMES:

        print()
        print("-" * 70)
        print(
            f"Analysing {expert_name}"
        )
        print("-" * 70)

        episode_regrets = []
        oracle_rewards = []
        expert_rewards = []
        realized_rewards = []

        for idx, stare_file in enumerate(
            paired
        ):

            (
                oracle,
                expert_expected,
                actions,
                realized,
            ) = run_expert_episode(
                expert_name=expert_name,
                episode_file=stare_file,
                scan_file=(
                    train_scan
                    / stare_file.name
                ),
                seed=idx,
            )

            # -------------------------------------------------------------
            # EXPECTED pseudo-regret.
            #
            # This is fundamentally different from:
            #
            #     oracle_truth - realized_hit
            #
            # which mixes deterministic truth with stochastic detection.
            # -------------------------------------------------------------

            instantaneous_regret = (
                oracle
                - expert_expected
            )

            # FAIL LOUDLY on negative instantaneous regret
            if np.any(instantaneous_regret < -1e-12):
                bad_t = int(np.argmin(instantaneous_regret))
                raise RuntimeError(
                    f"Negative instantaneous regret detected for "
                    f"{expert_name} at t={bad_t}: "
                    f"regret={instantaneous_regret[bad_t]:.8e}"
                )

            cumulative_regret = np.cumsum(
                instantaneous_regret
            )

            episode_regrets.append(
                cumulative_regret
            )

            oracle_rewards.append(
                oracle
            )

            expert_rewards.append(
                expert_expected
            )

            realized_rewards.append(
                realized
            )

            if (
                (idx + 1) % 100 == 0
                or idx + 1 == len(paired)
            ):
                print(
                    f"  {idx + 1}/{len(paired)}"
                )

        # ---------------------------------------------------------------------
        # Arrays
        # ---------------------------------------------------------------------

        regrets = np.asarray(
            episode_regrets,
            dtype=np.float64
        )

        oracle_array = np.asarray(
            oracle_rewards,
            dtype=np.float64
        )

        expert_array = np.asarray(
            expert_rewards,
            dtype=np.float64
        )

        realized_array = np.asarray(
            realized_rewards,
            dtype=np.float64
        )

        # ---------------------------------------------------------------------
        # Aggregate regret
        # ---------------------------------------------------------------------

        mean_regret_curve = (
            np.mean(
                regrets,
                axis=0
            )
        )

        selected_regret_curve = (
            np.quantile(
                regrets,
                ENVELOPE_QUANTILE,
                axis=0
            )
        )

        # ---------------------------------------------------------------------
        # Calculate empirical_C(alpha) for every alpha.
        #
        # DO NOT claim one was "selected" here.
        # ---------------------------------------------------------------------

        envelope_candidates = []

        for alpha in ALPHA_CANDIDATES:

            candidate = calculate_envelope(
                selected_regret_curve,
                alpha
            )

            envelope_candidates.append(
                candidate
            )

        # ---------------------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------------------

        final_regrets = (
            regrets[:, -1]
        )

        result = {

            "n_episodes": len(paired),

            "oracle_expected_reward_rate": float(
                np.mean(
                    oracle_array
                )
            ),

            "expert_expected_reward_rate": float(
                np.mean(
                    expert_array
                )
            ),

            "realized_detection_rate": float(
                np.mean(
                    realized_array
                )
            ),

            "mean_final_pseudoregret": float(
                np.mean(
                    final_regrets
                )
            ),

            "median_final_pseudoregret": float(
                np.median(
                    final_regrets
                )
            ),

            "max_final_pseudoregret": float(
                np.max(
                    final_regrets
                )
            ),

            "mean_regret_curve": (
                mean_regret_curve.tolist()
            ),

            "envelope_statistic": ENVELOPE_STATISTIC,

            "selected_regret_curve": (
                selected_regret_curve.tolist()
            ),

            "envelope_candidates": (
                envelope_candidates
            ),

            "parameter_status": (
                "empirical_oracle_relative_envelope"
            ),

            "theoretical_guarantee_claimed": (
                False
            ),
        }

        results[
            expert_name
        ] = result

        print()

        print(
            "Expected reward rate: "
            f"{result['expert_expected_reward_rate']:.6f}"
        )

        print(
            "Oracle reward rate:    "
            f"{result['oracle_expected_reward_rate']:.6f}"
        )

        print(
            "Realized hit rate:     "
            f"{result['realized_detection_rate']:.6f}"
        )

        print(
            "Mean final regret:     "
            f"{result['mean_final_pseudoregret']:.6f}"
        )

        print()
        print(
            "Empirical C(alpha):"
        )

        for candidate in envelope_candidates:

            print(
                f"  alpha={candidate['alpha']:.1f} "
                f" empirical_C={candidate['empirical_C']:.8f}"
            )

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    output = {

        "timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "analysis_type": (
            "Oracle-relative expected "
            "pseudo-regret"
        ),

        "training_split": (
            "stare/train_stare + scan/train_scan"
        ),

        "n_training_scenarios": (
            len(paired)
        ),

        "horizon": HORIZON,

        "band_count": BAND_COUNT,

        "reward_model": {

            "active_band_expected_reward": PD,

            "inactive_band_expected_reward": PFA,

            "description": (
                "Expected detection reward based "
                "on simulator Pd/Pfa."
            ),
        },

        "oracle_definition": (
            "Truth-aware immediate-reward oracle: "
            "at each time slot, select the band "
            "with maximum expected detection reward."
        ),

        "regret_definition": (
            "Cumulative expected pseudo-regret "
            "relative to the truth-aware oracle."
        ),

        "alpha_candidates": (
            ALPHA_CANDIDATES
        ),

        "envelope_statistic": (
            ENVELOPE_STATISTIC
        ),

        "envelope_quantile": (
            ENVELOPE_QUANTILE
        ),

        "experts": results,

        "meta_ucb_status": (
            "empirical_C(alpha) values are empirical "
            "oracle-relative envelopes. "
            "No R_i is specified here and "
            "no Cutkosky theoretical guarantee "
            "is claimed."
        ),

        "scientific_note": (
            "The Cutkosky high-probability theorem "
            "requires known putative regret bounds "
            "C_i t^alpha_i and a well-specified base "
            "algorithm. Empirical envelopes computed "
            "from TSRD trajectories do not establish "
            "those assumptions."
        ),
    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            indent=2
        )

    print()
    print("=" * 90)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 90)
    print(
        f"Saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()