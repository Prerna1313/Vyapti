from pathlib import Path
from uuid import uuid4

import numpy as np

from vyapti_simulator.core.receiver_observation import ReceiverMetadata, ReceiverObservation
from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState, PublicTransition, create_algorithm
from vyapti_simulator.system_b.tsrd.policies.contextual_thompson import (
    BASE_SLOTS,
    CONTEXT_DIM,
    CausalState,
    CausalPeriodicity,
    LinearGaussianThompsonSampler,
    TwoStateHMMBelief,
)
from vyapti_simulator.system_b.tsrd.prediction_evaluation import collect_prediction
from vyapti_simulator.system_b.tsrd.training_setup import resolve_setup


PROJECT = Path(__file__).parents[1]


def _setup():
    return resolve_setup(
        PROJECT / "training_setup/environments/train250_composed.json",
        PROJECT / "training_setup/algorithms/contextual_thompson.json",
        seed=29,
        episodes=1,
        checkpoint_every=1,
    )


def test_contextual_thompson_is_registered_with_train250_calibration_and_dwells():
    setup = _setup()
    policy = create_algorithm(setup, bands=36, seed=29)
    assert policy.ALGORITHM_NAME == "contextual_thompson_belief_periodic"
    assert policy.ts.dimension == CONTEXT_DIM == 50
    assert setup["algorithm"]["settings"]["thompson_sampling_scale"] == 1.0
    assert policy.settings["inactive_to_active_probability"] == setup["belief_model_calibration"]["inactive_to_active_probability"]
    np.testing.assert_array_equal(policy.native_dwell_slots, setup["action"]["dwell_slots_by_band"])


def test_shared_transition_reward_updates_posterior_and_hit_history():
    setup = _setup()
    policy = create_algorithm(setup, bands=36, seed=29)
    start = PublicState(0, None)
    action = policy.select_action(start, training=True)
    dwell = int(policy.native_dwell_slots[action])
    observation = ReceiverObservation(
        time_slot=dwell - 1,
        selected_band=action,
        hit=True,
        retune_cost_s=0.0003,
        dwell_elapsed_s=dwell * 0.05,
        receiver_metadata=ReceiverMetadata(dwell_time_ms=dwell * 50.0, retune_time_ms=0.3),
    )
    policy.observe(
        PublicTransition(start, action, 0.123, PublicState(dwell, observation), False),
        training=True,
    )
    assert policy.total_updates == 1
    assert policy.episode_reward == 0.123
    assert policy.end_episode(training=True)["policy_seed"] == 29
    assert policy.state.periodicity.events[action] == [float(dwell - 1)]


def test_periodicity_time_forecast_is_logged_without_claiming_interception():
    setup = _setup()
    policy = create_algorithm(setup, bands=36, seed=33)
    policy.state.elapsed_slots = 25
    policy.state.periodicity.events[4] = [0.0, 10.0, 20.0]
    public_state = PublicState(25, None)
    assert policy.predict_observed_hit_recurrence_eta(public_state) == 0.25
    prediction = collect_prediction(
        policy, public_state, bands=36
    )
    assert prediction["target_slot"] == 25
    assert prediction["predicted_observed_hit_recurrence_eta_s"] == 0.25
    assert "next_intercept_slot" not in prediction


def test_terminal_partial_native_dwell_is_clipped_consistently():
    setup = _setup()
    belief = TwoStateHMMBelief.from_train250()
    dwell = np.asarray(setup["action"]["dwell_slots_by_band"], dtype=np.int64)
    state = CausalState(
        belief=belief,
        periodicity=CausalPeriodicity(),
        native_dwell_slots=dwell,
        last_visit_slot=np.full(36, -1, dtype=np.int32),
        elapsed_slots=BASE_SLOTS - 1,
    )
    contexts = state.observe_contexts()
    band = int(np.flatnonzero(dwell == 2)[0])
    assert contexts[band, 44] == 0.5
    state.commit_dwell(band, False, 1)
    assert state.elapsed_slots == BASE_SLOTS


def test_deadline_interactions_vary_by_candidate_while_remaining_is_global():
    setup = _setup()
    dwell = np.asarray(setup["action"]["dwell_slots_by_band"], dtype=np.int64)
    state = CausalState(
        belief=TwoStateHMMBelief.from_train250(),
        periodicity=CausalPeriodicity(),
        native_dwell_slots=dwell,
        last_visit_slot=np.arange(36, dtype=np.int32) * 3,
        elapsed_slots=117,
        previous_action=11,
    )
    state.belief.belief = np.linspace(0.03, 0.97, 36)
    contexts = state.observe_contexts()
    remaining = 1.0 - state.elapsed_slots / BASE_SLOTS
    np.testing.assert_allclose(contexts[:, 43], remaining)
    np.testing.assert_allclose(contexts[:, 46], contexts[:, 38] * remaining)
    np.testing.assert_allclose(contexts[:, 47], contexts[:, 39] * remaining)
    np.testing.assert_allclose(contexts[:, 48], contexts[:, 42] * remaining)
    np.testing.assert_allclose(contexts[:, 49], contexts[:, 40] * remaining)
    assert np.ptp(contexts[:, 46]) > 0.0


def test_linear_gaussian_posterior_and_cholesky_updates_match_closed_form():
    model = LinearGaussianThompsonSampler(
        dimension=CONTEXT_DIM,
        prior_precision=1.0,
        sampling_scale=0.05,
        reward_noise_std=0.01,
    )
    rng = np.random.default_rng(11)
    for _ in range(8):
        context = rng.normal(size=CONTEXT_DIM)
        reward = float(rng.normal())
        model.update(context, reward)
    np.testing.assert_allclose(model.L @ model.L.T, model.A, rtol=1e-11, atol=1e-8)
    np.testing.assert_allclose(model.posterior_mean(), np.linalg.solve(model.A, model.b), rtol=1e-10, atol=1e-10)


def test_checkpoint_roundtrip_preserves_posterior_and_rng():
    setup = _setup()
    policy = create_algorithm(setup, bands=36, seed=31)
    policy.rng.random(7)
    checkpoint = PROJECT / "runs" / f"contextual-thompson-test-{uuid4().hex}.pt"
    try:
        policy.save(checkpoint)
        restored = create_algorithm(setup, bands=36, seed=999, checkpoint=checkpoint)
        np.testing.assert_array_equal(restored.ts.A, policy.ts.A)
        np.testing.assert_array_equal(restored.ts.b, policy.ts.b)
        np.testing.assert_array_equal(restored.ts.L, policy.ts.L)
        assert restored.total_actions == policy.total_actions
        np.testing.assert_array_equal(
            restored.ts.sample_theta(restored.rng),
            policy.ts.sample_theta(policy.rng),
        )
        fresh_eval = create_algorithm(
            setup, bands=36, seed=999, checkpoint=checkpoint, restore_rng=False
        )
        expected_rng = np.random.default_rng(999)
        expected = fresh_eval.ts.sample_theta(expected_rng)
        actual = fresh_eval.ts.sample_theta(fresh_eval.rng)
        np.testing.assert_array_equal(actual, expected)
    finally:
        checkpoint.unlink(missing_ok=True)


def _observed_hits(*, emitter_period, receiver_period, emitter_phase=0,
                   receiver_phase=0, seed=0, horizon=512, mode="periodic",
                   jitter=0, pd=0.9, pfa=0.05):
    rng = np.random.default_rng(seed)
    if receiver_period is None:
        visits = np.flatnonzero(rng.random(horizon) < 0.3).tolist()
    else:
        visits = []
        slot = receiver_phase
        while slot < horizon:
            perturbed = int(np.clip(slot + (rng.integers(-jitter, jitter + 1) if jitter else 0),
                                    0, horizon - 1))
            if not visits or perturbed > visits[-1]:
                visits.append(perturbed)
            slot += receiver_period
    hits = []
    for slot in visits:
        if mode == "constant":
            active = True
        elif mode == "random":
            active = bool(rng.random() < 0.3)
        elif mode == "empty":
            active = False
        else:
            active = (slot - emitter_phase) % emitter_period == 0
        if rng.random() < (pd if active else pfa):
            hits.append(slot)
    return visits, hits


def _cue(hits, now=512):
    estimator = CausalPeriodicity(max_events=128, tolerance_fraction=0.15)
    for slot in hits:
        estimator.observe(0, slot, True)
    return estimator.estimate(0, now)


def test_constant_illumination_with_periodic_revisits_can_look_periodic():
    visits, hits = _observed_hits(
        emitter_period=None, receiver_period=10, mode="constant", pd=1.0, pfa=0.0
    )
    assert hits == visits
    score, confidence = _cue(hits, now=visits[-1] + 10)
    assert score > 0.99 and confidence == 1.0


def test_missed_detections_keep_observed_recurrence_cue_finite():
    _, hits = _observed_hits(
        emitter_period=16, receiver_period=4, seed=18, pd=0.55, pfa=0.0
    )
    score, confidence = _cue(hits)
    assert len(hits) < 512 // 16
    assert np.isfinite(score) and np.isfinite(confidence)
    assert 0.0 <= score <= 1.0 and 0.0 <= confidence <= 1.0


def test_false_alarm_only_hits_produce_bounded_recurrence_cue():
    _, hits = _observed_hits(
        emitter_period=None, receiver_period=3, mode="empty", seed=7, pfa=0.35
    )
    score, confidence = _cue(hits)
    assert np.isfinite(score) and np.isfinite(confidence)
    assert 0.0 <= score <= 1.0 and 0.0 <= confidence <= 1.0


def test_jittered_receiver_revisits_are_not_labeled_as_exact_periods():
    _, hits = _observed_hits(
        emitter_period=None, receiver_period=10, mode="constant", jitter=2, pd=1.0, pfa=0.0
    )
    score, confidence = _cue(hits)
    assert np.isfinite(score) and np.isfinite(confidence)
    assert 0.0 <= score <= 1.0 and 0.0 <= confidence <= 1.0
    assert score < 1.0


def test_periodic_emitter_and_periodic_receiver_recurrence():
    _, hits = _observed_hits(
        emitter_period=16, receiver_period=8, emitter_phase=0, receiver_phase=0,
        pd=1.0, pfa=0.0,
    )
    score, confidence = _cue(hits)
    assert len(hits) > 3
    assert score > 0.99 and confidence == 1.0


def test_periodic_emitter_with_randomized_receiver_visits_is_bounded():
    _, hits = _observed_hits(
        emitter_period=16, receiver_period=None, seed=23, mode="periodic", pd=1.0, pfa=0.0
    )
    score, confidence = _cue(hits)
    assert np.isfinite(score) and np.isfinite(confidence)
    assert 0.0 <= score <= 1.0 and 0.0 <= confidence <= 1.0


def test_randomized_emitter_with_periodic_receiver_is_bounded():
    _, hits = _observed_hits(
        emitter_period=None, receiver_period=8, mode="random", seed=29
    )
    score, confidence = _cue(hits)
    assert np.isfinite(score) and np.isfinite(confidence)
    assert 0.0 <= score <= 1.0 and 0.0 <= confidence <= 1.0


def test_randomized_emitter_and_randomized_receiver_is_bounded():
    _, hits = _observed_hits(
        emitter_period=None, receiver_period=None, mode="random", seed=31
    )
    score, confidence = _cue(hits)
    assert np.isfinite(score) and np.isfinite(confidence)
    assert 0.0 <= score <= 1.0 and 0.0 <= confidence <= 1.0


def test_randomized_initial_phase_changes_observability_under_periodic_revisits():
    _, aligned = _observed_hits(
        emitter_period=16, receiver_period=8, emitter_phase=0, receiver_phase=0,
        pd=1.0, pfa=0.0,
    )
    _, offset = _observed_hits(
        emitter_period=16, receiver_period=8, emitter_phase=4, receiver_phase=0,
        pd=1.0, pfa=0.0,
    )
    assert len(aligned) > len(offset)
