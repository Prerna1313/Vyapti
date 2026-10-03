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
    assert policy.ts.dimension == CONTEXT_DIM == 46
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
    assert policy.state.periodicity.events[action] == [float(dwell - 1)]


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
    finally:
        checkpoint.unlink(missing_ok=True)
