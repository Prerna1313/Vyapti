from pathlib import Path

from vyapti_simulator.core.receiver_observation import ReceiverMetadata, ReceiverObservation
from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState, PublicTransition, create_algorithm
from vyapti_simulator.system_b.tsrd.training_setup import resolve_setup


def _observation(slot, band, hit):
    return ReceiverObservation(
        time_slot=slot, selected_band=band, hit=hit,
        retune_cost_s=0.0003, dwell_elapsed_s=0.0997,
        receiver_metadata=ReceiverMetadata(dwell_time_ms=100.0, retune_time_ms=0.3),
    )


def test_belief_ucb_uses_causal_aggregate_observation_and_requested_exploration():
    project = Path(__file__).parents[1]
    setup = resolve_setup(
        project / "training_setup/environments/train250_composed.json",
        project / "training_setup/algorithms/belief_ucb.json",
        seed=3, episodes=0, checkpoint_every=1, execution_mode="online_baseline",
    )
    policy = create_algorithm(setup, bands=36, seed=3)
    assert policy.exploration_c == 0.25
    policy.reset_episode(training=False)

    state = PublicState(0, None)
    action = policy.select_action(state, training=False)
    assert action == 0  # deterministic all-band warm start
    next_state = PublicState(2, _observation(1, action, True))
    policy.observe(PublicTransition(state, action, 0.7, next_state, False), training=False)
    assert policy.visits[0] == 1
    assert policy.belief.belief_active[0] > 0.5

    # The algorithm advances on receiver evidence and does not use reward.
    next_action = policy.select_action(next_state, training=False)
    assert next_action == 1
    end_state = PublicState(3, _observation(2, next_action, False))
    policy.observe(PublicTransition(next_state, next_action, -0.2, end_state, False), training=False)
    diagnostics = policy.end_episode(training=False)
    assert diagnostics["updates_from_reward"] is False
    assert diagnostics["adapts_from_receiver_observations"] is True


def test_belief_ucb_warm_start_false_uses_belief_and_dwell_rate():
    project = Path(__file__).parents[1]
    setup = resolve_setup(
        project / "training_setup/environments/train250_composed.json",
        project / "training_setup/algorithms/belief_ucb.json",
        seed=3, episodes=0, checkpoint_every=1, execution_mode="online_baseline",
    )
    setup["algorithm"]["settings"]["warm_start_all_bands"] = False
    setup["algorithm"]["settings"]["exploration_c"] = 0.0
    # Band 0's 100 ms look has lower expected hit rate than a 50 ms look.
    setup["action"]["dwell_slots_by_band"] = [2] + [1] * 35
    policy = create_algorithm(setup, bands=36, seed=3)
    policy.reset_episode(training=False)
    policy.belief.belief_active[5] = 0.9

    action = policy.select_action(PublicState(0, None), training=False)
    assert action == 5


def test_belief_ucb_normalizes_expected_detections_by_dwell_time():
    project = Path(__file__).parents[1]
    setup = resolve_setup(
        project / "training_setup/environments/train250_composed.json",
        project / "training_setup/algorithms/belief_ucb.json",
        seed=3, episodes=0, checkpoint_every=1, execution_mode="online_baseline",
    )
    setup["algorithm"]["settings"]["warm_start_all_bands"] = False
    setup["algorithm"]["settings"]["exploration_c"] = 0.0
    setup["action"]["dwell_slots_by_band"] = [2] + [1] * 35
    policy = create_algorithm(setup, bands=36, seed=3)
    policy.reset_episode(training=False)

    # With equal beliefs, the one-slot bands have a higher predicted hit rate
    # per elapsed second than band 0's two-slot dwell.
    action = policy.select_action(PublicState(0, None), training=False)
    assert action == 1
