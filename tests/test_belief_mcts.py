from pathlib import Path

import numpy as np

from vyapti_simulator.core.receiver_observation import ReceiverMetadata, ReceiverObservation
from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState, PublicTransition, create_algorithm
from vyapti_simulator.system_b.tsrd.policies.belief_mcts import (
    BASE_SLOTS,
    BayesianRewardModel,
    BeliefMCTS,
    BeliefState,
    CausalPeriodicity,
    CONTEXT_DIM,
    DEFAULT_PERIODICITY_HISTORY,
    DEFAULT_PERIODICITY_TOLERANCE,
    MCTSNode,
    N_BANDS,
    TwoStateHMMBelief,
    VyaptiBeliefMCTSPolicy,
)
from vyapti_simulator.system_b.tsrd.training_setup import resolve_setup


PROJECT = Path(__file__).parents[1]


def _resolved_setup():
    return resolve_setup(
        PROJECT / "training_setup/environments/train250_composed.json",
        PROJECT / "training_setup/algorithms/belief_mcts.json",
        seed=19,
        episodes=1,
        checkpoint_every=1,
    )


def _belief_state(elapsed_slots=0):
    belief = TwoStateHMMBelief.from_train250()
    dwell = np.array([2, 2] + [1] * 34, dtype=np.int64)
    return BeliefState(
        belief=belief,
        periodicity=CausalPeriodicity(
            DEFAULT_PERIODICITY_HISTORY,
            DEFAULT_PERIODICITY_TOLERANCE,
        ),
        native_dwell_slots=dwell,
        last_visit_slot=np.full(N_BANDS, -1, dtype=np.int32),
        elapsed_slots=elapsed_slots,
    )


def test_belief_mcts_uses_resolved_train250_calibration_and_environment_dwells():
    setup = _resolved_setup()
    assert setup["algorithm"]["settings"]["mcts_simulations"] == 128
    assert setup["algorithm"]["settings"]["mcts_horizon"] == 4
    assert setup["algorithm"]["settings"]["mcts_uct_c"] == 1.25
    assert setup["algorithm"]["settings"]["prior_active_probability"] == 0.3426563254340138
    assert setup["algorithm"]["settings"]["inactive_to_active_probability"] == 0.022195484398616423
    assert setup["algorithm"]["settings"]["active_to_active_probability"] == 0.9575318615765981

    setup["algorithm"]["settings"].update(mcts_simulations=4, mcts_horizon=2)
    policy = create_algorithm(setup, bands=36, seed=19)
    assert isinstance(policy, VyaptiBeliefMCTSPolicy)
    np.testing.assert_array_equal(policy.native_dwell_slots, setup["action"]["dwell_slots_by_band"])


def test_rollout_uses_a_copy_of_the_persistent_tree_child():
    state = _belief_state()
    reward_model = BayesianRewardModel(dimension=CONTEXT_DIM)
    planner = BeliefMCTS(reward_model, np.random.default_rng(4), simulations=1, horizon=2)
    rollout_inputs = []
    original_rollout = planner._rollout

    def tracked_rollout(rollout_state, remaining_depth):
        rollout_inputs.append(rollout_state)
        return original_rollout(rollout_state, remaining_depth)

    planner._rollout = tracked_rollout
    root = MCTSNode(state=state.clone(), depth=0)
    planner._simulate(root)

    edge = root.children[0]
    tree_child = edge.hit_child or edge.miss_child
    assert rollout_inputs and rollout_inputs[0] is not tree_child.state
    assert tree_child.state.elapsed_slots == 2
    assert rollout_inputs[0].elapsed_slots > tree_child.state.elapsed_slots
    assert state.elapsed_slots == 0


def test_terminal_dwell_is_clipped_to_remaining_mission_slot():
    state = _belief_state(elapsed_slots=BASE_SLOTS - 1)
    contexts = state.observe_contexts()
    assert contexts[0, 44] == 0.5  # effective 1-slot dwell, normalized by 2

    state.commit_dwell(action=0, aggregate_hit=False, dwell_slots=1)
    assert state.elapsed_slots == BASE_SLOTS

    planner = BeliefMCTS(BayesianRewardModel(CONTEXT_DIM), np.random.default_rng(7), simulations=1, horizon=2)
    terminal_root = MCTSNode(state=_belief_state(elapsed_slots=BASE_SLOTS - 1), depth=0)
    planner._simulate(terminal_root)
    terminal_edge = terminal_root.children[0]
    terminal_child = terminal_edge.hit_child or terminal_edge.miss_child
    assert terminal_child.state.elapsed_slots == BASE_SLOTS


def test_vector_contexts_match_original_per_band_feature_formula():
    state = _belief_state(elapsed_slots=117)
    state.belief.belief = np.linspace(0.03, 0.97, N_BANDS)
    state.last_visit_slot[:] = np.arange(N_BANDS) * 3
    state.previous_action = 11
    for band, times in ((0, [10, 30, 50, 70]), (11, [12, 31, 49, 72])):
        for time_slot in times:
            state.periodicity.observe(band, time_slot, True)

    actual = state.observe_contexts()
    expected = np.zeros((N_BANDS, CONTEXT_DIM), dtype=np.float64)
    period_score, period_conf = state.periodicity.features(state.elapsed_slots)
    remaining = np.clip(1.0 - state.elapsed_slots / BASE_SLOTS, 0.0, 1.0)
    for band in range(N_BANDS):
        dwell = min(int(state.native_dwell_slots[band]), BASE_SLOTS - state.elapsed_slots)
        expected[band] = [
            1.0,
            *[float(i == band) for i in range(N_BANDS)],
            state.belief.belief[band],
            state.belief.predict_band(band, dwell),
            state.belief.aggregate_hit_probability(band, dwell),
            period_score[band],
            period_conf[band],
            1.0 if state.last_visit_slot[band] < 0 else np.clip(
                (state.elapsed_slots - state.last_visit_slot[band]) / BASE_SLOTS, 0.0, 1.0
            ),
            remaining,
            dwell / 2.0,
            float(state.previous_action >= 0 and state.previous_action != band),
            state.candidate_visit_count[band] / max(state.total_decisions, 1),
            state.candidate_hit_count[band] / max(state.candidate_visit_count[band], 1),
            state.candidate_hit_count[band] / max(state.total_decisions, 1),
            np.clip((state.elapsed_slots - state.last_hit_slot[band]) / BASE_SLOTS, 0.0, 1.0)
            if state.last_hit_slot[band] >= 0 else 1.0,
            np.clip((state.elapsed_slots - state.last_visit_slot[band]) / BASE_SLOTS, 0.0, 1.0)
            if state.last_visit_slot[band] >= 0 else 1.0,
            state.total_decisions / BASE_SLOTS,
            state.elapsed_slots / BASE_SLOTS,
            state.cumulative_observed_reward / max(state.total_decisions, 1),
            state.total_observed_hits / max(state.total_decisions, 1),
        ]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_policy_consumes_shared_reward_and_public_transition():
    setup = _resolved_setup()
    setup["algorithm"]["settings"].update(mcts_simulations=4, mcts_horizon=2, warmup_actions=36)
    policy = create_algorithm(setup, bands=36, seed=19)
    policy.reset_episode(training=True)
    policy.total_actions = policy.warmup_actions

    state = PublicState(0, None)
    action = policy.select_action(state, training=True)
    dwell = int(policy.native_dwell_slots[action])
    obs = ReceiverObservation(
        time_slot=dwell - 1,
        selected_band=action,
        hit=True,
        retune_cost_s=0.0003,
        dwell_elapsed_s=dwell * 0.05,
        receiver_metadata=ReceiverMetadata(
            dwell_time_ms=dwell * 50.0,
            retune_time_ms=0.3,
        ),
    )
    transition = PublicTransition(
        state, action, 0.123, PublicState(dwell, obs), False
    )
    policy.observe(transition, training=True)
    diagnostics = policy.end_episode(training=True)

    assert diagnostics["reward_model_updates"] == 1
    assert diagnostics["episode_reward"] == 0.123
    assert diagnostics["planning_calls"] == 1
    assert diagnostics["planning_seconds_total"] > 0.0
    assert policy.state.cumulative_observed_reward == 0.123


def test_history_updates_are_causal_and_clone_is_independent():
    state = _belief_state()
    state.commit_dwell(action=0, aggregate_hit=True, dwell_slots=2, observed_reward=0.25)
    assert state.candidate_visit_count[0] == 1
    assert state.candidate_hit_count[0] == 1
    assert state.last_visit_slot[0] == 0
    assert state.last_hit_slot[0] == 1
    assert state.total_observed_hits == state.total_decisions == 1
    assert state.cumulative_observed_reward == 0.25
    clone = state.clone()
    clone.candidate_visit_count[0] += 1
    clone.last_hit_slot[0] = 50
    assert state.candidate_visit_count[0] == 1
    assert state.last_hit_slot[0] == 1


def test_mcts_explicitly_represents_both_weighted_observation_branches():
    state = _belief_state()
    planner = BeliefMCTS(
        BayesianRewardModel(CONTEXT_DIM), np.random.default_rng(13), simulations=1, horizon=1
    )
    root = MCTSNode(state=state.clone(), depth=0)
    planner._simulate(root)
    edge = root.children[0]
    assert edge.hit_child is not None
    assert edge.miss_child is not None
    assert edge.hit_child.state.candidate_hit_count[0] == 1
    assert edge.miss_child.state.candidate_hit_count[0] == 0
    assert edge.hit_child.state.candidate_visit_count[0] == 1
    assert edge.miss_child.state.candidate_visit_count[0] == 1


def test_expansion_order_uses_each_nodes_own_history_context():
    state = _belief_state()
    state.commit_dwell(action=0, aggregate_hit=False, dwell_slots=2)
    planner = BeliefMCTS(
        BayesianRewardModel(CONTEXT_DIM), np.random.default_rng(17), simulations=1, horizon=1
    )
    planner.reward_weights[:] = 0.0
    planner.reward_weights[50] = 1.0  # normalized time since candidate's last visit
    node = MCTSNode(state=state, depth=1)
    assert planner._select_action(node) != 0
