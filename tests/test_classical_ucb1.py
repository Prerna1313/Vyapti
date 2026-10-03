"""Research baseline checks for classical UCB1 on the Mode-B v2 reward."""
import math

import pytest

from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState, PublicTransition
from vyapti_simulator.system_b.tsrd.policies.classical_ucb1 import ClassicalUCB1


def _transition(action, reward, slot=0):
    return PublicTransition(PublicState(slot, None), action, float(reward),
                            PublicState(slot + 1, None), False)


def test_ucb1_forces_first_visit_then_uses_scaled_environment_reward():
    policy = ClassicalUCB1(3, seed=7)
    state = PublicState(0, None)
    actions = []
    # UCB1 forces one sample of every band, in deterministic order.
    for slot, reward in enumerate((-0.01, 1.0, 0.495)):
        action = policy.select_action(state, training=False)
        actions.append(action)
        policy.observe(_transition(action, reward, slot), training=False)
    assert actions == [0, 1, 2]
    assert policy.counts.tolist() == [1, 1, 1]
    assert policy.reward_sums.tolist() == pytest.approx([0.0, 1.0, 0.5])
    assert policy.select_action(state, training=False) == 1
    diagnostics = policy.end_episode(training=False)
    assert diagnostics["raw_reward_means"] == pytest.approx([-0.01, 1.0, 0.495])


def test_ucb1_resets_between_worlds_and_rejects_out_of_bounds_rewards():
    policy = ClassicalUCB1(2, seed=1, exploration_coefficient=math.sqrt(2))
    policy.observe(_transition(0, 0.5), training=False)
    policy.reset_episode(training=False)
    assert policy.counts.tolist() == [0, 0]
    assert policy.select_action(PublicState(0, None), training=False) == 0
    with pytest.raises(ValueError, match="outside the frozen UCB1 scaling interval"):
        policy.observe(_transition(0, -0.010001), training=False)


def test_ucb1_factory_rejects_checkpoint_state():
    from vyapti_simulator.system_b.tsrd.policies.classical_ucb1 import create

    with pytest.raises(ValueError, match="does not load checkpoints"):
        create(bands=2, seed=5, settings={}, checkpoint=object())
