from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vyapti_simulator.core.receiver_observation import ReceiverMetadata, ReceiverObservation
from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState, PublicTransition
from vyapti_simulator.system_b.tsrd.policies import discrete_sac_belief_periodic as periodic


def _policy(**overrides):
    settings = {
        "device": "cpu",
        "hidden_units": 16,
        "replay_capacity": 16,
        "batch_size": 2,
        "warmup_actions": 1,
        "min_replay_size": 2,
        "gamma_per_base_slot": 0.99,
        "updates_per_action": 1,
        "update_every_actions": 1,
    }
    settings.update(overrides)
    return periodic.DiscreteSACBeliefPeriodicityPolicy(
        bands=periodic.ACTION_DIM, seed=7, settings=settings
    )


def _observation(slot, band):
    return ReceiverObservation(
        time_slot=slot,
        selected_band=band,
        hit=False,
        retune_cost_s=0.0003,
        dwell_elapsed_s=0.0497,
        receiver_metadata=ReceiverMetadata(dwell_time_ms=50.0, retune_time_ms=0.3),
    )


def test_replay_sampling_runs_gradient_update_after_warmup():
    policy = _policy()
    policy.reset_episode(training=True)
    state = PublicState(0, None)

    for slot in range(2):
        action = policy.select_action(state, training=True)
        next_state = PublicState(slot + 1, _observation(slot, action))
        policy.observe(
            PublicTransition(state, action, 0.25, next_state, False), training=True
        )
        state = next_state

    assert len(policy.replay) == 2
    assert policy.total_updates == 1


def test_alpha_increases_when_entropy_is_below_target():
    agent = periodic.DiscreteSAC(
        periodic.OBS_DIM, periodic.ACTION_DIM, 11,
        {"hidden_units": 16}, torch.device("cpu"),
    )
    alpha_before = float(agent.alpha.detach())
    entropy = torch.full((4,), agent.target_entropy - 0.5)

    agent._update_temperature(entropy)

    assert float(agent.alpha.detach()) > alpha_before


def test_discount_uses_elapsed_slots_and_not_retune_time():
    policy = _policy()
    before = PublicState(5, _observation(4, 2))
    after = PublicState(7, _observation(6, 8))  # band change plus two slots
    transition = PublicTransition(before, 8, 0.0, after, False)

    assert policy._transition_discount(transition) == pytest.approx(0.99 ** 2)
    assert periodic._actual_gamma(2) == pytest.approx(periodic.GAMMA_BASE ** 2)


def test_standalone_loader_accepts_shared_train250_calibration():
    calibration_path = (Path(__file__).parents[1]
                        / "training_setup/belief_models/train250_observed_band_hmm.json")
    belief = periodic.TwoStateHMMBelief.from_json(calibration_path)

    assert belief.prior[0] == pytest.approx(0.3426563254340138)
    assert belief.transition[0, 0, 1] == pytest.approx(0.022195484398616423)
    assert belief.transition[0, 1, 1] == pytest.approx(0.9575318615765981)
