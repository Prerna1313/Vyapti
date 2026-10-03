import pytest
import json
import shutil
import itertools
from pathlib import Path

torch = pytest.importorskip("torch")

from vyapti_simulator.core.receiver_observation import ReceiverMetadata, ReceiverObservation
from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState, PublicTransition, create_algorithm
from tests.test_train250_cache import _small_pool
from vyapti_simulator.system_b.tsrd.experiment import evaluate, plot_run, train
from vyapti_simulator.system_b.tsrd.training_setup import resolve_setup


def _transition(slot, band, *, hit=False, done=False):
    before = PublicState(slot, None if slot == 0 else _observation(slot - 1, band, hit))
    after = PublicState(slot + 1, _observation(slot, band, hit))
    return before, after


def _observation(slot, band, hit):
    return ReceiverObservation(
        time_slot=slot,
        selected_band=band,
        hit=hit,
        retune_cost_s=0.0003 if band else 0.0,
        dwell_elapsed_s=0.05,
        receiver_metadata=ReceiverMetadata(dwell_time_ms=50.0, retune_time_ms=0.3),
    )


def test_discrete_sac_consumes_public_reward_and_roundtrips_checkpoint(tmp_path):
    config = {
        "contract_version": "train250_recorded_pdw_v2",
        "algorithm": {
            "name": "discrete_sac",
            "module": "vyapti_simulator.system_b.tsrd.policies.discrete_sac",
            "api": "public_transitions",
            "settings": {
                "device": "cpu", "hidden_units": 16, "replay_capacity": 16,
                "batch_size": 2, "warmup_actions": 0, "detection_probability": 0.9,
                "false_alarm_probability": 0.05,
            },
        },
    }
    policy = create_algorithm(config, bands=4, seed=7)
    policy.reset_episode(training=True)
    for slot in range(3):
        state, next_state = _transition(slot, band=slot % 4, hit=(slot == 1), done=(slot == 2))
        action = policy.select_action(state, training=True)
        # The transition's action must match the policy decision; its scalar
        # reward comes from the shared Mode-B world runner.
        _, next_state = _transition(slot, band=action, hit=(slot == 1), done=(slot == 2))
        policy.observe(PublicTransition(state, action, 0.01, next_state, slot == 2), training=True)
    diagnostics = policy.end_episode(training=True)
    assert diagnostics["total_training_actions"] == 3
    assert diagnostics["training_updates"] > 0
    assert diagnostics["replay_size"] == 3
    assert sum(diagnostics["band_selection_counts"]) == 3

    checkpoint = tmp_path / "sac.pt"
    policy.save(checkpoint)
    restored = create_algorithm(config, bands=4, seed=7, checkpoint=checkpoint)
    assert restored.total_actions == policy.total_actions
    assert restored.total_updates == policy.total_updates
    assert restored.select_action(PublicState(0, None), training=False) in range(4)


def test_discrete_sac_validates_detector_operating_point():
    config = {
        "algorithm": {
            "name": "discrete_sac",
            "module": "vyapti_simulator.system_b.tsrd.policies.discrete_sac",
            "api": "public_transitions",
            "settings": {"device": "cpu", "detection_probability": 0.05,
                         "false_alarm_probability": 0.9},
        }
    }
    with pytest.raises(ValueError, match="Pfa must be lower"):
        create_algorithm(config, bands=4, seed=1)


def test_discrete_sac_temperature_increases_alpha_below_target_entropy():
    config = {
        "algorithm": {
            "name": "discrete_sac",
            "module": "vyapti_simulator.system_b.tsrd.policies.discrete_sac",
            "api": "public_transitions",
            "settings": {"device": "cpu", "hidden_units": 16},
        }
    }
    policy = create_algorithm(config, bands=4, seed=3)
    alpha_before = float(policy.alpha.detach())
    entropy_below_target = torch.full((4,), policy.target_entropy - 0.5)
    policy._update_temperature(entropy_below_target)
    assert float(policy.alpha.detach()) > alpha_before


def test_discrete_sac_discount_uses_environment_slot_clock_only():
    config = {
        "algorithm": {
            "name": "discrete_sac",
            "module": "vyapti_simulator.system_b.tsrd.policies.discrete_sac",
            "api": "public_transitions",
            "settings": {"device": "cpu", "hidden_units": 16, "gamma_per_base_slot": 0.99},
        }
    }
    policy = create_algorithm(config, bands=4, seed=4)
    before, after = _transition(4, band=1)
    # A 0.3 ms retune is included in the 50 ms base slot's elapsed clock.
    observation = ReceiverObservation(
        time_slot=4, selected_band=1, hit=False, retune_cost_s=0.0003,
        dwell_elapsed_s=0.0497,
        receiver_metadata=ReceiverMetadata(dwell_time_ms=50.0, retune_time_ms=0.3),
    )
    transition = PublicTransition(before, 1, 0.0, PublicState(5, observation), False)
    assert policy._elapsed_slots(transition) == 1
    assert policy._transition_discount(transition) == pytest.approx(0.99)


@pytest.mark.parametrize("aggregate_hit", [False, True])
def test_discrete_sac_two_slot_aggregate_hmm_matches_path_enumeration(aggregate_hit):
    config = {
        "algorithm": {
            "name": "discrete_sac",
            "module": "vyapti_simulator.system_b.tsrd.policies.discrete_sac",
            "api": "public_transitions",
            "settings": {
                "device": "cpu", "hidden_units": 16,
                "inactive_to_active_probability": 0.2,
                "active_to_active_probability": 0.7,
                "detection_probability": 0.8,
                "false_alarm_probability": 0.1,
            },
        }
    }
    policy = create_algorithm(config, bands=4, seed=5)
    initial_active = 0.35
    policy.belief[:] = initial_active
    observation = _observation(1, 1, aggregate_hit)
    policy._update_belief(observation, slots=2)

    # Independently enumerate (z_t, z_{t+1}, z_{t+2}) and the two hidden
    # receiver outcomes, conditioning only on their aggregate any-hit bit.
    transitions = {
        (0, 0): 1.0 - policy.p01, (0, 1): policy.p01,
        (1, 0): 1.0 - policy.p11, (1, 1): policy.p11,
    }
    emission = {0: policy.pfa, 1: policy.pd}
    weighted_terminal = [0.0, 0.0]
    for states in itertools.product((0, 1), repeat=3):
        path_probability = ((initial_active if states[0] else 1.0 - initial_active)
                            * transitions[(states[0], states[1])]
                            * transitions[(states[1], states[2])])
        for hits in itertools.product((False, True), repeat=2):
            if (any(hits) != aggregate_hit):
                continue
            probability = path_probability
            for state, hit in zip(states[:2], hits):
                p_hit = emission[state]
                probability *= p_hit if hit else 1.0 - p_hit
            weighted_terminal[states[2]] += probability
    expected_active = weighted_terminal[1] / sum(weighted_terminal)
    assert policy.belief[1] == pytest.approx(expected_active)

    # Non-selected bands are predicted through both latent transitions without
    # receiving the selected band's observation.
    expected_unobserved = initial_active
    for _ in range(2):
        expected_unobserved = ((1.0 - expected_unobserved) * policy.p01
                               + expected_unobserved * policy.p11)
    assert policy.belief[0] == pytest.approx(expected_unobserved)


def test_discrete_sac_uses_shared_train_val_runner_and_saves_scorecards(tmp_path):
    data, cache = _small_pool(tmp_path)
    fixture = Path(__file__).parent / "fixtures/tsrd/config_0_stare.h5"
    val = data / "stare/val_stare/config_9.h5"
    val.parent.mkdir(parents=True)
    shutil.copyfile(fixture, val)
    root = Path(__file__).parents[1] / "training_setup"
    config = resolve_setup(root / "environments/train250_source_replay.json",
                           root / "algorithms/discrete_sac.json", seed=29,
                           episodes=1, checkpoint_every=1)
    config.update(data_root=str(data), cache_root=str(cache), expected_train_configs=2,
                  train_episodes=1, checkpoint_file="final.pt")
    config["algorithm"]["settings"].update(device="cpu", hidden_units=16, replay_capacity=16,
                                            batch_size=2, warmup_actions=100_000)
    config["evaluation"]["val_config_ids"] = ["config_9"]
    config["evaluation"]["expected_val_configs"] = 1
    run = tmp_path / "runs/discrete_sac"
    train(config, run)
    reports = evaluate(run, "val", condition="all")
    assert set(reports["conditions"]) == {"normal", "beam_periodic", "beam_stochastic"}
    report = reports["conditions"]["normal"]
    assert report["summary"]["worlds"] == 1
    assert report["checkpoint_file"] == "final.pt"
    world_row = json.loads((run / "eval/val/normal/per_world.jsonl").read_text().splitlines()[0])
    assert world_row["scorecard"]["event_level"]
    assert world_row["policy_diagnostics"]["algorithm"] == "discrete_sac"
    assert (run / "eval/val/normal/steps.jsonl").is_file()
    assert (run / "eval/val/beam_periodic/summary.json").is_file()
    assert (run / "eval/val/beam_stochastic/summary.json").is_file()
    plots = plot_run(run)
    assert (run / "plots").is_dir()
    assert run / "plots/train_reward.png" in plots
