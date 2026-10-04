import json
import math
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from vyapti_simulator.core.receiver_observation import ReceiverMetadata, ReceiverObservation
from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState, PublicTransition, create_algorithm
from vyapti_simulator.system_b.tsrd.checkpoints import freeze_selection
from vyapti_simulator.system_b.tsrd.experiment import evaluate, plot_run, train
from vyapti_simulator.system_b.tsrd.training_setup import resolve_setup
from tests.test_train250_cache import _small_pool


def _receiver_observation(slot, band, hit=False):
    return ReceiverObservation(
        time_slot=slot, selected_band=band, hit=hit,
        retune_cost_s=0.0003 if band else 0.0, dwell_elapsed_s=0.0497,
        receiver_metadata=ReceiverMetadata(dwell_time_ms=50.0, retune_time_ms=0.3),
    )


def _config():
    return {
        "action": {"dwell_slots_by_band": [1, 2] * 18},
        "algorithm": {
            "name": "ppo_lstm",
            "module": "vyapti_simulator.system_b.tsrd.policies.ppo_lstm",
            "api": "public_transitions",
            "settings": {
                "device": "cpu", "hidden_units": 16, "ppo_epochs": 1,
                "bptt_chunk": 2, "sequence_minibatch": 2,
                "periodicity_history": 8, "training_actions_target": 4,
            },
        }
    }


def test_ppo_lstm_public_transition_update_and_checkpoint_roundtrip(tmp_path):
    config = _config()
    policy = create_algorithm(config, bands=36, seed=13)
    assert policy.observation_dim == 253
    assert not hasattr(policy, "band_reward_ema")
    assert not hasattr(policy, "previous_reward")
    assert policy.native_dwell_feature.tolist() == [0.5, 1.0] * 18
    policy.reset_episode(training=True)
    state = PublicState(0, None)
    action = policy.select_action(state, training=True)
    next_state = PublicState(1, _receiver_observation(0, action, hit=True))
    policy.observe(PublicTransition(state, action, 0.125, next_state, False), training=True)
    features = policy._features(1)
    assert features.shape == (253,)
    assert features[181 + action] == policy.native_dwell_feature[action]
    assert features[217 + action] == 0.0
    assert policy._rewards == [0.125]
    diagnostics = policy.end_episode(training=True)
    assert diagnostics["training_updates"] == 1
    assert diagnostics["episode_decisions"] == 1
    assert diagnostics["band_selection_counts"][action] == 1
    assert math.isfinite(diagnostics["mean_loss"])
    assert math.isfinite(diagnostics["mean_kl"])
    assert diagnostics["mean_learning_rate_fraction"] == 0.75

    policy.reset_episode(training=True)
    state = PublicState(0, None)
    action = policy.select_action(state, training=True)
    next_state = PublicState(1, _receiver_observation(0, action, hit=False))
    policy.observe(PublicTransition(state, action, 0.0, next_state, False), training=True)
    next_diagnostics = policy.end_episode(training=True)
    assert next_diagnostics["mean_learning_rate_fraction"] == 0.5

    checkpoint = tmp_path / "ppo_lstm.pt"
    policy.save(checkpoint)
    restored = create_algorithm(config, bands=36, seed=13, checkpoint=checkpoint)
    assert restored.total_actions == policy.total_actions
    assert restored.total_updates == policy.total_updates
    assert restored.completed_training_episodes == 2
    assert restored.select_action(PublicState(0, None), training=False) in range(36)


def test_ppo_lstm_runs_shared_train_val_test_and_plot_pipeline(tmp_path):
    data, cache = _small_pool(tmp_path)
    fixture = Path(__file__).parent / "fixtures/tsrd/config_0_stare.h5"
    for split in ("val", "test"):
        target = data / "stare" / f"{split}_stare" / "config_9.h5"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture, target)

    project = Path(__file__).parents[1]
    setup = resolve_setup(
        project / "training_setup/environments/train250_source_replay.json",
        project / "training_setup/algorithms/ppo_lstm.json", seed=37,
        episodes=1, checkpoint_every=1,
    )
    setup.update(data_root=str(data), cache_root=str(cache), expected_train_configs=2,
                 train_episodes=1, checkpoint_file="final.pt")
    setup["algorithm"]["settings"].update(
        device="cpu", hidden_units=16, ppo_epochs=1, bptt_chunk=64,
        sequence_minibatch=2, periodicity_history=8,
    )
    setup["evaluation"].update(
        val_config_ids=["config_9"], expected_val_configs=1,
        test_config_ids=["config_9"], expected_test_configs=1,
    )
    run = tmp_path / "runs/ppo_lstm"

    train(setup, run)
    val_report = evaluate(run, "val", condition="normal")
    assert val_report["checkpoint_file"] == "final.pt"
    assert val_report["summary"]["worlds"] == 1
    assert (run / "train/algorithm_updates.jsonl").is_file()
    freeze_selection(run, "final.pt", reason="Fixture PPO-LSTM VAL selection")

    test_report = evaluate(run, "test", final=True, condition="all")
    assert set(test_report["conditions"]) == {"normal", "beam_periodic", "beam_stochastic"}
    for condition in test_report["conditions"]:
        row = json.loads((run / "eval/test" / condition / "per_world.jsonl")
                         .read_text().splitlines()[0])
        assert row["policy_diagnostics"]["algorithm"] == "ppo_lstm"
    plot_run(run)
    assert (run / "plots/train_ppo_lstm_metrics.png").is_file()
    assert (run / "plots/test_ppo_lstm_band_counts.png").is_file()
