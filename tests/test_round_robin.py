import json
import shutil
from pathlib import Path

import pytest

from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState, PublicTransition
from vyapti_simulator.system_b.tsrd.policies.round_robin import RoundRobinPolicy, create
from vyapti_simulator.system_b.tsrd.checkpoints import freeze_selection
from vyapti_simulator.system_b.tsrd.experiment import evaluate, plot_run, train
from vyapti_simulator.system_b.tsrd.training_setup import resolve_setup
from tests.test_train250_cache import _small_pool


def _transition(action, slot):
    return PublicTransition(PublicState(slot, None), action, 0.25,
                            PublicState(slot + 1, None), False)


def test_round_robin_cycles_and_resets_for_each_world():
    policy = RoundRobinPolicy(4, seed=23)
    actions = []
    for slot in range(6):
        action = policy.select_action(PublicState(slot, None), training=False)
        actions.append(action)
        policy.observe(_transition(action, slot), training=False)
    assert actions == [0, 1, 2, 3, 0, 1]
    diagnostics = policy.end_episode(training=False)
    assert diagnostics["band_selection_counts"] == [2, 2, 1, 1]
    assert diagnostics["total_decisions"] == 6
    assert diagnostics["total_reward"] == pytest.approx(1.5)

    policy.reset_episode(training=False)
    assert policy.select_action(PublicState(0, None), training=False) == 0


def test_round_robin_factory_rejects_checkpoints_and_bad_start_band():
    with pytest.raises(ValueError, match="has no checkpoints"):
        create(bands=4, seed=1, settings={}, checkpoint=object())
    with pytest.raises(ValueError, match="valid band index"):
        RoundRobinPolicy(4, seed=1, start_band=4)


def test_round_robin_runs_shared_val_test_pipeline_and_saves_its_plots(tmp_path):
    data, cache = _small_pool(tmp_path)
    fixture = Path(__file__).parent / "fixtures/tsrd/config_0_stare.h5"
    for split in ("val", "test"):
        target = data / "stare" / f"{split}_stare" / "config_9.h5"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(fixture, target)

    project = Path(__file__).parents[1]
    setup = resolve_setup(
        project / "training_setup/environments/train250_source_replay.json",
        project / "training_setup/algorithms/round_robin.json",
        seed=31, episodes=0, checkpoint_every=1, execution_mode="online_baseline",
    )
    setup.update(data_root=str(data), cache_root=str(cache), expected_train_configs=2)
    setup["evaluation"].update(
        val_config_ids=["config_9"], expected_val_configs=1,
        test_config_ids=["config_9"], expected_test_configs=1,
    )
    run = tmp_path / "runs/round_robin"

    train(setup, run)
    val_report = evaluate(run, "val", condition="normal")
    assert val_report["summary"]["worlds"] == 1
    selection = freeze_selection(run, baseline=True)
    assert selection["metric"] == "VAL_NORMAL.opportunity_interception_ratio"

    test_report = evaluate(run, "test", final=True, condition="all")
    assert set(test_report["conditions"]) == {"normal", "beam_periodic", "beam_stochastic"}
    for condition in test_report["conditions"]:
        world_row = json.loads((run / "eval/test" / condition / "per_world.jsonl")
                               .read_text().splitlines()[0])
        assert world_row["policy_diagnostics"]["algorithm"] == "round_robin"
        assert sum(world_row["policy_diagnostics"]["band_selection_counts"]) == \
            world_row["policy_diagnostics"]["total_decisions"]

    plot_run(run)
    assert (run / "plots/test_round_robin_band_counts.png").is_file()
    assert not (run / "checkpoints").exists()
