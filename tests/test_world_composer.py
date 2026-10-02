"""Mode-B composition invariants, independent of receiver implementation."""
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from vyapti_simulator.system_b.tsrd.world_composer import EmitterRealization, WorldComposer


def realization(source, label=0):
    return EmitterRealization(f"train:{source}:{label}", source, label, f"{source}.h5",
                              np.array([[0, 1000, 2, 30, -80], [12, 2500, 3, 31, -81],
                                        [39, 1000, 2, 32, -82]], dtype=float),
                              type_id="recorded type", operating_mode=None)


def test_source_dedup_is_enforced_even_with_many_emitters_per_source():
    entries = [realization("a", i) for i in range(50)] + [realization("b")]
    composer = WorldComposer()
    for seed in range(20):
        selected = composer.select(entries, lambda e: e.source_config_id, seed=seed, count=2)
        assert {e.source_config_id for e in selected} == {"a", "b"}
    with pytest.raises(ValueError, match="distinct source"):
        composer.select(entries, lambda e: e.source_config_id, seed=0, count=3)
    with pytest.raises(ValueError, match="one emitter"):
        composer.compose(entries[:2], seed=0)


def test_rephasing_keeps_full_agile_trace_and_relative_timing():
    emitters = [realization("a"), realization("b")]
    composer = WorldComposer(time_offset_us=100)
    data, labels, audit = composer.compose(emitters, seed=8)
    repeated = composer.compose(emitters, seed=8)
    np.testing.assert_array_equal(data, repeated[0])
    assert audit == repeated[2]
    assert np.all(np.diff(data[:, 0]) >= 0)
    assert len(data) == 6  # No crop, even when ToA leaves the mission window.
    assert audit[0]["time_offset_us"] != audit[1]["time_offset_us"]
    for i, emitter in enumerate(emitters):
        placed = data[labels == i]
        np.testing.assert_array_equal(placed[:, 1:], emitter.pdws[:, 1:])
        np.testing.assert_array_equal(placed[:, 0] - audit[i]["time_offset_us"], emitter.pdws[:, 0])
        np.testing.assert_array_equal(np.diff(placed[:, 0]), np.diff(emitter.pdws[:, 0]))
    assert composer.compose(emitters, seed=9)[2] != audit


def test_realization_is_closed_and_immutable():
    emitter = realization("a")
    with pytest.raises(FrozenInstanceError):
        emitter.type_id = "changed"
    with pytest.raises(ValueError):
        emitter.pdws.setflags(write=True)
    with pytest.raises(ValueError):
        emitter.pdws[0, 1] = 999
    with pytest.raises(ValueError, match="sorted"):
        EmitterRealization("x", "a", 0, "a.h5", emitter.pdws[::-1])
    with pytest.raises(ValueError):
        WorldComposer(time_offset_us=-1)
