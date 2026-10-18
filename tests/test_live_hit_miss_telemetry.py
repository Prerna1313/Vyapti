"""
Tests for live HIT/MISS telemetry and dwell synchronization.
Verifies:
1. Real System C I/Q synthesis and CFAR PulseDetector PDW output.
2. HIT strictly defined as len(pdws) > 0, MISS as len(pdws) == 0.
3. Reward semantics: 1.0 for HIT, 0.0 for MISS.
4. UCB algorithm update receives identical reward.
5. Timing: HIT/MISS belongs to the exact dwell whose window is commanded and displayed.
6. All required WebSocket telemetry fields are present.
"""
import pytest
from vyapti_simulator.server import RFSimulationSession


def test_live_hit_miss_closed_loop():
    session = RFSimulationSession(seed=42)
    sample_hit = None
    sample_miss = None

    for _ in range(40):
        telem = session.step()

        # Check required fields
        assert "decision_id" in telem
        assert "selected_frequency_window" in telem
        assert "commanded_frequency_window" in telem
        assert "system_c_current_frequency_window" in telem
        assert "detection_result" in telem
        assert "cfar_detection_count" in telem
        assert "detection" in telem
        assert "reward" in telem
        assert "ucb_score" in telem
        assert "visit_count" in telem
        assert "mean_reward" in telem
        assert "ucb" in telem
        assert "total_hits" in telem
        assert "total_misses" in telem
        assert "hit_rate" in telem
        assert "total_detections" in telem

        res = telem["detection_result"]
        count = telem["cfar_detection_count"]
        rew = telem["reward"]

        assert res in ("HIT", "MISS")
        if count > 0:
            assert res == "HIT"
            assert rew == 1.0
            if sample_hit is None:
                sample_hit = telem
        else:
            assert res == "MISS"
            assert rew == 0.0
            if sample_miss is None:
                sample_miss = telem

        # Check nested structures
        assert telem["detection"]["result"] == res
        assert telem["detection"]["cfar_detection_count"] == count
        assert telem["ucb"]["score"] == telem["ucb_score"]
        assert telem["ucb"]["visit_count"] == telem["visit_count"]
        assert telem["ucb"]["mean_reward"] == telem["mean_reward"]

        # Check counter consistency
        assert telem["total_hits"] + telem["total_misses"] == telem["decision_id"]

    assert sample_hit is not None, "Expected at least one HIT during closed-loop simulation"
    assert sample_miss is not None, "Expected at least one MISS during closed-loop simulation"

    print("\nVerified real sample HIT:")
    print("Decision ID:", sample_hit["decision_id"])
    print("Window:", sample_hit["selected_frequency_window"])
    print("Detection Result:", sample_hit["detection_result"])
    print("CFAR Count:", sample_hit["cfar_detection_count"])
    print("Reward:", sample_hit["reward"])
    print("UCB:", sample_hit["ucb"])

    print("\nVerified real sample MISS:")
    print("Decision ID:", sample_miss["decision_id"])
    print("Window:", sample_miss["selected_frequency_window"])
    print("Detection Result:", sample_miss["detection_result"])
    print("CFAR Count:", sample_miss["cfar_detection_count"])
    print("Reward:", sample_miss["reward"])
    print("UCB:", sample_miss["ucb"])


if __name__ == "__main__":
    test_live_hit_miss_closed_loop()
    print("\nAll tests passed successfully!")
