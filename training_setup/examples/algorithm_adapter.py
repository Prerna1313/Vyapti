"""Interface skeleton for the user's chosen implementation; not runnable as-is."""
from pathlib import Path
from vyapti_simulator.system_b.tsrd.algorithm_interface import PublicState, PublicTransition


def create(*, bands: int, seed: int, settings: dict, checkpoint: Path | None = None):
    """Construct your chosen model/optimizer and restore its checkpoint here."""
    raise NotImplementedError("Connect the actual algorithm implementation before training")


class AlgorithmAdapter:
    def reset_episode(self, *, training: bool):
        """Clear episode memory; preserve replay buffers and learned weights."""
        raise NotImplementedError

    def select_action(self, state: PublicState, *, training: bool) -> int:
        """Use exploration only when requested; return one discrete band."""
        raise NotImplementedError

    def observe(self, transition: PublicTransition, *, training: bool):
        """Store a rollout/replay transition and run your algorithm's updates."""
        raise NotImplementedError

    def end_episode(self, *, training: bool) -> dict | None:
        """Finish updates/terminal handling and optionally return diagnostics."""
        raise NotImplementedError

    def save(self, path: Path):
        """Serialize the implementation's weights and training state to this path."""
        raise NotImplementedError

    # Optional prediction hooks; remove or leave returning None if your
    # architecture has no corresponding head. Called after the last public
    # observation: state.time_slot is the next-slot forecast target.
    def predict_band_activity(self, state: PublicState):
        return None

    def predict_next_intercept_slot(self, state: PublicState):
        return None
