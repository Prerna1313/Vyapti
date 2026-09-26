import random
from typing import List, Dict, Any

class ExplainableAILogger:
    """
    Cognitive EW Tactical Logger.
    Translates the mathematical decisions of the AI scheduler into natural language
    military intelligence logs to provide Explainable AI (XAI) for commanders.
    """

    def __init__(self):
        self._history = []

    def generate_log(self, receiver_id: int, selected_band: int, features: Dict[str, Any]) -> str:
        """
        Generates a human-readable justification for why the AI chose a specific band,
        based on the statistical tensors output by the BandFeatureExtractor.
        """
        # Pull features
        hit_prob = features.get("hit_probability", 0.0)
        revisit_time = features.get("avg_revisit_time", 0.0)
        time_since_last = features.get("time_since_last_hit", 0.0)

        # Rule-based Explainable AI Logic
        reasoning = ""

        if hit_prob > 0.8:
            reasoning = f"Tracking high-probability target (Pd={hit_prob:.2f})."
        elif time_since_last > (revisit_time * 1.5) and revisit_time > 0:
            reasoning = f"Predicted intercept window active (Historical revisit={revisit_time:.1f} slots)."
        elif hit_prob == 0.0:
            reasoning = "Executing wide-area background search for unknown emitters."
        else:
            reasoning = f"Investigating anomalies (Time since last hit={time_since_last:.1f} slots)."

        log_entry = f"[TACTICAL AI] Receiver {receiver_id} tuned to Band {selected_band}. Justification: {reasoning}"
        self._history.append(log_entry)
        return log_entry

    def get_recent_logs(self, count: int = 5) -> List[str]:
        return self._history[-count:]
