import numpy as np
from typing import List, Dict, Any

class BandFeatureExtractor:
    \"\"\"
    A dedicated ML pipeline component that translates raw physical RF observations
    (hits, snr, timestamps) into rich statistical tensors for Reinforcement Learning agents.
    \"\"\"
    def __init__(self, band_count: int, history_window: int = 50):
        self.band_count = band_count
        self.history_window = history_window
        
        # History buffers
        self.hit_history = [[] for _ in range(band_count)]
        self.time_history = [[] for _ in range(band_count)]
        
    def update(self, band_idx: int, timestamp: int, hit: bool):
        self.hit_history[band_idx].append(hit)
        self.time_history[band_idx].append(timestamp)
        
        # Truncate to window
        if len(self.hit_history[band_idx]) > self.history_window:
            self.hit_history[band_idx].pop(0)
            self.time_history[band_idx].pop(0)
            
    def get_features(self) -> np.ndarray:
        \"\"\"
        Returns a normalized tensor shape (band_count, 3):
        [hit_probability, avg_revisit_time, recent_activity]
        \"\"\"
        features = np.zeros((self.band_count, 3), dtype=np.float32)
        
        for b in range(self.band_count):
            hits = self.hit_history[b]
            times = self.time_history[b]
            
            if len(hits) == 0:
                continue
                
            # Feature 1: Hit probability
            features[b, 0] = sum(hits) / len(hits)
            
            # Feature 2: Avg Revisit time (normalized)
            if len(times) > 1:
                intervals = np.diff(times)
                features[b, 1] = np.mean(intervals) / 100.0  # arbitrary normalization
                
            # Feature 3: Recent Activity (exponential decay)
            if hits[-1]:
                features[b, 2] = 1.0
                
        return features
