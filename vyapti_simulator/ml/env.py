import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Tuple

# Import the core physics engine and configs
from vyapti_simulator.core.environment import SimulationConfig, RFEnvironment
from vyapti_simulator.core.scenario_registry import ScenarioRegistry

class VyaptiRFEnv(gym.Env):
    """
    Standard OpenAI Gymnasium wrapper for the Vyapti High-Fidelity RF Physics Engine.
    This exposes the simulator natively to ML frameworks like RLlib and Stable Baselines 3.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config: SimulationConfig = None, scenario_name: str = "TSRD_Baseline"):
        super().__init__()
        
        self.config = config or SimulationConfig()
        registry = ScenarioRegistry(self.config)
        self.scenario = registry.get_scenario(scenario_name)
        self.env = RFEnvironment(self.config, self.scenario)
        
        # Action space: select one of the available frequency bands to scan
        self.action_space = spaces.Discrete(self.config.band_count)
        
        # Observation space: binary hit (0 or 1), or could be expanded to PSD features
        # We start with a simple Box for the observation dictionary
        self.observation_space = spaces.Dict({
            "band_idx": spaces.Discrete(self.config.band_count),
            "hit": spaces.Discrete(2),
            "snr_estimate": spaces.Box(low=-100.0, high=100.0, shape=(1,), dtype=np.float32)
        })

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.config.scenario_seed = seed
            
        self.env.reset(seed=self.config.scenario_seed)
        
        # Initial dummy observation
        obs = {
            "band_idx": 0,
            "hit": 0,
            "snr_estimate": np.array([-100.0], dtype=np.float32)
        }
        return obs, {}

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        obs_dict, hit = self.env.step(action)
        
        obs = {
            "band_idx": action,
            "hit": int(hit),
            "snr_estimate": np.array([obs_dict.get("snr", -100.0)], dtype=np.float32)
        }
        
        reward = 1.0 if hit else 0.0
        terminated = self.env.done()
        truncated = False
        
        return obs, reward, terminated, truncated, obs_dict

    def render(self):
        # Could integrate matplotlib or Plotly here for 'human' rendering
        pass
