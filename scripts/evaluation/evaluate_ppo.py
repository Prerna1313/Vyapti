import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from causal_harness import N_BANDS, DEVICE, load_prior, build_replay_registry
from vyapti_simulator.core.metrics import TrajectoryStep
from vyapti_simulator.tsrd.benchmark_protocol import score_recorded_replay, _assert_scorecard_consistent, _seed_for_file
from scripts.training.vyapti_ppo_components import PPOFeatureBuilder

# Need to import the model definitions from their respective files
from scripts.training.vyapti_ppo_gru_500pool import GRUActorCritic, evaluate_model as eval_gru
from scripts.training.vyapti_ppo_gtrxl_500pool import GTrXLActorCritic, evaluate_model as eval_gtrxl
from scripts.training.vyapti_ppo_lstm_500pool import LSTMActorCritic, evaluate_model as eval_lstm

TEST_FILES = 100
VAL_REPLAY_SEED = 42

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["gru", "gtrxl", "lstm"], required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--corpus-root", required=True)
    ap.add_argument("--prior-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--dwell-slots", type=int, default=2)
    ap.add_argument("--feature-mode", default="belief_periodic")
    
    args = ap.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    print(f"Loading prior from {args.prior_dir}...")
    transition, prior_active = load_prior(Path(args.prior_dir))
    
    print("Loading test registry...")
    test_registry = build_replay_registry(Path(args.corpus_root), "test", TEST_FILES)
    
    print(f"Loading model weights from {model_path}...")
    blob = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    # Infer obs dimension
    obs_dim = blob["model"]["actor.0.weight"].shape[1]
    
    if args.algo == "gru":
        print("Building GRU Actor Critic...")
        # Note: hidden dim in the original training script might be 512, check from weight shapes
        hidden_dim = blob["model"]["gru.weight_ih_l0"].shape[0] // 3
        model = GRUActorCritic(obs_dim, N_BANDS, hidden_dim).to(DEVICE)
        model.load_state_dict(blob["model"])
        test_summary = eval_gru(model, Path(args.corpus_root), test_registry, transition, prior_active, args.feature_mode, args.dwell_slots, VAL_REPLAY_SEED)
        
    elif args.algo == "gtrxl":
        print("Building GTrXL Actor Critic...")
        d_model = blob["model"]["value.0.weight"].shape[1]
        n_layers = 0
        for k in blob["model"].keys():
            if k.startswith("gtrxl.layers.") and k.endswith(".mha.q_proj.weight"):
                n_layers = max(n_layers, int(k.split(".")[2]) + 1)
        # default memory_len to 64 if we can't infer it easily
        memory_len = 64
        model = GTrXLActorCritic(obs_dim, N_BANDS, d_model=d_model, n_layers=n_layers, n_heads=8, memory_len=memory_len).to(DEVICE)
        model.load_state_dict(blob["model"])
        test_summary = eval_gtrxl(model, Path(args.corpus_root), test_registry, transition, prior_active, args.feature_mode, args.dwell_slots, VAL_REPLAY_SEED)
        
    elif args.algo == "lstm":
        print("Building LSTM Actor Critic...")
        hidden_dim = blob["model"]["lstm.weight_ih_l0"].shape[0] // 4
        model = LSTMActorCritic(obs_dim, N_BANDS, hidden_dim).to(DEVICE)
        model.load_state_dict(blob["model"])
        test_summary = eval_lstm(model, Path(args.corpus_root), test_registry, transition, prior_active, args.feature_mode, args.dwell_slots, VAL_REPLAY_SEED)
    
    save_path = output_dir / "TEST_final.json"
    with open(save_path, "w") as f:
        json.dump(test_summary, f, indent=2)
        
    print(f"\\nEvaluation complete! Saved to {save_path}")
    print(json.dumps(test_summary["summary"], indent=2))

if __name__ == "__main__":
    main()
