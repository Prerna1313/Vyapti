import argparse
import json
import sys
from pathlib import Path

# Add scripts/training to Python path so all imports resolve flawlessly
HERE = Path(__file__).resolve().parent
TRAINING_DIR = HERE.parent / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from causal_harness import N_BANDS, build_replay_registry

import vyapti_ppo_gru_500pool as gru_module
import vyapti_ppo_gtrxl_500pool as gtrxl_module
import vyapti_ppo_lstm_500pool_colab as lstm_module

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
    
    print("Loading test registry...")
    test_registry = build_replay_registry(Path(args.corpus_root), "test", TEST_FILES)
    
    print(f"Loading model weights from {model_path}...")
    blob = torch.load(model_path, map_location=DEVICE, weights_only=False)
    
    obs_dim = 544
    for k, v in blob["model"].items():
        if "encoder" in k and "weight" in k:
            if v.dim() == 2 and v.shape[1] > 500:
                obs_dim = v.shape[1]
                break
    print(f"Loading prior from {args.prior_dir}...")
    prior_path = Path(args.prior_dir) / "pormab_transition.json"
    fingerprint = json.loads(prior_path.read_text(encoding="utf-8")).get("source_pool_fingerprint", "")
    
    if args.algo == "gru":
        print("Building GRU Actor Critic...")
        transition, prior_active = gru_module.load_prior(Path(args.prior_dir), fingerprint)
        
        hidden_dim = blob["model"]["gru.weight_ih_l0"].shape[0] // 3
        model = gru_module.GRUActorCritic(obs_dim, N_BANDS, hidden_dim).to(DEVICE)
        model.load_state_dict(blob["model"])
        test_summary = gru_module.evaluate_model(model, Path(args.corpus_root), test_registry, transition, prior_active, args.feature_mode, args.dwell_slots, VAL_REPLAY_SEED)
        
    elif args.algo == "gtrxl":
        print("Building GTrXL Actor Critic...")
        transition, prior_active = gtrxl_module.load_prior(Path(args.prior_dir), fingerprint)
        
        d_model = blob["model"]["value.0.weight"].shape[1]
        n_layers = 0
        for k in blob["model"].keys():
            if k.startswith("gtrxl.layers.") and k.endswith(".mha.q_proj.weight"):
                n_layers = max(n_layers, int(k.split(".")[2]) + 1)
        memory_len = 64
        model = gtrxl_module.GTrXLActorCritic(obs_dim, N_BANDS, d_model=d_model, n_layers=n_layers, n_heads=8, memory_len=memory_len).to(DEVICE)
        model.load_state_dict(blob["model"])
        test_summary = gtrxl_module.evaluate_model(model, Path(args.corpus_root), test_registry, transition, prior_active, args.feature_mode, args.dwell_slots, VAL_REPLAY_SEED)
        
    elif args.algo == "lstm":
        print("Building LSTM Actor Critic...")
        transition, prior_active = lstm_module.load_prior(Path(args.prior_dir), fingerprint)
        
        hidden_dim = blob["model"]["lstm.weight_ih_l0"].shape[0] // 4
        model = lstm_module.LSTMActorCritic(obs_dim, N_BANDS, hidden_dim).to(DEVICE)
        model.load_state_dict(blob["model"])
        test_summary = lstm_module.evaluate_model(model, Path(args.corpus_root), test_registry, transition, prior_active, args.feature_mode, args.dwell_slots, VAL_REPLAY_SEED)
    
    save_path = output_dir / "TEST_final.json"
    with open(save_path, "w") as f:
        json.dump(test_summary, f, indent=2)
        
    print(f"\\nEvaluation complete! Saved to {save_path}")
    print(json.dumps(test_summary["summary"], indent=2))

if __name__ == "__main__":
    main()
