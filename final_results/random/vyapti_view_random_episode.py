#!/usr/bin/env python3
"""
View a random episode's complete metrics
"""

import json
import random
from pathlib import Path

# Load all episodes
output_dir = Path("vyapti_results")
all_file = output_dir / "rr_all_episodes_all_metrics.json"

print(f"Loading: {all_file}")
with open(all_file, 'r') as f:
    all_episodes = json.load(f)

print(f"Loaded {len(all_episodes)} episodes\n")

# Pick random episode
random_ep = random.choice(all_episodes)
ep_num = random_ep['episode']
config = random_ep['config']
metrics = random_ep['all_metrics']

print("="*80)
print(f"RANDOM EPISODE: {ep_num} ({config})")
print("="*80)

# Print all metric categories
print("\n📊 DETECTION:")
det = metrics.get('comprehensive_detection', {})
for key, val in det.items():
    print(f"  {key}: {val}")

print("\n📊 COVERAGE:")
cov = metrics.get('comprehensive_coverage', {})
for key, val in cov.items():
    print(f"  {key}: {val}")

print("\n📊 SCHEDULER:")
sched = metrics.get('comprehensive_scheduler', {})
for key, val in sched.items():
    print(f"  {key}: {val}")

print("\n📊 EFFICIENCY:")
eff = metrics.get('comprehensive_efficiency', {})
for key, val in eff.items():
    print(f"  {key}: {val}")

print("\n📊 LATENCY:")
lat = metrics.get('comprehensive_latency', {})
for key, val in lat.items():
    print(f"  {key}: {val}")

print("\n📊 DISCOVERY:")
disc = metrics.get('discovery_metrics', {})
for key, val in disc.items():
    print(f"  {key}: {val}")

print("\n📊 DETECTION (raw):")
det_raw = metrics.get('detection_metrics', {})
for key, val in det_raw.items():
    print(f"  {key}: {val}")

print("\n📊 THREAT ASSESSMENT:")
threat = metrics.get('threat_assessment', {})
for key, val in threat.items():
    print(f"  {key}: {val}")

print("\n📊 SPATIAL:")
spatial = metrics.get('spatial_analysis', {})
for key, val in spatial.items():
    print(f"  {key}: {val}")

print("\n📊 SPECTRAL:")
spectral = metrics.get('spectral_environment', {})
for key, val in spectral.items():
    print(f"  {key}: {val}")

print("\n📊 EMITTER POPULATION:")
pop = metrics.get('emitter_population', {})
for key, val in pop.items():
    print(f"  {key}: {val}")

print("\n📊 REWARD:")
reward = metrics.get('reward_composite', {})
for key, val in reward.items():
    print(f"  {key}: {val}")

print("\n📊 PER-BAND (first 5 bands):")
per_band = metrics.get('per_band_metrics', {})
for i in range(min(5, len(per_band))):
    print(f"  Band {i}: {per_band.get(str(i), 'N/A')}")

print("\n📊 TEMPORAL:")
temporal = metrics.get('temporal_metrics', {})
for key, val in temporal.items():
    print(f"  {key}: {val}")

print("\n" + "="*80)
print(f"Total metric categories: {len(metrics)}")
print(f"Total keys: {sum(len(v) if isinstance(v, dict) else 1 for v in metrics.values())}")
print("="*80)