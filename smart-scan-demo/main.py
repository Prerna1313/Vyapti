# =============================================================================
# PS26055 — ADVANCED SCHEDULER INTEGRATION PROTOTYPE (ROBUST, HONEST NAMING)
# =============================================================================
#
# main.py — CLI entry point
# --------------------------
# This preserves the ORIGINAL main.py's standalone behavior exactly:
# running it (`python main.py`) still downloads/loads the TSRD dataset,
# runs a 20-episode evaluation of AdvancedSchedulerPrototype, prints the
# same progress lines, and writes the same
# advanced_scheduler_prototype_tsrd_robust_results.json.
#
# What's different from the original single-file main.py is only WHERE
# the code lives:
#   - the algorithm classes now live in scheduler_core.py (byte-identical
#     to what was in the original main.py)
#   - the TSRD dataset download/scan now lives in dataset.py, and is
#     called explicitly below, instead of running at import time
#
# This means `import main` (or `import scheduler_core`) no longer
# triggers a HuggingFace download as a side effect — the FastAPI backend
# imports scheduler_core.py directly and only calls
# dataset.load_tsrd_dataset() when a simulation is actually started.
#
# Nothing about the scheduling algorithm, the evaluation loop, or the
# output format has changed.

import json
import time
from datetime import datetime, timezone
import numpy as np

from scheduler_core import (
    AdvancedSchedulerPrototype,
    SIM_CONFIG,
    BEST_DETECTION_CONFIG,
    run_episode_safe,
)
from dataset import load_tsrd_dataset
from vyapti_simulator.tsrd import TSRDEnvironment

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("🎬 RUNNING ADVANCED SCHEDULER PROTOTYPE (ROBUST, REAL TSRD)")
    print(f"{'='*80}")

    VALID_EPISODES = load_tsrd_dataset()

    all_pd = []
    all_dwell = []
    episode_details = []
    t0 = time.time()
    n_target_episodes = 20

    episode_index = 0
    max_episode_index = len(VALID_EPISODES) - 1
    consecutive_failures = 0
    max_consecutive_failures = 100

    while len(all_pd) < n_target_episodes and episode_index <= max_episode_index and consecutive_failures < max_consecutive_failures:
        try:
            result = VALID_EPISODES[episode_index]
            env = TSRDEnvironment(
                pdw_stream=result.pdw_stream,
                simulation_config=SIM_CONFIG,
                detection_config=BEST_DETECTION_CONFIG,
            )
            scheduler = AdvancedSchedulerPrototype(band_count=36)
            pd, metrics = run_episode_safe(scheduler, env, seed=episode_index)

            all_pd.append(pd)
            all_dwell.append(metrics['avg_dwell'])
            episode_details.append({
                'episode_index': episode_index,
                'pd': pd,
                'avg_dwell': metrics['avg_dwell'],
            })
            consecutive_failures = 0

        except Exception as e:
            print(f"⚠️  Episode index {episode_index} failed: {e}")
            consecutive_failures += 1

        episode_index += 1

        if len(all_pd) % 5 == 0 and len(all_pd) > 0:
            elapsed = time.time() - t0
            last = episode_details[-1]
            print(
                f"  Successful episodes: {len(all_pd)}/{n_target_episodes}, "
                f"last: ep_index={last['episode_index']}, Pd={last['pd']:.3f}, "
                f"Avg Dwell={last['avg_dwell']:.1f}ms [{elapsed:.1f}s]"
            )

    total_elapsed = time.time() - t0
    if len(all_pd) == 0:
        raise RuntimeError("No successful episodes to report.")

    mean_pd = float(np.mean(all_pd))
    std_pd = float(np.std(all_pd, ddof=1))
    mean_dwell = float(np.mean(all_dwell))

    print(f"\n{'='*80}")
    print(f"✅ EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"   Successful episodes: {len(all_pd)}")
    print(f"   Mean Pd: {mean_pd:.3f} ± {std_pd:.3f}")
    print(f"   Mean Dwell: {mean_dwell:.1f}ms")
    print(f"   Total time: {total_elapsed:.1f}s")

    with open('advanced_scheduler_prototype_tsrd_robust_results.json', 'w') as f:
        json.dump(
            {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'mean_pd': mean_pd,
                'std_pd': std_pd,
                'mean_dwell': mean_dwell,
                'n_episodes': len(all_pd),
                'episode_details': episode_details,
                'total_elapsed_s': total_elapsed,
            },
            f,
            indent=2,
        )

    print(f"\n✅ Saved to advanced_scheduler_prototype_tsrd_robust_results.json")
