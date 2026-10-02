"""Run and plot the System C IQ receiver sensitivity calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from vyapti_simulator.system_c.rf.calibration import (
    CFAR_OFFSET_SWEEP_DB,
    SNR_SWEEP_DB,
    run_receiver_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/system_c_receiver_calibration.json"))
    parser.add_argument("--trials", type=int, default=1000,
                        help="Monte Carlo ticks at each point; use at least 10000 for final reporting")
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--snr-db", type=float, nargs="+", default=list(SNR_SWEEP_DB))
    parser.add_argument("--cfar-db", type=float, nargs="+", default=list(CFAR_OFFSET_SWEEP_DB))
    args = parser.parse_args()

    report = run_receiver_calibration(
        trials=args.trials,
        seed=args.seed,
        snr_grid_db=args.snr_db,
        cfar_offset_grid_db=args.cfar_db,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    snr = report["pd_vs_snr"]
    axes[0].plot([row["snr_db"] for row in snr], [row["pd"] for row in snr], marker="o")
    axes[0].set(xlabel="Input SNR (dB)", ylabel="Probability of detection",
                ylim=(0, 1), title="System C detection sensitivity")
    false_alarm = report["false_alarm_vs_cfar_offset"]
    axes[1].plot([row["cfar_db"] for row in false_alarm],
                 [row["false_alarm_probability_per_tick"] for row in false_alarm], marker="o")
    axes[1].set(xlabel="CFAR threshold offset (dB)", ylabel="False-alarm probability per tick",
                ylim=(0, 1), title="Noise-only false alarms")
    fig.suptitle(f"{report['trials_per_point']} trials per calibration point")
    fig.tight_layout()
    figure_path = args.output.with_suffix(".png")
    fig.savefig(figure_path, dpi=160)
    plt.close(fig)
    print(f"Wrote {args.output} and {figure_path}")


if __name__ == "__main__":
    main()
