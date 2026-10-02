# System C receiver calibration protocol

## Purpose and scope

Measure the IQ receiver independently from scheduler performance. The primary
curve is probability of detection versus input SNR. A second curve measures
noise-only false-alarm probability versus the detector's `cfar_db` threshold
offset. The false-alarm ordinate is **per receiver tick**, since that is the
detector's operational output unit; it must not be labelled per-cell Pfa.

System C uses a one-dimensional matched-filter output and a CA-CFAR-style
threshold over time samples. It is calibrated with continuous circular complex
AWGN across each receiver tick. Noise is added once after emitter signals are
summed. The configured `snr_db` is referenced to the strongest received pulse;
weaker emitters retain their physical relative power.

## Frozen starting sweep

- Input SNR: `0, 5, 10, 15, 20, 25, 30, 35 dB`.
- CFAR threshold offsets: `3, 6, 9, 10, 12, 15, 18, 21 dB`.
- The detection curve holds the configured System C threshold at `10 dB`.
- The false-alarm curve uses noise-only ticks and sweeps the threshold offset.
- Use one fixed seed, record it in the JSON report, and use the same trial
  count at every point. `1,000` trials is a development run; use at least
  `10,000` per point for a final report and retain the Wilson 95% intervals.
- The injected pulse is a known LFM chirp. A detection counts only when its
  measured ToA is within two pulse widths of the injected pulse.

The SNR range follows Wang et al.'s published CA-CFAR detection comparison,
which evaluated SNR from 0 to 35 dB at a stated target `Pfa = 10^-6`. That paper
uses a **two-dimensional range-Doppler map**, so it grounds the sweep range,
not a claim that its detector or false-alarm performance transfers to Vyapti.
Vyapti's measured false-alarm curve is empirical; `Pfa = 10^-6` is not assumed.

## Why this remains one-dimensional

The cited two-dimensional method operates on range-Doppler cells. System C's
current detector receives a one-dimensional time series and produces a
matched-filter time trace; it does not produce a range-Doppler map. Adding a
second dimension would require defining a coherent burst, Doppler processing,
and target/clutter statistics first. The paper's result is therefore not an
apples-to-apples drop-in comparison. Keep the current one-dimensional detector
for this calibration and revisit a 2D detector only if the RF model gains a
validated range-Doppler processing stage.

## Reproduction

```powershell
python -m scripts.calibration.system_c_receiver `
  --trials 10000 `
  --seed 20261002 `
  --output results/system_c_receiver_calibration.json
```

The command writes JSON measurements and a two-panel PNG. Preserve both with
the experiment artifacts. Report the measured points and confidence intervals;
do not interpolate them into unsupported guarantees.

## Reference

Wang, Y. et al. (2019), “Modified reference window for two-dimensional CFAR in
radar target detection,” *The Journal of Engineering*.
[DOI: 10.1049/joe.2019.0687](https://doi.org/10.1049/joe.2019.0687).
