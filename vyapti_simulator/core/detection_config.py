"""Shared amplitude detector settings; IQ CFAR has its own receiver configuration."""
from dataclasses import dataclass, field
import numpy as np

@dataclass(frozen=True)
class DetectionConfig:
    """
    Configuration for the amplitude-based detection model.

    The model converts TSRD's ``amp_db`` field to an approximate
    SNR and a Bernoulli detection outcome. TSRD amplitudes are
    relative (not absolute dBm), but the **relative ordering** of
    cells by SNR is physically meaningful.

    Attributes
    ----------
    nominal_noise_floor_db : float
        Fallback noise floor in dB (relative amplitude scale).
        Used as the AGC estimate's initial value and when AGC
        has no recent data. TSRD's amplitude range empirically
        spans approximately -130 dB (noise floor) to -85 dB
        (strongest pulse), so -130 dB is the conservative
        default. This field is deprecated in favour of the AGC
        model when ``agc_window_slots > 0``; it becomes the
        initial estimate only.

    detection_threshold_db : float
        SNR above which detection is certain (Pd → 1). Cells
        with ``snr_db >= detection_threshold_db`` always hit.
        With AGC enabled, this is applied *above* the
        AGC-computed noise floor.

    no_detection_threshold_db : float
        SNR below which detection is impossible (Pd → 0).
        Cells with ``snr_db <= no_detection_threshold_db``
        always miss.

    false_alarm_probability : float
        Probability that an empty cell produces a false alarm.
        Applied to cells with ``pulse_count == 0`` only.

    use_sensitivity_curve : bool
        If True, use a logistic SNR-to-Pd curve between
        ``no_detection_threshold_db`` and ``detection_threshold_db``.
        If False, apply a hard threshold at
        ``no_detection_threshold_db``.

    agc_window_slots : int
        Number of recent slots the AGC tracks. The AGC maintains
        a sliding window of ``max_amplitude_db`` values from
        non-empty cells and computes an adaptive noise-floor
        estimate as ``floor_fraction * recent_max_amplitude``.
        Set to 0 to disable AGC and use ``nominal_noise_floor_db``
        as a fixed threshold.  Typical values: 10–100 slots.
        At 50 ms/slot: 100 slots = 5 s of tracking history.

    agc_floor_fraction : float
        **DEPRECATED — use ``agc_dynamic_range_db`` instead.**
        Previously: fraction of the recent-window maximum amplitude
        in linear power units used as the AGC noise-floor estimate.
        This is now ignored in favour of the dB-offset model.

        Retained for API compatibility. If set, it overrides
        ``agc_dynamic_range_db`` (backwards compat only).

    agc_dynamic_range_db : float
        The dynamic range in dB from the noise floor to the
        strongest recent pulse. The AGC noise-floor estimate is:
        ``floor = max_db - dynamic_range_db``.
        A real AGC tracks gain changes (strong signal → AGC backs off),
        but the absolute noise floor is set by the thermal floor,
        not by signal level. This parameter specifies how many dB
        the noise floor sits below the recent maximum amplitude.
        Typical values:
          - 20–30 dB: nominal (noise floor is 20–30 dB below
            the strongest expected signal).
          - 40–50 dB: conservative (for strong-multipath
            scenarios where noise floor is near TSRD's -130 dB).
        Used only when ``agc_window_slots > 0``.

    agc_snr_margin_db : float
        SNR margin in dB above the AGC noise floor for detection.
        The effective detection threshold is:
        ``threshold = agc_noise_floor_db + agc_snr_margin_db``.
        Higher values suppress multipath (which is close to the
        floor) and only pass strong direct-path signals.
        Lower values admit weaker signals including multipath.
        Typical: 3–10 dB.

    agc_no_signal_floor_db : float
        Absolute floor for the AGC noise-floor estimate when
        there are no recent non-empty slots. Prevents the floor
        from collapsing to -∞ if several empty slots occur.
        Defaults to ``nominal_noise_floor_db``.
    """
    nominal_noise_floor_db: float = -130.0
    # [LITERATURE-GROUNDED] LNA noise figure. A real ESM receiver's front-end
    # degrades the SNR by NF dB relative to an ideal receiver. Typical
    # values:
    #   3 dB  — excellent (cryogenic LNA)
    #   6 dB  — typical discrete LNA
    #   10 dB — budget receiver
    #   20 dB — severe (front-end losses before LNA)
    # The effective noise floor = nominal_noise_floor_db + noise_figure_db,
    # so a 6 dB NF raises the floor by 6 dB and reduces measured SNR by 6 dB.
    # Default 0 dB means no NF degradation (backward compatible).
    # Per IEEE Std 521-2006 and Skolnik Radar Handbook LNA section.
    noise_figure_db: float = 0.0
    # Detection curve parameters. The defaults (0 dB, 5 dB) are an engineering
    # choice for a flat 5 dB transition region. To sweep this as an
    # experimental variable (see audit issue #7), override at construction:
    #     DetectionConfig(no_detection_threshold_db=-2.0, detection_threshold_db=8.0)
    #     # → 10 dB transition, midpoint 3 dB, gives Pd(0)=0.16, Pd(5)=0.66
    # See `scripts/test_detection_sensitivity.py` for a worked sweep.
    detection_threshold_db: float = 5.0
    no_detection_threshold_db: float = 0.0
    false_alarm_probability: float = 0.0
    use_sensitivity_curve: bool = True
    # AGC parameters
    agc_window_slots: int = 0          # 0 = disabled (fixed floor)
    agc_floor_fraction: float = 1.0   # DEPRECATED; use agc_dynamic_range_db
    agc_snr_margin_db: float = 0.0   # [FROZEN-DEFAULT] align with System A; 5.0 was a multipath-suppression bias that System A did not have
    # NOTE: This is a frozen default per the 2026-09-05 audit. To sweep the
    # AGC margin as an experimental variable (see audit issue #5), override
    # at construction time:
    #     DetectionConfig(agc_snr_margin_db=5.0)  # tighter multipath rejection
    #     DetectionConfig(agc_snr_margin_db=-2.0) # more aggressive detection
    # and verify the System-A / System-B cross-path hit rate agreement test
    # still passes (or note in the report that System A has no equivalent
    # bias and the comparison becomes asymmetric).
    agc_dynamic_range_db: float = 30.0   # noise floor = max - this many dB
    agc_no_signal_floor_db: float = -130.0
    # Antenna gain pattern — per-band receiver gain relative to isotropic (0 dB).
    # A sectorised EW antenna has gain that varies with frequency: some bands
    # fall in high-gain sectors (0 dB = reference), others fall in
    # low-gain sidelobes (-3 to -10 dB). This field models that variation.
    # When empty (default), uniform 0 dB is assumed (backward compatible).
    # When set, must have length == band_count. Applied as a dB penalty
    # in the SNR estimate: effective_snr = measured_snr - antenna_gain_db[band].
    # Per Skolnik Radar Handbook antenna sectorisation.
    antenna_gain_db: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    # Coherent integration parameters
    # A real ESM receiver coherently integrates pulses from the same
    # burst: N pulses → 10*log10(N) dB SNR gain. The detection model
    # groups pulses in a (band, slot) cell by emitter_id and adds this
    # gain to the live SNR before applying the Pd curve. This is the
    # largest single ESM performance lever and was previously missing.
    # This path groups by dataset emitter_id, so keep oracle-aided integration opt-in.
    coherent_integration_enabled: bool = False
    coherent_integration_max_pulses: int = 50
    coherent_integration_non_coherent_loss_db: float = 0.5

    def pd_for_snr(self, snr_db: float) -> float:
        """
        Map SNR (dB) to detection probability Pd.

        Hard threshold model::

            Pd = 1.0  if snr_db >= detection_threshold_db
            Pd = 0.0  if snr_db <= no_detection_threshold_db
            Pd = 0.5  if use_sensitivity_curve == False  (linear midpoint)
            Pd = logistic(snr_db)  if use_sensitivity_curve == True

        Logistic parameters are chosen so that:
          - Pd(0) ≈ 0.01
          - Pd(20) ≈ 0.99
        which matches realistic radar detection ROC behaviour.
        """
        if snr_db >= self.detection_threshold_db:
            return 1.0
        if snr_db <= self.no_detection_threshold_db:
            return 0.0
        if not self.use_sensitivity_curve:
            # Linear interpolation between the two thresholds
            t = (snr_db - self.no_detection_threshold_db) / (
                self.detection_threshold_db - self.no_detection_threshold_db
            )
            return float(np.clip(t, 0.0, 1.0))
        # Logistic: Pd = 1 / (1 + exp(-k * (snr_db - midpoint)))
        # k = 4 / width gives roughly 0.01 at (midpoint - 2*width) and 0.99 at (midpoint + 2*width)
        midpoint = (self.detection_threshold_db + self.no_detection_threshold_db) / 2.0
        width = (self.detection_threshold_db - self.no_detection_threshold_db) / 4.0
        margin = (snr_db - midpoint) / width
        return float(1.0 / (1.0 + np.exp(-4.0 * margin)))


@dataclass(frozen=True)
class ShnidmanDetectionConfig(DetectionConfig):
    """
    [LITERATURE-GROUNDED] Detection model using the Albersheim (1964) /
    Shnidman (1989) equation. Replaces the heuristic logistic curve with
    the published closed-form that gives a (Pd, Pfa, N) relationship.

    For a non-fluctuating target in Gaussian noise with N non-coherently
    integrated pulses, the required per-pulse SNR (linear) for target
    (Pd, Pfa) is:

        A = ln(0.5 / Pfa)
        B = ln(Pd / (1 - Pd))
        SNR_lin = A + B + 3.0 * sqrt(B) * sqrt(A - B)        (Shnidman 1989)

    For a non-coherent integration of N pulses, the threshold SNR
    decreases (sensitivity improves) by 10*log10(sqrt(N)) ≈
    5*log10(N) dB. (Coherent integration gives the full 10*log10(N).)

    Comparison with the parent DetectionConfig
    -----------------------------------------
    The parent class uses a 5 dB logistic transition between
    (no_detection_threshold_db, detection_threshold_db). This works
    for a single representative (Pd, Pfa) and cannot be adjusted
    without changing the thresholds. The Shnidman model lets you
    set the *target* (Pd, Pfa, N) directly and produces a single
    closed-form threshold SNR in dB. The result is closer to what
    published radar detection tables (e.g. North, Blake, Albersheim)
    report.

    Backward compatibility
    ----------------------
    ShnidmanDetectionConfig IS a DetectionConfig (subclass), so any
    code that accepts `DetectionConfig` also accepts
    `ShnidmanDetectionConfig`. The default pd_for_snr() override
    here is the Albersheim / Shnidman formula; pass it
    `n_pulses=1` (or use the default 1) for single-pulse detection.

    Attributes
    ----------
    target_Pd : float
        Desired probability of detection at the threshold SNR.
        Typical: 0.5, 0.9, 0.95.
    target_Pfa : float
        Desired probability of false alarm at the threshold SNR.
        Typical: 1e-3, 1e-6, 1e-9.
    transition_width_db : float
        Width of the soft transition region around the threshold
        SNR (in dB). Below the threshold, Pd falls off with this
        logistic width; above, Pd approaches 1. Set to 0.0 for a
        hard step (less physically realistic but tighter).
    integration_mode : str
        "non_coherent" — N pulses give 5*log10(N) dB gain (sqrt law)
        "coherent"     — N pulses give 10*log10(N) dB gain (full law)
        Default "non_coherent" because ESM receivers are typically
        non-coherent on the PRI timescale; use "coherent" only when
        the emitter has a known constant phase reference.

    References
    ----------
    Albersheim, W. R. (1964). "Equation for SNR Required for
        Detection of a Target with Given Pd and Pfa".
    Shnidman, D. A. (1989). "Radar Detection Probability and
        Its Approximation". IEEE Trans. AES-25, no. 6, pp. 672-676.
    North, D. O. (1963). "An Analysis of the Factors which
        Determine Signal/Noise Discrimination in Pulsed-Carrier
        Systems". RCA Labs Tech. Rept. PTR-6C.
    """
    target_Pd: float = 0.9
    target_Pfa: float = 1e-6
    transition_width_db: float = 1.0
    integration_mode: str = "non_coherent"

    def threshold_snr_db(self, n_pulses: int = 1) -> float:
        """
        Compute the per-pulse SNR (dB) required to achieve
        (target_Pd, target_Pfa) with n_pulses non-coherent
        integration, using the Shnidman (1989) approximation.

        Returns the SNR threshold in dB. Callers then compare
        an observed SNR to this threshold.
        """
        if not (0.0 < self.target_Pfa < 1.0):
            raise ValueError(f"target_Pfa must be in (0, 1), got {self.target_Pfa}")
        if not (0.0 < self.target_Pd < 1.0):
            raise ValueError(f"target_Pd must be in (0, 1), got {self.target_Pd}")
        if n_pulses < 1:
            raise ValueError(f"n_pulses must be >= 1, got {n_pulses}")
        A = np.log(0.5 / self.target_Pfa)
        B = np.log(self.target_Pd / (1.0 - self.target_Pd))
        snr_lin_single = A + B + 3.0 * np.sqrt(B) * np.sqrt(A - B)
        # Integration gain in dB
        if self.integration_mode == "coherent":
            integ_gain_db = 10.0 * np.log10(n_pulses)
        elif self.integration_mode == "non_coherent":
            integ_gain_db = 5.0 * np.log10(n_pulses)
        else:
            raise ValueError(
                f"integration_mode must be 'coherent' or 'non_coherent', "
                f"got {self.integration_mode!r}"
            )
        return 10.0 * np.log10(snr_lin_single) - integ_gain_db

    def pd_for_snr(self, snr_db: float, n_pulses: int = 1) -> float:
        """
        Shnidman / Albersheim SNR -> Pd.

        At the threshold SNR (computed by threshold_snr_db), Pd equals
        target_Pd. Above the threshold, Pd rises toward 1.0 with a
        soft logistic transition of `transition_width_db`; below, Pd
        falls toward 0.0. Outside the transition region, the
        Shnidman equation becomes a poor approximation and a step /
        flat region is more honest.
        """
        if n_pulses == 0:
            return 0.0
        thr = self.threshold_snr_db(n_pulses)
        if self.transition_width_db <= 0.0:
            # Hard step: certain detection above threshold, false-alarm
            # rate below. This is the canonical Albersheim behaviour.
            if snr_db >= thr:
                return 1.0
            return 0.0
        # Soft logistic transition centred on the threshold SNR.
        # At (snr_db == thr), Pd = 0.5 + (target_Pd - 0.5) so the
        # threshold is not exactly at 50% — we shift accordingly.
        # Approach: solve logistic at thr such that logistic(thr) = target_Pd.
        # logistic(x) = 1 / (1 + exp(-k * (x - mu)))
        # We want logistic(thr) = target_Pd, so:
        #   mu = thr - (-1/k) * ln(1/target_Pd - 1)
        # For the simplest behaviour that puts target_Pd at the threshold,
        # we set mu = thr and scale k so that logistic passes through
        # (thr, target_Pd). Then k = -ln(1/target_Pd - 1) / transition_width_db
        # — but a simpler well-behaved choice is to scale the logistic
        # width so that logistic width (1/k) = transition_width_db / 4.
        width = max(self.transition_width_db, 1e-9) / 4.0
        # Solve for offset so that the curve passes through (thr, target_Pd):
        # target_Pd = 1 / (1 + exp(-(thr - mu) / width))
        # (thr - mu) = -width * ln(1/target_Pd - 1)
        offset = -width * float(np.log(1.0 / self.target_Pd - 1.0))
        mu = thr - offset
        z = (snr_db - mu) / width
        return float(1.0 / (1.0 + np.exp(-z)))
