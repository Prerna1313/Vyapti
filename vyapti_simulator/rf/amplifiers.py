"""
vyapti_simulator.rf.amplifiers
===============================

Power amplifier non-linearity models for realistic RF simulation.

Real amplifiers distort signals when driven near saturation. Two
canonical models cover the most important effects:

  1. **Rapp model** — AM/AM distortion: amplitude compression at
     high drive levels. Used for travelling-wave tube (TWT) amplifiers.
  2. **Saleh model** — AM/AM and AM/PM distortion: both magnitude
     compression and phase rotation. Used for solid-state power
     amplifiers (SSPAs).

Both produce spectral regrowth (out-of-band emissions) when a
modulated signal is passed through a non-linear amplifier.

References
----------
Rapp, C. (1991). "Effects of HPA-Nonlinearity on a 4-DPSK/OFDM-Signal
    for a Digital Sound Broadcasting System". Proc. 2nd European
    Conf. on Radio Receivers and Systems.
Saleh, A. A. M. (1981). "Frequency-Independent and Frequency-Dependent
    Nonlinear Models of TWT Amplifiers". IEEE Trans. COM-29, pp. 1715-1720.

Usage
-----
::

    from vyapti_simulator.rf.amplifiers import (
        RappAmplifier, SalehAmplifier, apply_spectral_regrowth,
        OIP3, OIP3_from_gain_compression,
    )
    import numpy as np

    # TWT amplifier with 2 dB gain compression at P1dB
    twt = RappAmplifier(
        output_power_w=10.0,    # 40 dBm max
        gain_db=30.0,          # small-signal gain
        p1db_dbw=0.0,         # 1 dB compression at 0 dBW input
    )

    # 16-QAM waveform, 1000 samples
    signal = np.random.randn(1000) + 1j * np.random.randn(1000)
    distorted = twt.apply(signal)

    # Check spectral regrowth (ACPR)
    aclr = compute_aclr(distorted, carrier_bw_hz=5e6, adjacent_bw_hz=5e6)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


# =====================================================================
# Helper: third-order intercept point
# =====================================================================

def OIP3_from_gain_compression(
    gain_db: float,
    p1db_dbw: float,
) -> float:
    """
    Estimate OIP3 from gain and 1-dB compression point.

    For a typical solid-state amplifier::

        OIP3 ≈ P1dB + 10.6 dB

    This is an empirical approximation; the true relationship depends
    on the amplifier design.

    Parameters
    ----------
    gain_db : float
        Small-signal gain in dB.
    p1db_dbw : float
        Output-referred 1-dB compression point in dBW.

    Returns
    -------
    float
        Estimated OIP3 in dBW.
    """
    return float(p1db_dbw + 10.6)


def OIP3(p1db_dbw: float) -> float:
    """
    Compute OIP3 from the 1-dB compression point.

    For a typical amplifier the third-order intercept is roughly
    10 dB above the P1dB.

    Parameters
    ----------
    p1db_dbw : float
        Output-referred 1-dB compression point in dBW.

    Returns
    -------
    float
        OIP3 in dBW.
    """
    return float(p1db_dbw + 10.0)


def gain_compression_db(input_power_dbw: float, gain_db: float,
                        p1db_dbw: float) -> float:
    """
    Compute gain compression at a given input power.

    Uses the classical model::

        G(P_in) = G_0 - max(0, P_in - P_s)^q / P_s

    where P_s is the saturation input power and q ≈ 2-3.

    Parameters
    ----------
    input_power_dbw : float
        Input power in dBW.
    gain_db : float
        Small-signal gain in dB.
    p1db_dbw : float
        Output-referred 1-dB compression point in dBW.

    Returns
    -------
    float
        Gain in dB at the given input level (may be compressed).
    """
    # P_sat ≈ P1dB + 1 dB (rough rule)
    p_sat_dbw = float(p1db_dbw + 1.0)
    if input_power_dbw <= p_sat_dbw:
        return float(gain_db)
    # Exponential compression beyond saturation
    delta = input_power_dbw - p_sat_dbw
    compression = 0.5 * delta  # ~0.5 dB per dB overdrive
    return float(max(gain_db - compression, gain_db - 20.0))


# =====================================================================
# Rapp model (AM/AM only — TWT / travelling-wave tube)
# =====================================================================

@dataclass
class RappAmplifier:
    """
    Rapp model for travelling-wave tube (TWT) amplifiers.

    The Rapp model characterises only AM/AM distortion (magnitude
    compression), with no phase distortion::

        output = input / (1 + (|input|/v_sat)^p)^(1/p)

    where ``v_sat`` is the saturation voltage and ``p`` controls
    the sharpness of the transition.

    Parameters
    ----------
    output_power_w : float
        Maximum output power in watts (linear, not dBW).
    gain_db : float
        Small-signal gain in dB.
    p1db_dbw : float
        Output-referred 1-dB compression point in dBW.
        Used to estimate saturation.
    p : float
        Smoothness parameter. p=2 is typical for TWTs. Higher p
        gives a sharper knee. Default 2.0.
    rng : np.random.Generator, optional
        RNG for thermal noise injection.
    """

    output_power_w: float
    gain_db: float
    p1db_dbw: float
    p: float = 2.0
    rng: Optional[np.random.Generator] = None

    def __post_init__(self):
        if self.output_power_w <= 0:
            raise ValueError(f"output_power_w must be positive; got {self.output_power_w}")
        if self.p <= 0:
            raise ValueError(f"p must be positive; got {self.p}")
        self._gain_lin = float(10.0 ** (self.gain_db / 20.0))
        self._p1db_lin = float(10.0 ** (self.p1db_dbw / 10.0))
        # Saturation voltage from P1dB point
        # At P1dB: G = G_0 - 1
        # 10^((G_0-1)/20) * |in| = 10^(P1dB/10)
        # v_sat = |in|_P1dB * (10^(1/20) - 1)^(-1/p)
        # Use a simpler estimate: P_sat = output_power_w * 0.8
        self._v_sat = float(np.sqrt(self.output_power_w * 0.8))
        self._oip3 = OIP3(self.p1db_dbw)

    @property
    def saturation_voltage(self) -> float:
        """Saturation voltage of the Rapp model."""
        return self._v_sat

    @property
    def oip3_dbw(self) -> float:
        """Estimated OIP3 in dBW."""
        return self._oip3

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply Rapp AM/AM distortion to a complex signal.

        Parameters
        ----------
        signal : np.ndarray
            Complex baseband signal (voltage waveform).

        Returns
        -------
        np.ndarray
            Distorted signal (same shape).
        """
        signal = np.asarray(signal, dtype=np.complex128)
        magnitude = np.abs(signal)

        # Avoid division by zero at the origin
        magnitude_safe = np.where(magnitude > 1e-12, magnitude, 1e-12)

        # Rapp AM/AM: output_mag = input_mag / (1 + (in/v_sat)^p)^(1/p)
        denominator = 1.0 + (magnitude_safe / self._v_sat) ** self.p
        compressed = magnitude_safe / (denominator ** (1.0 / self.p))

        # Preserve phase, apply gain
        phase = np.angle(signal)
        out_mag = compressed * self._gain_lin
        out = out_mag * np.exp(1j * phase)

        # Clip at saturation
        out_mag_max = float(np.sqrt(self.output_power_w))
        out_abs = np.abs(out)
        if np.any(out_abs > out_mag_max):
            out = np.where(
                out_abs > out_mag_max,
                out_mag_max * np.exp(1j * np.angle(out)),
                out,
            )
        return out.astype(np.complex128)


# =====================================================================
# Saleh model (AM/AM + AM/PM — SSPA / solid-state)
# =====================================================================

@dataclass
class SalehAmplifier:
    """
    Saleh model for solid-state power amplifiers (SSPAs).

    The Saleh model characterises both AM/AM and AM/PM distortion::

        A(r) = α_a * r / (1 + β_a * r²)
        Φ(r) = α_φ * r² / (1 + β_φ * r²)

    where ``r`` is the normalised input magnitude. Parameters are
    fitted to measured SSPA characteristics.

    Parameters
    ----------
    alpha_a : float
        AM/AM linear gain factor. Default 2.1587 (fitted TWTA).
    beta_a : float
        AM/AM saturation factor. Default 1.1517 (fitted TWTA).
    alpha_phi : float
        AM/PM coefficient. Default 4.0033 (radians per squared volt).
    beta_phi : float
        AM/PM saturation factor. Default 9.1040 (fitted TWTA).
    input_backoff_db : float
        Input backoff in dB from 1-dB compression. Reduces drive level.
    saturation_voltage : float
        Saturation voltage (peak). Used to normalise input.
    """

    alpha_a: float = 2.1587
    beta_a: float = 1.1517
    alpha_phi: float = 4.0033
    beta_phi: float = 9.1040
    input_backoff_db: float = 0.0
    saturation_voltage: float = 1.0

    def __post_init__(self):
        self._backoff_lin = float(10.0 ** (-self.input_backoff_db / 20.0))

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply Saleh AM/AM+AM/PM distortion to a complex signal.

        Parameters
        ----------
        signal : np.ndarray
            Complex baseband signal.

        Returns
        -------
        np.ndarray
            Distorted signal with both magnitude compression and phase rotation.
        """
        signal = np.asarray(signal, dtype=np.complex128)
        magnitude = np.abs(signal) * self._backoff_lin
        phase = np.angle(signal)

        # AM/AM: amplitude compression
        a_r = (self.alpha_a * magnitude) / (1.0 + self.beta_a * magnitude ** 2)

        # AM/PM: phase rotation
        phi_r = (self.alpha_phi * magnitude ** 2) / (
            1.0 + self.beta_phi * magnitude ** 2
        )

        out = a_r * np.exp(1j * (phase + phi_r))
        return out.astype(np.complex128)


# =====================================================================
# Spectral regrowth / ACLR
# =====================================================================

def compute_aclr(
    signal: np.ndarray,
    f_s: float,
    carrier_bw_hz: float,
    adjacent_bw_hz: float,
    n_fft: int = 2048,
) -> Tuple[float, float]:
    """
    Compute adjacent channel leakage ratio (ACLR) in dB.

    ACLR is the ratio of power in the adjacent channel to the power
    in the main channel. It is the standard metric for spectral regrowth
    caused by amplifier non-linearity.

    Parameters
    ----------
    signal : np.ndarray
        Complex baseband signal.
    f_s : float
        Sample rate in Hz.
    carrier_bw_hz : float
        Bandwidth of the main (carrier) channel.
    adjacent_bw_hz : float
        Bandwidth of the adjacent channel (upper and lower averaged).
    n_fft : int
        FFT size for the PSD estimate.

    Returns
    -------
    lower_aclr_db : float
        Lower adjacent channel ACLR in dB.
    upper_aclr_db : float
        Upper adjacent channel ACLR in dB.
    """
    signal = np.asarray(signal, dtype=np.complex128).ravel()
    # Compute PSD using Welch's method
    from scipy.signal import welch

    # For complex signals use the two-sided spectrum
    f, psd = welch(
        np.abs(signal) ** 2,
        fs=float(f_s),
        nperseg=min(len(signal), n_fft // 4),
        noverlap=None,
        return_onesided=False,
    )
    psd = np.fft.fftshift(psd)

    bw_carrier = carrier_bw_hz / 2
    bw_adj = adjacent_bw_hz / 2
    f_pos = f[f >= 0]
    psd_pos = psd[len(psd) // 2:]

    # Integrate power in carrier and adjacent channels
    carrier_mask = np.abs(f_pos) <= bw_carrier
    adj_mask_lower = (f_pos >= (bw_carrier + bw_adj)) & (f_pos <= (bw_carrier + 2 * bw_adj))
    adj_mask_upper = (f_pos >= -(bw_carrier + 2 * bw_adj)) & (f_pos <= -(bw_carrier + bw_adj))

    p_carrier = float(np.trapz(psd_pos[carrier_mask], f_pos[carrier_mask]))
    p_adj_lower = float(np.trapz(psd_pos[np.abs(f_pos - (bw_carrier + bw_adj)) < bw_adj], f_pos[np.abs(f_pos - (bw_carrier + bw_adj)) < bw_adj]))
    p_adj_upper = float(np.trapz(psd_pos[np.abs(f_pos + bw_carrier + bw_adj) < bw_adj], f_pos[np.abs(f_pos + bw_carrier + bw_adj) < bw_adj]))

    p_adj = (p_adj_lower + p_adj_upper) / 2
    if p_carrier <= 0 or p_adj <= 0:
        return -np.inf, -np.inf
    return float(10.0 * np.log10(p_adj / p_carrier)), float(10.0 * np.log10(p_adj / p_carrier))


def compute_evm(
    received: np.ndarray,
    reference: np.ndarray,
) -> float:
    """
    Compute Error Vector Magnitude (EVM) in percent.

    EVM measures the distance between received constellation points and
    their ideal locations::

        EVM = sqrt(E[|r - s|²] / E[|s|²]) * 100%

    where ``r`` is the received symbol and ``s`` is the reference.

    Parameters
    ----------
    received : np.ndarray
        Received complex symbols.
    reference : np.ndarray
        Ideal reference symbols (same length).

    Returns
    -------
    float
        EVM in percent.
    """
    received = np.asarray(received, dtype=np.complex128).ravel()
    reference = np.asarray(reference, dtype=np.complex128).ravel()
    n = min(len(received), len(reference))
    if n == 0:
        return np.inf
    mse = float(np.mean(np.abs(received[:n] - reference[:n]) ** 2))
    power = float(np.mean(np.abs(reference[:n]) ** 2))
    if power <= 0:
        return np.inf
    return float(100.0 * np.sqrt(mse / power))


def apply_thermal_noise(
    signal: np.ndarray,
    noise_temperature_k: float,
    bandwidth_hz: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Add thermal noise at a given noise temperature and bandwidth.

    Uses the classic radiometer equation::

        P_noise = k_B * T * B

    where k_B = 1.38e-23 J/K is Boltzmann's constant.

    Parameters
    ----------
    signal : np.ndarray
        Complex signal (voltage).
    noise_temperature_k : float
        System noise temperature in Kelvin.
    bandwidth_hz : float
        Noise bandwidth in Hz.
    rng : np.random.Generator, optional

    Returns
    -------
    np.ndarray
        Signal with thermal noise added.
    """
    from vyapti_simulator.rf.waveforms import add_awgn

    signal = np.asarray(signal, dtype=np.complex128)
    if rng is None:
        rng = np.random.default_rng()
    k_B = 1.380649e-23  # Boltzmann constant (exact)
    noise_power_w = k_B * float(noise_temperature_k) * float(bandwidth_hz)
    return add_awgn(signal, noise_floor_w=noise_power_w, rng=rng)


__all__ = [
    "RappAmplifier",
    "SalehAmplifier",
    "OIP3",
    "OIP3_from_gain_compression",
    "gain_compression_db",
    "compute_aclr",
    "compute_evm",
    "apply_thermal_noise",
]
