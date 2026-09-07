"""
vyapti_simulator.rf.receiver_impairments
========================================

Hardware receiver impairment models for I/Q complex-baseband signals.

The models in this module transform a clean complex envelope by
applying one or more of the following impairments:

  1. **Phase noise** — random walk of the carrier phase due to
     oscillator instability. Modelled as a Wiener process on phase
     with variance controlled by ``phase_noise_dbc_per_hz`` × sample
     rate × sample interval.
  2. **IQ gain imbalance** — amplitude mismatch between the I and Q
     ADCs (typical 0.1–0.5 dB for a real receiver).
  3. **IQ phase imbalance** — orthogonality error between I and Q
     (typical 0.5°–3° for a real receiver).
  4. **DC offset** — non-zero mean on I and Q from the ADC (typical
     1–100 LSBs depending on the AC-coupling network).
  5. **Quantization noise** — finite ADC resolution. Applied as
     uniform quantisation across ``2**quantization_bits`` levels.

These are the most common imperfections a real ESM/ELINT receiver
introduces. Including them in the synthetic I/Q path lets the
researcher study their effect on the matched filter, coherent
integration, and deinterleaver.

References
----------
Tsui, J. B. Y. (2004). "Digital Techniques for Wideband Receivers".
    2nd ed. SciTech Publishing. (Chapter 7: I/Q channel errors.)
Schreier, R. & Temes, G. C. (2005). "Understanding Delta-Sigma
    Data Converters". Wiley. (Quantization noise analysis.)
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# =====================================================================
# Individual impairment functions
# =====================================================================

def apply_phase_noise(
    iq_samples: np.ndarray,
    *,
    phase_noise_dbc_per_hz: float = -100.0,
    sample_rate_hz: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply oscillator phase noise to a complex envelope.

    Phase noise is modelled as a Wiener-process random walk on
    the carrier phase, with step variance

        sigma_phase_step^2 = 2 * pi * 10**(phase_noise_dbc_per_hz / 10)
                             * sample_rate_hz * dt

    where ``dt = 1 / sample_rate_hz`` and the dBc/Hz value
    represents the single-sideband phase noise floor of the
    oscillator. For a typical TCXO at 1 kHz offset: -100 dBc/Hz.
    For an OCXO: -130 dBc/Hz. A free-running VCO: -70 dBc/Hz.

    Parameters
    ----------
    iq_samples : np.ndarray
        Complex envelope, shape ``(n,)`` or ``(n, k)``.
    phase_noise_dbc_per_hz : float
        Phase noise floor in dBc/Hz. Default -100 dBc/Hz (TCXO).
        Set to 0 to disable.
    sample_rate_hz : float
        Sample rate in Hz. Used to convert dBc/Hz to per-sample
        phase variance. For an under-sampled I/Q (one sample per
        pulse), set this to the PRI in Hz. Default 1.0 (no
        accumulation between samples).
    rng : np.random.Generator, optional
        Random number generator. If None, a default_rng() is
        created.

    Returns
    -------
    np.ndarray
        Complex envelope with phase noise applied. Same shape as
        input.

    Notes
    -----
    The Wiener-process model is the simplest phase-noise model and
    captures the line-spread broadening; it does NOT model
    specific spurs or 1/f^3 close-in noise. For high-precision
    work, replace with a coloured phase-noise generator that
    filters white noise by a 1/f^2 PSD.
    """
    if rng is None:
        rng = np.random.default_rng()
    if phase_noise_dbc_per_hz == 0.0:
        return iq_samples.copy()
    iq = iq_samples.astype(np.complex128, copy=True)
    # Phase noise power per Hz → variance per sample step
    # The single-sideband phase noise model gives:
    #   d_phi^2 = 2 * pi * 10**(PN/10) * sample_rate * dt
    # where dt = 1 / sample_rate, so:
    #   d_phi^2 = 2 * pi * 10**(PN/10)
    sigma_per_step = np.sqrt(2.0 * np.pi * 10.0 ** (phase_noise_dbc_per_hz / 10.0))
    n = iq.size
    # Wiener process: cumulative sum of N(0, sigma) steps
    if iq.ndim == 1:
        dphi = rng.normal(0.0, sigma_per_step, size=n)
        phase = np.cumsum(dphi)
        iq = iq * np.exp(1j * phase)
    else:
        for k in range(iq.shape[1]):
            dphi = rng.normal(0.0, sigma_per_step, size=iq.shape[0])
            phase = np.cumsum(dphi)
            iq[:, k] = iq[:, k] * np.exp(1j * phase)
    return iq


def apply_iq_imbalance(
    iq_samples: np.ndarray,
    *,
    gain_imbalance_db: float = 0.0,
    phase_imbalance_deg: float = 0.0,
) -> np.ndarray:
    """
    Apply IQ gain and phase imbalance to a complex envelope.

    Real receivers have amplitude and orthogonality errors between
    the I and Q ADC channels. The model multiplies Q by a complex
    factor ``(1 + g) * exp(j*phi)`` where ``g`` is the gain
    error (linear, not dB) and ``phi`` is the phase error
    (radians).

    Parameters
    ----------
    iq_samples : np.ndarray
        Complex envelope, shape ``(n,)`` or ``(n, k)``.
    gain_imbalance_db : float
        Amplitude mismatch between I and Q in dB. Positive
        means Q is hotter than I. Typical 0.1–0.5 dB.
        Default 0.0 (no imbalance).
    phase_imbalance_deg : float
        Phase orthogonality error in degrees. Typical 0.5°–3°.
        Default 0.0 (perfect orthogonality).

    Returns
    -------
    np.ndarray
        Complex envelope with IQ imbalance applied. Same shape
        as input.
    """
    if gain_imbalance_db == 0.0 and phase_imbalance_deg == 0.0:
        return iq_samples.astype(np.complex128, copy=True)
    g_lin = 10.0 ** (gain_imbalance_db / 20.0) - 1.0
    phi_rad = np.deg2rad(phase_imbalance_deg)
    q_factor = (1.0 + g_lin) * np.exp(1j * phi_rad)
    iq = iq_samples.astype(np.complex128, copy=True)
    iq = 0.5 * (np.real(iq) + 1j * np.imag(iq) * q_factor) * 2.0
    return iq


def apply_dc_offset(
    iq_samples: np.ndarray,
    *,
    dc_i: float = 0.0,
    dc_q: float = 0.0,
) -> np.ndarray:
    """
    Add a DC offset to the I and Q components.

    Parameters
    ----------
    iq_samples : np.ndarray
        Complex envelope, shape ``(n,)`` or ``(n, k)``.
    dc_i, dc_q : float
        DC offset to add to the I (real) and Q (imaginary)
        parts. Default 0.0.

    Returns
    -------
    np.ndarray
        Complex envelope with DC offset applied. Same shape as
        input.
    """
    if dc_i == 0.0 and dc_q == 0.0:
        return iq_samples.astype(np.complex128, copy=True)
    iq = iq_samples.astype(np.complex128, copy=True)
    return iq + (dc_i + 1j * dc_q)


def apply_quantization(
    iq_samples: np.ndarray,
    *,
    bits: int = 12,
) -> np.ndarray:
    """
    Apply uniform ADC quantisation to the I/Q samples.

    The full-scale range is assumed to be the magnitude of the
    largest sample, with a small headroom margin. The number of
    quantisation levels is ``2 ** bits``.

    Parameters
    ----------
    iq_samples : np.ndarray
        Complex envelope, shape ``(n,)`` or ``(n, k)``.
    bits : int
        Number of ADC bits. Typical 8–16 for an ESM receiver.
        Default 12.

    Returns
    -------
    np.ndarray
        Quantised complex envelope. Same shape as input. The
        dtype is float64 (the discrete levels are stored as
        floating-point; the precision is what the ADC gives).
    """
    if bits < 1:
        raise ValueError(f"bits must be >= 1, got {bits}")
    n_levels = float(2 ** bits)
    iq = iq_samples.astype(np.complex128, copy=True)
    # Full-scale: max of |I| and |Q| over all samples, with
    # 5% headroom to avoid clipping.
    fs = 1.05 * float(np.max(np.abs(iq))) if iq.size > 0 else 1.0
    if fs <= 0.0:
        return iq
    # Quantise: round to nearest level
    step = 2.0 * fs / n_levels
    # Real and imaginary parts independently
    i_part = np.round(iq.real / step) * step
    q_part = np.round(iq.imag / step) * step
    return i_part + 1j * q_part


# =====================================================================
# Combined impairment model
# =====================================================================

def apply_receiver_impairments(
    iq_samples: np.ndarray,
    *,
    phase_noise_dbc_per_hz: float = 0.0,
    iq_gain_imbalance_db: float = 0.0,
    iq_phase_imbalance_deg: float = 0.0,
    dc_offset_i: float = 0.0,
    dc_offset_q: float = 0.0,
    quantization_bits: Optional[int] = None,
    sample_rate_hz: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply a chain of hardware receiver impairments to a complex
    I/Q envelope. The order matches the signal flow in a real
    receiver: phase noise first (in the LO), then IQ
    imbalance, then DC offset, then ADC quantisation.

    Parameters
    ----------
    iq_samples : np.ndarray
        Complex envelope, shape ``(n,)`` or ``(n, k)``.
    phase_noise_dbc_per_hz : float
        Phase noise floor in dBc/Hz. Set to 0 to disable.
        Default 0.
    iq_gain_imbalance_db : float
        Amplitude mismatch between I and Q. Set to 0 to disable.
        Default 0.
    iq_phase_imbalance_deg : float
        Phase orthogonality error in degrees. Set to 0 to disable.
        Default 0.
    dc_offset_i, dc_offset_q : float
        DC offset on the I and Q ADCs. Set to 0 to disable.
        Default 0.
    quantization_bits : int, optional
        Number of ADC bits. If None, no quantisation is applied
        (infinite precision). Default None.
    sample_rate_hz : float
        Sample rate in Hz. Used by the phase noise model. For
        a single complex sample per pulse, set this to 1/PRI.
        Default 1.0.
    rng : np.random.Generator, optional
        Random number generator for the phase noise. If None,
        a default_rng() is created.

    Returns
    -------
    np.ndarray
        Complex envelope with all enabled impairments applied.

    Notes
    -----
    The impairments are applied in this order (signal flow):

      1. Phase noise (LO) — affects all subsequent stages.
      2. IQ imbalance (analogue baseband).
      3. DC offset (ADC front-end).
      4. Quantisation (ADC).

    This matches a real receiver's signal path: the LO drives
    the I/Q demodulator, which is followed by the baseband
    anti-alias filter, then the ADC.
    """
    if rng is None:
        rng = np.random.default_rng()
    iq = iq_samples.astype(np.complex128, copy=True)
    if phase_noise_dbc_per_hz != 0.0:
        iq = apply_phase_noise(
            iq, phase_noise_dbc_per_hz=phase_noise_dbc_per_hz,
            sample_rate_hz=sample_rate_hz, rng=rng,
        )
    if iq_gain_imbalance_db != 0.0 or iq_phase_imbalance_deg != 0.0:
        iq = apply_iq_imbalance(
            iq,
            gain_imbalance_db=iq_gain_imbalance_db,
            phase_imbalance_deg=iq_phase_imbalance_deg,
        )
    if dc_offset_i != 0.0 or dc_offset_q != 0.0:
        iq = apply_dc_offset(iq, dc_i=dc_offset_i, dc_q=dc_offset_q)
    if quantization_bits is not None:
        iq = apply_quantization(iq, bits=quantization_bits)
    return iq


__all__ = [
    "apply_phase_noise",
    "apply_iq_imbalance",
    "apply_dc_offset",
    "apply_quantization",
    "apply_receiver_impairments",
]
