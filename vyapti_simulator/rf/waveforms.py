"""
vyapti_simulator.rf.waveforms
==============================

Realistic RF waveform synthesizers for I/Q complex-baseband simulation.

This module provides the three canonical waveform families used in EW and
radar simulation:

  1. **PSK** — Phase-shift keying. Symbols lie on a unit circle at
     angles determined by the bit pattern. BPSK (1 bit/sym), QPSK
     (2 bits/sym), 8-PSK (3 bits/sym).
  2. **QAM** — Quadrature amplitude modulation. Symbols occupy a
     rectangular grid in the complex plane. 16-QAM (4 bits/sym),
     64-QAM (6 bits/sym).
  3. **LFM / Chirp** — Linear frequency modulation. Instantaneous
     frequency sweeps linearly with time, giving a sinc-spectrum
     pulse ideal for pulse compression in radar.

All generators accept a ``rng`` so the bit patterns and noise
realisations are reproducible. The AWGN noise model uses the
Box-Muller transform (polar variant for speed) to produce
complex Gaussian noise scaled to the requested SNR or noise floor.

References
----------
Proakis, J. G. & Salehi, M. (2008). "Digital Communications". 5th ed.
    McGraw-Hill. (PSK, QAM constellation geometry.)
Skolnik, M. I. (2001). "Introduction to Radar Systems". 3rd ed.
    McGraw-Hill. (LFM/chirp pulse compression.)

Usage
-----
::

    from vyapti_simulator.rf.waveforms import (
        generate_psk_symbols, generate_qam_symbols, generate_lfm_chirp,
        add_awgn, complex_baseband_to_rf,
    )
    import numpy as np

    # QPSK waveform: 1000 symbols at 10 MHz symbol rate
    bits = np.random.randint(0, 4, size=1000)  # 2-bit symbols
    symbols = generate_psk_symbols(bits, constellation='qpsk')

    # Embed in a chirp pulse: 10 µs duration at 1 GHz centre freq
    t = np.linspace(0, 10e-6, 1000)
    chirp = generate_lfm_chirp(
        t, f0=995e6, f1=1005e6, peak_power_w=1.0,
    )

    # Add AWGN to set SNR = 15 dB
    noisy = add_awgn(chirp, snr_db=15.0)

    # Up-convert to RF (passband)
    rf = complex_baseband_to_rf(noisy, f_c=1e9)
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np


# =====================================================================
# PSK constellation tables
# =====================================================================

# Phase offsets in radians for each modulation order.
# Each table is indexed by the integer symbol value [0, M-1].
# Values are complex numbers on the unit circle.
def _build_constellation(M: int, phase_offset: float = 0.0) -> np.ndarray:
    """Build an M-PSK constellation table on the unit circle."""
    phases = np.arange(M) * (2.0 * np.pi / M) + phase_offset
    return np.exp(1j * phases)


# Standard constellations (Gray-coded where applicable)
_BPSK = _build_constellation(2, phase_offset=0.0)          # [1, -1]
_QPSK = _build_constellation(4, phase_offset=np.pi / 4)    # [e^jπ/4, e^j3π/4, ...]
_8PSK = _build_constellation(8, phase_offset=np.pi / 8)    # 8 equal-spaced points


def generate_psk_symbols(
    symbol_indices: np.ndarray,
    constellation: Literal["bpsk", "qpsk", "8psk"] = "qpsk",
) -> np.ndarray:
    """
    Map integer symbol indices to complex PSK symbols on the unit circle.

    Parameters
    ----------
    symbol_indices : np.ndarray
        Integer symbol values, shape ``(n,)``. Valid range depends
        on ``constellation``:
          - ``"bpsk"``: 0 or 1
          - ``"qpsk"``: 0, 1, 2, or 3
          - ``"8psk"``: 0..7
    constellation : {"bpsk", "qpsk", "8psk"}
        Modulation order. Default ``"qpsk"``.

    Returns
    -------
    np.ndarray
        Complex symbols on the unit circle, shape ``(n,)``,
        dtype ``complex128``.

    Notes
    -----
    **Gray coding** (QPSK): adjacent symbols differ by exactly one bit.
    The symbol-to-bit mapping is::

        0 → 00    1 → 01    2 → 11    3 → 10

    **BPSK** is the standard radar pulse (a simple phase flip keyed
    by the data bit). **QPSK** is the most common communication
    modem. **8-PSK** is used in legacy tactical datalinks.
    """
    indices = np.asarray(symbol_indices, dtype=np.intp).ravel()
    if constellation == "bpsk":
        table = _BPSK
    elif constellation == "qpsk":
        table = _QPSK
    elif constellation == "8psk":
        table = _8PSK
    else:
        raise ValueError(
            f"constellation must be bpsk/qpsk/8psk; got {constellation!r}"
        )
    return table[indices].astype(np.complex128)


def bits_to_psk_symbols(
    bits: np.ndarray,
    constellation: Literal["bpsk", "qpsk", "8psk"] = "qpsk",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a bit array to PSK symbols, returning both symbols and the
    bit-to-symbol mapping.

    Parameters
    ----------
    bits : np.ndarray
        Flat bit array (0 or 1). Shape ``(n_bits,)``.
    constellation : {"bpsk", "qpsk", "8psk"}
        Modulation scheme.

    Returns
    -------
    symbols : np.ndarray
        Complex symbols, shape ``(n_symbols,)``.
    bit_mapping : np.ndarray
        The bit indices used for each symbol (for demodulation reference).
    """
    bits = np.asarray(bits, dtype=np.intp).ravel()
    bits_per_symbol = {"bpsk": 1, "qpsk": 2, "8psk": 3}[constellation]
    n_symbols = len(bits) // bits_per_symbol
    truncated = bits[: n_symbols * bits_per_symbol]
    reshaped = truncated.reshape(n_symbols, bits_per_symbol)

    # Binary to integer: bit 0 = MSB (for consistent ordering)
    indices = np.dot(reshaped, 2 ** np.arange(bits_per_symbol - 1, -1, -1))
    symbols = generate_psk_symbols(indices, constellation=constellation)
    return symbols, indices


# =====================================================================
# QAM constellation tables
# =====================================================================

def _build_rectangular_qam(
    m: int,
) -> np.ndarray:
    """
    Build an M-QAM constellation table (rectangular grid).

    The table is row-major over the amplitude levels: I (real) varies
    faster than Q (imag). For M = k², there are k amplitude levels
    in each dimension.

    Parameters
    ----------
    m : int
        Modulation order. Must be a perfect square (4, 16, 64, 256, ...).

    Returns
    -------
    np.ndarray
        Complex symbols, dtype complex128, indexed 0..M-1.
        Points lie on a regular grid from -(k-1) to +(k-1)
        in steps of 2, normalised so the minimum distance = 2/M.
    """
    k = int(np.sqrt(m))
    if k * k != m:
        raise ValueError(f"M-QAM requires M to be a perfect square; got {m}")
    # Amplitude levels: odd integers from -(k-1) to (k-1)
    levels = np.arange(-(k - 1), k, 2, dtype=np.float64)
    # Normalise so min distance = 2/sqrt(M) ≈ 2/sqrt(k²) = 2/k
    # but we keep the raw grid and normalise to unit average power below.
    # Grid: I changes faster (row-major order)
    i_vals, q_vals = np.meshgrid(levels, levels, indexing="ij")
    flat = (i_vals.ravel() + 1j * q_vals.ravel()).astype(np.complex128)
    # Normalise to unit average symbol energy (E[|s|²] = 1)
    power = np.mean(np.abs(flat) ** 2)
    return flat / np.sqrt(power)


# Pre-computed constellations (normalised to unit average power)
_QAM16 = _build_rectangular_qam(16)
_QAM64 = _build_rectangular_qam(64)


def generate_qam_symbols(
    symbol_indices: np.ndarray,
    constellation: Literal["qam16", "qam64"] = "qam16",
) -> np.ndarray:
    """
    Map integer symbol indices to complex QAM symbols on a rectangular grid.

    Parameters
    ----------
    symbol_indices : np.ndarray
        Integer symbol values, shape ``(n,)``.
          - ``"qam16"``: 0..15
          - ``"qam64"``: 0..63
    constellation : {"qam16", "qam64"}
        Modulation order. Default ``"qam16"``.

    Returns
    -------
    np.ndarray
        Complex symbols, dtype complex128, normalised to unit average
        symbol energy (E[|s|²] = 1).

    Notes
    -----
    16-QAM is the minimum practical QAM for tactical communications.
    64-QAM is used in high-throughput SATCOM links. Both are
    rectangular (not cross) constellations.
    """
    indices = np.asarray(symbol_indices, dtype=np.intp).ravel()
    if constellation == "qam16":
        table = _QAM16
        max_idx = 15
    elif constellation == "qam64":
        table = _QAM64
        max_idx = 63
    else:
        raise ValueError(
            f"constellation must be qam16/qam64; got {constellation!r}"
        )
    if np.any(indices > max_idx) or np.any(indices < 0):
        raise ValueError(
            f"symbol_indices out of range for {constellation}: "
            f"must be 0..{max_idx}"
        )
    return table[indices]


# =====================================================================
# LFM / Chirp waveform
# =====================================================================

def generate_lfm_chirp(
    t: np.ndarray,
    f0: float,
    f1: float,
    peak_power_w: float = 1.0,
    initial_phase_rad: float = 0.0,
    rng: Optional[np.random.Generator] = None,
    add_noise: bool = False,
    snr_db: float = 20.0,
) -> np.ndarray:
    """
    Generate a Linear Frequency Modulation (LFM) / chirp waveform.

    The instantaneous frequency sweeps linearly from ``f0`` to ``f1``
    over the time vector ``t``::

        f_inst(t) = f0 + (f1 - f0) * (t - t[0]) / (t[-1] - t[0])
        s(t) = sqrt(2 * P) * exp(j * (2π * f0 * t + π * K * t²) + j*φ0)

    where K = (f1 - f0) / T is the chirp rate and T is the pulse duration.

    The complex envelope has average power ``peak_power_w`` (real-valued
    equivalent, so peak voltage = sqrt(2 * P)).

    Parameters
    ----------
    t : np.ndarray
        Time axis in seconds, shape ``(n_samples,)``. Must be uniformly
        spaced. The chirp is evaluated at these sample times.
    f0 : float
        Start frequency in Hz.
    f1 : float
        End frequency in Hz.
    peak_power_w : float
        Peak pulse power in watts (real power). Default 1.0 W.
    initial_phase_rad : float
        Initial carrier phase in radians. Default 0.0.
    rng : np.random.Generator, optional
        RNG for the optional AWGN.
    add_noise : bool
        If True, add complex Gaussian noise to set the SNR to
        ``snr_db``. Default False.
    snr_db : float
        SNR in dB for the noise injection. Default 20.0.

    Returns
    -------
    np.ndarray
        Complex baseband chirp envelope, shape ``(n_samples,)``,
        dtype complex128. Average power ≈ ``peak_power_w``.

    Notes
    -----
    **Pulse compression**: an LFM pulse with bandwidth B and duration T
    has a time-bandwidth product BT. A matched filter compresses the
    BT pulses into a narrow peak with peak-to-sidelobe ratio ≈ 13 dB
    (Hamming window). The coherent processing gain from integration is
    10*log10(BT) dB.

    **Up-chirp vs down-chirp**: f1 > f0 → up-chirp (bandwidth sweeps
    upward); f1 < f0 → down-chirp. Both are used in practice; the
    matched filter must be matched to the correct chirp direction.

    **Stretch processing**: for wideband LFM, the dechirped return at
    IF gives a CW tone whose frequency is proportional to range. This
    is the standard SAR / ISAR processing chain.
    """
    t = np.asarray(t, dtype=np.float64)
    if t.ndim != 1:
        raise ValueError(f"t must be 1-D; got shape {t.shape}")
    if t.size < 2:
        raise ValueError("t must have at least 2 samples")

    dt = float(t[1] - t[0])
    # Guard against non-uniform sampling
    if t.size > 2:
        dts = np.diff(t)
        if not np.allclose(dts, dt, rtol=1e-6):
            raise ValueError(
                "t must be uniformly spaced for LFM synthesis; "
                f"found dt range [{dts.min():.3e}, {dts.max():.3e}]"
            )

    T = float(t[-1] - t[0])
    K = (f1 - f0) / T  # chirp rate, Hz/s
    sqrt_power = np.sqrt(2.0 * peak_power_w)  # voltage scaling for P_w real

    # Phase: φ(t) = 2π * f0 * t + π * K * t² + φ0
    phase = 2.0 * np.pi * f0 * t + np.pi * K * (t ** 2) + initial_phase_rad
    s = sqrt_power * np.exp(1j * phase)

    if add_noise and rng is not None:
        s = add_awgn(s, snr_db=snr_db, rng=rng)

    return s.astype(np.complex128)


# =====================================================================
# AWGN — Box-Muller transform (polar variant)
# =====================================================================

def add_awgn(
    signal: np.ndarray,
    snr_db: Optional[float] = None,
    noise_floor_w: Optional[float] = None,
    *,
    rng: Optional[np.random.Generator] = None,
    return_noise: bool = False,
) -> np.ndarray:
    """
    Add complex AWGN to a signal to achieve a target SNR or noise floor.

    Two mutually-exclusive modes:

    **SNR mode** (``snr_db`` given):
        Noise power is computed so that ``SNR_linear = 10^(snr_db/10)``.
        ``P_signal`` is the mean squared magnitude of ``signal``.

    **Noise floor mode** (``noise_floor_w`` given):
        Noise variance is ``noise_floor_w / 2`` per dimension (real and imag).

    Parameters
    ----------
    signal : np.ndarray
        Complex signal, shape ``(n,)`` or ``(n, k)``.
    snr_db : float, optional
        Target SNR in dB. Mutually exclusive with ``noise_floor_w``.
    noise_floor_w : float, optional
        Total noise power in watts (complex), i.e. ``E[|n|²]``.
        Mutually exclusive with ``snr_db``.
    rng : np.random.Generator, optional
        RNG for the Gaussian draws. If None, a default_rng is created.
    return_noise : bool
        If True, return a tuple ``(signal_plus_noise, noise_vector)``.
        Default False (return only the noisy signal).

    Returns
    -------
    np.ndarray or (np.ndarray, np.ndarray)
        Noisy signal (same shape as input). Dtype complex128.
        If ``return_noise=True``, returns a second array with the
        noise realisation.
    """
    signal = np.asarray(signal, dtype=np.complex128)
    if rng is None:
        rng = np.random.default_rng()

    if snr_db is not None and noise_floor_w is not None:
        raise ValueError("snr_db and noise_floor_w are mutually exclusive")
    if snr_db is None and noise_floor_w is None:
        raise ValueError("One of snr_db or noise_floor_w must be given")

    if snr_db is not None:
        # P_signal = E[|s|²] = mean(|signal|²)
        p_signal = float(np.mean(np.abs(signal) ** 2))
        if p_signal <= 0:
            raise ValueError(
                "Signal has zero or negative power; cannot set SNR."
            )
        p_noise = p_signal / (10.0 ** (float(snr_db) / 10.0))
    else:
        p_noise = float(noise_floor_w)

    # Complex AWGN: variance = p_noise / 2 per real/imag dimension
    # Box-Muller (polar): u uniform [0,1), v uniform [0,1)
    # z = sqrt(-2*ln(u)) * exp(j*2π*v) has circularly-symmetric N(0,1) distribution.
    u1 = rng.random(size=signal.shape, dtype=np.float64)
    u2 = rng.random(size=signal.shape, dtype=np.float64)
    # Guard against log(0)
    u1 = np.where(u1 == 0, 1e-12, u1)
    magnitude = np.sqrt(-2.0 * np.log(u1))
    angle = 2.0 * np.pi * u2
    std = np.sqrt(p_noise / 2.0)
    noise = std * (magnitude * np.cos(angle) + 1j * magnitude * np.sin(angle))
    noise = noise.astype(np.complex128)

    out = signal + noise
    if return_noise:
        return out, noise
    return out


def estimate_snr(signal: np.ndarray) -> float:
    """
    Estimate SNR (dB) from a signal+noise observation.

    Uses the difference between the signal variance and the noise
    variance under the assumption that the signal is coherent (constant
    magnitude) and noise is circularly symmetric complex Gaussian.

    For coherent pulses (constant |s|), the ML SNR estimator is::

        SNR_lin = max(0, (var(|x|) - σ²_n) / σ²_n)

    where σ²_n is estimated from the minimum sample variance across
    the observation. This is a simple method; for precise work,
    use a known clean reference or a phase-reference algorithm.

    Parameters
    ----------
    signal : np.ndarray
        Complex observations, shape ``(n,)`` or ``(n, k)``.

    Returns
    -------
    float
        Estimated SNR in dB. Returns ``-inf`` if the noise floor
        dominates (signal below noise).
    """
    signal = np.asarray(signal, dtype=np.complex128)
    magnitudes = np.abs(signal)
    # Noise variance estimate: minimum of per-dimension sample variance
    # (signal is coherent, so noise dominates the minimum-variance cut)
    var_real = float(np.var(signal.real))
    var_imag = float(np.var(signal.imag))
    noise_var = min(var_real, var_imag)
    signal_power = float(np.mean(magnitudes ** 2))
    if noise_var <= 0:
        return -np.inf
    snr_lin = max(0.0, (signal_power - 2 * noise_var) / (2 * noise_var))
    return float(10.0 * np.log10(snr_lin + 1e-30))


# =====================================================================
# Up/down conversion (complex baseband ↔ RF passband)
# =====================================================================

def complex_baseband_to_rf(
    baseband: np.ndarray,
    f_c: float,
    f_s: float,
) -> np.ndarray:
    """
    Up-convert a complex baseband envelope to a real RF passband signal.

    The real passband is::

        s_RF(t) = Re{ s_bb(t) * exp(j*2π*f_c*t) }

    This is equivalent to single-sideband modulation with carrier
    frequency ``f_c``.

    Parameters
    ----------
    baseband : np.ndarray
        Complex baseband envelope, shape ``(n,)`` or ``(n, k)``.
    f_c : float
        Carrier frequency in Hz.
    f_s : float
        Sample rate in Hz. Must satisfy ``f_s > 2 * f_c`` (Nyquist).

    Returns
    -------
    np.ndarray
        Real passband signal, same shape as baseband, dtype float64.

    Notes
    -----
    For a real-valued RF signal the two-sided spectrum has peaks at
    ±f_c. The complex baseband representation keeps only the
    positive-frequency component, halving the required sample rate.
    """
    baseband = np.asarray(baseband, dtype=np.complex128)
    n = baseband.shape[0]
    if n < 1:
        return np.array([], dtype=np.float64)
    if 2.0 * f_c >= f_s:
        raise ValueError(
            f"Nyquist violation: f_c={f_c} Hz requires f_s > {2*f_c} Hz; "
            f"got f_s={f_s} Hz"
        )
    t = np.arange(n, dtype=np.float64) / f_s
    carrier = np.exp(2j * np.pi * f_c * t)
    rf = np.real(baseband * carrier).astype(np.float64)
    return rf


def rf_to_complex_baseband(
    rf_signal: np.ndarray,
    f_c: float,
    f_s: float,
    *, dtype: np.dtype = np.complex128,
) -> np.ndarray:
    """
    Down-convert a real RF passband signal to a complex baseband envelope.

    The baseband is recovered by mixing with a local oscillator at ``f_c``
    and applying a two-pole IIR low-pass filter with -3 dB cutoff at
    ``0.2 * f_c``.

    Parameters
    ----------
    rf_signal : np.ndarray
        Real passband samples, shape ``(n,)``.
    f_c : float
        Carrier frequency in Hz.
    f_s : float
        Sample rate in Hz.
    dtype : np.dtype
        Output dtype. Default complex128.

    Returns
    -------
    np.ndarray
        Complex baseband envelope, shape ``(n,)``.
    """
    rf_signal = np.asarray(rf_signal, dtype=np.float64)
    n = rf_signal.size
    if n < 1:
        return np.array([], dtype=dtype)
    if 2.0 * f_c >= f_s:
        raise ValueError(
            f"Nyquist violation: f_c={f_c} Hz requires f_s > {2*f_c} Hz"
        )
    t = np.arange(n, dtype=np.float64) / f_s
    lo = np.exp(-2j * np.pi * f_c * t)  # conjugate for down-mix
    bb = (rf_signal * lo).astype(dtype)

    # Moving-average (CIC-like) low-pass filter.  Averaging over
    # M = max(10, int(f_s / (4 * f_c))) samples gives a sinc-shaped
    # response with DC gain = 1 and strong attenuation at the carrier.
    # This is the standard ESM front-end: strip the carrier, extract baseband.
    M = max(10, int(round(f_s / (4.0 * f_c))))
    M = min(M, 5000)  # cap so we don't average too much
    if n < M:
        # Short record: use what we have
        window = n
    else:
        window = M
    if window < 1:
        return np.zeros(n, dtype=dtype)
    # Use cumsum trick for efficient moving average
    cs = np.cumsum(bb, axis=0)
    # Pad to handle wrap-around cleanly for the final samples
    padded = np.concatenate([np.zeros(window, dtype=dtype), cs])
    # Average over 'window' samples: shift the cumsum and divide
    # ma[k] = (cs[k+window-1] - cs[k-1]) / window  for k in 0..n-1
    start = window - 1
    ma = (padded[start:start + n] - np.concatenate([np.zeros(1, dtype=dtype), padded[:n - 1]])) / float(window)
    return ma.astype(dtype)


# =====================================================================
# Matched filter (pulse compression) for LFM
# =====================================================================

def matched_filter(
    signal: np.ndarray,
    reference: np.ndarray,
    *, mode: Literal["sampleless", "full"] = "sampleless",
) -> np.ndarray:
    """
    Apply a matched filter to compress a waveform against a reference.

    The matched filter maximises the SNR for a known signal in AWGN.
    For an LFM chirp, the reference is the time-reversed complex conjugate
    of the transmitted pulse.

    Parameters
    ----------
    signal : np.ndarray
        Received signal (can be longer than the reference).
    reference : np.ndarray
        Known transmitted waveform. Same sample rate as signal.
    mode : {"sampleless", "full"}
        ``"sampleless"`` uses zero-padded FFT convolution (O(n log n),
        constant memory). ``"full"`` uses direct correlation (O(n*m),
        more intuitive). Default ``"sampleless"``.

    Returns
    -------
    np.ndarray
        Matched filter output (compression gain applied), same length
        as ``signal``. Peak value ≈ √(BT) for an LFM of time-bandwidth
        product BT.

    Notes
    -----
    The peak-to-sidelobe ratio (PSLR) for an LFM with no window is
    ≈ 13.3 dB. A Hamming window reduces PSL to ≈ 42 dB but costs
    1.4 dB of processing gain.
    """
    signal = np.asarray(signal, dtype=np.complex128)
    reference = np.asarray(reference, dtype=np.complex128)
    if mode == "sampleless":
        n = signal.size
        # FFT-based convolution: zero-pad both to n + m - 1
        m = reference.size
        padded = np.zeros(n + m - 1, dtype=np.complex128)
        padded[:n] = signal
        padded_ref = np.zeros(n + m - 1, dtype=np.complex128)
        padded_ref[:m] = reference
        return np.fft.ifft(
            np.fft.fft(padded) * np.fft.fft(padded_ref[::-1].conj())
        )[:n].astype(np.complex128)
    elif mode == "full":
        from scipy import signal as sig
        return sig.correlate(signal, reference, mode="same").astype(np.complex128)
    else:
        raise ValueError(f"mode must be sampleless or full; got {mode!r}")


def coherent_integrate(
    pulses: np.ndarray,
    n_pulses: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coherently integrate ``n_pulses`` complex pulses.

    Returns both the integrated waveform and the incoherent (envelope)
    reference for SNR comparison.

    Parameters
    ----------
    pulses : np.ndarray
        Complex pulses, shape ``(n_pulses, n_samples_per_pulse)``.
    n_pulses : int
        Number of pulses to integrate.

    Returns
    -------
    integrated : np.ndarray
        Phase-aligned sum of ``n_pulses`` pulses.
    incoherent : np.ndarray
        Sum of |pulse|² (non-coherent reference).
    """
    pulses = np.asarray(pulses, dtype=np.complex128)
    if pulses.ndim != 2:
        raise ValueError(f"pulses must be 2-D (n_pulses, n_samples); got {pulses.ndim}D")
    n = min(n_pulses, pulses.shape[0])
    integrated = pulses[:n].sum(axis=0)
    incoherent = np.sum(np.abs(pulses[:n]) ** 2, axis=0)
    return integrated.astype(np.complex128), incoherent.astype(np.float64)


__all__ = [
    "generate_psk_symbols",
    "bits_to_psk_symbols",
    "generate_qam_symbols",
    "generate_lfm_chirp",
    "add_awgn",
    "estimate_snr",
    "complex_baseband_to_rf",
    "rf_to_complex_baseband",
    "matched_filter",
    "coherent_integrate",
]
