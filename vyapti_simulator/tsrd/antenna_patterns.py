"""
vyapti_simulator.tsrd.antenna_patterns
======================================

Antenna gain pattern generators for realistic EW receiver modelling.

In a real EW system the antenna gain is not uniform: a sectorised
rotating antenna has high-gain main lobes and lower-gain sidelobes,
producing a non-uniform gain across the frequency spectrum. This module
provides generator functions that produce per-band gain arrays for use
with ``DetectionConfig.antenna_gain_db`` and ``SimulatorConfig.antenna_gain_db``.

The gain is applied as an SNR penalty:
    effective_snr_db = measured_snr_db − antenna_gain_db[band]

A band with gain = −6 dB means every pulse in that band appears
6 dB weaker than it would with an isotropic reference antenna.

Usage
-----
::

    from vyapti_simulator.tsrd.antenna_patterns import (
        sectorised_antenna_gain,
        realistic_antenna_gain,
        uniform_antenna_gain,
    )
    from vyapti_simulator.tsrd import DetectionConfig

    gains = sectorised_antenna_gain(band_count=36, n_sectors=4, sector_gain_db=-6.0)
    cfg = DetectionConfig(antenna_gain_db=gains)

References
----------
Skolnik, M. I. "Radar Handbook", 3rd ed. — Antenna gain and sectorised
arrays, Chapter 7. IEEE Std 521-2006 — S-band antenna gain standard.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def uniform_antenna_gain(band_count: int) -> np.ndarray:
    """
    Uniform isotropic reference gain: 0 dB for all bands.

    This is the default (no antenna pattern applied). Use this as the
    baseline when sweeping antenna gain as an experimental variable.

    Parameters
    ----------
    band_count : int
        Number of bands (must match the simulation grid).

    Returns
    -------
    np.ndarray
        Shape ``(band_count,)``, dtype float64. All values are 0.0.
    """
    return np.zeros(band_count, dtype=np.float64)


def sectorised_antenna_gain(
    band_count: int,
    n_sectors: int = 4,
    sector_gain_db: float = -6.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Rotating sector antenna gain pattern.

    A mechanically rotating EW antenna has a main beam that sweeps
    through 360°. At any instant, the receiver sees:
      - One active sector at ~0 dB (main beam)
      - ``n_sectors − 1`` inactive sectors at ``sector_gain_db`` dB
        (sidelobes)

    The sectors are evenly spaced in band space (not angular space),
    which models a frequency-sectorised array (not a rotating dish).
    For a frequency-scanning radar or a broadband EW antenna, the
    sectors map directly to frequency bands.

    Parameters
    ----------
    band_count : int
        Number of bands.
    n_sectors : int
        Number of sectors (4 = quadrant, 6 = 60° sectors).
    sector_gain_db : float
        Gain of non-active sectors in dB relative to the main beam.
        Typical values:
          - −3 dB  — mild sidelobes
          - −6 dB  — typical sectorised array
          - −10 dB — aggressive sidelobe suppression
    seed : int, optional
        RNG seed for slight randomisation of sector boundaries.

    Returns
    -------
    np.ndarray
        Shape ``(band_count,)``. Active sector = 0.0 dB,
        inactive sectors = ``sector_gain_db``.
    """
    if n_sectors < 1:
        raise ValueError(f"n_sectors must be >= 1, got {n_sectors}")
    if sector_gain_db > 0:
        raise ValueError(
            f"sector_gain_db must be <= 0 (relative to main beam), got {sector_gain_db}"
        )
    bands_per_sector = band_count / n_sectors
    rng = np.random.default_rng(seed)
    gains = np.full(band_count, sector_gain_db, dtype=np.float64)
    # One sector is the "active" (main beam) sector
    active_sector = rng.integers(0, n_sectors)
    start = int(round(active_sector * bands_per_sector))
    end = int(round((active_sector + 1) * bands_per_sector))
    gains[start:end] = 0.0
    return gains


def realistic_antenna_gain(
    band_count: int,
    seed: int = 0,
    base_gain_db: float = 0.0,
    ripple_db: float = 3.0,
    n_ripples: int = 2,
    sidelobe_db: float = -8.0,
) -> np.ndarray:
    """
    Realistic antenna gain pattern with smooth sinusoidal variation and sidelobes.

    Models a real wideband antenna (e.g. conical spiral, log-periodic)
    whose gain varies with frequency. The pattern has:
      - A smooth sinusoidal ripple (modelling the antenna's frequency response)
      - Occasional deep nulls (sidelobes, −8 dB relative to mean)

    This is the most physically realistic option for EW receiver modelling.

    Parameters
    ----------
    band_count : int
        Number of bands.
    seed : int
        RNG seed for reproducibility.
    base_gain_db : float
        Mean gain in dB. Default 0.0 = isotropic reference.
    ripple_db : float
        Peak-to-peak amplitude of the smooth sinusoidal ripple in dB.
        Default 3.0 dB peak-to-peak ≈ ±1.5 dB around the mean.
    n_ripples : int
        Number of full sinusoidal cycles across the band.
        Default 2 = two gain peaks across the frequency range.
    sidelobe_db : float
        Gain of deep nulls (sidelobes) in dB. Default −8.0.
        Typical real antennas have −10 to −15 dB sidelobes.

    Returns
    -------
    np.ndarray
        Shape ``(band_count,)``. Values range from
        ``base_gain_db − ripple_db/2`` to ``base_gain_db + ripple_db/2``,
        with occasional drops to ``sidelobe_db``.
    """
    rng = np.random.default_rng(seed)
    bands = np.arange(band_count, dtype=np.float64)
    # Smooth sinusoidal ripple across the frequency range
    ripple = (ripple_db / 2.0) * np.sin(
        2.0 * np.pi * n_ripples * bands / band_count
    )
    # Random deep nulls (sidelobes) at ~1 per n_ripples bands
    gain = base_gain_db + ripple
    null_prob_per_band = 0.05
    for i in range(band_count):
        if rng.random() < null_prob_per_band:
            gain[i] = sidelobe_db
    # Clip extreme values for physical sanity
    gain = np.clip(gain, sidelobe_db - 2.0, base_gain_db + ripple_db)
    return gain.astype(np.float64)


def uniform_sectorised_antenna_gain(
    band_count: int,
    n_sectors: int = 4,
    sector_gain_db: float = -6.0,
) -> np.ndarray:
    """
    Deterministic sectorised gain with no randomisation.

    Like ``sectorised_antenna_gain`` but without the RNG seed.
    Sector 0 is always the active sector. Use this for
    deterministic reproducible experiments.

    Parameters
    ----------
    band_count, n_sectors, sector_gain_db
        Same as ``sectorised_antenna_gain``.

    Returns
    -------
    np.ndarray
        Sector 0 is active (0 dB), sectors 1..n_sectors−1 are
        at ``sector_gain_db``.
    """
    if band_count % n_sectors != 0:
        raise ValueError(
            f"band_count ({band_count}) must be divisible by n_sectors ({n_sectors})"
        )
    bands_per_sector = band_count // n_sectors
    gains = np.full(band_count, sector_gain_db, dtype=np.float64)
    gains[:bands_per_sector] = 0.0
    return gains


# =====================================================================
# Frequency-dependent antenna gain models
# =====================================================================
# The original three helpers (sectorised / realistic / uniform) only
# produce a per-band gain array indexed 0..band_count-1. The new
# models in this section take an actual frequency axis (MHz) so the
# gain is an honest function of the pulse frequency, not just the
# band index. They are useful for hardware-in-the-loop studies where
# the antenna's real response is known.

def cosine_taper_antenna_gain(
    freq_mhz: np.ndarray,
    *,
    center_freq_mhz: Optional[float] = None,
    peak_gain_db: float = 6.0,
    edge_attenuation_db: float = -10.0,
) -> np.ndarray:
    """
    Cosine-tapered antenna gain across frequency.

    Models a typical wideband EW antenna (e.g. log-periodic or
    discone) whose gain peaks at the centre frequency and rolls
    off at the band edges. The shape is::

        G(f) = edge + (peak - edge) * cos(pi * (f - f_lo) / (f_hi - f_lo))^2

    where ``f_lo`` and ``f_hi`` are the min and max of ``freq_mhz``.

    Parameters
    ----------
    freq_mhz : np.ndarray
        Centre frequency of each band in MHz, shape ``(band_count,)``.
    center_freq_mhz : float, optional
        Centre of the passband. Default = midpoint of ``freq_mhz``.
    peak_gain_db : float
        Gain at the centre frequency in dB. Default +6 dB.
    edge_attenuation_db : float
        Gain at the band edges in dB. Default -10 dB.

    Returns
    -------
    np.ndarray
        Per-band gain in dB, shape ``(band_count,)``.
    """
    freq_mhz = np.asarray(freq_mhz, dtype=np.float64)
    if freq_mhz.size == 0:
        return np.array([], dtype=np.float64)
    f_lo = float(freq_mhz.min())
    f_hi = float(freq_mhz.max())
    if f_hi <= f_lo:
        return np.full(freq_mhz.shape, peak_gain_db, dtype=np.float64)
    if center_freq_mhz is None:
        center_freq_mhz = 0.5 * (f_lo + f_hi)
    # Normalised distance from centre, in [-1, +1]
    x = (freq_mhz - center_freq_mhz) / (0.5 * (f_hi - f_lo))
    x = np.clip(x, -1.0, 1.0)
    gain = edge_attenuation_db + (peak_gain_db - edge_attenuation_db) * np.cos(0.5 * np.pi * x) ** 2
    return gain.astype(np.float64)


def sinc_antenna_gain(
    freq_mhz: np.ndarray,
    *,
    center_freq_mhz: Optional[float] = None,
    peak_gain_db: float = 8.0,
    null_depth_db: float = -25.0,
) -> np.ndarray:
    """
    Sinc-squared (uniform-aperture) antenna gain across frequency.

    The radiation pattern of a uniformly-illuminated linear aperture
    is sinc-squared off boresight. Across frequency, a band-limited
    feed produces a similar roll-off with nulls at harmonics of
    the design frequency.

        G(f) = peak + null_depth * sinc(pi * N * (f - f_centre) / f_centre)

    with ``N=2`` for a typical 2-element sub-array. Nulls give a
    realistic deep fade signature.

    Parameters
    ----------
    freq_mhz : np.ndarray
        Centre frequency of each band in MHz, shape ``(band_count,)``.
    center_freq_mhz : float, optional
        Centre frequency of the passband. Default = midpoint.
    peak_gain_db : float
        Gain at the centre frequency. Default +8 dB.
    null_depth_db : float
        Gain at the first sinc null. Default -25 dB.

    Returns
    -------
    np.ndarray
        Per-band gain in dB, shape ``(band_count,)``.
    """
    freq_mhz = np.asarray(freq_mhz, dtype=np.float64)
    if freq_mhz.size == 0:
        return np.array([], dtype=np.float64)
    f_lo = float(freq_mhz.min())
    f_hi = float(freq_mhz.max())
    if f_hi <= f_lo:
        return np.full(freq_mhz.shape, peak_gain_db, dtype=np.float64)
    if center_freq_mhz is None:
        center_freq_mhz = 0.5 * (f_lo + f_hi)
    # Arg to sinc: zero at center_freq_mhz, first null at the band edge
    # Choose N so first null sits at the band edge.
    n_aperture = 2.0
    arg = np.pi * n_aperture * (freq_mhz - center_freq_mhz) / max(center_freq_mhz, 1e-9)
    # sinc(x) = sin(x) / x, with sinc(0) = 1
    sinc = np.ones_like(arg)
    nz = np.abs(arg) > 1e-12
    sinc[nz] = np.sin(arg[nz]) / arg[nz]
    # Sinc-squared envelope in linear, then map to dB
    env_lin = sinc ** 2
    # Map: peak (env_lin=1) -> peak_gain_db; first null (env_lin=0) -> null_depth_db
    # Use a floor so deep nulls don't go to -inf
    env_lin = np.clip(env_lin, 10.0 ** (null_depth_db / 10.0), 1.0)
    gain = peak_gain_db + 10.0 * np.log10(env_lin)
    return gain.astype(np.float64)


def realistic_ew_antenna_gain(
    freq_mhz: np.ndarray,
    *,
    center_freq_mhz: Optional[float] = None,
    peak_gain_db: float = 5.0,
    edge_attenuation_db: float = -8.0,
    ripple_amplitude_db: float = 1.5,
    ripple_period_mhz: float = 200.0,
    sidelobe_db: float = -15.0,
    sidelobe_probability: float = 0.04,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Realistic broadband EW antenna with cosine taper + ripple + sidelobes.

    This is the frequency-domain equivalent of ``realistic_antenna_gain``
    (which is band-index-based). The frequency axis is used so the
    ripple period is in MHz, making the model portable across
    different band grids.

    The model sums three components:
      1. **Cosine taper**: passband shape with peak at the centre.
      2. **Sinusoidal ripple**: standing-wave pattern at
         ``ripple_period_mhz`` intervals (typical for log-periodic
         feeds where element resonances interact).
      3. **Random deep nulls (sidelobes)**: at ``sidelobe_probability``
         per band.

    Parameters
    ----------
    freq_mhz : np.ndarray
        Centre frequency of each band in MHz, shape ``(band_count,)``.
    center_freq_mhz : float, optional
        Centre of the passband. Default = midpoint.
    peak_gain_db : float
        Gain at the centre frequency. Default +5 dB.
    edge_attenuation_db : float
        Gain at the band edges. Default -8 dB.
    ripple_amplitude_db : float
        Peak-to-peak amplitude of the standing-wave ripple. Default 1.5 dB.
    ripple_period_mhz : float
        Period of the ripple in MHz. Default 200 MHz.
    sidelobe_db : float
        Gain of randomly-placed deep nulls. Default -15 dB.
    sidelobe_probability : float
        Probability of a sidelobe null per band. Default 0.04.
    seed : int, optional
        RNG seed for the sidelobe placement.

    Returns
    -------
    np.ndarray
        Per-band gain in dB, shape ``(band_count,)``.
    """
    freq_mhz = np.asarray(freq_mhz, dtype=np.float64)
    if freq_mhz.size == 0:
        return np.array([], dtype=np.float64)

    # 1. Cosine taper (passband shape)
    taper = cosine_taper_antenna_gain(
        freq_mhz,
        center_freq_mhz=center_freq_mhz,
        peak_gain_db=peak_gain_db,
        edge_attenuation_db=edge_attenuation_db,
    )

    # 2. Sinusoidal ripple (standing-wave pattern)
    phase = 2.0 * np.pi * (freq_mhz - freq_mhz.min()) / max(ripple_period_mhz, 1e-9)
    ripple = (ripple_amplitude_db / 2.0) * np.sin(phase)
    gain = taper + ripple

    # 3. Random deep nulls (sidelobes)
    rng = np.random.default_rng(seed)
    if sidelobe_probability > 0:
        mask = rng.random(freq_mhz.shape) < sidelobe_probability
        gain = np.where(mask, sidelobe_db, gain)

    return gain.astype(np.float64)
