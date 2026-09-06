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
