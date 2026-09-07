"""
vyapti_simulator.rf.swerling
=============================

Swerling target fluctuation models for per-pulse RCS (radar cross
section) variation. The Swerling models describe the statistical
distribution of received amplitude from a fluctuating target and are
the canonical extension of the Marcum (non-fluctuating) case used in
classical radar detection theory.

Models
------
Swerling 0 (Marcum, no fluctuation):
    Constant RCS, deterministic amplitude. The amplitude is unchanged
    from the link budget.

Swerling I:
    Slow fluctuation. RCS is constant within a burst (set of pulses
    with the same beam dwell) and decorrelates between bursts. PDF:
    exponential. Decorrelation time >> PRI.

Swerling II:
    Fast fluctuation. RCS decorrelates per pulse. PDF: exponential.
    Decorrelation time << PRI.

Swerling III:
    Slow fluctuation. PDF: 4-DoF chi-squared (chi-square with 4 DoF
    is equivalent to the square of a Rician envelope with K=0). The
    PDF is concentrated near the mean with relatively fewer low-amplitude
    samples compared to Swerling I/II.

Swerling IV:
    Fast fluctuation. PDF: 4-DoF chi-squared (same as Swerling III
    distribution but with per-pulse decorrelation instead of per-burst).

References
----------
Swerling, P. (1954). "Detection of Fluctuating Pulsed Signals in the
    Presence of Noise". IRE Trans. IT-3, pp. 175-183.
Swerling, P. (1960). "Probability of Detection for Fluctuating Targets".
    IRE Trans. IT-6, pp. 269-308.
DiFranco, J. V. & Rubin, W. L. (1980). "Radar Detection". Artech House.
Skolnik, M. I. (2001). "Introduction to Radar Systems". 3rd ed.
    McGraw-Hill. (Swerling cases I-IV discussed in Chapter 2.)

Usage
-----
::

    from vyapti_simulator.rf.swerling import apply_swerling_fluctuation
    import numpy as np

    # 50 pulses from one emitter, Swerling II (per-pulse exponential)
    rng = np.random.default_rng(42)
    base_amp_db = np.full(50, -90.0)  # -90 dBm link budget
    amp_db = apply_swerling_fluctuation(
        amp_db=base_amp_db,
        model="swerling_2",
        rng=rng,
    )
    # amp_db has per-pulse fluctuation ~ N(0, 5.57 dB) (exp -> -1.5 dB mean)

For Swerling I/III (per-burst): pass the same burst_size as the
deinterleaver's expected pulses-per-burst (e.g. ``burst_size=10`` to
group every 10 pulses into one burst with shared RCS).
"""

from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np


# Type alias for the supported Swerling model names.
SwerlingModel = Literal[
    "swerling_0", "swerling_1", "swerling_2", "swerling_3", "swerling_4",
]


# Standard deviation of 10*log10(X) where X ~ Exponential(1).
# E[X] = 1, Var[X] = 1, so E[10*log10(X)] = 10*log10(e) * E[ln(X)] =
# -1.506 dB (the mean Swerling loss); std = 10*log10(e) * sqrt(pi^2/6)
# = 5.571 dB.
_EXP_MEAN_DB = -1.506
_EXP_STD_DB = 5.571


def _spectrum_exponential_fluctuation(
    n_pulses: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return ``n_pulses`` linear power samples from the standard
    exponential distribution (mean = 1, unit variance).
    """
    return rng.exponential(scale=1.0, size=n_pulses).astype(np.float64)


def _chi_squared_4dof(
    n_pulses: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Return ``n_pulses`` linear power samples from a chi-squared
    distribution with 4 degrees of freedom, scaled to mean 1.

    A chi-squared 4-DoF variable is the sum of squares of two
    independent standard normal variables. The mean of chi-squared
    with 4 DoF is 4, so we divide by 4 to normalise to mean 1.
    """
    chi2_4 = rng.chisquare(df=4, size=n_pulses).astype(np.float64)
    return chi2_4 / 4.0


def apply_swerling_fluctuation(
    amp_db: np.ndarray,
    model: Union[str, SwerlingModel] = "swerling_0",
    rng: Optional[np.random.Generator] = None,
    *,
    burst_size: int = 1,
) -> np.ndarray:
    """
    Apply Swerling target fluctuation to a sequence of pulse amplitudes
    in dB.

    Parameters
    ----------
    amp_db : np.ndarray
        Per-pulse received amplitude in dBm (or dB above noise floor
        for a relative scale). Shape ``(n_pulses,)``.
    model : str
        One of ``"swerling_0"`` (Marcum, no fluctuation), ``"swerling_1"``
        (slow exponential), ``"swerling_2"`` (fast exponential),
        ``"swerling_3"`` (slow 4-DoF chi-squared), ``"swerling_4"``
        (fast 4-DoF chi-squared).
    rng : np.random.Generator, optional
        Random number generator. If None, a default_rng() is created.
    burst_size : int
        For slow-fluctuation models (Swerling I and III), the number
        of consecutive pulses that share the same RCS draw. After
        every ``burst_size`` pulses, a new independent RCS sample is
        drawn. For fast-fluctuation models (II and IV), this is
        ignored — each pulse gets its own draw. Default 1 (treats
        slow models as fast for convenience).

    Returns
    -------
    np.ndarray
        Amplitude in dB with fluctuation applied, same shape as
        ``amp_db``. The mean RCS is preserved (Swerling 0) or has the
        classical -1.5 dB Swerling loss (cases I-IV).

    Raises
    ------
    ValueError
        If ``model`` is not a valid Swerling case, ``amp_db`` is not
        a 1-D array, or ``burst_size`` < 1.

    Notes
    -----
    The Swerling loss (-1.5 dB mean) means that at the same link
    budget, a fluctuating target requires ~1.5 dB more SNR to achieve
    the same Pd as a Marcum target. The standard deviation of the
    dB-domain amplitude is ~5.6 dB for the exponential cases and
    ~3.4 dB for the 4-DoF chi-squared cases.
    """
    if rng is None:
        rng = np.random.default_rng()
    amp_db = np.asarray(amp_db, dtype=np.float64)
    if amp_db.ndim != 1:
        raise ValueError(
            f"amp_db must be 1-D (one entry per pulse); got shape {amp_db.shape}"
        )
    n_pulses = int(amp_db.size)
    if n_pulses == 0:
        return amp_db.copy()
    if burst_size < 1:
        raise ValueError(f"burst_size must be >= 1; got {burst_size}")

    model_lc = str(model).lower()
    if model_lc not in (
        "swerling_0", "swerling_1", "swerling_2", "swerling_3", "swerling_4",
    ):
        raise ValueError(
            f"model must be one of 'swerling_0'..'swerling_4'; got {model!r}"
        )

    if model_lc == "swerling_0":
        return amp_db.copy()

    # Per-pulse linear power fluctuation. Cases I/III share a draw
    # across `burst_size` consecutive pulses; II/IV draw per pulse.
    if model_lc in ("swerling_2", "swerling_4"):
        n_draws = n_pulses
        draws = np.empty(n_draws, dtype=np.float64)
        for k in range(n_draws):
            if model_lc == "swerling_2":
                draws[k] = _spectrum_exponential_fluctuation(1, rng)[0]
            else:  # swerling_4
                draws[k] = _chi_squared_4dof(1, rng)[0]
    else:  # swerling_1, swerling_3
        n_bursts = (n_pulses + burst_size - 1) // burst_size
        if model_lc == "swerling_1":
            burst_draws = _spectrum_exponential_fluctuation(n_bursts, rng)
        else:  # swerling_3
            burst_draws = _chi_squared_4dof(n_bursts, rng)
        # Tile each burst's draw across `burst_size` pulses.
        draws = np.repeat(burst_draws, burst_size)[:n_pulses]

    # Convert to dB and add to base amplitude.
    # 10*log10(power_fluctuation) is in dB; mean of
    # 10*log10(Exp(1)) = -1.506 dB; mean of
    # 10*log10(ChiSq(4)/4) = 10*log10(1) = 0 dB (since 4-DoF chi-sq
    # with 4 DoF divided by 4 has mean exactly 1, by construction).
    # Note: in the dB domain, the *median* loss differs from the
    # mean, and the distribution is asymmetric. We expose the raw
    # dB draw so the user can see the full empirical distribution.
    fluctuation_db = 10.0 * np.log10(np.maximum(draws, 1e-30))
    return amp_db + fluctuation_db


__all__ = [
    "SwerlingModel",
    "apply_swerling_fluctuation",
]
