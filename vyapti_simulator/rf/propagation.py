"""
vyapti_simulator.rf.propagation
================================

RF propagation physics: free-space path loss, kinematic motion, and
statistical fading models for EW and radar simulation.

The module is organised into three layers matching the physics hierarchy:

  1. **Kinematics** — positions, velocities, and angles between
     emitter and receiver. Drives the time-varying propagation delay
     and Doppler shift.
  2. **Path Loss** — Friis free-space loss, shadowing, and diffraction.
     Applied as a scalar multiplier on received power.
  3. **Fading** — Rayleigh and Rician multipath channel models.
     Applied as a multiplicative complex gain on the signal envelope.

References
----------
Balanis, C. A. (2012). "Antenna Theory". 3rd ed. Wiley. (Friis, ray tracing.)
Proakis, J. G. & Salehi, M. (2008). "Digital Communications". 5th ed.
    (Rayleigh/Rician channel models.)
Skolnik, M. I. (2001). "Introduction to Radar Systems". 3rd ed.
    (Doppler shift.)

Usage
-----
::

    from vyapti_simulator.rf.propagation import (
        FreeSpacePathLoss, RayleighFadingChannel, RicianFadingChannel,
        KinematicEmitter, compute_doppler_shift,
    )
    import numpy as np

    # Emitter moving towards receiver at 150 m/s
    emitter = KinematicEmitter(
        position_m=np.array([500.0, 0.0, 0.0]),  # 500 m away on x-axis
        velocity_m_s=np.array([-150.0, 0.0, 0.0]),  # approaching
    )
    receiver_pos = np.array([0.0, 0.0, 0.0])
    doppler = compute_doppler_shift(emitter, receiver_pos, f_c=3e9)
    print(f"Doppler shift: {doppler:.1f} Hz")

    # Rayleigh multipath fading
    channel = RayleighFadingChannel(f_c=3e9, sample_rate_hz=1e6, rng=np.random.default_rng(0))
    complex_gain = channel.step()  # returns complex multiplicative gain
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np


# =====================================================================
# Physical constants
# =====================================================================
SPEED_OF_LIGHT = 299792458.0  # m/s (exact by definition since 1983)
SPEED_OF_LIGHT_KM_S = SPEED_OF_LIGHT / 1000.0  # km/s


# =====================================================================
# Kinematics
# =====================================================================

@dataclass
class KinematicEmitter:
    """
    A kinematic state for an emitter or target.

    The emitter is tracked in 3D Cartesian space (x, y, z) with
    an instantaneous velocity vector. Position and velocity are updated
    externally by the tick engine; this class only stores and queries them.

    All units are SI: metres, metres/second, seconds.
    """
    position_m: np.ndarray          # shape (3,)
    velocity_m_s: np.ndarray = field(default=None)
    _position_history: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.position_m = np.asarray(self.position_m, dtype=np.float64).reshape(3)
        if self.velocity_m_s is None:
            self.velocity_m_s = np.zeros(3, dtype=np.float64)
        else:
            self.velocity_m_s = np.asarray(self.velocity_m_s, dtype=np.float64).reshape(3)

    def range_m(self, receiver_position_m: np.ndarray) -> float:
        """Current range to the receiver in metres."""
        diff = self.position_m - np.asarray(receiver_position_m, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(diff))

    def range_m(self, receiver_position_m: np.ndarray) -> float:
        """Current range to the receiver in metres."""
        diff = self.position_m - np.asarray(receiver_position_m, dtype=np.float64).reshape(3)
        return float(np.linalg.norm(diff))

    def range_rate_m_s(
        self, receiver_position_m: np.ndarray
    ) -> float:
        """
        Rate of change of range (positive = receding, negative = approaching).

        dr/dt = v · (p - r) / |p - r|
        """
        p = self.position_m
        r = np.asarray(receiver_position_m, dtype=np.float64).reshape(3)
        diff = p - r
        dist = float(np.linalg.norm(diff))
        if dist < 1e-12:
            return 0.0
        return float(np.dot(self.velocity_m_s, diff) / dist)

    def angle_of_arrival_deg(
        self, receiver_position_m: np.ndarray
    ) -> Tuple[float, float]:
        """
        AoA from the emitter to the receiver as (azimuth_deg, elevation_deg).

        Azimuth: 0° = +x axis, 90° = +y axis (ENU frame).
        Elevation: 0° = horizon, +90° = zenith.
        """
        p = self.position_m
        r = np.asarray(receiver_position_m, dtype=np.float64).reshape(3)
        diff = p - r
        x, y, z = diff[0], diff[1], diff[2]
        horizontal = np.sqrt(x**2 + y**2)
        azimuth = np.rad2deg(np.arctan2(y, x))
        elevation = np.rad2deg(np.arctan2(z, horizontal))
        return float(azimuth), float(elevation)

    def update_position(self, dt_s: float) -> None:
        """
        Advance the emitter by ``dt`` seconds using Euler integration.

        Stores the previous position in the history buffer.
        """
        self._position_history.append(self.position_m.copy())
        self.position_m = self.position_m + self.velocity_m_s * dt_s


def compute_doppler_shift(
    emitter: KinematicEmitter,
    receiver_position_m: np.ndarray,
    f_c: float,
) -> float:
    """
    Compute the one-way Doppler frequency shift in Hz.

    For a moving emitter, the received carrier frequency shifts by::

        f_d = (v_r / c) * f_c

    where ``v_r = dr/dt`` is the range rate (negative = approaching,
    positive = receding).

    Parameters
    ----------
    emitter : KinematicEmitter
        Moving emitter state.
    receiver_position_m : np.ndarray
        Receiver 3D position, shape (3,).
    f_c : float
        Carrier frequency in Hz.

    Returns
    -------
    float
        Doppler shift in Hz. Positive = emitter receding (received freq lower).

    Notes
    -----
    For a bistatic radar or a moving receiver, the total Doppler is
    the sum of emitter and receiver contributions::

        f_d = (v_re / c) * f_c + (v_te / c) * f_c

    The one-way form here is the standard ESM case (stationary receiver,
    moving emitter/target).
    """
    v_r = emitter.range_rate_m_s(receiver_position_m)
    return float((v_r / SPEED_OF_LIGHT) * f_c)


def time_of_flight(
    emitter_position_m: np.ndarray,
    receiver_position_m: np.ndarray,
) -> float:
    """
    Compute the one-way time-of-flight delay in seconds.

    τ = |p_e - p_r| / c

    Parameters
    ----------
    emitter_position_m, receiver_position_m : np.ndarray
        3D positions in metres, shape (3,).

    Returns
    -------
    float
        Time of flight in seconds.
    """
    diff = (
        np.asarray(emitter_position_m, dtype=np.float64).reshape(3)
        - np.asarray(receiver_position_m, dtype=np.float64).reshape(3)
    )
    return float(np.linalg.norm(diff) / SPEED_OF_LIGHT)


def fractional_delay_filter(
    n_taps: int,
    delay_samples: float,
    window: Literal["kaiser", "hamming", "blackman"] = "kaiser",
    beta: float = 8.0,
) -> np.ndarray:
    """
    Build an FIR low-pass filter that implements a fractional-sample delay.

    The filter approximates a pure delay of ``delay_samples`` by using
    sinc interpolation with a window. This is needed when the ToA between
    emitter and receiver falls between sample boundaries.

    Parameters
    ----------
    n_taps : int
        Number of FIR taps. Must be odd. Higher = more accurate delay.
    delay_samples : float
        Fractional delay in samples. Can be non-integer. Positive
        delays the signal.
    window : {"kaiser", "hamming", "blackman"}
        Windowing function. Kaiser is best for controlling sidelobes.
    beta : float
        Kaiser beta parameter. Higher = narrower mainlobe, higher sidelobes.
        Default 8.0.

    Returns
    -------
    np.ndarray
        FIR filter coefficients, dtype complex128.
    """
    if n_taps % 2 == 0:
        raise ValueError(f"n_taps must be odd; got {n_taps}")
    n = n_taps
    centre = (n - 1) / 2  # group delay in samples

    # Ideal sinc impulse response: h[k] = sinc(k - delay)
    k = np.arange(n, dtype=np.float64)  # tap indices 0..n-1
    h_ideal = np.sinc(k - delay_samples)

    # Window
    if window == "kaiser":
        win = np.kaiser(n, beta)
    elif window == "hamming":
        win = np.hamming(n)
    elif window == "blackman":
        win = np.blackman(n)
    else:
        raise ValueError(f"window must be kaiser/hamming/blackman; got {window!r}")

    h_windowed = (h_ideal * win).astype(np.complex128)
    # Normalise to unity gain at DC
    return h_windowed / np.sum(h_windowed)


# =====================================================================
# Free Space Path Loss (Friis)
# =====================================================================

class FreeSpacePathLoss:
    """
    Free-space path loss using the Friis transmission equation.

    The loss from isotropic antennas in free space is::

        FSPL_dB = 20*log10(d_km) + 20*log10(f_MHz) + 32.44

    With directional antennas (gain G_t, G_r in dBi)::

        P_r = P_t + G_t + G_r - FSPL_dB

    Parameters
    ----------
    frequency_hz : float
        Carrier frequency in Hz.
    receiver_position_m : np.ndarray, optional
        3D position of the receiver. When set, range is computed
        dynamically. When None, the user supplies range manually.
    speed_of_light : float
        Speed of light in m/s. Default = 299792458.
    """

    def __init__(
        self,
        frequency_hz: float,
        *,
        receiver_position_m: Optional[np.ndarray] = None,
        speed_of_light: float = SPEED_OF_LIGHT,
    ):
        self.frequency_hz = float(frequency_hz)
        self.receiver_position_m = (
            None if receiver_position_m is None
            else np.asarray(receiver_position_m, dtype=np.float64).reshape(3)
        )
        self._c = float(speed_of_light)

    def __repr__(self) -> str:
        return (
            f"FreeSpacePathLoss(frequency_hz={self.frequency_hz:.3e}, "
            f"receiver_position={self.receiver_position_m})"
        )

    def fspl_db(
        self,
        range_m: float,
        *,
        f_mhz: Optional[float] = None,
    ) -> float:
        """
        Compute free-space path loss in dB.

        Parameters
        ----------
        range_m : float
            Range in metres.
        f_mhz : float, optional
            Frequency in MHz. Computed from ``frequency_hz`` if not given.

        Returns
        -------
        float
            FSPL in dB.
        """
        if range_m <= 0:
            raise ValueError(f"range_m must be positive; got {range_m}")
        if f_mhz is None:
            f_mhz = self.frequency_hz / 1e6
        return float(20.0 * np.log10(range_m / 1000.0) + 20.0 * np.log10(f_mhz) + 32.44)

    def received_power_dbw(
        self,
        transmitted_power_dbw: float,
        tx_gain_dbi: float,
        rx_gain_dbi: float,
        range_m: float,
    ) -> float:
        """
        Compute received power using the Friis equation.

        Parameters
        ----------
        transmitted_power_dbw : float
            Transmitted power in dBW.
        tx_gain_dbi, rx_gain_dbi : float
            Transmit and receive antenna gains in dBi.
        range_m : float
            Range in metres.

        Returns
        -------
        float
            Received power in dBW.
        """
        fspl = self.fspl_db(range_m)
        return float(transmitted_power_dbw + tx_gain_dbi + rx_gain_dbi - fspl)

    def received_power_dbm(
        self,
        transmitted_power_dbm: float,
        tx_gain_dbi: float,
        rx_gain_dbi: float,
        range_m: float,
    ) -> float:
        """Same as ``received_power_dbw`` but with dBm input/output."""
        return float(
            self.received_power_dbw(
                transmitted_power_dbm - 30.0,
                tx_gain_dbi,
                rx_gain_dbi,
                range_m,
            ) + 30.0
        )


@dataclass
class PathLossResult:
    """Container for all path loss components."""
    fspl_db: float
    shadowing_db: float
    diffraction_db: float
    total_loss_db: float
    range_m: float
    doppler_shift_hz: float


def compute_path_loss(
    emitter: KinematicEmitter,
    receiver_position_m: np.ndarray,
    f_c: float,
    *,
    shadowing_std_db: float = 8.0,
    diffraction_std_db: float = 4.0,
    rng: Optional[np.random.Generator] = None,
) -> PathLossResult:
    """
    Compute the full path loss from emitter to receiver.

    Combines deterministic FSPL with statistical shadowing and
    diffraction losses::

        L_total = FSPL + shadowing + diffraction

    Parameters
    ----------
    emitter : KinematicEmitter
        Emitter kinematic state.
    receiver_position_m : np.ndarray
        Receiver 3D position, shape (3,).
    f_c : float
        Carrier frequency in Hz.
    shadowing_std_db : float
        Standard deviation of the terrain shadowing loss in dB.
        Default 8.0 dB. From TSRDEmitterSampler.
    diffraction_std_db : float
        Standard deviation of the diffraction loss in dB. Default 4.0 dB.
    rng : np.random.Generator, optional
        RNG for the shadowing/diffraction draws. If None, the draws
        are deterministic (same values every call).

    Returns
    -------
    PathLossResult
        Container with all loss components and the Doppler shift.
    """
    fspl = FreeSpacePathLoss(f_c)
    range_m = emitter.range_m(receiver_position_m)
    fspl_db = fspl.fspl_db(range_m)

    # Shadowing and diffraction are drawn once per emitter and
    # remain constant over the mission (slow variation).
    if rng is None:
        shadowing_db = 0.0
        diffraction_db = 0.0
    else:
        shadowing_db = float(rng.normal(0.0, shadowing_std_db))
        diffraction_db = float(rng.normal(0.0, diffraction_std_db))

    doppler = compute_doppler_shift(emitter, receiver_position_m, f_c)
    total = fspl_db + shadowing_db + diffraction_db

    return PathLossResult(
        fspl_db=fspl_db,
        shadowing_db=shadowing_db,
        diffraction_db=diffraction_db,
        total_loss_db=total,
        range_m=range_m,
        doppler_shift_hz=doppler,
    )


# =====================================================================
# Multipath fading
# =====================================================================

class RayleighFadingChannel:
    """
    Rayleigh fading channel (no direct line-of-sight).

    Models environments where the received signal is the sum of many
    uncorrelated reflections. The complex gain follows a circularly-symmetric
    complex normal distribution::

        h ~ CN(0, σ²)

    The envelope |h| follows a Rayleigh distribution with
    E[|h|²] = 2σ².

    The channel is updated at each sample. The Doppler frequency
    determines the rate of envelope variation via Clarke's flat-fading
    model.

    Parameters
    ----------
    f_c : float
        Carrier frequency in Hz.
    doppler_hz : float
        Maximum Doppler frequency in Hz (from relative motion).
        Sets the fading rate.
    sample_rate_hz : float
        Simulation sample rate in Hz.
    rng : np.random.Generator, optional
        RNG for reproducibility.
    """

    def __init__(
        self,
        f_c: float,
        *,
        doppler_hz: float = 10.0,
        sample_rate_hz: float = 1e6,
        rng: Optional[np.random.Generator] = None,
    ):
        self.f_c = float(f_c)
        self.doppler_hz = float(doppler_hz)
        self.sample_rate_hz = float(sample_rate_hz)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._omega_d = 2.0 * np.pi * float(doppler_hz)
        self._last_t_s: float = 0.0

    def step(self, n_samples: int = 1) -> np.ndarray:
        """
        Advance the channel by ``n_samples`` and return the complex gain.

        Uses a sum-of-sinusoids (Jakes) approximation with N=16 paths
        for Clarke's flat-fading Doppler spectrum. The standard
        complex-valued form is::

            h(t) = (1/√N) * Σ_{n=0}^{N-1} exp(j * (ω_d * t * cos(α_n) + φ_n))

        where ``α_n = 2πn/N`` are the equally-spaced arrival angles
        and ``φ_n`` are independent uniform [0, 2π) phases. This
        form gives E[|h|²] = 1 by construction.

        Parameters
        ----------
        n_samples : int
            Number of samples to advance.

        Returns
        -------
        np.ndarray
            Complex channel gain, shape ``(n_samples,)``.
            Multiply the received signal by this gain.
        """
        N = 16
        omega_d = self._omega_d
        if not hasattr(self, "_path_phases"):
            self._path_phases = self.rng.uniform(0.0, 2.0 * np.pi, size=N)
            self._path_thetas = (2.0 * np.pi * np.arange(N)) / N

        # Time axis for this call, starting from the last-end point.
        # Using cumulative time so phase stays continuous across
        # multiple step() calls.
        t_offset = self._last_t_s
        t_end = t_offset + float(n_samples) / self.sample_rate_hz
        t = np.linspace(t_offset, t_end, n_samples, endpoint=False)
        self._last_t_s = t_end

        # Outer product of time and path cosines
        cos_th = np.cos(self._path_thetas)
        angles = (
            omega_d * t[:, None] * cos_th[None, :]
            + self._path_phases[None, :]
        )
        # Complex sum, divided by √N so E[|h|²] = 1
        h = np.exp(1j * angles).sum(axis=1) / np.sqrt(N)
        return h.astype(np.complex128)

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply the channel gain to a signal array (one gain per sample)."""
        return signal * self.step(signal.size)


class RicianFadingChannel:
    """
    Rician fading channel (dominant LOS + diffuse scatter).

    Models environments where a strong direct line-of-sight is present
    in addition to many weak reflections::

        h = sqrt(K / (K+1)) * exp(j*phi_LOS) + sqrt(1 / (K+1)) * h_scatter

    where ``h_scatter ~ CN(0, 1)`` and ``K`` is the Rice factor
    (ratio of LOS power to scatter power, linear).

    Parameters
    ----------
    f_c : float
        Carrier frequency in Hz.
    k_factor : float
        Rice K-factor in linear (not dB). K = P_LOS / P_scatter.
        K=0 → Rayleigh; K→∞ → no fading.
    los_phase_rad : float
        Phase of the dominant LOS component. Default 0.
    doppler_hz : float
        Maximum Doppler frequency in Hz. Applied to both LOS and scatter.
    sample_rate_hz : float
        Simulation sample rate in Hz.
    rng : np.random.Generator, optional
        RNG for reproducibility.
    """

    def __init__(
        self,
        f_c: float,
        k_factor: float = 3.0,
        *,
        los_phase_rad: float = 0.0,
        doppler_hz: float = 10.0,
        sample_rate_hz: float = 1e6,
        rng: Optional[np.random.Generator] = None,
    ):
        if k_factor < 0:
            raise ValueError(f"k_factor must be >= 0; got {k_factor}")
        self.f_c = float(f_c)
        self.k_factor = float(k_factor)
        self.los_phase_rad = float(los_phase_rad)
        self.doppler_hz = float(doppler_hz)
        self.sample_rate_hz = float(sample_rate_hz)
        self.rng = rng if rng is not None else np.random.default_rng()

        # Scatter component (Rayleigh part)
        self._scatter = RayleighFadingChannel(
            f_c, doppler_hz=doppler_hz,
            sample_rate_hz=sample_rate_hz, rng=rng,
        )

    @property
    def k_factor_db(self) -> float:
        """K-factor in dB."""
        return float(10.0 * np.log10(self.k_factor))

    def step(self, n_samples: int = 1) -> np.ndarray:
        """
        Advance the channel and return the complex gain.

        Returns
        -------
        np.ndarray
            Complex channel gain, shape ``(n_samples,)``.
        """
        scatter_gain = self._scatter.step(n_samples)
        # LOS component
        amplitude = np.sqrt(self.k_factor / (self.k_factor + 1.0))
        los = amplitude * np.exp(1j * self.los_phase_rad)
        # Normalise scatter so total E[|h|²] = 1
        # Rayleigh already normalises to E[|h|²] = 1, so weight by 1/sqrt(K+1)
        scatter_weight = 1.0 / np.sqrt(self.k_factor + 1.0)
        return (los + scatter_weight * scatter_gain).astype(np.complex128)

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply the channel gain to a signal."""
        return signal * self.step(signal.size)


class MultipathChannel:
    """
    Multi-path channel with discrete tap delays.

    Models a finite impulse response (FIR) channel where each tap has
    its own complex gain (Rayleigh or Rician) and delay. This is the
    most general tapped-delay-line model used in wideband ESM simulation.

    Parameters
    ----------
    f_c : float
        Carrier frequency in Hz.
    taps : dict[int, float]
        Mapping from ``tap_index`` (integer, 0 = LOS) to relative power
        in dB. Powers are relative to the LOS tap (index 0). E.g.::

            {0: 0.0,   # LOS: 0 dB (reference)
             3: -8.0,  # Tap 3: 8 dB below LOS
             7: -15.0} # Tap 7: 15 dB below LOS

    sample_rate_hz : float
        Chip/sample rate. Tap delays are integer multiples of 1/f_s.
    rng : np.random.Generator, optional
        RNG for per-tap Rayleigh fading.
    """

    def __init__(
        self,
        f_c: float,
        taps: dict,
        *,
        sample_rate_hz: float = 1e6,
        k_factors: Optional[dict[int, float]] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.f_c = float(f_c)
        self.taps = dict(taps)
        self.sample_rate_hz = float(sample_rate_hz)
        self.k_factors = dict(k_factors) if k_factors else {}
        self.rng = rng if rng is not None else np.random.default_rng()

        # Initialise per-tap channel models
        self._tap_channels: dict[int, RayleighFadingChannel | RicianFadingChannel] = {}
        for idx, power_db in self.taps.items():
            k = self.k_factors.get(idx, 0.0)
            if k > 0:
                self._tap_channels[idx] = RicianFadingChannel(
                    f_c, k_factor=k, rng=rng,
                    doppler_hz=0.0,  # Doppler handled at outer level
                    sample_rate_hz=sample_rate_hz,
                )
            else:
                self._tap_channels[idx] = RayleighFadingChannel(
                    f_c, rng=rng,
                    doppler_hz=0.0,
                    sample_rate_hz=sample_rate_hz,
                )

        # Relative amplitudes from dB
        self._tap_gains = {
            idx: float(10.0 ** (power_db / 20.0))
            for idx, power_db in self.taps.items()
        }

    def step(self, n_samples: int) -> np.ndarray:
        """
        Advance all taps and return the combined channel impulse response.

        Returns
        -------
        np.ndarray
            Complex channel impulse response (CIR), shape ``(n_samples,)``.
            The taps are placed at their integer delays.
        """
        cir = np.zeros(n_samples, dtype=np.complex128)
        max_delay = max(self.taps.keys())
        actual = min(n_samples, max_delay + 1)
        for idx, power_lin in self._tap_gains.items():
            if idx < actual:
                gain = self._tap_channels[idx].step(n_samples)
                cir[idx] = gain[0] * power_lin
        return cir

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Convolve the signal with the channel impulse response.

        Returns
        -------
        np.ndarray
            Received signal with multipath applied.
        """
        cir = self.step(signal.size)
        return np.convolve(signal, cir, mode="same").astype(np.complex128)


__all__ = [
    "SPEED_OF_LIGHT",
    "SPEED_OF_LIGHT_KM_S",
    "KinematicEmitter",
    "compute_doppler_shift",
    "time_of_flight",
    "fractional_delay_filter",
    "FreeSpacePathLoss",
    "compute_path_loss",
    "PathLossResult",
    "RayleighFadingChannel",
    "RicianFadingChannel",
    "MultipathChannel",
]
