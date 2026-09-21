import numpy as np

class HardwareReceiverModel:
    """
    Simulates analog front-end imperfections in RF receivers.
    Applying these distortions closes the Simulation-to-Reality (Sim2Real) gap.
    """
    
    @staticmethod
    def apply_phase_noise(signal: np.ndarray, phase_noise_std_deg: float, rng: np.random.Generator) -> np.ndarray:
        """
        Simulates oscillator jitter (Phase Noise).
        Random walk of phase over time across the signal.
        """
        if phase_noise_std_deg <= 0.0:
            return signal
            
        std_rad = np.radians(phase_noise_std_deg)
        # Random walk phase noise (cumulative sum of Gaussian steps)
        phase_steps = rng.normal(0, std_rad, size=signal.shape)
        phase_noise = np.cumsum(phase_steps)
        
        # Apply phase rotation
        return signal * np.exp(1j * phase_noise)

    @staticmethod
    def apply_iq_imbalance(signal: np.ndarray, amplitude_imbalance_db: float, phase_imbalance_deg: float) -> np.ndarray:
        """
        Simulates analog mixer asymmetry (I/Q Imbalance).
        Causes the signal to leak into its mirrored negative frequency.
        """
        if amplitude_imbalance_db == 0.0 and phase_imbalance_deg == 0.0:
            return signal
            
        # Convert dB to linear scale (voltage ratio)
        g = 10 ** (amplitude_imbalance_db / 20.0)
        phi = np.radians(phase_imbalance_deg)
        
        # Separate I and Q
        I = np.real(signal)
        Q = np.imag(signal)
        
        # Apply imbalance
        # I_new = I
        # Q_new = g * (Q * np.cos(phi) - I * np.sin(phi))
        I_new = I
        Q_new = g * (Q * np.cos(phi) + I * np.sin(phi))
        
        return I_new + 1j * Q_new

    @staticmethod
    def apply_adc_clipping(signal: np.ndarray, max_voltage: float) -> np.ndarray:
        """
        Simulates Analog-to-Digital Converter (ADC) saturation.
        If a signal is too loud, the tops of the sine waves are clipped, generating harmonics.
        """
        if max_voltage <= 0.0:
            return signal
            
        magnitude = np.abs(signal)
        # Find where magnitude exceeds max_voltage
        clip_mask = magnitude > max_voltage
        
        if not np.any(clip_mask):
            return signal
            
        # Clip magnitude while preserving phase
        clipped_signal = signal.copy()
        clipped_signal[clip_mask] = (signal[clip_mask] / magnitude[clip_mask]) * max_voltage
        return clipped_signal

    @staticmethod
    def apply_ip3_nonlinearity(signal: np.ndarray, iip3_dbm: float) -> np.ndarray:
        """
        Simulates Third-Order Intercept Point (IP3) non-linearities in the LNA.
        Causes strong signals to mix and spawn "ghost" signals (intermodulation products).
        """
        # Convert IIP3 to linear voltage scale approximation
        # (Simplified polynomial model: V_out = a1*V_in + a3*V_in^3)
        # For a 50 ohm system, power to voltage mapping applies.
        # This is a rapid prototype model of 3rd order distortion.
        
        # In this simplified model, a3 is derived inversely from IIP3.
        # If IIP3 is high (e.g., 30 dBm), the distortion is very small.
        if iip3_dbm > 50.0: # Basically perfect amplifier
            return signal
            
        a1 = 1.0
        # Rough empirical scaling for the cubic term based on IIP3
        a3 = - (10 ** (-iip3_dbm / 10.0)) 
        
        # V_out = V_in * (a1 + a3 * |V_in|^2)
        # This causes compression and intermodulation
        distortion = a3 * (np.abs(signal) ** 2)
        return signal * (a1 + distortion)

