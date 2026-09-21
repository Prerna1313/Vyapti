/**
 * WaveForge RF DSP Simulation Engine
 * Generates synthetic Power Spectral Density (PSD) data and STFT Spectrogram rows.
 * Replicates the exact emitter IDs, frequencies, and layout from WaveForge screenshots.
 */

export const FREQ_MIN = 0.1; // kHz
export const FREQ_MAX = 1.0; // kHz
export const NUM_BINS = 800; // Match screen pixel density for visible jaggedness

// Exact emitters matching the screenshot:
export const EXACT_EMITTERS = [
  { id: 'MARKER_ONLY_1', freq: 0.275, basePower: -95, bw: 0.0005, type: 'Scan Marker', markerOnly: true },
  { id: 'ID 7', freq: 0.400, basePower: -56, bw: 0.0015, type: 'Multi-Tone', markerOnly: false },
  { id: 'ID 2', freq: 0.435, basePower: -72, bw: 0.0012, type: 'Multi-Tone', markerOnly: false },
  { id: 'MARKER_ONLY_2', freq: 0.480, basePower: -68, bw: 0.0008, type: 'Scan Marker', markerOnly: true },
  { id: 'ID 4', freq: 0.520, basePower: -57, bw: 0.0015, type: 'Multi-Tone', markerOnly: false },
  { id: 'ID 1', freq: 0.635, basePower: -60, bw: 0.0012, type: 'Multi-Tone', markerOnly: false },
  { id: 'ID 5', freq: 0.695, basePower: -58, bw: 0.0014, type: 'Multi-Tone', markerOnly: false },
  { id: 'MARKER_ONLY_3', freq: 0.955, basePower: -72, bw: 0.0008, type: 'Pulsed', markerOnly: true }
];

export const BASE_EMITTERS = EXACT_EMITTERS;

export const SCENARIOS = {
  DYNAMIC: 'Dynamic Spectrum',
  BUSY: 'Busy Spectrum',
  QUIET: 'Quiet Spectrum',
  HIGH_NOISE: 'High Noise Dynamic Spectrum',
  ANOMALOUS: 'Anomalous Activity'
};

/**
 * Generate a single frame of Spectrum Data (dBm per frequency bin)
 */

let livePsd = new Float32Array(NUM_BINS).fill(-97.0);

if (typeof window !== 'undefined') {
  const ws = new window.WebSocket('ws://localhost:8000/ws/system-c');
  ws.onmessage = (e) => {
    try {
      const data = JSON.parse(e.data);
      if (data.type === 'fft_frame' && data.psd) {
        for (let i = 0; i < NUM_BINS && i < data.psd.length; i++) {
          livePsd[i] = data.psd[i];
        }
      }
    } catch(err){}
  };
}

export function generateSpectrumFrame(scenario = SCENARIOS.DYNAMIC, time = 0, anomalyBurst = false) {
  const psd = new Float32Array(livePsd);
  // Add a tiny bit of random jitter so the baseline looks alive even if ws pauses
  for (let i = 0; i < NUM_BINS; i++) {
    psd[i] += (Math.random() - 0.5) * 1.5;
  }
  let activeEmitters = [...EXACT_EMITTERS];
  return { psd, activeEmitters };
}



/**
 * Detect peaks crossing thresholdDbm
 */
export function detectPeaks(psd, thresholdDbm, activeEmitters = EXACT_EMITTERS) {
  const uniquePeaks = new Map();

  for (let i = 3; i < NUM_BINS - 3; i++) {
    const val = psd[i];
    
    if (
      val >= thresholdDbm &&
      val > psd[i - 1] &&
      val > psd[i - 2] &&
      val >= psd[i + 1] &&
      val >= psd[i + 2]
    ) {
      const freq = FREQ_MIN + (i / NUM_BINS) * (FREQ_MAX - FREQ_MIN);
      const matched = activeEmitters.find(e => Math.abs(e.freq - freq) < 0.020);

      if (matched && !matched.markerOnly) {
        if (!uniquePeaks.has(matched.id) || val > uniquePeaks.get(matched.id).power) {
          uniquePeaks.set(matched.id, {
            binIndex: i,
            freq: matched.freq,
            power: val,
            id: matched.id,
            labelLine1: matched.id,
            labelLine2: matched.type,
            signalType: matched.type
          });
        }
      }
    }
  }

  return Array.from(uniquePeaks.values());
}

/**
 * Get all vertical cursor line positions (red dashed lines)
 */
export function getVerticalCursorBins(activeEmitters = EXACT_EMITTERS) {
  return activeEmitters.map(e => {
    const binIndex = Math.round(((e.freq - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)) * NUM_BINS);
    return { freq: e.freq, binIndex, id: e.id, markerOnly: e.markerOnly };
  });
}

/**
 * Map dBm power (-120 to -20) to RGBA Thermal Spectrogram color
 */
export function dbmToThermalColor(dbm) {
  // Normalize dbm specifically for the visual range of the video (-100 to -40)
  const norm = Math.max(0, Math.min(1, (dbm - (-105)) / 65));

  let r = 0, g = 0, b = 0;

  if (norm < 0.15) {
    // Deep indigo noise floor
    r = 25;
    g = 15;
    b = 45;
  } else if (norm < 0.4) {
    // Dark purple/magenta
    const t = (norm - 0.15) / 0.25;
    r = Math.floor(25 + t * 95);
    g = Math.floor(15 + t * 5);
    b = Math.floor(45 + t * 65);
  } else if (norm < 0.7) {
    // Bright magenta vertical lines
    const t = (norm - 0.4) / 0.3;
    r = Math.floor(120 + t * 115);
    g = Math.floor(20 + t * 40);
    b = Math.floor(110 + t * 60);
  } else {
    // Intense hot pink center
    const t = (norm - 0.7) / 0.3;
    r = 255;
    g = Math.floor(60 + t * 150);
    b = Math.floor(170 + t * 85);
  }

  return [r, g, b, 255];
}
