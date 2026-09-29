/**
 * VYAPTI - RF PHYSICS & LIVE SIMULATION
 * High-Precision Tactical Radar, Spectrum & Physics Simulation Engine
 */

(function () {
  'use strict';

  // Seeded PRNG fallback
  let prng;
  try {
    if (window.Random && window.Random.MersenneTwister19937) {
      const engine = window.Random.MersenneTwister19937.seed(74231);
      prng = {
        real: (min, max) => window.Random.real(min, max, true)(engine),
        integer: (min, max) => window.Random.integer(min, max)(engine),
        bool: (prob = 0.5) => window.Random.bool(prob)(engine),
      };
    } else {
      throw new Error('Fallback PRNG');
    }
  } catch (e) {
    prng = {
      real: (min, max) => min + Math.random() * (max - min),
      integer: (min, max) => Math.floor(min + Math.random() * (max - min + 1)),
      bool: (prob = 0.5) => Math.random() < prob,
    };
  }

  // Simulation State
  const state = {
    isRunning: true,
    simTime: 12.482,
    wallTime: 1.276,
    decisions: 842,
    episodes: 250,
    runId: 'RUN-2026-0922-001',
    lastFrame: performance.now(),
    waterfallOffset: 0,
    iqPhase: 0,
    noiseGrass: new Float32Array(500),
  };

  // Prepopulate noise grass
  for (let i = 0; i < state.noiseGrass.length; i++) {
    state.noiseGrass[i] = Math.random() * 8;
  }

  // Canvas References
  const cPsd = document.getElementById('psdCanvas');
  const cWaterfall = document.getElementById('waterfallCanvas');
  const cRfScanHistory = document.getElementById('rfScanHistoryCanvas');
  const cIqTime = document.getElementById('iqTimeCanvas');
  const cCc0Evm = document.getElementById('cc0EvmCanvas');
  const cQCarrier = document.getElementById('qCarrierCanvas');
  const cDoppler = document.getElementById('dopplerCanvas');
  const cConstellation = document.getElementById('constellationCanvas');
  const cCCDF = document.getElementById('ccdfCanvas');

  // DOM Elements
  const elSimTime = document.getElementById('valSimTime');
  const elWallTime = document.getElementById('valWallTime');
  const elDecisions = document.getElementById('valDecisions');
  const elEpisodes = document.getElementById('valEpisodes');
  const elIqIRms = document.getElementById('iqValIRms');
  const elIqQRms = document.getElementById('iqValQRms');
  const elIqPeak = document.getElementById('iqValPeak');
  const elIqPower = document.getElementById('iqValPower');
  const elIqPhase = document.getElementById('iqValPhase');
  const elIqFreqOff = document.getElementById('iqValFreqOff');
  const elIqSnr = document.getElementById('iqValSnr');
  const elIqSampleRate = document.getElementById('iqValSampleRate');

  // Buttons
  const btnStartTop = document.getElementById('btnStartTop');
  const btnPauseTop = document.getElementById('btnPauseTop');
  const btnStopTop = document.getElementById('btnStopTop');
  const btnCtrlStart = document.getElementById('btnCtrlStart');
  const btnCtrlPause = document.getElementById('btnCtrlPause');
  const btnCtrlStop = document.getElementById('btnCtrlStop');
  const btnCtrlReset = document.getElementById('btnCtrlReset');

  // Helper: Setup High DPI Canvas with intelligent size caching
  function setupCanvas(canvas) {
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return null;
    const dpr = window.devicePixelRatio || 1;
    const targetW = Math.round(rect.width * dpr);
    const targetH = Math.round(rect.height * dpr);
    if (canvas.width !== targetW || canvas.height !== targetH) {
      canvas.width = targetW;
      canvas.height = targetH;
    }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);
    return { ctx, width: rect.width, height: rect.height, dpr };
  }

  /* ==========================================================================
     1. RF SPECTRUM (PSD)
     ========================================================================== */
  /* ==========================================================================
     1. EXACT SDR++ RF SPECTRUM ANALYZER (First Graph)
     ========================================================================== */
  function sincSq(x) {
    if (Math.abs(x) < 0.0001) return 1.0;
    const px = Math.PI * x;
    return Math.pow(Math.sin(px) / px, 2);
  }

  // High-Resolution 1024-bin FFT buffer with temporal smoothing
  const FFT_BINS = 1024;
  const fftPowerHistory = new Float32Array(FFT_BINS);

  function drawPsd() {
    const setup = setupCanvas(cPsd);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 24;
    const padBottom = 14;
    const padTop = 10;
    const padRight = 4;
    const plotW = Math.max(10, width - padLeft - padRight);
    const plotH = Math.max(10, height - padBottom - padTop);

    // 1. Pure Pitch Black Instrument Background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // 2. Y-Axis Ticks (0, -10, -20, -30, -40, -50, -60, -70, -80, -90, -100 dB)
    ctx.font = '500 6.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let db = 0; db >= -100; db -= 10) {
      const y = padTop + ((0 - db) / 100) * plotH;
      ctx.fillText(db.toString(), padLeft - 3, y);

      // Subtle horizontal instrument grid line
      ctx.strokeStyle = '#131b26';
      ctx.lineWidth = 0.7;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    }

    // 3. X-Axis Frequency Ticks (118.6M to 120.8M, step 0.2M)
    const fMin = 118.45;
    const fMax = 120.95;
    const fSpan = fMax - fMin;

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.font = '500 6.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#cbd5e1';

    const freqTicks = [118.6, 118.8, 119.0, 119.2, 119.4, 119.6, 119.8, 120.0, 120.2, 120.4, 120.6, 120.8];
    freqTicks.forEach(f => {
      const x = padLeft + ((f - fMin) / fSpan) * plotW;
      const label = (f % 1 === 0 ? f.toFixed(0) : f.toFixed(1)) + 'M';
      ctx.fillText(label, x, padTop + plotH + 2);

      // Subtle vertical instrument grid line
      ctx.strokeStyle = '#131b26';
      ctx.lineWidth = 0.7;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();
    });

    // 4. Update Dynamic Overlay Positions in HTML (Station Badges, Lines & Air Band)
    const x1188 = padLeft + ((118.80 - fMin) / fSpan) * plotW;
    const xCenter = padLeft + ((119.625 - fMin) / fSpan) * plotW;
    const x12003 = padLeft + ((120.03 - fMin) / fSpan) * plotW;
    const x12063 = padLeft + ((120.63 - fMin) / fSpan) * plotW;

    const elBadge1 = document.getElementById('badgeBerlin');
    const elBadge2 = document.getElementById('badgeBremenApproach');
    const elBadge3 = document.getElementById('badgeBremen12063');
    const elV1 = document.getElementById('vlineBerlin');
    const elVcenter = document.getElementById('vlineCenterLO');
    const elV2 = document.getElementById('vlineBremenApproach');
    const elV3 = document.getElementById('vlineBremen12063');
    const elAirband = document.getElementById('airBandBox');

    if (elBadge1) elBadge1.style.left = `${x1188}px`;
    if (elBadge2) elBadge2.style.left = `${x12003}px`;
    if (elBadge3) elBadge3.style.left = `${x12063}px`;
    if (elV1) elV1.style.left = `${x1188}px`;
    if (elVcenter) elVcenter.style.left = `${xCenter}px`;
    if (elV2) elV2.style.left = `${x12003}px`;
    if (elV3) elV3.style.left = `${x12063}px`;
    if (elAirband) {
      const yAirTop = padTop + ((0 - (-88)) / 100) * plotH;
      const yAirBot = padTop + ((0 - (-100)) / 100) * plotH;
      elAirband.style.left = `${padLeft}px`;
      elAirband.style.width = `${plotW}px`;
      elAirband.style.top = `${yAirTop}px`;
      elAirband.style.height = `${yAirBot - yAirTop}px`;
    }

    // 5. Build High-Density Instrument FFT Trace (1024 bins, ultra-fine 1px line)
    const curvePoints = [];

    for (let i = 0; i < FFT_BINS; i++) {
      const frac = i / (FFT_BINS - 1);
      const px = frac * plotW;
      const f = fMin + frac * fSpan;

      // Organic center noise floor dome matching target SDR++ image
      const centerDome = 11.0 * Math.exp(-Math.pow((f - 119.65) / 0.52, 2));
      
      // Fine-grained Gaussian noise floor jitter (tight ±1.5 dB fuzz)
      const u1 = Math.max(0.0001, Math.random());
      const u2 = Math.random();
      const gaussNoise = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2) * 1.35;
      const ripple = 0.8 * Math.sin(f * 28.0) + 0.5 * Math.cos(f * 62.0);

      let targetPower = -83.5 + centerDome + gaussNoise + ripple;

      // Peak 1: 118.80 MHz (BERLIN TOWER ~ -43.5 dBm, needle sharp)
      const d1 = Math.abs(f - 118.80);
      if (d1 < 0.08) {
        targetPower += 40.0 * sincSq(d1 / 0.009);
      }

      // Center LO residual spike at 119.625 MHz (~ -78 dBm)
      const dLO = Math.abs(f - 119.625);
      if (dLO < 0.02) {
        targetPower += 5.0 * sincSq(dLO / 0.005);
      }

      // Peak 2: 120.03 MHz (BREMEN RADAR ~ -38.0 dBm) + sideband at 119.98 MHz (~ -64.0 dBm)
      const d2 = Math.abs(f - 120.03);
      if (d2 < 0.09) {
        targetPower += 45.5 * sincSq(d2 / 0.010);
      }
      const d2b = Math.abs(f - 119.98);
      if (d2b < 0.06) {
        targetPower += 19.5 * sincSq(d2b / 0.008);
      }

      // Peak 3: 120.63 MHz (BREMEN RADAR ~ -71.5 dBm)
      const d3 = Math.abs(f - 120.63);
      if (d3 < 0.06) {
        targetPower += 12.0 * sincSq(d3 / 0.008);
      }

      // Exponential moving average filter for natural SDR phosphor decay
      if (fftPowerHistory[i] === 0) {
        fftPowerHistory[i] = targetPower;
      } else {
        fftPowerHistory[i] = fftPowerHistory[i] * 0.45 + targetPower * 0.55;
      }

      const powerDbm = fftPowerHistory[i];
      const y = Math.min(padTop + plotH, Math.max(padTop, padTop + ((0 - powerDbm) / 100) * plotH));
      curvePoints.push({ x: padLeft + px, y });
    }

    // 6. Draw Ultra-Fine Electric Cyan Spectrum Trace (1.0px hairline)
    ctx.save();
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 1.0;
    ctx.shadowColor = 'rgba(0, 212, 255, 0.4)';
    ctx.shadowBlur = 2;
    ctx.beginPath();
    curvePoints.forEach((p, idx) => {
      if (idx === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    ctx.restore();
  }

  /* ==========================================================================
     2. WATERFALL (Frequency x Time) - HIGH DEFINITION JET SPECTROGRAM
     ========================================================================== */
  // Precomputed 256-step Jet Look-Up Table
  const JET_LUT = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const v = i / 255;
    let r = 0, g = 0, b = 0;
    if (v < 0.15) {
      r = 3; g = Math.floor(10 + v * 80); b = Math.floor(40 + v * 600);
    } else if (v < 0.4) {
      const n = (v - 0.15) / 0.25;
      r = 2; g = Math.floor(80 + n * 170); b = Math.floor(200 - n * 80);
    } else if (v < 0.65) {
      const n = (v - 0.4) / 0.25;
      r = Math.floor(n * 255); g = 255; b = Math.floor(120 * (1 - n));
    } else if (v < 0.88) {
      const n = (v - 0.65) / 0.23;
      r = 255; g = Math.floor(255 - n * 160); b = 0;
    } else {
      const n = (v - 0.88) / 0.12;
      r = 255; g = Math.floor(25 * (1 - n)); b = Math.floor(25 * (1 - n));
    }
    JET_LUT[i * 3] = r;
    JET_LUT[i * 3 + 1] = g;
    JET_LUT[i * 3 + 2] = b;
  }

  // Master Offscreen Waterfall Spectrogram Buffer (Hardware-accelerated)
  const WF_BUF_W = 1200;
  const WF_BUF_H = 300;
  let wfOffscreenCanvas = null;

  function initWaterfallBuffer() {
    wfOffscreenCanvas = document.createElement('canvas');
    wfOffscreenCanvas.width = WF_BUF_W;
    wfOffscreenCanvas.height = WF_BUF_H;
    const offCtx = wfOffscreenCanvas.getContext('2d');
    const imgData = offCtx.createImageData(WF_BUF_W, WF_BUF_H);
    const data = imgData.data;

    // High quality pseudo-random noise generator
    function hashNoise(u, v) {
      const s = Math.sin(u * 12.9898 + v * 78.233) * 43758.5453123;
      return s - Math.floor(s);
    }

    for (let y = 0; y < WF_BUF_H; y++) {
      const freqGhz = 18.0 * (1.0 - y / WF_BUF_H);

      for (let x = 0; x < WF_BUF_W; x++) {
        const timeSec = (x / WF_BUF_W) * 12.0;

        // 1. Thermal Noise Floor: realistic multi-scale RF speckle
        const n1 = hashNoise(x * 1.3, y * 1.7);
        const n2 = hashNoise(x * 0.4 + 19.3, y * 0.8 + 43.1);
        const n3 = hashNoise(x * 2.7 + 101.1, y * 3.1 + 87.4);
        
        // Deep navy blue noise baseline (~ -118 dBm to -102 dBm)
        let power = 0.07 + n1 * 0.06 + n2 * 0.05 + n3 * 0.03;

        // Subtle horizontal RF raster striations (ADC/FFT bin ripples)
        power += 0.015 * Math.sin(y * 1.2) + 0.012 * Math.cos(y * 2.8);

        // 2. Persistent Narrowband Carriers (Horizontal Tracks)
        // E5: 2.78 GHz (Narrow carrier with natural jitter)
        const dE5 = Math.abs(freqGhz - 2.78);
        if (dE5 < 0.22) {
          const mod = 0.85 + 0.15 * hashNoise(x * 0.2, 5.0);
          power += 0.48 * Math.exp(-(dE5 * dE5) / 0.0035) * mod;
        }

        // E1: 3.25 GHz (Primary High-Intensity Radar Carrier)
        const dE1 = Math.abs(freqGhz - 3.25);
        if (dE1 < 0.32) {
          const mod = 0.90 + 0.10 * hashNoise(x * 0.25, 12.0);
          power += 0.72 * Math.exp(-(dE1 * dE1) / 0.0055) * mod;
        }

        // E2: 7.86 GHz (Agile Intercept Emitter - active 3.6s to 10.6s)
        const dE2 = Math.abs(freqGhz - 7.86);
        if (dE2 < 0.25 && timeSec >= 3.6 && timeSec <= 10.6) {
          const edgeFade = Math.min(1.0, (timeSec - 3.6) * 4.0) * Math.min(1.0, (10.6 - timeSec) * 4.0);
          const mod = 0.85 + 0.15 * hashNoise(x * 0.3, 22.0);
          power += 0.58 * Math.exp(-(dE2 * dE2) / 0.004) * mod * edgeFade;
        }

        // E3: 11.42 GHz (Bursty Emitter - active 4.6s to 10.4s)
        const dE3 = Math.abs(freqGhz - 11.42);
        if (dE3 < 0.26 && timeSec >= 4.6 && timeSec <= 10.4) {
          const edgeFade = Math.min(1.0, (timeSec - 4.6) * 3.0) * Math.min(1.0, (10.4 - timeSec) * 3.0);
          const mod = 0.85 + 0.15 * hashNoise(x * 0.3, 33.0);
          power += 0.62 * Math.exp(-(dE3 * dE3) / 0.0045) * mod * edgeFade;
        }

        // E6: 15.23 GHz (High-Frequency Radar - active 8.0s to 11.6s)
        const dE6 = Math.abs(freqGhz - 15.23);
        if (dE6 < 0.24 && timeSec >= 8.0 && timeSec <= 11.6) {
          const edgeFade = Math.min(1.0, (timeSec - 8.0) * 3.0) * Math.min(1.0, (11.6 - timeSec) * 3.0);
          const mod = 0.85 + 0.15 * hashNoise(x * 0.3, 44.0);
          power += 0.52 * Math.exp(-(dE6 * dE6) / 0.004) * mod * edgeFade;
        }

        // Secondary harmonic spurs / LO leakage lines
        if (Math.abs(freqGhz - 0.75) < 0.15) power += 0.18 * Math.exp(-Math.pow(freqGhz - 0.75, 2) / 0.003);
        if (Math.abs(freqGhz - 5.45) < 0.15) power += 0.16 * Math.exp(-Math.pow(freqGhz - 5.45, 2) / 0.003);
        if (Math.abs(freqGhz - 9.15) < 0.18) power += 0.22 * Math.exp(-Math.pow(freqGhz - 9.15, 2) / 0.004);
        if (Math.abs(freqGhz - 13.65) < 0.16) power += 0.18 * Math.exp(-Math.pow(freqGhz - 13.65, 2) / 0.003);

        // 3. Broadband Radar Chirp Bursts (Vertical Columns matching reference)
        // Chirp 1 at t ~ 2.12s
        const dt1 = Math.abs(timeSec - 2.12);
        if (dt1 < 0.15) {
          const vShape = 1.0 - (freqGhz / 18.0) * 0.40;
          power += 0.58 * Math.exp(-(dt1 * dt1) / 0.005) * vShape;
        }

        // Chirp 2 at t ~ 3.52s
        const dt2 = Math.abs(timeSec - 3.52);
        if (dt2 < 0.12) {
          const vShape = 1.0 - (freqGhz / 18.0) * 0.30;
          power += 0.52 * Math.exp(-(dt2 * dt2) / 0.004) * vShape;
        }

        // Chirp 3 at t ~ 4.82s
        const dt3 = Math.abs(timeSec - 4.82);
        if (dt3 < 0.11) {
          power += 0.46 * Math.exp(-(dt3 * dt3) / 0.0035);
        }

        // Chirp 4 at t ~ 7.32s
        const dt4 = Math.abs(timeSec - 7.32);
        if (dt4 < 0.12) {
          power += 0.50 * Math.exp(-(dt4 * dt4) / 0.004);
        }

        // Chirp 5 at t ~ 8.02s (INSIDE R1 DWELL BOX: 8.0 GHz)
        const dt5 = Math.abs(timeSec - 8.02);
        if (dt5 < 0.16) {
          // Broadband vertical column
          power += 0.52 * Math.exp(-(dt5 * dt5) / 0.005);
          // Intense focal pulse inside R1 (7.2 to 9.5 GHz)
          const dF8 = Math.abs(freqGhz - 8.2);
          if (dF8 < 1.8) {
            power += 0.40 * Math.exp(-(dF8 * dF8) / 0.9) * Math.exp(-(dt5 * dt5) / 0.003);
          }
        }

        // Chirp 6 at t ~ 9.60s (INSIDE R2 DWELL BOX: 15.0 GHz)
        const dt6 = Math.abs(timeSec - 9.60);
        if (dt6 < 0.18) {
          // Huge vertical broadband burst from 3.5 to 16.5 GHz
          power += 0.65 * Math.exp(-(dt6 * dt6) / 0.006);
          if (freqGhz >= 3.5 && freqGhz <= 16.5) {
            power += 0.28 * Math.exp(-(dt6 * dt6) / 0.004);
          }
        }

        // Chirp 7 at t ~ 11.42s
        const dt7 = Math.abs(timeSec - 11.42);
        if (dt7 < 0.12) {
          power += 0.54 * Math.exp(-(dt7 * dt7) / 0.004);
        }

        // 4. Map Normalized Power to Jet Color LUT
        const val = Math.min(255, Math.max(0, Math.floor(power * 255)));
        const lutIdx = val * 3;
        const pIdx = (y * WF_BUF_W + x) * 4;
        data[pIdx] = JET_LUT[lutIdx];
        data[pIdx + 1] = JET_LUT[lutIdx + 1];
        data[pIdx + 2] = JET_LUT[lutIdx + 2];
        data[pIdx + 3] = 255;
      }
    }

    offCtx.putImageData(imgData, 0, 0);
  }

  // Pre-initialize offscreen spectrogram buffer
  try {
    initWaterfallBuffer();
  } catch (e) {
    console.error('Failed to init waterfall buffer:', e);
  }

  function drawWaterfall() {
    const setup = setupCanvas(cWaterfall);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 28;
    const padBottom = 20;
    const plotW = width - padLeft;
    const plotH = height - padBottom;

    if (plotW <= 0 || plotH <= 0) return;

    // Y Axis Frequencies: 18.0 to 0.0 GHz
    ctx.font = '500 8px Inter, sans-serif';
    ctx.fillStyle = '#557297';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yFreqs = [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0];
    yFreqs.forEach((f) => {
      const y = plotH - (f / 18.0) * plotH;
      ctx.fillText(f.toFixed(1), padLeft - 4, y);
    });

    // X Axis Time: 0, 2, 4, 6, 8, 10, 12 s
    ctx.textBaseline = 'top';
    for (let t = 0; t <= 12; t += 2) {
      const x = padLeft + (t / 12) * plotW;
      ctx.textAlign = t === 12 ? 'right' : (t === 0 ? 'left' : 'center');
      const textX = t === 12 ? x - 1 : (t === 0 ? x + 1 : x);
      ctx.fillText(t.toString(), textX, plotH + 4);
    }

    // Clip to Spectrogram Canvas Plot Area
    ctx.save();
    ctx.beginPath();
    ctx.rect(padLeft, 0, plotW, plotH);
    ctx.clip();

    // Ensure offscreen buffer exists
    if (!wfOffscreenCanvas) {
      initWaterfallBuffer();
    }

    // Blit hardware-accelerated spectrogram texture
    if (wfOffscreenCanvas) {
      ctx.drawImage(wfOffscreenCanvas, 0, 0, WF_BUF_W, WF_BUF_H, padLeft, 0, plotW, plotH);
    } else {
      ctx.fillStyle = '#020b24';
      ctx.fillRect(padLeft, 0, plotW, plotH);
    }

    // Subtle Live Tactical Pulse Sweep / Time Cursor
    if (state.isRunning) {
      const sweepFrac = (state.simTime % 12.0) / 12.0;
      const sweepX = padLeft + sweepFrac * plotW;
      ctx.save();
      ctx.strokeStyle = 'rgba(0, 255, 136, 0.45)';
      ctx.lineWidth = 1.5;
      ctx.shadowColor = 'rgba(0, 255, 136, 0.8)';
      ctx.shadowBlur = 5;
      ctx.beginPath();
      ctx.moveTo(sweepX, 0);
      ctx.lineTo(sweepX, plotH);
      ctx.stroke();
      ctx.restore();
    }

    // Subtle Tactical Grid lines
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.16)';
    ctx.lineWidth = 1;
    for (let f = 3; f <= 15; f += 3) {
      const y = plotH - (f / 18.0) * plotH;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    }
    for (let t = 2; t <= 10; t += 2) {
      const x = padLeft + (t / 12) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, plotH);
      ctx.stroke();
    }

    // Outer border of plot area
    ctx.strokeStyle = '#101e33';
    ctx.lineWidth = 1;
    ctx.strokeRect(padLeft, 0, plotW, plotH);

    ctx.restore();
  }

  /* ==========================================================================
     3. I/Q TIME DOMAIN (Dual Sinusoidal Waveforms)
     ========================================================================== */
  function drawIqTimeDomain() {
    const setup = setupCanvas(cIqTime);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 28;
    const padBottom = 20;
    const padTop = 10;
    const plotW = width - padLeft - 8;
    const plotH = height - padBottom - padTop;

    // Y Axis: 1.0, 0.5, 0.0, -0.5, -1.0
    ctx.font = '500 8px Inter, sans-serif';
    ctx.fillStyle = '#557297';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yVals = [1.0, 0.5, 0.0, -0.5, -1.0];
    yVals.forEach((v) => {
      const y = padTop + ((1.0 - v) / 2.0) * plotH;
      ctx.fillText(v.toFixed(1), padLeft - 4, y);

      ctx.strokeStyle = v === 0.0 ? '#182b45' : '#0c1726';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    // X Axis: 0, 10, 20, 30, 40, 50 μs
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let t = 0; t <= 50; t += 10) {
      const x = padLeft + (t / 50) * plotW;
      ctx.fillText(t.toString(), x, padTop + plotH + 4);
    }

    const phase = state.iqPhase;

    // 1. Draw In-phase (I) Wave (Cyan)
    ctx.save();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.4;
    ctx.shadowColor = 'rgba(56, 189, 248, 0.4)';
    ctx.shadowBlur = 4;
    ctx.beginPath();
    for (let px = 0; px <= plotW; px++) {
      const t = (px / plotW) * 50;
      const envelope = 0.82 + 0.14 * Math.sin(t * 0.18);
      const amp = envelope * Math.sin(t * 1.8 + phase);
      const y = padTop + ((1.0 - amp) / 2.0) * plotH;
      const x = padLeft + px;
      if (px === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // 2. Draw Quadrature (Q) Wave (Yellow / Amber)
    ctx.save();
    ctx.strokeStyle = '#facc15';
    ctx.lineWidth = 1.4;
    ctx.shadowColor = 'rgba(250, 204, 21, 0.4)';
    ctx.shadowBlur = 4;
    ctx.beginPath();
    for (let px = 0; px <= plotW; px++) {
      const t = (px / plotW) * 50;
      const envelope = 0.8 + 0.16 * Math.cos(t * 0.18);
      const amp = envelope * Math.cos(t * 1.8 + phase);
      const y = padTop + ((1.0 - amp) / 2.0) * plotH;
      const x = padLeft + px;
      if (px === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  /* ==========================================================================
     4. FREQUENCY OFFSET / DOPPLER (Instantaneous) - Exact Attached Image
     ========================================================================== */
  function drawDoppler() {
    const setup = setupCanvas(cDoppler);
    if (!setup) return;
    const { ctx, width, height } = setup;

    // Padding for axes, tick marks, and labels
    const padLeft = 38;
    const padRight = 14;
    const padTop = 10;
    const padBottom = 22;
    const plotW = Math.max(10, width - padLeft - padRight);
    const plotH = Math.max(10, height - padTop - padBottom);

    // 1. Pitch Black Background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // 2. Y-Axis Label: "Freq Offset (kHz)"
    ctx.save();
    ctx.translate(11, padTop + plotH * 0.5);
    ctx.rotate(-Math.PI / 2);
    ctx.font = '600 8.5px "Inter", sans-serif';
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Freq Offset (kHz)', 0, 0);
    ctx.restore();

    // 3. X-Axis Label: "Time (μs)"
    ctx.font = '600 8px "Inter", sans-serif';
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('Time (μs)', padLeft + plotW * 0.5, padTop + plotH + 11);

    // 4. Grid & Y-Axis Ticks: 10, 5, 0, -5, -10
    const yTicks = [
      { v: 10, lbl: '10' },
      { v: 5, lbl: '5' },
      { v: 0, lbl: '0' },
      { v: -5, lbl: '-5' },
      { v: -10, lbl: '-10' }
    ];

    ctx.font = '600 8px "JetBrains Mono", monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    yTicks.forEach((item) => {
      const frac = (10 - item.v) / 20.0;
      const y = padTop + frac * plotH;

      // Tick label number
      ctx.fillText(item.lbl, padLeft - 4, y);

      // Horizontal grid line
      ctx.strokeStyle = item.v === 0 ? '#1e293b' : '#101a26';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();

      // Tick mark on left axis
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(padLeft - 3, y);
      ctx.lineTo(padLeft, y);
      ctx.stroke();
    });

    // 5. X-Axis Ticks: 0, 10, 20, 30, 40, 50
    const xTicks = [0, 10, 20, 30, 40, 50];
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    xTicks.forEach((t) => {
      const frac = t / 50.0;
      const x = padLeft + frac * plotW;

      // Tick number
      ctx.fillText(t.toString(), x, padTop + plotH + 2);

      // Vertical grid line
      if (t > 0 && t < 50) {
        ctx.strokeStyle = '#101a26';
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.moveTo(x, padTop);
        ctx.lineTo(x, padTop + plotH);
        ctx.stroke();
      }

      // Tick mark on bottom axis
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(x, padTop + plotH);
      ctx.lineTo(x, padTop + plotH + 3);
      ctx.stroke();
    });

    // 6. Outer Plot Box Border
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 1.0;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    // 7. Instantaneous Frequency Offset / Doppler Waveform (Green)
    ctx.save();
    ctx.strokeStyle = '#22c55e'; // Bright tactical green
    ctx.lineWidth = 1.4;
    ctx.lineJoin = 'round';
    ctx.shadowColor = 'rgba(34, 197, 94, 0.4)';
    ctx.shadowBlur = 3;
    ctx.beginPath();

    const tSim = state.simTime;
    const numPoints = Math.max(80, Math.floor(plotW * 1.2));

    for (let px = 0; px <= numPoints; px++) {
      const frac = px / numPoints;
      const tMicro = frac * 50; // 0 to 50 μs
      const x = padLeft + frac * plotW;

      // Deterministic instantaneous Doppler jitter & modulation profile matching image:
      // Macro envelope: rise near 7-10 μs, dips near 40 μs, mean around 0
      const macroRise = 2.2 * Math.sin(tMicro * 0.12 + tSim * 1.5) * Math.exp(-Math.pow((tMicro - 8) / 7.5, 2));
      const macroDip = -3.2 * Math.sin(tMicro * 0.14 + tSim * 1.2) * Math.exp(-Math.pow((tMicro - 40) / 6.5, 2));

      // High-frequency instantaneous RF phase jitter / frequency noise
      const highFreqNoise = 
        1.3 * Math.sin(tMicro * 1.7 + tSim * 4.2) +
        1.0 * Math.cos(tMicro * 3.3 - tSim * 6.5) +
        0.7 * Math.sin(tMicro * 7.2 + tSim * 9.8) +
        0.4 * Math.cos(tMicro * 12.8 + tSim * 14.5);

      const offsetKHz = Math.max(-9.2, Math.min(9.2, macroRise + macroDip + highFreqNoise));
      const y = padTop + ((10 - offsetKHz) / 20.0) * plotH;

      if (px === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.stroke();
    ctx.restore();
  }

  /* ==========================================================================
     5. CONSTELLATION (Dynamic Demodulated I/Q Scatter Diagram)
     ========================================================================== */
  function drawConstellation() {
    const setup = setupCanvas(cConstellation);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 20;
    const padBottom = 16;
    const plotW = width - padLeft;
    const plotH = height - padBottom;

    // Y Axis: 2, 1, 0, -1, -2
    ctx.font = '500 7.5px Inter, sans-serif';
    ctx.fillStyle = '#557297';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const axisTicks = [2, 1, 0, -1, -2];
    axisTicks.forEach((v) => {
      const y = plotH - ((v + 2) / 4.0) * plotH;
      ctx.fillText(v.toString(), padLeft - 3, y);

      // Grid line
      ctx.strokeStyle = v === 0 ? '#1b2f48' : '#0c1726';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    // X Axis: -2, -1, 0, 1, 2
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    axisTicks.forEach((v) => {
      const x = padLeft + ((v + 2) / 4.0) * plotW;
      ctx.fillText(v.toString(), x, plotH + 2);

      ctx.strokeStyle = v === 0 ? '#1b2f48' : '#0c1726';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, plotH);
      ctx.stroke();
    });

    const sel = document.getElementById('physicsModulationSelect') || document.querySelector('.card-constellation select');
    const mod = sel ? sel.value : 'QPSK';
    const modUpper = mod.toUpperCase();

    // Check if backend real data is available
    const bData = window.VyaptiConstellationData;
    let points = [];

    if (bData && bData.points && bData.points.length && bData.modulation && bData.modulation.toUpperCase().includes(modUpper.substring(0, 3))) {
      // Use backend-calculated points with smooth frame-level RF jitter
      const isRunning = state.isRunning;
      points = bData.points.map((pt, idx) => {
        let jI = 0, jQ = 0;
        if (isRunning) {
          const t = state.simTime * 8.0 + idx * 0.31;
          jI = (Math.sin(t * 1.7) + Math.cos(t * 3.1)) * 0.012;
          jQ = (Math.cos(t * 1.9) + Math.sin(t * 2.7)) * 0.012;
        }
        return {
          i: pt.i + jI,
          q: pt.q + jQ,
          alpha: pt.alpha || 0.85
        };
      });
    } else {
      // Dynamic local RF physical simulation grounded in state.simTime and active modulation
      let centers = [];
      let sigma = 0.14;

      if (modUpper.includes('16')) {
        const grid = [-1.5, -0.5, 0.5, 1.5];
        grid.forEach(iVal => {
          grid.forEach(qVal => {
            centers.push({ i: iVal, q: qVal });
          });
        });
        sigma = 0.08;
      } else if (modUpper.includes('BPSK')) {
        centers = [{ i: -1.0, q: 0.0 }, { i: 1.0, q: 0.0 }];
        sigma = 0.16;
      } else if (modUpper.includes('64')) {
        const grid8 = [-1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75];
        grid8.forEach(iVal => {
          grid8.forEach(qVal => {
            centers.push({ i: iVal, q: qVal });
          });
        });
        sigma = 0.05;
      } else {
        // QPSK default
        centers = [
          { i: 1.0, q: 1.0 },
          { i: -1.0, q: 1.0 },
          { i: -1.0, q: -1.0 },
          { i: 1.0, q: -1.0 },
        ];
        sigma = 0.15;
      }

      const count = 240;
      const t = state.simTime;
      const phaseDrift = Math.sin(t * 1.2) * 0.04;
      const cosP = Math.cos(phaseDrift);
      const sinP = Math.sin(phaseDrift);
      const snrMod = 1.0 + 0.15 * Math.sin(t * 2.4);

      for (let k = 0; k < count; k++) {
        const c = centers[k % centers.length];
        const seed1 = Math.sin(k * 12.34 + t * 4.1 + c.i * 7.7) * 43758.5453;
        const seed2 = Math.cos(k * 43.21 + t * 3.7 + c.q * 5.3) * 23421.6312;
        const u1 = Math.max(0.0001, seed1 - Math.floor(seed1));
        const u2 = Math.max(0.0001, seed2 - Math.floor(seed2));
        const r = (sigma * snrMod) * Math.sqrt(-2.0 * Math.log(u1));
        const theta = 2.0 * Math.PI * u2;
        const dI = r * Math.cos(theta);
        const dQ = r * Math.sin(theta);

        const rawI = c.i + dI;
        const rawQ = c.q + dQ;
        const rotI = rawI * cosP - rawQ * sinP;
        const rotQ = rawI * sinP + rawQ * cosP;
        const dist = Math.hypot(dI, dQ);
        const alpha = Math.max(0.35, Math.min(0.98, 1.0 - dist * 2.2));

        points.push({ i: rotI, q: rotQ, alpha: alpha });
      }
    }

    // Render scatter points with cyan glow
    ctx.shadowBlur = 3;
    ctx.shadowColor = 'rgba(0, 212, 255, 0.7)';

    points.forEach((pt) => {
      const x = padLeft + ((pt.i + 2) / 4.0) * plotW;
      const y = plotH - ((pt.q + 2) / 4.0) * plotH;

      if (x >= padLeft && x <= padLeft + plotW && y >= 0 && y <= plotH) {
        ctx.fillStyle = `rgba(0, 212, 255, ${pt.alpha || 0.8})`;
        ctx.fillRect(x - 0.9, y - 0.9, 1.8, 1.8);

        // Core bright center for high-confidence clusters
        if (pt.alpha > 0.65) {
          ctx.fillStyle = `rgba(224, 248, 255, ${(pt.alpha || 0.8) * 0.9})`;
          ctx.fillRect(x - 0.45, y - 0.45, 0.9, 0.9);
        }
      }
    });

    ctx.shadowBlur = 0;
  }

  /* ==========================================================================
     5b. 3 LAYER CC0 ERROR VECTOR SPECTRUM (First Attached Image)
     ========================================================================== */
  function drawCc0Evm() {
    const setup = setupCanvas(cCc0Evm);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 24;
    const padBottom = 16;
    const padTop = 12;
    const padRight = 6;
    const plotW = Math.max(10, width - padLeft - padRight);
    const plotH = Math.max(10, height - padTop - padBottom);

    // 1. Pitch Black Background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // 2. Y-Axis Ticks (4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 500m)
    ctx.font = '500 6.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const yVals = [
      { v: 4.5, lbl: '4.5' },
      { v: 4.0, lbl: '4' },
      { v: 3.5, lbl: '3.5' },
      { v: 3.0, lbl: '3' },
      { v: 2.5, lbl: '2.5' },
      { v: 2.0, lbl: '2' },
      { v: 1.5, lbl: '1.5' },
      { v: 1.0, lbl: '1' },
      { v: 0.5, lbl: '500m' },
    ];

    const maxVal = 5.0;

    yVals.forEach((item) => {
      const y = padTop + ((maxVal - item.v) / maxVal) * plotH;
      ctx.fillText(item.lbl, padLeft - 2, y);

      // Horizontal grid lines
      ctx.strokeStyle = '#223042';
      ctx.lineWidth = 0.7;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    // 3. Vertical Grid Lines (10 divisions across -300 to +300 carriers)
    for (let i = 1; i < 10; i++) {
      const x = padLeft + (i / 10) * plotW;
      ctx.strokeStyle = '#223042';
      ctx.lineWidth = 0.7;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();
    }

    // Outer grid border
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 0.8;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    // 4. Draw Green & Cyan EVM Error Vector Stems & Baseline
    const numCarriers = Math.floor(plotW);
    const whitePoints = [];
    const t = state.simTime || 0;
    const isRunning = state.isRunning;
    const slotIdx = Math.floor(t * 14);

    // Dynamic SNR scaling from backend or simulation
    const currentSnr = (window.VyaptiConstellationData && window.VyaptiConstellationData.snr_db) ? window.VyaptiConstellationData.snr_db : (12.4 + 1.2 * Math.sin(t * 1.6));
    const snrScale = Math.max(0.65, Math.min(1.65, 14.0 / currentSnr));

    for (let i = 0; i < numCarriers; i++) {
      const frac = i / Math.max(1, numCarriers - 1);
      const x = padLeft + frac * plotW;
      const carrierIdx = (frac - 0.5) * 600;

      // Base upward curve at the edges (subcarrier rolloff filter response)
      const edgeCurve = (0.22 + 0.16 * Math.pow(Math.abs(carrierIdx) / 300, 2.4)) * snrScale + (isRunning ? 0.02 * Math.sin(t * 2.0 + carrierIdx * 0.02) : 0);
      
      // Dynamic per-subcarrier EVM height
      const seedVal = Math.sin(i * 12.9898 + slotIdx * 3.71 + carrierIdx * 0.04) * 43758.5453;
      const noise = (seedVal - Math.floor(seedVal));
      const fastJitter = isRunning ? Math.sin(i * 4.3 + t * 14.0) * 0.05 : 0;
      const greenHeightMag = Math.max(0.18, edgeCurve + noise * 0.55 * snrScale + fastJitter);
      const yGreen = padTop + ((maxVal - Math.min(maxVal, greenHeightMag)) / maxVal) * plotH;

      // Green vertical stem
      ctx.strokeStyle = '#22c55e';
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(x, padTop + plotH);
      ctx.lineTo(x, yGreen);
      ctx.stroke();

      // Cyan sporadic high stems with dot heads (peaks & pilot carriers)
      const isPilot = (Math.abs(Math.round(carrierIdx)) % 12 === 0);
      if (noise > 0.65 || isPilot) {
        const boost = isPilot ? 0.55 : (noise - 0.65) * 1.7 * snrScale;
        const cyanHeightMag = greenHeightMag + boost + (isRunning ? Math.sin(t * 11.0 + i * 0.5) * 0.08 : 0);
        const yCyan = padTop + ((maxVal - Math.min(maxVal, cyanHeightMag)) / maxVal) * plotH;

        ctx.strokeStyle = '#00e5ff';
        ctx.lineWidth = 1.0;
        ctx.beginPath();
        ctx.moveTo(x, yGreen);
        ctx.lineTo(x, yCyan);
        ctx.stroke();

        // Dot head
        ctx.fillStyle = '#00e5ff';
        ctx.fillRect(x - 0.6, yCyan - 0.6, 1.2, 1.2);
      }

      // White lower envelope profile
      const yWhite = padTop + ((maxVal - edgeCurve) / maxVal) * plotH;
      whitePoints.push({ x, y: yWhite });
    }

    // 5. Draw Thick White Lower Envelope Curve
    ctx.save();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.6;
    ctx.shadowColor = 'rgba(255, 255, 255, 0.7)';
    ctx.shadowBlur = 3;
    ctx.beginPath();
    whitePoints.forEach((p, idx) => {
      if (idx === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
    ctx.restore();
  }

  /* ==========================================================================
     5c. Q-CHANNEL CARRIER MODULATION: Q(t)·sin(2π·fc·t) (Second Attached Image)
     ========================================================================== */
  function drawQCarrier() {
    const setup = setupCanvas(cQCarrier);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 28;
    const padBottom = 16;
    const padTop = 18;
    const padRight = 10;
    const plotW = Math.max(10, width - padLeft - padRight);
    const plotH = Math.max(10, height - padTop - padBottom);

    // 1. Pure Pitch Black Background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // 2. Y-Axis Ticks (1, 0, -1)
    ctx.font = '600 8.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const yTicks = [
      { v: 1.0, lbl: '1' },
      { v: 0.0, lbl: '0' },
      { v: -1.0, lbl: '-1' },
    ];

    yTicks.forEach((item) => {
      const y = padTop + ((1.0 - item.v) / 2.0) * plotH;
      ctx.fillText(item.lbl, padLeft - 4, y);

      // Subtle horizontal baseline
      ctx.strokeStyle = item.v === 0 ? '#1e293b' : '#0f172a';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    // 3. X-Axis Time Ticks (0, 0.1, 0.2)
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xTicks = [
      { t: 0.0, lbl: '0' },
      { t: 0.1, lbl: '0.1' },
      { t: 0.2, lbl: '0.2' },
    ];

    xTicks.forEach((item) => {
      const x = padLeft + (item.t / 0.2) * plotW;
      ctx.fillText(item.lbl, x, padTop + plotH + 2);

      // Subtle vertical grid line
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();
    });

    // Outer border around plot
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 0.8;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    // 4. Modulated Carrier Waveform: Q(t) * sin(2*pi*fc*t)
    // Symbol sequence Q(t) in {-1, +1} across t in [0, 0.2]
    const bitBoundaries = [0.0, 0.024, 0.052, 0.076, 0.098, 0.126, 0.148, 0.174, 0.20];
    const bitValues = [-1, 1, 1, -1, 1, -1, 1, -1];

    function getQ(tSec) {
      for (let b = 0; b < bitValues.length; b++) {
        if (tSec >= bitBoundaries[b] && tSec < bitBoundaries[b + 1]) {
          return bitValues[b];
        }
      }
      return 1;
    }

    const fc = 120; // 24 cycles across 0.2s
    const phaseOffset = (state.iqPhase || 0) * 1.5;

    ctx.save();
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1.6;
    ctx.shadowColor = 'rgba(239, 68, 68, 0.5)';
    ctx.shadowBlur = 3;
    ctx.beginPath();

    for (let px = 0; px <= plotW; px++) {
      const t = (px / plotW) * 0.2;
      const qVal = getQ(t);
      const carrier = Math.sin(2 * Math.PI * fc * t + phaseOffset);
      const val = qVal * carrier * 0.95;

      const x = padLeft + px;
      const y = padTop + ((1.0 - val) / 2.0) * plotH;

      if (px === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }

    ctx.stroke();
    ctx.restore();
  }

  /* ==========================================================================
     5d. RF BAND SCAN HISTORY & PO-RMAB SCHEDULER (vyapti_pormab_500pool.py)
     ========================================================================== */
  const N_BANDS_PO = 36;
  const MODEL_PD_PO = 0.90;
  const MODEL_PFA_PO = 0.05;
  const WHITTLE_GAMMA_PO = 0.997;

  // 36 continuous 500 MHz bands spanning 0.00 GHz to 18.00 GHz
  const PORMAB_BANDS_PHYSICS = [];
  for (let b = 0; b < N_BANDS_PO; b++) {
    const fMin = b * 0.50;
    const fMax = (b + 1) * 0.50;
    const center = 0.25 + b * 0.50;
    PORMAB_BANDS_PHYSICS.push({
      bandId: b + 1,
      bandIdx: b,
      fMin: fMin,
      fMax: fMax,
      center: center,
      name: `${fMin.toFixed(2)} \u2013 ${fMax.toFixed(2)} GHz`,
    });
  }

  // Active emitter profiles in the mission scenario
  const EMITTERS_PHYSICS = [
    { band: 3,  period: 20, width: 5, phase: 0 },
    { band: 7,  period: 25, width: 6, phase: 6 },
    { band: 10, period: 18, width: 4, phase: 3 },
    { band: 17, period: 22, width: 5, phase: 12 },
    { band: 24, period: 30, width: 7, phase: 2 },
    { band: 30, period: 16, width: 4, phase: 8 },
    { band: 33, period: 28, width: 6, phase: 15 }
  ];

  class PormabPhysicsEngine {
    constructor() {
      this.nBands = N_BANDS_PO;
      this.transition = new Array(this.nBands);
      this.belief = new Float64Array(this.nBands);
      this.hitTimestamps = Array.from({ length: this.nBands }, () => []);
      this.lastVisit = new Int32Array(this.nBands).fill(-1);
      this.visitCounts = new Int32Array(this.nBands).fill(0);
      this.elapsedSlots = 0;

      for (let b = 0; b < this.nBands; b++) {
        const isEm = EMITTERS_PHYSICS.some(e => e.band === b);
        const p01 = isEm ? 0.12 : 0.02;
        const p11 = isEm ? 0.82 : 0.20;
        this.transition[b] = { p01, p11 };
        this.belief[b] = p01 / (p01 + 1.0 - p11);
      }
    }

    whittleIndex(band, p) {
      const t = this.transition[band];
      const pClip = Math.max(1e-6, Math.min(1.0 - 1e-6, p));
      const delta = t.p11 - t.p01;
      const denom = 1.0 - WHITTLE_GAMMA_PO * delta;
      return (delta * pClip + t.p01) / Math.max(denom, 1e-6);
    }

    estimatePeriodicity(band, nowSlot) {
      const hist = this.hitTimestamps[band];
      if (hist.length < 2) return 0.0;
      const gaps = [];
      for (let i = 1; i < hist.length; i++) gaps.push(hist[i] - hist[i - 1]);
      const period = gaps[gaps.length - 1] || 10;
      const last = hist[hist.length - 1];
      const elapsed = Math.max(0, nowSlot - last);
      const rem = elapsed % period;
      const phaseErr = Math.min(rem, period - rem);
      return Math.exp(-phaseErr / Math.max(0.2 * period, 1.0));
    }

    selectNextBand(prevAction = -1) {
      let bestBand = 0;
      let bestScore = -Infinity;
      for (let b = 0; b < this.nBands; b++) {
        let p = this.belief[b];
        let lookahead = 0;
        const t = this.transition[b];
        for (let k = 0; k < 2; k++) {
          lookahead += (WHITTLE_GAMMA_PO ** k) * this.whittleIndex(b, p);
          p = (1.0 - p) * t.p01 + p * t.p11;
        }
        const pScore = this.estimatePeriodicity(b, this.elapsedSlots);
        const staleness = this.lastVisit[b] >= 0 ? Math.min(1.0, (this.elapsedSlots - this.lastVisit[b]) / 40.0) : 1.0;
        let score = lookahead + 0.25 * pScore + 0.40 * staleness;
        if (prevAction >= 0 && b !== prevAction) score -= 0.05;

        if (score > bestScore) {
          bestScore = score;
          bestBand = b;
        }
      }
      return bestBand;
    }

    stepObservation(action, isHit) {
      const y = isHit ? 1 : 0;
      if (isHit) {
        this.hitTimestamps[action].push(this.elapsedSlots);
        if (this.hitTimestamps[action].length > 16) this.hitTimestamps[action].shift();
      }
      this.lastVisit[action] = this.elapsedSlots;
      this.visitCounts[action]++;
      this.elapsedSlots += 2;

      const p = this.belief[action];
      const likeAct = y ? MODEL_PD_PO : (1.0 - MODEL_PD_PO);
      const likeInact = y ? MODEL_PFA_PO : (1.0 - MODEL_PFA_PO);
      const denom = p * likeAct + (1.0 - p) * likeInact;
      const post = (p * likeAct) / Math.max(denom, 1e-12);

      for (let b = 0; b < this.nBands; b++) {
        const p0 = (b === action) ? post : this.belief[b];
        const t = this.transition[b];
        this.belief[b] = Math.max(1e-6, Math.min(1.0 - 1e-6, (1.0 - p0) * t.p01 + p0 * t.p11));
      }
    }
  }

  const pormabPhysicsEngine = new PormabPhysicsEngine();

  const pormabPhysicsState = {
    engine: pormabPhysicsEngine,
    currentBandIdx: 17, // Start at Band 18 (8.50 - 9.00 GHz)
    nextBandIdx: pormabPhysicsEngine.selectNextBand(17),
    lastScannedBand: {
      bandIdx: 15,
      name: '7.50 \u2013 8.00 GHz',
      center: 7.750,
      result: 'MISS',
      snr: 8.4,
      time: 0.12
    },
    dwellDuration: 1.2, // seconds per live visualization dwell
    dwellElapsed: 0.0,
    scannedBandsCount: 18,
    occupiedBandsCount: 7,
    history: [
      { tStart: 0.1, tEnd: 1.3, bandIdx: 3, fMin: 1.5, fMax: 2.0, hit: true, name: '1.50 \u2013 2.00 GHz' },
      { tStart: 1.4, tEnd: 2.6, bandIdx: 7, fMin: 3.5, fMax: 4.0, hit: true, name: '3.50 \u2013 4.00 GHz' },
      { tStart: 2.7, tEnd: 3.9, bandIdx: 10, fMin: 5.0, fMax: 5.5, hit: false, name: '5.00 \u2013 5.50 GHz' },
      { tStart: 4.0, tEnd: 5.2, bandIdx: 17, fMin: 8.5, fMax: 9.0, hit: true, name: '8.50 \u2013 9.00 GHz' }
    ]
  };

  function updatePormabPhysicsScanning(dt) {
    if (!state.isRunning || state.isPaused) return;

    pormabPhysicsState.dwellElapsed += dt;
    const progress = Math.min(1.0, pormabPhysicsState.dwellElapsed / pormabPhysicsState.dwellDuration);
    const pct = Math.floor(progress * 100);

    const elCurPct = document.getElementById('rfValCurBandProgressPct');
    const elCurBar = document.getElementById('rfValCurBandProgressBar');
    if (elCurPct) elCurPct.textContent = `${pct}%`;
    if (elCurBar) elCurBar.style.width = `${pct}%`;

    // Complete dwell and advance PO-RMAB
    if (pormabPhysicsState.dwellElapsed >= pormabPhysicsState.dwellDuration) {
      const curIdx = pormabPhysicsState.currentBandIdx;
      const curBand = PORMAB_BANDS_PHYSICS[curIdx];

      // Check if emitter is active during this slot
      const em = EMITTERS_PHYSICS.find(e => e.band === curIdx);
      const isActive = em ? (((pormabPhysicsEngine.elapsedSlots + em.phase) % em.period) < em.width) : false;
      const isHit = isActive ? (Math.random() < MODEL_PD_PO) : (Math.random() < MODEL_PFA_PO);

      pormabPhysicsEngine.stepObservation(curIdx, isHit);

      const tStart = Math.max(0, state.simTime - pormabPhysicsState.dwellDuration);
      const tEnd = state.simTime;

      // Update history
      pormabPhysicsState.history.push({
        tStart: tStart,
        tEnd: tEnd,
        bandIdx: curIdx,
        fMin: curBand.fMin,
        fMax: curBand.fMax,
        hit: isHit,
        name: curBand.name
      });
      if (pormabPhysicsState.history.length > 40) pormabPhysicsState.history.shift();

      // Update last scanned band
      const snrVal = (isHit ? (12.4 + Math.random() * 3.2) : (7.5 + Math.random() * 2.0)).toFixed(1);
      pormabPhysicsState.lastScannedBand = {
        bandIdx: curIdx,
        name: curBand.name,
        center: curBand.center,
        result: isHit ? 'HIT' : 'MISS',
        snr: snrVal,
        time: state.simTime.toFixed(2)
      };

      if (isHit) {
        pormabPhysicsState.occupiedBandsCount = Math.min(36, pormabPhysicsState.occupiedBandsCount + (Math.random() > 0.6 ? 1 : 0));
      }
      pormabPhysicsState.scannedBandsCount = Math.min(36, Math.max(pormabPhysicsState.scannedBandsCount, Math.floor(state.simTime * 2.5) % 36 + 1));

      // Advance: current band becomes next, and select new next via PO-RMAB Whittle lookahead
      pormabPhysicsState.currentBandIdx = pormabPhysicsState.nextBandIdx;
      pormabPhysicsState.nextBandIdx = pormabPhysicsEngine.selectNextBand(pormabPhysicsState.currentBandIdx);
      pormabPhysicsState.dwellElapsed = 0.0;

      const newCur = PORMAB_BANDS_PHYSICS[pormabPhysicsState.currentBandIdx];
      const newNext = PORMAB_BANDS_PHYSICS[pormabPhysicsState.nextBandIdx];
      const lastBand = pormabPhysicsState.lastScannedBand;

      // Update CURRENT BAND DOM
      const elCurRange = document.getElementById('rfValCurBandRange');
      const elCurCF = document.getElementById('rfValCurBandCF');
      const elCurDwell = document.getElementById('rfValCurBandDwell');
      if (elCurRange) elCurRange.textContent = newCur.name;
      if (elCurCF) elCurCF.textContent = `${newCur.center.toFixed(3)} GHz`;
      if (elCurDwell) elCurDwell.textContent = '100 ms';

      // Update LAST SCANNED DOM
      const elLastRange = document.getElementById('rfValLastBandRange');
      const elLastCF = document.getElementById('rfValLastBandCF');
      const elLastResult = document.getElementById('rfValLastBandResult');
      const elLastSnr = document.getElementById('rfValLastBandSnr');
      const elLastTime = document.getElementById('rfValLastBandTime');

      if (elLastRange) elLastRange.textContent = lastBand.name;
      if (elLastCF) elLastCF.textContent = `${lastBand.center.toFixed(3)} GHz`;
      if (elLastSnr) elLastSnr.textContent = `${lastBand.snr} dB`;
      if (elLastTime) elLastTime.textContent = `${lastBand.time} s`;
      if (elLastResult) {
        elLastResult.textContent = lastBand.result;
        elLastResult.className = lastBand.result === 'HIT' ? 'rf-badge-hit' : 'rf-badge-miss';
      }

      // Update NEXT BAND DOM
      const elNextRange = document.getElementById('rfValNextBandRange');
      const elNextCF = document.getElementById('rfValNextBandCF');
      const elNextDwell = document.getElementById('rfValNextBandDwell');
      const elNextPriority = document.getElementById('rfValNextBandPriority');
      const elNextStatus = document.getElementById('rfValNextBandStatus');

      if (elNextRange) elNextRange.textContent = newNext.name;
      if (elNextCF) elNextCF.textContent = `${newNext.center.toFixed(3)} GHz`;
      if (elNextDwell) elNextDwell.textContent = '100 ms';
      if (elNextStatus) elNextStatus.textContent = 'QUEUED';
      if (elNextPriority) {
        const isPri = EMITTERS_PHYSICS.some(e => e.band === pormabPhysicsState.nextBandIdx);
        elNextPriority.textContent = isPri ? 'HIGH' : 'MEDIUM';
        elNextPriority.className = isPri ? 'rf-mini-v highlight-priority' : 'rf-mini-v highlight-cyan';
      }

      // Update Environment Counts
      const elEnvTotal = document.getElementById('rfEnvTotalBands');
      const elEnvScanned = document.getElementById('rfEnvScannedBands');
      const elEnvOcc = document.getElementById('rfEnvOccupiedBands');
      const elEnvEmpty = document.getElementById('rfEnvEmptyBands');
      const elEvtBand = document.getElementById('rfEventBand');
      const elStatsScanned = document.getElementById('rfStatsBandsScanned');
      const elStatsOcc = document.getElementById('rfStatsOccBands');

      if (elEnvTotal) elEnvTotal.textContent = '36';
      if (elEnvScanned) elEnvScanned.textContent = `${pormabPhysicsState.scannedBandsCount} / 36`;
      if (elEnvOcc) elEnvOcc.textContent = `${pormabPhysicsState.occupiedBandsCount}`;
      if (elEnvEmpty) elEnvEmpty.textContent = `${36 - pormabPhysicsState.occupiedBandsCount}`;
      if (elEvtBand) elEvtBand.textContent = `${newCur.bandId}`;
      if (elStatsScanned) elStatsScanned.textContent = `${pormabPhysicsState.scannedBandsCount} / 36`;
      if (elStatsOcc) elStatsOcc.textContent = `${pormabPhysicsState.occupiedBandsCount} / 36`;
    }
  }

  function drawRfScanHistory() {
    const setup = setupCanvas(cRfScanHistory);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 28;
    const padBottom = 16;
    const padTop = 8;
    const padRight = 8;
    const plotW = Math.max(10, width - padLeft - padRight);
    const plotH = Math.max(10, height - padTop - padBottom);

    // 1. Pitch Black Background
    ctx.fillStyle = '#01050e';
    ctx.fillRect(0, 0, width, height);

    // 2. Y-Axis Frequency (GHz): 18.0, 15.0, 12.0, 9.0, 6.0, 3.0, 0.0
    ctx.font = '500 7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const yFreqs = [18.0, 15.0, 12.0, 9.0, 6.0, 3.0, 0.0];
    const minF = 0.0;
    const maxF = 18.0;

    yFreqs.forEach((f) => {
      const y = padTop + ((maxF - f) / (maxF - minF)) * plotH;
      ctx.fillText(f.toFixed(1), padLeft - 4, y);

      ctx.strokeStyle = '#0c1626';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    // 3. X-Axis Time (s)
    const tCurrent = state.simTime;
    const tWindowEnd = Math.max(12.0, tCurrent + 3.0);
    const tWindowStart = tWindowEnd - 12.0;

    function timeToX(t) { return padLeft + ((t - tWindowStart) / 12.0) * plotW; }
    function freqToY(f) { return padTop + ((maxF - f) / (maxF - minF)) * plotH; }

    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const firstTick = Math.ceil(tWindowStart / 2) * 2;
    for (let t = firstTick; t <= tWindowEnd; t += 2) {
      const x = timeToX(t);
      if (x >= padLeft && x <= padLeft + plotW) {
        ctx.fillText(t.toFixed(0), x, padTop + plotH + 3);
        ctx.strokeStyle = '#0c1626';
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.moveTo(x, padTop);
        ctx.lineTo(x, padTop + plotH);
        ctx.stroke();
      }
    }

    // 4. Draw 36 Horizontal Band Tracks
    const numBands = 36;
    for (let b = 0; b < numBands; b++) {
      const bandFreq = 0.25 + b * 0.50;
      const y = freqToY(bandFreq);
      ctx.strokeStyle = 'rgba(30, 58, 95, 0.25)';
      ctx.lineWidth = 0.5;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    }

    // Outer Border
    ctx.strokeStyle = '#142338';
    ctx.lineWidth = 0.8;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    // Clip to plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(padLeft, padTop, plotW, plotH);
    ctx.clip();

    // 5. Draw Historical Completed Dwells
    pormabPhysicsState.history.forEach(d => {
      if (d.tEnd < tWindowStart || d.tStart > tWindowEnd) return;
      const x1 = timeToX(d.tStart);
      const x2 = timeToX(d.tEnd);
      const bw = Math.max(3, x2 - x1);
      const bandFreq = 0.25 + d.bandIdx * 0.50;
      const y = freqToY(bandFreq);

      if (d.hit) {
        // Yellow detection marker
        ctx.strokeStyle = '#facc15';
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.moveTo(x1, y);
        ctx.lineTo(x2, y);
        ctx.stroke();

        ctx.fillStyle = '#facc15';
        ctx.beginPath();
        ctx.arc(x1 + bw / 2, y, 2.4, 0, Math.PI * 2);
        ctx.fill();
      } else {
        // Cyan/Slate scanned line
        ctx.strokeStyle = 'rgba(56, 189, 248, 0.65)';
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(x1, y);
        ctx.lineTo(x2, y);
        ctx.stroke();
      }
    });

    // 6. Highlight Current Active Band (Bright Green / Cyan)
    const curBand = PORMAB_BANDS_PHYSICS[pormabPhysicsState.currentBandIdx] || PORMAB_BANDS_PHYSICS[0];
    const curY = freqToY(curBand.center);
    const curX = timeToX(tCurrent);

    ctx.save();
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 2.2;
    ctx.shadowColor = 'rgba(0, 255, 136, 0.85)';
    ctx.shadowBlur = 6;
    ctx.beginPath();
    ctx.moveTo(Math.max(padLeft, curX - 30), curY);
    ctx.lineTo(curX, curY);
    ctx.stroke();

    ctx.fillStyle = '#00ff88';
    ctx.beginPath();
    ctx.arc(curX, curY, 2.8, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // 7. Highlight Next Band (Dashed Blue)
    const nextBand = PORMAB_BANDS_PHYSICS[pormabPhysicsState.nextBandIdx] || PORMAB_BANDS_PHYSICS[1];
    const nextY = freqToY(nextBand.center);

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.4;
    ctx.setLineDash([3, 2]);
    ctx.beginPath();
    ctx.moveTo(curX, nextY);
    ctx.lineTo(Math.min(padLeft + plotW, curX + 30), nextY);
    ctx.stroke();
    ctx.setLineDash([]);

    // 8. Vertical Cursor
    ctx.strokeStyle = 'rgba(0, 255, 136, 0.65)';
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    ctx.moveTo(curX, padTop);
    ctx.lineTo(curX, padTop + plotH);
    ctx.stroke();

    ctx.restore();
  }

  /* ==========================================================================
     5.5 SIGNAL QUALITY: 3-LAYER CCDF (Complementary Cumulative Distribution)
     ========================================================================== */
  function drawCCDF() {
    const setup = setupCanvas(cCCDF);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 32;
    const padRight = 10;
    const padTop = 8;
    const padBottom = 20;
    const plotW = Math.max(10, width - padLeft - padRight);
    const plotH = Math.max(10, height - padTop - padBottom);

    // 1. Pitch Black Background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // 2. Y-Axis Label: "Prob (%)"
    ctx.save();
    ctx.translate(10, padTop + plotH * 0.5);
    ctx.rotate(-Math.PI / 2);
    ctx.font = '600 7.5px "Inter", sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Prob (%)', 0, 0);
    ctx.restore();

    // 3. X-Axis Label: "Power > Avg (dB)"
    ctx.font = '600 7.5px "Inter", sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('Power > Avg (dB)', padLeft + plotW * 0.5, padTop + plotH + 9);

    // 4. Y-Axis Log Scale Decades: 100%, 10%, 1%, 0.1%, 0.01% (4 decades: log10 = 0 to -4)
    const logDecades = [
      { exp: 0, lbl: '100%' },
      { exp: -1, lbl: '10%' },
      { exp: -2, lbl: '1%' },
      { exp: -3, lbl: '0.1%' },
      { exp: -4, lbl: '0.01%' }
    ];

    ctx.font = '600 7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    logDecades.forEach((d) => {
      const frac = -d.exp / 4.0;
      const y = padTop + frac * plotH;

      ctx.fillText(d.lbl, padLeft - 3, y);

      // Horizontal decade grid line
      ctx.strokeStyle = d.exp === 0 ? '#1e293b' : '#141c28';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();

      // Tick mark
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(padLeft - 3, y);
      ctx.lineTo(padLeft, y);
      ctx.stroke();
    });

    // 5. X-Axis Ticks & Grid: 0, 2, 4, 6, 8, 10, 12 dB
    const xTicks = [0, 2, 4, 6, 8, 10, 12];
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    xTicks.forEach((db) => {
      const frac = db / 12.0;
      const x = padLeft + frac * plotW;

      ctx.fillText(db.toString(), x, padTop + plotH + 2);

      // Vertical grid line
      if (db > 0 && db < 12) {
        ctx.strokeStyle = '#141c28';
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.moveTo(x, padTop);
        ctx.lineTo(x, padTop + plotH);
        ctx.stroke();
      }

      // Tick mark
      ctx.strokeStyle = '#64748b';
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(x, padTop + plotH);
      ctx.lineTo(x, padTop + plotH + 3);
      ctx.stroke();
    });

    // 6. Outer Plot Box Border
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1.0;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    // 7. Gaussian Reference Curve (Grey dashed)
    ctx.save();
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    for (let px = 0; px <= plotW; px++) {
      const db = (px / plotW) * 12.0;
      const linRatio = Math.pow(10, db / 10.0);
      const prob = Math.exp(-linRatio); // Rayleigh / Gaussian CCDF
      const logP = prob > 0 ? Math.log10(prob) : -4;
      const clampedLogP = Math.max(-4, Math.min(0, logP));
      const fracY = -clampedLogP / 4.0;
      const y = padTop + fracY * plotH;
      const x = padLeft + px;

      if (px === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    const tSim = state.simTime;
    const ripple1 = 0.08 * Math.sin(tSim * 2.5);
    const ripple2 = 0.05 * Math.cos(tSim * 3.2);

    // Helper to map log probability to Y coordinate
    const logPToY = (logP) => {
      const clamped = Math.max(-4, Math.min(0, logP));
      return padTop + (-clamped / 4.0) * plotH;
    };

    // 8. Q-Component Curve (Amber #facc15)
    ctx.save();
    ctx.strokeStyle = '#facc15';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let px = 0; px <= plotW; px++) {
      const db = (px / plotW) * 12.0;
      // Q-curve: sharp knee around 6.8 dB
      const linP = Math.exp(-Math.pow(db / 3.4, 2.2) + ripple2 * Math.sin(db * 1.5));
      const logP = linP > 0.0001 ? Math.log10(linP) : -4;
      const y = logPToY(logP);
      const x = padLeft + px;
      if (px === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // 9. I-Component Curve (Teal #06b6d4)
    ctx.save();
    ctx.strokeStyle = '#06b6d4';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let px = 0; px <= plotW; px++) {
      const db = (px / plotW) * 12.0;
      // I-curve: sharp knee around 7.0 dB
      const linP = Math.exp(-Math.pow(db / 3.55, 2.15) + ripple1 * Math.cos(db * 1.4));
      const logP = linP > 0.0001 ? Math.log10(linP) : -4;
      const y = logPToY(logP);
      const x = padLeft + px;
      if (px === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // 10. Total Signal Measured Curve (Cyan #38bdf8, glow, PAPR knee at ~7.4 dB)
    ctx.save();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.6;
    ctx.shadowColor = 'rgba(56, 189, 248, 0.4)';
    ctx.shadowBlur = 4;
    ctx.beginPath();
    for (let px = 0; px <= plotW; px++) {
      const db = (px / plotW) * 12.0;
      // Realistic multi-carrier/QAM signal PAPR roll-off
      let linP;
      if (db < 4.0) {
        linP = Math.exp(-Math.pow(db / 3.8, 1.85));
      } else {
        linP = Math.exp(-Math.pow(db / 3.8, 1.85) - Math.pow((db - 4.0) / 1.6, 2.4) + ripple1 * 0.5);
      }
      const logP = linP > 0.0001 ? Math.log10(linP) : -4;
      const y = logPToY(logP);
      const x = padLeft + px;
      if (px === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  /* ==========================================================================
     5.7 LIVE METRICS (SYSTEM C) SPARKLINES
     ========================================================================== */
  const sparkConfigs = [
    { id: 'sparkPd', color: '#22c55e', fill: 'rgba(34, 197, 94, 0.12)', seed: 0.1 },
    { id: 'sparkPfa', color: '#38bdf8', fill: 'rgba(56, 189, 248, 0.12)', seed: 0.2 },
    { id: 'sparkIntercept', color: '#d946ef', fill: 'rgba(217, 70, 239, 0.12)', seed: 0.3 },
    { id: 'sparkSnr', color: '#eab308', fill: 'rgba(234, 179, 8, 0.12)', seed: 0.4 },
    { id: 'sparkTti', color: '#f59e0b', fill: 'rgba(245, 158, 11, 0.12)', seed: 0.5 },
    { id: 'sparkCoverage', color: '#06b6d4', fill: 'rgba(6, 182, 212, 0.12)', seed: 0.6 },
    { id: 'sparkSwitches', color: '#e2e8f0', fill: 'rgba(226, 232, 240, 0.12)', seed: 0.7 }
  ];

  function drawLiveMetricsSparklines() {
    const t = state.simTime;
    sparkConfigs.forEach((cfg) => {
      const cvs = document.getElementById(cfg.id);
      if (!cvs) return;
      const setup = setupCanvas(cvs);
      if (!setup) return;
      const { ctx, width, height } = setup;

      const numPoints = 18;
      const stepX = width / (numPoints - 1);
      ctx.beginPath();
      
      for (let i = 0; i < numPoints; i++) {
        const progress = i / (numPoints - 1);
        const baseline = height * (0.82 - 0.62 * Math.pow(progress, 1.2));
        const ripple = Math.sin(t * 3.0 + i * 0.9 + cfg.seed * 10) * 1.5 + Math.cos(t * 1.5 + i * 1.4) * 0.8;
        const y = Math.max(1, Math.min(height - 1, baseline + ripple));
        const x = i * stepX;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }

      ctx.strokeStyle = cfg.color;
      ctx.lineWidth = 1.3;
      ctx.lineJoin = 'round';
      ctx.stroke();

      // Subtle fill underneath
      ctx.lineTo(width, height);
      ctx.lineTo(0, height);
      ctx.closePath();
      ctx.fillStyle = cfg.fill;
      ctx.fill();
    });
  }

  /* ==========================================================================
     6. MAIN RENDER ALL
     ========================================================================== */
  function renderAll() {
    drawPsd();
    drawWaterfall();
    drawRfScanHistory();
    drawIqTimeDomain();
    drawCc0Evm();
    drawQCarrier();
    drawDoppler();
    drawConstellation();
    drawCCDF();
    drawLiveMetricsSparklines();
  }

  window.addEventListener('resize', () => {
    requestAnimationFrame(renderAll);
  });

  /* ==========================================================================
     7. SIMULATION LOOP & TELEMETRY ADVANCE
     ========================================================================== */
  function tickSimulation(now) {
    const dt = (now - state.lastFrame) / 1000;
    state.lastFrame = now;

    if (state.isRunning) {
      state.simTime += dt * 0.4;
      state.wallTime += dt;
      state.decisions += dt * 12;
      state.waterfallOffset += dt * 2.8;
      state.iqPhase += dt * 8.0;

      // Update UI Telemetry
      if (elSimTime) elSimTime.textContent = state.simTime.toFixed(3) + ' s';
      if (elWallTime) elWallTime.textContent = state.wallTime.toFixed(3) + ' s';
      if (elDecisions) elDecisions.textContent = Math.floor(state.decisions).toString();
      if (elEpisodes) elEpisodes.textContent = state.episodes.toString();

      // Advance PO-RMAB Band Scanning Engine & Dynamic UI
      updatePormabPhysicsScanning(0.016);

      const activeBandObj = PORMAB_BANDS_PHYSICS[pormabPhysicsState.currentBandIdx] || PORMAB_BANDS_PHYSICS[0];
      const curCF = activeBandObj.center.toFixed(3);
      const curSnr = (12.7 + 0.3 * Math.sin(state.simTime * 2.5)).toFixed(1);
      const totalTime = state.simTime.toFixed(2);
      const scanPct = Math.floor(Math.min(1.0, pormabPhysicsState.dwellElapsed / pormabPhysicsState.dwellDuration) * 100);

      const elCurCF = document.getElementById('rfValCurBandCF');
      const elCurSnr = document.getElementById('rfValCurBandSnr');
      const elEventTime = document.getElementById('rfEventTime');
      const elStatsTotalTime = document.getElementById('rfStatsTotalTime');
      const elStatsProgress = document.getElementById('rfStatsProgress');
      const elEnvSnr = document.getElementById('rfEnvCurrentSnr');

      if (elCurCF) elCurCF.textContent = `${curCF} GHz`;
      if (elCurSnr) elCurSnr.textContent = `${curSnr} dB`;
      if (elEventTime) elEventTime.textContent = `${(state.simTime - 0.001).toFixed(3)} s`;
      if (elStatsTotalTime) elStatsTotalTime.textContent = `${totalTime} s`;
      if (elStatsProgress) elStatsProgress.textContent = `${scanPct} %`;
      if (elEnvSnr) elEnvSnr.textContent = `${curSnr} dB`;

      // Receiver State & Processing Pipeline Live Telemetry
      const elRxTuner = document.getElementById('rxTunerStatus');
      const elRxLO = document.getElementById('rxStateLO');
      const elRxAgc = document.getElementById('rxStateAgc');
      const elRxNF = document.getElementById('rxStateNF');
      const elRxSfdr = document.getElementById('rxStateSfdr');
      const elRxJitter = document.getElementById('rxStateJitter');
      const elRxIqImb = document.getElementById('rxStateIqImb');
      const elRxFifo = document.getElementById('rxStateFifo');

      if (elRxTuner) elRxTuner.textContent = `LOCKED (${curCF} GHz)`;
      if (elRxLO) elRxLO.textContent = `${curCF} GHz`;
      if (elRxAgc) elRxAgc.textContent = `${(12.0 + 0.4 * Math.sin(state.simTime * 1.4)).toFixed(1)} dB`;
      if (elRxNF) elRxNF.textContent = `${(2.15 + 0.03 * Math.sin(state.simTime * 0.8)).toFixed(2)} dB`;
      if (elRxSfdr) elRxSfdr.textContent = `${(78.4 + 0.5 * Math.sin(state.simTime * 1.1)).toFixed(1)} dBc`;
      if (elRxJitter) elRxJitter.textContent = `${Math.floor(120 + 4 * Math.sin(state.simTime * 2.3))} fs`;
      if (elRxIqImb) elRxIqImb.textContent = `${(0.05 + 0.01 * Math.sin(state.simTime)).toFixed(2)} dB / ${(0.4 + 0.05 * Math.cos(state.simTime)).toFixed(1)}°`;
      if (elRxFifo) elRxFifo.textContent = `${(18.4 + 2.5 * Math.sin(state.simTime * 2.8)).toFixed(1)} %`;

      // Update Live System C Metrics
      const elValPd = document.getElementById('lmValPd');
      const elValPfa = document.getElementById('lmValPfa');
      const elValIntercept = document.getElementById('lmValIntercept');
      const elValSnr = document.getElementById('lmValSnr');
      const elValTti = document.getElementById('lmValTti');
      const elValCoverage = document.getElementById('lmValCoverage');
      const elValSwitches = document.getElementById('lmValSwitches');

      if (elValPd) elValPd.textContent = (0.672 + 0.012 * Math.sin(state.simTime * 1.8)).toFixed(3);
      if (elValPfa) elValPfa.textContent = (0.041 + 0.003 * Math.sin(state.simTime * 2.2 + 1.0)).toFixed(3);
      if (elValIntercept) elValIntercept.textContent = (0.718 + 0.015 * Math.sin(state.simTime * 1.5 + 2.0)).toFixed(3);
      if (elValSnr) elValSnr.textContent = (12.4 + 0.3 * Math.sin(state.simTime * 2.0 + 0.5)).toFixed(1);
      if (elValTti) elValTti.textContent = Math.floor(146 + 3 * Math.sin(state.simTime * 1.3)).toString();
      if (elValCoverage) elValCoverage.textContent = (0.612 + 0.014 * Math.sin(state.simTime * 1.9 + 3.0)).toFixed(3);
      if (elValSwitches) elValSwitches.textContent = Math.floor(28 + 2 * Math.sin(state.simTime * 0.7)).toString();

      // Dynamic Signal Quality & EVM Telemetry
      const elEvmRms = document.getElementById('evmValRms');
      const elEvmPeak = document.getElementById('evmValPeak');
      const elEvmMer = document.getElementById('evmValMer');
      const elEvmPapr = document.getElementById('evmValPapr');
      const elEvmPhaseErr = document.getElementById('evmValPhaseErr');
      const elEvmMagErr = document.getElementById('evmValMagErr');
      const elEvmFreqErr = document.getElementById('evmValFreqErr');
      const elEvmLeakage = document.getElementById('evmValLeakage');
      const elEvmSnr = document.getElementById('evmValSnr');
      const elEvmRmsErr = document.getElementById('evmValRmsErr');

      if (elEvmRms) {
        const rmsVal = (3.4 + 0.15 * Math.sin(state.simTime * 2.1)).toFixed(1);
        const subDb = (-29.4 + 0.3 * Math.sin(state.simTime * 1.8)).toFixed(1);
        elEvmRms.innerHTML = `${rmsVal} % <span class="evm-sub-db">(${subDb} dB)</span>`;
      }
      if (elEvmPeak) elEvmPeak.textContent = `${(7.8 + 0.3 * Math.sin(state.simTime * 1.7)).toFixed(1)} %`;
      if (elEvmMer) elEvmMer.textContent = `${(29.4 + 0.3 * Math.sin(state.simTime * 1.9)).toFixed(1)} dB`;
      if (elEvmPapr) elEvmPapr.textContent = `${(7.42 + 0.04 * Math.sin(state.simTime * 0.9)).toFixed(2)} dB`;
      if (elEvmPhaseErr) elEvmPhaseErr.textContent = `${(1.8 + 0.1 * Math.sin(state.simTime * 2.4)).toFixed(1)}°`;
      if (elEvmMagErr) elEvmMagErr.textContent = `${(1.9 + 0.1 * Math.sin(state.simTime * 1.6)).toFixed(1)} %`;
      if (elEvmFreqErr) elEvmFreqErr.textContent = `+${(1.8 + 0.2 * Math.sin(state.simTime * 3.1)).toFixed(1)} kHz`;
      if (elEvmLeakage) elEvmLeakage.textContent = `${(-42.6 + 0.4 * Math.sin(state.simTime * 1.1)).toFixed(1)} dBc`;
      if (elEvmSnr) elEvmSnr.textContent = `${curSnr} dB`;
      if (elEvmRmsErr) elEvmRmsErr.textContent = (0.028 + 0.001 * Math.sin(state.simTime * 2.8)).toFixed(3);

      // Periodic AI Log streaming
      if (!state.lastAiLogTime || state.simTime - state.lastAiLogTime > 1.6) {
        state.lastAiLogTime = state.simTime;
        appendLiveAiLog();
      }

      // Slightly shift noise grass
      for (let i = 0; i < state.noiseGrass.length; i += 8) {
        state.noiseGrass[i] = Math.random() * 8;
      }

      // Redraw Dynamic Canvases
      drawPsd();
      drawWaterfall();
      drawRfScanHistory();
      drawIqTimeDomain();
      drawCc0Evm();
      drawQCarrier();
      drawDoppler();
      drawCCDF();
      drawLiveMetricsSparklines();
    }

    requestAnimationFrame(tickSimulation);
  }

  const aiLogTemplates = [
    () => `Decision: Band ${((Math.floor(state.simTime * 2) % 36) + 1)} selected | Dwell: 50 ms | Predicted Activity: ${(0.65 + Math.random() * 0.3).toFixed(2)}`,
    () => `HIT: Pulse detected (CF ${(8.0 + Math.random() * 2.5).toFixed(2)} GHz, PW ${(0.8 + Math.random() * 0.8).toFixed(1)} μs, SNR ${(8.0 + Math.random() * 6).toFixed(1)} dB)`,
    () => `Scheduler: Switching to Band ${((Math.floor(state.simTime * 2 + 3) % 36) + 1)} | Reason: High activity probability`,
    () => `MISS: No detection (Band ${((Math.floor(state.simTime * 2) % 36) + 1)})`,
    () => `Emitter E${Math.floor(1 + Math.random() * 6)}: State change &rarr; ${Math.random() > 0.5 ? 'Bursty ON' : 'Agile Frequency Hop'}`,
    () => `PDW extracted: CF ${(9.0 + Math.random() * 3).toFixed(2)} GHz, PW ${(0.6 + Math.random() * 0.8).toFixed(1)} μs, AoA ${(20 + Math.random() * 20).toFixed(1)}°, Amp -${Math.floor(60 + Math.random() * 20)} dBm`,
    () => `Belief update: Band ${((Math.floor(state.simTime * 2) % 36) + 1)} &uarr; (${(0.2 + Math.random() * 0.2).toFixed(2)} &rarr; ${(0.45 + Math.random() * 0.4).toFixed(2)})`,
    () => `Retune cost applied: ${(0.04 + Math.random() * 0.05).toFixed(2)}`,
    () => `Decision: Band ${((Math.floor(state.simTime * 2 + 1) % 36) + 1)} selected | Dwell: 50 ms`,
  ];

  function appendLiveAiLog() {
    if (window.VyaptiRealDataActive) return;
    const container = document.getElementById('aiLogsContainer');
    if (!container) return;
    const msgGen = aiLogTemplates[Math.floor(Math.random() * aiLogTemplates.length)];
    const entry = document.createElement('div');
    entry.className = 'ai-log-entry';
    entry.innerHTML = `
      <span class="log-ts">[${state.simTime.toFixed(3)}]</span>
      <span class="log-msg">${msgGen()}</span>
    `;
    container.appendChild(entry);
    while (container.children.length > 25) {
      container.removeChild(container.firstChild);
    }
    container.scrollTop = container.scrollHeight;
  }

  /* ==========================================================================
     8. INTERACTION & ACCORDION BEHAVIORS
     ========================================================================== */
  function setupInteractions() {
    // Accordion headers toggle
    document.querySelectorAll('.accordion-header').forEach((hdr) => {
      hdr.addEventListener('click', () => {
        const item = hdr.parentElement;
        const arrow = hdr.querySelector('.acc-arrow');
        const content = item.querySelector('.accordion-content');
        if (content.style.display === 'none') {
          content.style.display = 'flex';
          if (arrow) arrow.textContent = '▼';
        } else {
          content.style.display = 'none';
          if (arrow) arrow.textContent = '▶';
        }
      });
    });

    // Start / Pause / Stop Handlers
    function setRunning(running) {
      state.isRunning = running;
      if (running) {
        btnStartTop?.classList.add('active');
        btnCtrlStart?.classList.add('active');
        btnPauseTop?.classList.remove('active');
        btnCtrlPause?.classList.remove('active');
      } else {
        btnStartTop?.classList.remove('active');
        btnCtrlStart?.classList.remove('active');
        btnPauseTop?.classList.add('active');
        btnCtrlPause?.classList.add('active');
      }
    }

    btnStartTop?.addEventListener('click', () => setRunning(true));
    btnCtrlStart?.addEventListener('click', () => setRunning(true));

    btnPauseTop?.addEventListener('click', () => setRunning(false));
    btnCtrlPause?.addEventListener('click', () => setRunning(false));

    function resetSimulation() {
      setRunning(false);
      state.simTime = 12.482;
      state.wallTime = 1.276;
      state.decisions = 842;
      if (elSimTime) elSimTime.textContent = '12.482 s';
      if (elWallTime) elWallTime.textContent = '1.276 s';
      if (elDecisions) elDecisions.textContent = '842';
      renderAll();
    }

    btnStopTop?.addEventListener('click', resetSimulation);
    btnCtrlStop?.addEventListener('click', resetSimulation);
    btnCtrlReset?.addEventListener('click', resetSimulation);

    // Random seed regenerate
    document.getElementById('btnRegenSeed')?.addEventListener('click', () => {
      const seedEl = document.getElementById('seedVal');
      if (seedEl) {
        seedEl.value = Math.floor(10000 + Math.random() * 89999).toString();
      }
    });

    // Slider interactive labels
    document.querySelectorAll('.slider-param-row').forEach((row) => {
      const slider = row.querySelector('.tactical-slider');
      const valEl = row.querySelector('.slider-val');
      if (slider && valEl) {
        slider.addEventListener('input', () => {
          const unit = valEl.textContent.replace(/[\d.-]/g, '').trim();
          valEl.textContent = `${parseFloat(slider.value).toFixed(1)} ${unit}`;
        });
      }
    });

    // Smooth scroll event handlers
    const scrollDownCue = document.getElementById('scrollDownCue');
    const btnScrollToControls = document.getElementById('btnScrollToControls');
    const btnBackTop = document.getElementById('btnBackTop');
    const detailsSection = document.getElementById('detailedAnalyticsSection');
    const mainSection = document.getElementById('mainScreenSection');

    function scrollToDetails(e) {
      if (e) e.preventDefault();
      detailsSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function scrollToMain(e) {
      if (e) e.preventDefault();
      mainSection?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    scrollDownCue?.addEventListener('click', scrollToDetails);
    btnScrollToControls?.addEventListener('click', scrollToDetails);
    btnBackTop?.addEventListener('click', scrollToMain);

    // Responsive Canvas Resize on Window change
    window.addEventListener('resize', () => {
      renderAll();
    });
  }

  /* ==========================================================================
     9. ENTRANCE ANIMATION (Anime.js & Motion One)
     ========================================================================== */
  function runEntrance() {
    renderAll();
    setupInteractions();

    if (window.anime) {
      window.anime({
        targets: '.sidebar-panel, .viz-card',
        opacity: [0, 1],
        translateY: [10, 0],
        delay: window.anime.stagger(40, { start: 100 }),
        easing: 'easeOutCubic',
        duration: 500,
      });

      // Subtle pulse on active hit indicator
      window.anime({
        targets: '.glow-status-hit',
        opacity: [0.75, 1],
        direction: 'alternate',
        loop: true,
        duration: 1200,
        easing: 'easeInOutQuad',
      });
    }

    state.lastFrame = performance.now();
    requestAnimationFrame(tickSimulation);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', runEntrance);
  } else {
    runEntrance();
  }
})();
