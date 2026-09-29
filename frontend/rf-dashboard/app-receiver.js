/**
 * VYAPTI | RF RECEIVER & SCHEDULER DASHBOARD - MASTER ENGINE
 * Precision 60 FPS HTML5 Canvas Rendering, Anime.js, and Random.js Engine
 */

(function () {
  'use strict';

  // ==========================================================================
  // 1. RANDOM NUMBER GENERATOR (Random-js or Fallback)
  // ==========================================================================
  let rng;
  try {
    if (typeof Random !== 'undefined' && Random.Random) {
      rng = new Random.Random(Random.MersenneTwister19937.autoSeed());
    } else {
      throw new Error('Random-js not available');
    }
  } catch (e) {
    rng = {
      real: (min, max) => min + Math.random() * (max - min),
      integer: (min, max) => Math.floor(min + Math.random() * (max - min + 1)),
      bool: (prob = 0.5) => Math.random() < prob
    };
  }

  // Box-Muller Gaussian Noise Generator
  function gaussianRandom(mean = 0, stdDev = 1) {
    let u1 = 0, u2 = 0;
    while (u1 === 0) u1 = Math.random();
    while (u2 === 0) u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + z0 * stdDev;
  }

  // ==========================================================================
  // 2. STATE & TELEMETRY MODEL
  // ==========================================================================
  const state = {
    isRunning: true,
    isPaused: false,
    simTime: 8.420,
    activeReceiver: 1,
    showReceiverWindow: true,
    activeRfTab: 'spectrum',
    
    // Receiver Presets
    receivers: {
      1: { id: 'R1', freq: 9.182, ibw: 50, dwell: 20, snr: 8.6, status: 'HIT', detCount: 3, color: '#facc15' },
      2: { id: 'R2', freq: 8.995, ibw: 40, dwell: 25, snr: 10.4, status: 'HIT', detCount: 5, color: '#38bdf8' },
      3: { id: 'R3', freq: 9.772, ibw: 60, dwell: 15, snr: 6.8, status: 'HIT', detCount: 2, color: '#ef4444' }
    }
  };

  // ==========================================================================
  // 3. COLORMAP HELPERS
  // ==========================================================================
  function getWaterfallRGB(dBm) {
    const norm = Math.max(0, Math.min(1, (dBm - (-120)) / 100));
    let r = 0, g = 0, b = 0;

    if (norm < 0.18) {
      const t = norm / 0.18;
      r = Math.round(2 + t * 4);
      g = Math.round(8 + t * 24);
      b = Math.round(55 + t * 145);
    } else if (norm < 0.38) {
      const t = (norm - 0.18) / 0.20;
      r = 0;
      g = Math.round(32 + t * 220);
      b = Math.round(200 + t * 55);
    } else if (norm < 0.62) {
      const t = (norm - 0.38) / 0.24;
      r = Math.round(t * 220);
      g = 255;
      b = Math.round(255 * (1 - t));
    } else if (norm < 0.82) {
      const t = (norm - 0.62) / 0.20;
      r = 255;
      g = Math.round(255 - t * 125);
      b = 0;
    } else {
      const t = (norm - 0.82) / 0.18;
      r = 255;
      g = Math.round(130 * (1 - t));
      b = Math.round(t * 20);
    }
    return [r, g, b];
  }

  // ==========================================================================
  // 4. CANVAS MANAGEMENT & DPI SCALING
  // ==========================================================================
  const canvases = {
    spectrum: document.getElementById('canvasSpectrum'),
    waterfall: document.getElementById('canvasWaterfall'),
    bandScan: document.getElementById('canvasBandScan'),
    timeline: document.getElementById('canvasTimeline'),
    rxWindow: document.getElementById('canvasRxWindow'),
    constellation: document.getElementById('canvasConstellation'),
    // Section 2 Canvases
    sec2Spectrogram: document.getElementById('canvasSec2Spectrogram'),
    sec2Spectrum: document.getElementById('canvasSec2Spectrum'),
    sec2Iq: document.getElementById('canvasSec2Iq'),
    sec2Timeline: document.getElementById('canvasSec2Timeline'),
    sec2Constellation: document.getElementById('canvasSec2Constellation'),
    scanningView: document.getElementById('canvasScanningView')
  };

  function resizeCanvas(canvas) {
    if (!canvas) return { w: 0, h: 0 };
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);

    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext('2d');
    ctx.resetTransform();
    ctx.scale(dpr, dpr);
    return { w, h, ctx };
  }

  // ==========================================================================
  // 5. RF SPECTRUM POWER MODEL
  // ==========================================================================
  function computeSpectrumPower(freq, isLive = true) {
    let power = -106 + gaussianRandom(0, isLive ? 2.0 : 1.2);

    function addPeak(centerF, peakDbm, sigma) {
      const diff = freq - centerF;
      const effective = -106 + (peakDbm - (-106)) * Math.exp(-(diff * diff) / (2 * sigma * sigma));
      power = Math.max(power, effective);
    }

    addPeak(8.18, -48, 0.007);
    addPeak(8.40, -68, 0.006);
    addPeak(8.60, -68, 0.006);
    addPeak(9.00, -62, 0.007);
    addPeak(9.20, -38, 0.009);
    addPeak(9.77, -56, 0.007);
    addPeak(10.00, -62, 0.007);

    return Math.max(-122, Math.min(-20, power));
  }

  // ==========================================================================
  // 6. WATERFALL BUFFER INITIALIZATION
  // ==========================================================================
  const WF_COLS = 512;
  const WF_ROWS = 220;
  const waterfallHistory = [];

  function initWaterfallHistory() {
    for (let r = 0; r < WF_ROWS; r++) {
      const row = new Float32Array(WF_COLS);
      const rowTime = (r / WF_ROWS) * 12.0;
      
      for (let c = 0; c < WF_COLS; c++) {
        const freq = 8.0 + (c / (WF_COLS - 1)) * 2.0;
        let p = -112 + gaussianRandom(0, 3.5);

        const isNear = (targetF, bw = 0.008) => Math.abs(freq - targetF) < bw;
        if (isNear(8.18)) p = Math.max(p, -55 + gaussianRandom(0, 6));
        if (isNear(8.40)) p = Math.max(p, -72 + gaussianRandom(0, 5));
        if (isNear(9.00)) p = Math.max(p, -65 + gaussianRandom(0, 5));
        if (isNear(9.20, 0.012)) p = Math.max(p, -42 + gaussianRandom(0, 5));
        if (isNear(9.60)) p = Math.max(p, -68 + gaussianRandom(0, 6));
        if (isNear(9.77)) p = Math.max(p, -60 + gaussianRandom(0, 5));
        if (isNear(10.00)) p = Math.max(p, -65 + gaussianRandom(0, 5));

        const burstTimes = [2.8, 4.2, 5.5, 7.2, 10.2];
        burstTimes.forEach(bt => {
          if (Math.abs(rowTime - bt) < 0.15 && freq >= 9.05 && freq <= 9.45) {
            p = Math.max(p, -38 + gaussianRandom(0, 4));
          }
        });

        if ((Math.abs(rowTime - 4.0) < 0.12 || Math.abs(rowTime - 10.0) < 0.12) && Math.abs(freq - 8.18) < 0.15) {
          p = Math.max(p, -45 + gaussianRandom(0, 5));
        }
        if ((Math.abs(rowTime - 3.2) < 0.12 || Math.abs(rowTime - 6.8) < 0.12) && Math.abs(freq - 9.00) < 0.12) {
          p = Math.max(p, -50 + gaussianRandom(0, 5));
        }
        if (Math.abs(rowTime - 6.2) < 0.12 && Math.abs(freq - 9.60) < 0.18) {
          p = Math.max(p, -48 + gaussianRandom(0, 5));
        }
        if ((Math.abs(rowTime - 2.2) < 0.12 || Math.abs(rowTime - 6.8) < 0.12) && Math.abs(freq - 8.60) < 0.20) {
          p = Math.max(p, -46 + gaussianRandom(0, 5));
        }

        let chirpTargetF = 8.40;
        if (rowTime > 5.0) {
          const frac = (rowTime - 5.0) / 5.0;
          chirpTargetF = 8.40 - 0.10 * Math.pow(frac, 1.4);
        } else if (rowTime < 2.0) {
          chirpTargetF = 8.40 + 0.20 * (2.0 - rowTime) / 2.0;
        }

        if (rowTime >= 1.8 && rowTime <= 10.5 && Math.abs(freq - chirpTargetF) < 0.018) {
          p = Math.max(p, -42 + gaussianRandom(0, 4));
        }

        row[c] = Math.max(-125, Math.min(-15, p));
      }
      waterfallHistory.push(row);
    }
  }

  // ==========================================================================
  // 7. RENDERERS FOR SECTION 1 (TOP DASHBOARD)
  // ==========================================================================

  // Aligned Plot Margins for Upper Spectrum & Lower Waterfall
  const PAD_LEFT_S1 = 36;
  const PAD_RIGHT_S1_SPECTRUM = 48; // Accounts for 44px colorbar container below it!
  const PAD_RIGHT_S1_WATERFALL = 4; // Waterfall canvas is inside a container that has colorbar to its right

  function renderSpectrum() {
    const { w, h, ctx } = resizeCanvas(canvases.spectrum);
    if (!ctx || w <= 0 || h <= 0) return;

    const padLeft = PAD_LEFT_S1;
    const padRight = PAD_RIGHT_S1_SPECTRUM;
    const padTop = 8;
    const padBottom = 16;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    ctx.clearRect(0, 0, w, h);

    // Y Axis Grid & Labels: -20, -40, -60, -80, -100, -120 dBm
    ctx.strokeStyle = '#1e1f24';
    ctx.lineWidth = 1;
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.fillStyle = '#71717a';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const yTicks = [-20, -40, -60, -80, -100, -120];
    yTicks.forEach(val => {
      const normY = (val - (-120)) / 100;
      const y = padTop + plotH * (1 - normY);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
      ctx.fillText(val.toString(), padLeft - 4, y);
    });

    function freqToX(f) { return padLeft + ((f - 8.0) / 2.0) * plotW; }
    function dbmToY(p) {
      const normY = Math.max(0, Math.min(1, (p - (-120)) / 100));
      return padTop + plotH * (1 - normY);
    }

    // Receiver Window Overlays (Dashed vertical guidelines & shaded region)
    if (state.showReceiverWindow) {
      ctx.setLineDash([4, 3]);
      ctx.lineWidth = 1.2;

      function drawRxBox(f1, f2, color, alpha = 0.08) {
        const x1 = freqToX(f1), x2 = freqToX(f2);
        ctx.fillStyle = color === '#facc15' ? `rgba(250, 204, 21, ${alpha})` :
                        color === '#38bdf8' ? `rgba(56, 189, 248, ${alpha})` : `rgba(239, 68, 68, ${alpha})`;
        ctx.fillRect(x1, padTop, x2 - x1, plotH);
        ctx.strokeStyle = color;
        ctx.beginPath();
        ctx.moveTo(x1, padTop); ctx.lineTo(x1, padTop + plotH);
        ctx.moveTo(x2, padTop); ctx.lineTo(x2, padTop + plotH);
        ctx.stroke();
      }

      drawRxBox(8.16, 8.20, '#facc15', 0.12);
      drawRxBox(9.20, 9.24, '#facc15', 0.15);
      drawRxBox(8.98, 9.02, '#38bdf8', 0.12);
      drawRxBox(9.75, 9.79, '#ef4444', 0.12);
      ctx.setLineDash([]);
    }

    // Trace
    const points = [];
    const numSteps = Math.min(plotW * 1.5, 450);
    for (let i = 0; i <= numSteps; i++) {
      const frac = i / numSteps;
      const freq = 8.0 + frac * 2.0;
      points.push({ x: padLeft + frac * plotW, y: dbmToY(computeSpectrumPower(freq, true)) });
    }

    ctx.save();
    ctx.shadowBlur = 5;
    ctx.shadowColor = 'rgba(250, 204, 21, 0.5)';
    ctx.strokeStyle = '#ffd000';
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    points.forEach((pt, idx) => { idx === 0 ? ctx.moveTo(pt.x, pt.y) : ctx.lineTo(pt.x, pt.y); });
    ctx.stroke();
    ctx.restore();

    ctx.strokeStyle = '#27272a';
    ctx.lineWidth = 1;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);
  }

  function renderWaterfall() {
    const { w, h, ctx } = resizeCanvas(canvases.waterfall);
    if (!ctx || w <= 0 || h <= 0) return;

    const padLeft = PAD_LEFT_S1;
    const padRight = PAD_RIGHT_S1_WATERFALL;
    const padTop = 2;
    const padBottom = 22;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    ctx.clearRect(0, 0, w, h);

    if (state.isRunning && !state.isPaused && Math.random() < 0.15) {
      const topRow = waterfallHistory[0];
      const newRow = new Float32Array(WF_COLS);
      for (let c = 0; c < WF_COLS; c++) newRow[c] = topRow[c] + gaussianRandom(0, 0.6);
      waterfallHistory.unshift(newRow);
      waterfallHistory.pop();
    }

    const imgData = ctx.createImageData(WF_COLS, WF_ROWS);
    const data = imgData.data;
    for (let r = 0; r < WF_ROWS; r++) {
      const row = waterfallHistory[r] || waterfallHistory[0];
      for (let c = 0; c < WF_COLS; c++) {
        const [red, green, blue] = getWaterfallRGB(row ? row[c] : -112);
        const idx = (r * WF_COLS + c) * 4;
        data[idx] = red; data[idx + 1] = green; data[idx + 2] = blue; data[idx + 3] = 255;
      }
    }

    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = WF_COLS; tempCanvas.height = WF_ROWS;
    tempCanvas.getContext('2d').putImageData(imgData, 0, 0);
    ctx.drawImage(tempCanvas, 0, 0, WF_COLS, WF_ROWS, padLeft, padTop, plotW, plotH);

    // Overlay Emitter Tracks
    ctx.save();
    ctx.shadowBlur = 6; ctx.shadowColor = '#00ff66'; ctx.strokeStyle = '#00ff66'; ctx.lineWidth = 1.8;
    ctx.beginPath();
    const pStart = { x: padLeft + ((8.30 - 8.0) / 2.0) * plotW, y: padTop + (10.0 / 12.0) * plotH };
    const pCp1   = { x: padLeft + ((8.38 - 8.0) / 2.0) * plotW, y: padTop + (6.0 / 12.0) * plotH };
    const pMid   = { x: padLeft + ((8.40 - 8.0) / 2.0) * plotW, y: padTop + (3.4 / 12.0) * plotH };
    const pCp2   = { x: padLeft + ((8.43 - 8.0) / 2.0) * plotW, y: padTop + (2.4 / 12.0) * plotH };
    const pEnd   = { x: padLeft + ((8.62 - 8.0) / 2.0) * plotW, y: padTop + (2.0 / 12.0) * plotH };
    ctx.moveTo(pStart.x, pStart.y);
    ctx.quadraticCurveTo(pCp1.x, pCp1.y, pMid.x, pMid.y);
    ctx.quadraticCurveTo(pCp2.x, pCp2.y, pEnd.x, pEnd.y);
    ctx.stroke();

    ctx.fillStyle = '#00ff66';
    [pStart, pMid, pEnd].forEach(pt => { ctx.beginPath(); ctx.arc(pt.x, pt.y, 2, 0, Math.PI * 2); ctx.fill(); });
    ctx.restore();

    if (state.showReceiverWindow) {
      ctx.setLineDash([4, 3]); ctx.lineWidth = 1.2;
      function drawWfGuide(f1, f2, color) {
        const x1 = padLeft + ((f1 - 8.0) / 2.0) * plotW, x2 = padLeft + ((f2 - 8.0) / 2.0) * plotW;
        ctx.strokeStyle = color; ctx.beginPath();
        ctx.moveTo(x1, padTop); ctx.lineTo(x1, padTop + plotH);
        ctx.moveTo(x2, padTop); ctx.lineTo(x2, padTop + plotH); ctx.stroke();
      }
      drawWfGuide(8.16, 8.20, '#facc15');
      drawWfGuide(9.20, 9.24, '#facc15');
      drawWfGuide(8.98, 9.02, '#38bdf8');
      drawWfGuide(9.75, 9.79, '#ef4444');
      ctx.setLineDash([]);
    }

    ctx.font = '8px "JetBrains Mono", monospace'; ctx.fillStyle = '#71717a'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    [0, 2, 4, 6, 8, 10, 12].forEach(t => ctx.fillText(t.toString(), padLeft - 4, padTop + (t / 12) * plotH));

    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    [8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8, 10.0].forEach(f => {
      const x = padLeft + ((f - 8.0) / 2.0) * plotW;
      ctx.fillText(f.toFixed(1), x, padTop + plotH + 4);
      ctx.strokeStyle = '#27272a'; ctx.beginPath(); ctx.moveTo(x, padTop + plotH); ctx.lineTo(x, padTop + plotH + 3); ctx.stroke();
    });

    ctx.strokeStyle = '#27272a'; ctx.strokeRect(padLeft, padTop, plotW, plotH);
  }

  // ==========================================================================
  // RF BAND SCAN TIMELINE (RECEIVER) - VYAPTI PO-RMAB SCHEDULER ENGINE
  // (Partially Observable Restless Multi-Armed Bandit with Bayesian Belief & Periodicity)
  // Reference: vyapti_pormab_500pool.py / causal_harness.py
  // ==========================================================================
  const N_BANDS = 36;
  const MODEL_PD = 0.90;
  const MODEL_PFA = 0.05;
  const WHITTLE_GAMMA = 0.997;
  const PERIODIC_WEIGHT = 0.20;
  const DEFAULT_DWELL_SLOTS = 2;

  // 36 continuous 500 MHz bands spanning 0.00 GHz to 18.00 GHz
  const PORMAB_BANDS = [];
  for (let b = 0; b < N_BANDS; b++) {
    const fMin = b * 0.50;
    const fMax = (b + 1) * 0.50;
    const center = 0.25 + b * 0.50;
    PORMAB_BANDS.push({
      bandId: b + 1,
      bandIdx: b,
      fMin: fMin,
      fMax: fMax,
      center: center,
      name: `${fMin.toFixed(2)} - ${fMax.toFixed(2)} GHz`
    });
  }

  // Active mission emitter profiles
  const EMITTERS_RECEIVER = [
    { band: 3,  period: 20, width: 5, phase: 0 },
    { band: 7,  period: 25, width: 6, phase: 6 },
    { band: 10, period: 18, width: 4, phase: 3 },
    { band: 17, period: 22, width: 5, phase: 12 },
    { band: 24, period: 30, width: 7, phase: 2 },
    { band: 30, period: 16, width: 4, phase: 8 },
    { band: 33, period: 28, width: 6, phase: 15 }
  ];

  class PormabEngine {
    constructor() {
      this.nBands = N_BANDS;
      this.pd = MODEL_PD;
      this.pfa = MODEL_PFA;
      this.gamma = WHITTLE_GAMMA;
      this.periodicWeight = PERIODIC_WEIGHT;
      this.dwellSlots = DEFAULT_DWELL_SLOTS;

      this.transition = new Array(this.nBands);
      this.belief = new Float64Array(this.nBands);
      this.hitTimestamps = Array.from({ length: this.nBands }, () => []);
      this.lastVisit = new Int32Array(this.nBands).fill(-1);
      this.visitCounts = new Int32Array(this.nBands).fill(0);
      this.elapsedSlots = 0;

      for (let b = 0; b < this.nBands; b++) {
        const isEm = EMITTERS_RECEIVER.some(e => e.band === b);
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
      const denom = 1.0 - this.gamma * delta;
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

    computeScore(band, nowSlot, prevAction) {
      let p = this.belief[band];
      let lookaheadTotal = 0.0;
      const t = this.transition[band];

      for (let k = 0; k < this.dwellSlots; k++) {
        lookaheadTotal += (this.gamma ** k) * this.whittleIndex(band, p);
        p = (1.0 - p) * t.p01 + p * t.p11;
      }

      const pScore = this.estimatePeriodicity(band, nowSlot);
      const staleness = this.lastVisit[band] >= 0 ? Math.min(1.0, (nowSlot - this.lastVisit[band]) / 40.0) : 1.0;
      let total = lookaheadTotal + this.periodicWeight * pScore + 0.40 * staleness;

      if (prevAction >= 0 && band !== prevAction) {
        total -= 0.05;
      }
      return total;
    }

    selectNextBand(prevAction = -1) {
      let bestBand = 0;
      let bestScore = -Infinity;
      for (let b = 0; b < this.nBands; b++) {
        const score = this.computeScore(b, this.elapsedSlots, prevAction);
        if (score > bestScore) {
          bestScore = score;
          bestBand = b;
        }
      }
      return bestBand;
    }

    stepObservation(scannedBand, isHit) {
      const y = isHit ? 1 : 0;
      const nowSlot = this.elapsedSlots;

      if (isHit) {
        this.hitTimestamps[scannedBand].push(nowSlot);
        if (this.hitTimestamps[scannedBand].length > 16) {
          this.hitTimestamps[scannedBand].shift();
        }
      }

      this.lastVisit[scannedBand] = nowSlot;
      this.visitCounts[scannedBand] += 1;
      this.elapsedSlots += this.dwellSlots;

      const p = this.belief[scannedBand];
      const likeActive = y ? this.pd : (1.0 - this.pd);
      const likeInactive = y ? this.pfa : (1.0 - this.pfa);
      const denom = p * likeActive + (1.0 - p) * likeInactive;
      const post = (p * likeActive) / Math.max(denom, 1e-12);

      for (let b = 0; b < this.nBands; b++) {
        const p0 = (b === scannedBand) ? post : this.belief[b];
        const t = this.transition[b];
        this.belief[b] = Math.max(1e-6, Math.min(1.0 - 1e-6, (1.0 - p0) * t.p01 + p0 * t.p11));
      }
    }
  }

  const pormabEngine = new PormabEngine();

  const bandScanState = {
    engine: pormabEngine,
    currentBandIdx: 17, // Band 18 (8.50 - 9.00 GHz)
    nextBandIdx: pormabEngine.selectNextBand(17),
    dwellDuration: 1.2, // seconds per live visualization dwell
    dwellStartTime: 0.1,
    dwellElapsed: 0.0,
    detectedBandsCount: 9,
    totalBandsCount: 36,
    lastScanned: { name: '7.50 - 8.00 GHz', result: 'Miss' },
    history: [
      { tStart: 0.1, tEnd: 1.3, fMin: 1.5, fMax: 2.0, state: 'detected', hit: true, name: '1.50 - 2.00 GHz' },
      { tStart: 1.4, tEnd: 2.6, fMin: 3.5, fMax: 4.0, state: 'detected', hit: true, name: '3.50 - 4.00 GHz' },
      { tStart: 2.7, tEnd: 3.9, fMin: 5.0, fMax: 5.5, state: 'scanned', hit: false, name: '5.00 - 5.50 GHz' },
      { tStart: 4.0, tEnd: 5.2, fMin: 8.5, fMax: 9.0, state: 'detected', hit: true, name: '8.50 - 9.00 GHz' }
    ]
  };

  function initBandScanData() {
    const cur = PORMAB_BANDS[bandScanState.currentBandIdx];
    const nxt = PORMAB_BANDS[bandScanState.nextBandIdx];
    const curBandEl = document.getElementById('siCurrentBand');
    if (curBandEl) curBandEl.textContent = cur.name;
    const centerFreqEl = document.getElementById('siCenterFreq');
    if (centerFreqEl) centerFreqEl.textContent = cur.center.toFixed(3) + ' GHz';
    const nextBandEl = document.getElementById('siNextBand');
    if (nextBandEl) nextBandEl.textContent = nxt.name;
    const lastScannedEl = document.getElementById('siLastScanned');
    if (lastScannedEl) lastScannedEl.textContent = bandScanState.lastScanned.name;
    const lastResultEl = document.getElementById('siLastResult');
    if (lastResultEl) {
      lastResultEl.textContent = bandScanState.lastScanned.result;
      lastResultEl.className = bandScanState.lastScanned.result === 'HIT' ? 'si-val status-hit' : 'si-val status-miss';
    }
  }

  function updateBandScanEngine(dt) {
    if (!state.isRunning || state.isPaused) return;

    bandScanState.dwellElapsed += dt;

    const currentBand = PORMAB_BANDS[bandScanState.currentBandIdx] || PORMAB_BANDS[0];
    const nextBand = PORMAB_BANDS[bandScanState.nextBandIdx] || PORMAB_BANDS[1];
    const progress = Math.min(1.0, bandScanState.dwellElapsed / bandScanState.dwellDuration);

    // Update Progress Bar
    const progressFill = document.getElementById('siScanProgressFill');
    const progressVal = document.getElementById('siScanProgressVal');
    if (progressFill && progressVal) {
      const pct = Math.floor(progress * 100);
      progressVal.textContent = pct + ' %';
      progressFill.style.width = pct + '%';
    }

    // When current dwell finishes:
    if (bandScanState.dwellElapsed >= bandScanState.dwellDuration) {
      const tStart = Math.max(0, state.simTime - bandScanState.dwellDuration);
      const tEnd = state.simTime;

      // Determine detection outcome based on emitter burst arrival
      const em = EMITTERS_RECEIVER.find(e => e.band === bandScanState.currentBandIdx);
      const isActive = em ? (((pormabEngine.elapsedSlots + em.phase) % em.period) < em.width) : false;
      const isHit = isActive ? (Math.random() < MODEL_PD) : (Math.random() < MODEL_PFA);
      const resultState = isHit ? 'detected' : 'scanned';

      // Update PO-RMAB belief filter with observation
      pormabEngine.stepObservation(bandScanState.currentBandIdx, isHit);

      if (isHit) {
        bandScanState.detectedBandsCount = Math.min(
          bandScanState.totalBandsCount || 36,
          bandScanState.detectedBandsCount + 1
        );
      }

      // Append completed dwell to history
      bandScanState.history.push({
        tStart: tStart,
        tEnd: tEnd,
        fMin: currentBand.fMin,
        fMax: currentBand.fMax,
        state: resultState,
        hit: isHit,
        name: currentBand.name
      });

      // Keep last 40 dwells in history
      if (bandScanState.history.length > 40) {
        bandScanState.history.shift();
      }

      // Update Last Scanned and Last Result
      bandScanState.lastScanned = {
        name: currentBand.name,
        result: isHit ? 'HIT' : 'Miss'
      };

      // Add corresponding Hit / Miss and Switch events to the Hit / Miss Timeline
      if (currentBand.bandIdx !== bandScanState.nextBandIdx) {
        addTimelineEvent('switch', tEnd - 0.04, 0.86 + Math.random() * 0.12);
      }
      addTimelineEvent(isHit ? 'hit' : 'miss', tEnd, isHit ? (0.75 + Math.random() * 0.22) : (0.45 + Math.random() * 0.20));

      // Advance PO-RMAB: current becomes next, and select new next
      bandScanState.currentBandIdx = bandScanState.nextBandIdx;
      bandScanState.nextBandIdx = pormabEngine.selectNextBand(bandScanState.currentBandIdx);
      bandScanState.dwellStartTime = tEnd + 0.10;
      bandScanState.dwellElapsed = 0;

      const newCurrent = PORMAB_BANDS[bandScanState.currentBandIdx];
      const newNext = PORMAB_BANDS[bandScanState.nextBandIdx];

      // Update DOM Information Panel
      const curBandEl = document.getElementById('siCurrentBand');
      if (curBandEl) curBandEl.textContent = newCurrent.name;

      const centerFreqEl = document.getElementById('siCenterFreq');
      if (centerFreqEl) centerFreqEl.textContent = newCurrent.center.toFixed(3) + ' GHz';

      const nextBandEl = document.getElementById('siNextBand');
      if (nextBandEl) nextBandEl.textContent = newNext.name;

      const lastScannedEl = document.getElementById('siLastScanned');
      if (lastScannedEl) lastScannedEl.textContent = bandScanState.lastScanned.name;

      const lastResultEl = document.getElementById('siLastResult');
      if (lastResultEl) {
        lastResultEl.textContent = bandScanState.lastScanned.result;
        lastResultEl.className = bandScanState.lastScanned.result === 'HIT' ? 'si-val status-hit' : 'si-val status-miss';
      }

      const detBandsEl = document.getElementById('siDetectedBands');
      if (detBandsEl) detBandsEl.textContent = `${bandScanState.detectedBandsCount} / ${bandScanState.totalBandsCount}`;

      // Update Observation panel if R1 active
      if (state.activeReceiver === 1) {
        const tunedFreqEl = document.getElementById('valTunedFreq');
        if (tunedFreqEl) tunedFreqEl.textContent = newCurrent.center.toFixed(3) + ' GHz';

        const detectorStatusEl = document.getElementById('valDetectorStatus');
        if (detectorStatusEl) {
          detectorStatusEl.textContent = isHit ? 'HIT' : 'SCAN';
          detectorStatusEl.className = isHit ? 'detector-badge status-hit' : 'detector-badge status-scan';
        }
      }

      // Add entry to Event Log in Section 2
      const eventLogBody = document.getElementById('sec2EventLogBody');
      if (eventLogBody) {
        const newRow = document.createElement('tr');
        const detailText = isHit
          ? `ToA: ${(state.simTime * 1.47).toFixed(2)} ms, CF: ${newCurrent.center.toFixed(2)} GHz`
          : `Window: ${newCurrent.name}`;
        const eventType = isHit ? 'PDW Extracted' : 'PO-RMAB Dwell Complete';
        newRow.innerHTML = `
          <td>${state.simTime.toFixed(2)}</td>
          <td>${eventType}</td>
          <td>${detailText}</td>
        `;
        eventLogBody.insertBefore(newRow, eventLogBody.firstChild);
        while (eventLogBody.children.length > 5) {
          eventLogBody.removeChild(eventLogBody.lastChild);
        }
      }
    }
  }

  function renderBandScan() {
    const { w, h, ctx } = resizeCanvas(canvases.bandScan);
    if (!ctx || w <= 0 || h <= 0) return;

    const padLeft = 24, padRight = 8, padTop = 14, padBottom = 16;
    const plotW = Math.max(10, w - padLeft - padRight);
    const plotH = Math.max(10, h - padTop - padBottom);
    ctx.clearRect(0, 0, w, h);

    // Compute rolling 12s time window based on live simulation time
    const tCurrent = state.simTime;
    const tWindowEnd = Math.max(12.0, tCurrent + 3.0);
    const tWindowStart = tWindowEnd - 12.0;

    function timeToX(t) { return padLeft + ((t - tWindowStart) / 12.0) * plotW; }
    function freqToY(f) { return padTop + (1.0 - (f / 18.0)) * plotH; }

    // 1. Grid Lines (Horizontal & Vertical Dashed)
    ctx.setLineDash([2, 3]);
    ctx.strokeStyle = '#141822';
    ctx.lineWidth = 1;

    // Horizontal grid lines at [3.0, 6.0, 9.0, 12.0, 15.0, 18.0]
    [3.0, 6.0, 9.0, 12.0, 15.0, 18.0].forEach(f => {
      const y = freqToY(f);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    // Vertical grid lines across visible time range
    const firstTick = Math.ceil(tWindowStart / 2) * 2;
    for (let t = firstTick; t <= tWindowEnd; t += 2) {
      const x = timeToX(t);
      if (x >= padLeft && x <= padLeft + plotW) {
        ctx.beginPath();
        ctx.moveTo(x, padTop);
        ctx.lineTo(x, padTop + plotH);
        ctx.stroke();
      }
    }
    ctx.setLineDash([]);

    // 2. Current Time Vertical Dashed Cursor Line
    const curX = timeToX(tCurrent);
    if (curX >= padLeft && curX <= padLeft + plotW) {
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = 'rgba(0, 255, 136, 0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(curX, padTop);
      ctx.lineTo(curX, padTop + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Clip to plot area for drawing blocks
    ctx.save();
    ctx.beginPath();
    ctx.rect(padLeft, padTop - 12, plotW, plotH + 12);
    ctx.clip();

    // 3. Draw Completed Past Dwells from Live History
    bandScanState.history.forEach(b => {
      if (b.tEnd < tWindowStart || b.tStart > tWindowEnd) return;

      const x1 = timeToX(b.tStart);
      const x2 = timeToX(b.tEnd);
      const bw = Math.max(3, x2 - x1);
      const yTop = freqToY(b.fMax);
      const yBottom = freqToY(b.fMin);
      const bh = Math.max(3, yBottom - yTop);
      const r = 2;

      if (b.state === 'scanned') {
        // Cyan solid bar
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x1, yTop, bw, bh, r);
        else ctx.rect(x1, yTop, bw, bh);
        ctx.fill();
      } else if (b.state === 'detected') {
        // Cyan solid bar + glowing yellow detection marker dot
        ctx.fillStyle = '#38bdf8';
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x1, yTop, bw, bh, r);
        else ctx.rect(x1, yTop, bw, bh);
        ctx.fill();

        // Yellow detection marker dot
        ctx.save();
        ctx.shadowColor = 'rgba(250, 204, 21, 0.85)';
        ctx.shadowBlur = 4;
        ctx.fillStyle = '#facc15';
        ctx.beginPath();
        ctx.arc(x1 + bw / 2, yTop + bh / 2, 3.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      } else if (b.state === 'miss') {
        // Muted Red/Rose bar
        ctx.fillStyle = '#d94665';
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x1, yTop, bw, bh, r);
        else ctx.rect(x1, yTop, bw, bh);
        ctx.fill();

        ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
        ctx.beginPath();
        ctx.arc(x1 + bw / 2, yTop + bh / 2, 1.8, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    // 4. Draw Active CURRENT Dwell
    const currentBand = PORMAB_BANDS[bandScanState.currentBandIdx] || PORMAB_BANDS[0];
    if (currentBand) {
      const curTStart = bandScanState.dwellStartTime;
      const curTEnd = curTStart + bandScanState.dwellDuration;
      const x1 = timeToX(curTStart);
      const x2 = timeToX(curTEnd);
      const bw = Math.max(4, x2 - x1);
      const yTop = freqToY(currentBand.fMax);
      const yBottom = freqToY(currentBand.fMin);
      const bh = Math.max(4, yBottom - yTop);
      const r = 2;

      // Bright Neon Green bar with glow
      ctx.save();
      ctx.shadowColor = 'rgba(0, 255, 136, 0.85)';
      ctx.shadowBlur = 8;
      ctx.fillStyle = '#00ff88';
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x1, yTop, bw, bh, r);
      else ctx.rect(x1, yTop, bw, bh);
      ctx.fill();
      ctx.restore();

      // Callout Badge Label Above Bar: "CURRENT: fMin - fMax GHz"
      const labelText = `CURRENT: ${currentBand.name}`;
      ctx.font = '700 7px "JetBrains Mono", monospace';
      const tw = ctx.measureText(labelText).width;
      const bx = Math.max(padLeft, Math.min(padLeft + plotW - tw - 8, x1 + bw / 2 - tw / 2 - 4));
      const by = Math.max(1, yTop - 11);

      ctx.fillStyle = 'rgba(2, 18, 10, 0.92)';
      ctx.strokeStyle = '#00ff88';
      ctx.lineWidth = 1;
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(bx, by, tw + 8, 9.5, 2);
      else ctx.rect(bx, by, tw + 8, 9.5);
      ctx.fill();
      ctx.stroke();

      ctx.fillStyle = '#00ff88';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(labelText, bx + 4, by + 4.8);
    }

    // 5. Draw NEXT Band (Queued)
    const nextBand = PORMAB_BANDS[bandScanState.nextBandIdx] || PORMAB_BANDS[1];
    if (nextBand) {
      const nextTStart = bandScanState.dwellStartTime + bandScanState.dwellDuration + 0.15;
      const nextTEnd = nextTStart + bandScanState.dwellDuration;
      const x1 = timeToX(nextTStart);
      const x2 = timeToX(nextTEnd);
      const bw = Math.max(4, x2 - x1);
      const yTop = freqToY(nextBand.fMax);
      const yBottom = freqToY(nextBand.fMin);
      const bh = Math.max(4, yBottom - yTop);
      const r = 2;

      ctx.save();
      ctx.setLineDash([3, 2]);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.2;
      ctx.fillStyle = 'rgba(56, 189, 248, 0.12)';
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x1, yTop, bw, bh, r);
      else ctx.rect(x1, yTop, bw, bh);
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    ctx.restore(); // restore clip

    // 6. Grid Bounding Frame & Ticks
    ctx.strokeStyle = '#1e222b';
    ctx.lineWidth = 1;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    // Y-Axis Ticks & Labels: 0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0
    ctx.font = '400 7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#a1a1aa';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0].forEach(f => {
      const y = freqToY(f);
      ctx.strokeStyle = '#272b36';
      ctx.beginPath();
      ctx.moveTo(padLeft - 2, y);
      ctx.lineTo(padLeft, y);
      ctx.stroke();
      ctx.fillText(f.toFixed(1), padLeft - 3, y);
    });

    // X-Axis Ticks & Labels: dynamic rolling time
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let t = firstTick; t <= tWindowEnd; t += 2) {
      const x = timeToX(t);
      if (x >= padLeft && x <= padLeft + plotW) {
        ctx.strokeStyle = '#272b36';
        ctx.beginPath();
        ctx.moveTo(x, padTop + plotH);
        ctx.lineTo(x, padTop + plotH + 2);
        ctx.stroke();
        ctx.fillText(t.toFixed(0), x, padTop + plotH + 3);
      }
    }
  }

  // Section 1 Timeline - Dynamic Hit / Miss / Switch Intercept Engine
  const timelineEvents = [];
  let lastTimelineEventTime = 0;

  function initTimeline() {
    timelineEvents.length = 0;
    const windowSpan = 300;
    const count = 140;
    const timeStep = windowSpan / count;
    const tCurrent = state.simTime;
    const tStart = Math.max(0, tCurrent - 280);

    for (let i = 0; i < count; i++) {
      const t = tStart + i * timeStep;
      if (t > tCurrent) continue;
      const rand = Math.random();
      const type = rand < 0.56 ? 'hit' : (rand < 0.76 ? 'switch' : 'miss');
      const h = type === 'switch' ? (0.80 + Math.random() * 0.18) : (type === 'hit' ? (0.60 + Math.random() * 0.38) : (0.42 + Math.random() * 0.25));
      timelineEvents.push({ time: t, type: type, heightFactor: h });
    }
    lastTimelineEventTime = tCurrent;
  }

  function addTimelineEvent(type, t, heightFactor) {
    const time = typeof t === 'number' ? t : state.simTime;
    const h = typeof heightFactor === 'number' ? heightFactor : (type === 'switch' ? (0.82 + Math.random() * 0.16) : (type === 'hit' ? (0.65 + Math.random() * 0.32) : (0.42 + Math.random() * 0.25)));
    timelineEvents.push({ time: time, type: type, heightFactor: h });
    if (timelineEvents.length > 350) {
      timelineEvents.shift();
    }
  }
  window.addReceiverTimelineEvent = addTimelineEvent;

  function updateTimelineEngine(dt) {
    if (!state.isRunning || state.isPaused) return;

    // Append continuous live events as simulation time progresses (every ~0.5s)
    if (state.simTime - lastTimelineEventTime >= 0.55) {
      lastTimelineEventTime = state.simTime;
      const rx = state.receivers[state.activeReceiver] || state.receivers[1];
      const isDet = rx.status === 'HIT' || Math.random() < 0.60;
      const isSwitch = Math.random() < 0.14;
      const type = isSwitch ? 'switch' : (isDet ? 'hit' : 'miss');
      addTimelineEvent(type, state.simTime);
    }
  }

  function renderTimeline() {
    const { w, h, ctx } = resizeCanvas(canvases.timeline);
    if (!ctx || w <= 0 || h <= 0) return;
    const padLeft = 14, padRight = 14, padTop = 4, padBottom = 16;
    const plotW = w - padLeft - padRight, plotH = h - padTop - padBottom;
    ctx.clearRect(0, 0, w, h);

    const windowSpan = 300;
    const tCurrent = state.simTime;
    const tWindowEnd = Math.max(windowSpan, tCurrent + 15);
    const tWindowStart = tWindowEnd - windowSpan;

    function timeToX(t) {
      return padLeft + ((t - tWindowStart) / windowSpan) * plotW;
    }

    // Center Baseline
    const centerAxis = padTop + plotH / 2;
    ctx.strokeStyle = '#1E2D4A';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padLeft, centerAxis);
    ctx.lineTo(padLeft + plotW, centerAxis);
    ctx.stroke();

    // Render Event Markers
    const barW = Math.max(2, Math.min(4, (plotW / 140)));
    timelineEvents.forEach(ev => {
      const x = timeToX(ev.time);
      if (x >= padLeft - barW && x <= padLeft + plotW + barW) {
        if (ev.type === 'hit') {
          ctx.fillStyle = '#10b981';
          ctx.fillRect(x - barW / 2, centerAxis - plotH * 0.4, barW, plotH * 0.4 - 2);
        } else if (ev.type === 'miss') {
          ctx.fillStyle = '#ef4444';
          ctx.fillRect(x - barW / 2, centerAxis + 2, barW, plotH * 0.4 - 2);
        } else if (ev.type === 'switch') {
          ctx.fillStyle = '#f59e0b';
          ctx.beginPath();
          ctx.arc(x, centerAxis, barW * 1.2, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    });

    // Live Playhead Indicator
    const curX = timeToX(tCurrent);
    if (curX >= padLeft && curX <= padLeft + plotW) {
      ctx.save();
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1.4;
      ctx.shadowColor = 'rgba(56, 189, 248, 0.8)';
      ctx.shadowBlur = 6;
      ctx.beginPath();
      ctx.moveTo(curX, padTop - 1);
      ctx.lineTo(curX, padTop + plotH + 1);
      ctx.stroke();

      ctx.fillStyle = '#38bdf8';
      ctx.beginPath();
      ctx.arc(curX, padTop + plotH / 2, 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // X-Axis Ticks & Labels: 0, 50, 100, 150, 200, 250, 300 (rolling)
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.fillStyle = '#71717a';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    const firstTick = Math.ceil(tWindowStart / 50) * 50;
    for (let t = firstTick; t <= tWindowEnd; t += 50) {
      const x = timeToX(t);
      if (x >= padLeft && x <= padLeft + plotW) {
        ctx.fillText(t.toString(), x, padTop + plotH + 2);
      }
    }
  }

  // Section 1 Rx Window - Exact Match to Reference Image
  function renderRxWindow() {
    const { w, h, ctx } = resizeCanvas(canvases.rxWindow);
    if (!ctx || w <= 0 || h <= 0) return;

    // Layout Margins: Leave space for Y-axis (left) and X-axis (bottom)
    const padLeft = 36;
    const padRight = 14;
    const padTop = 8;
    const padBottom = 22;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;

    ctx.clearRect(0, 0, w, h);

    function freqToX(f) { return padLeft + ((f - 7.50) / 2.00) * plotW; }
    function dbmToY(p) { return padTop + plotH * (1 - Math.max(0, Math.min(1, (p - (-120)) / 80))); }

    // 1. Grid Lines & Ticks
    // Horizontal Grid Lines & Y-Axis Labels: -40, -60, -80, -100, -120 dBm
    ctx.strokeStyle = '#181f2c';
    ctx.lineWidth = 1;
    ctx.font = '8.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#8e9bb0';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const yTicks = [-40, -60, -80, -100, -120];
    yTicks.forEach(val => {
      const y = dbmToY(val);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();

      ctx.fillText(val.toString(), padLeft - 4, y);
    });

    // Vertical Grid Lines & X-Axis Labels: 7.50 to 9.50 GHz (step 0.25)
    const fTicks = [7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00, 9.25, 9.50];
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    fTicks.forEach(f => {
      const x = freqToX(f);
      ctx.strokeStyle = '#181f2c';
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();

      ctx.fillText(f.toFixed(2), x, padTop + plotH + 4);

      // Tick mark extending down
      ctx.strokeStyle = '#384252';
      ctx.beginPath();
      ctx.moveTo(x, padTop + plotH);
      ctx.lineTo(x, padTop + plotH + 3);
      ctx.stroke();
    });

    // 2. Shaded Receiver Window Box (8.05 GHz to 8.78 GHz)
    const winX1 = freqToX(8.05);
    const winX2 = freqToX(8.78);
    const winYTop = dbmToY(-40);
    const winYBottom = dbmToY(-120);

    // Translucent Blue-Grey Shading
    ctx.fillStyle = 'rgba(45, 90, 145, 0.28)';
    ctx.fillRect(winX1, winYTop, winX2 - winX1, winYBottom - winYTop);

    // Dashed Window Boundary (Top, Left, Right)
    ctx.setLineDash([4, 3]);
    ctx.strokeStyle = 'rgba(215, 228, 245, 0.75)';
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    // Top border
    ctx.moveTo(winX1, winYTop); ctx.lineTo(winX2, winYTop);
    // Left border
    ctx.moveTo(winX1, winYTop); ctx.lineTo(winX1, winYBottom);
    // Right border
    ctx.moveTo(winX2, winYTop); ctx.lineTo(winX2, winYBottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // 3. Cyan RF Power Signal Trace
    const numPoints = Math.min(plotW * 2, 500);
    const tracePoints = [];

    for (let i = 0; i <= numPoints; i++) {
      const f = 7.50 + (i / numPoints) * 2.00;
      
      // Base noise floor centered around -100.5 dBm with realistic grass jitter
      let power = -100.5 + gaussianRandom(0, 1.8);

      // Comb / harmonic micro-spurs (~ -90 to -95 dBm)
      const spurs = [7.55, 7.68, 7.82, 8.16, 8.28, 8.35, 8.62, 8.85, 9.15, 9.35];
      spurs.forEach(sf => {
        const d = Math.abs(f - sf);
        if (d < 0.015) {
          const spurPwr = -92 - Math.random() * 3;
          power = Math.max(power, spurPwr * Math.exp(-Math.pow(d / 0.004, 2)));
        }
      });

      // 3 Main Sharp Needle Peaks:
      // Peak 1: 7.925 GHz (-56 dBm)
      const d1 = Math.abs(f - 7.925);
      if (d1 < 0.035) {
        const p1 = -100.5 + (-56 - (-100.5)) * Math.exp(-Math.pow(d1 / 0.005, 2));
        power = Math.max(power, p1);
      }

      // Peak 2 (Main Signal inside window): 8.412 GHz (-46 dBm)
      const d2 = Math.abs(f - 8.412);
      if (d2 < 0.04) {
        const p2 = -100.5 + (-46 - (-100.5)) * Math.exp(-Math.pow(d2 / 0.0055, 2));
        power = Math.max(power, p2);
      }

      // Peak 3: 9.025 GHz (-58 dBm)
      const d3 = Math.abs(f - 9.025);
      if (d3 < 0.035) {
        const p3 = -100.5 + (-58 - (-100.5)) * Math.exp(-Math.pow(d3 / 0.005, 2));
        power = Math.max(power, p3);
      }

      tracePoints.push({ x: freqToX(f), y: dbmToY(power) });
    }

    // Glow effect on electric cyan signal trace
    ctx.save();
    ctx.shadowBlur = 3.5;
    ctx.shadowColor = 'rgba(0, 210, 255, 0.7)';
    ctx.strokeStyle = '#00d2ff';
    ctx.lineWidth = 1.35;
    ctx.beginPath();
    tracePoints.forEach((pt, idx) => {
      if (idx === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    });
    ctx.stroke();
    ctx.restore();

    // 4. Outer Graph Bounding Box
    ctx.strokeStyle = '#2b3442';
    ctx.lineWidth = 1;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);
  }

  // ==========================================================================
  // 7. CONSTELLATION ENGINE & MULTI-MODULATION RENDERER
  // ==========================================================================
  const MODULATION_CENTERS = {
    'QPSK': [
      { i: -1.0, q: 1.0 },
      { i: 1.0, q: 1.0 },
      { i: -1.0, q: -1.0 },
      { i: 1.0, q: -1.0 }
    ],
    'BPSK': [
      { i: -1.0, q: 0.0 },
      { i: 1.0, q: 0.0 }
    ],
    '16-QAM': [
      { i: -1.2, q: 1.2 }, { i: -0.4, q: 1.2 }, { i: 0.4, q: 1.2 }, { i: 1.2, q: 1.2 },
      { i: -1.2, q: 0.4 }, { i: -0.4, q: 0.4 }, { i: 0.4, q: 0.4 }, { i: 1.2, q: 0.4 },
      { i: -1.2, q: -0.4 }, { i: -0.4, q: -0.4 }, { i: 0.4, q: -0.4 }, { i: 1.2, q: -0.4 },
      { i: -1.2, q: -1.2 }, { i: -0.4, q: -1.2 }, { i: 0.4, q: -1.2 }, { i: 1.2, q: -1.2 }
    ]
  };

  const constellationState = {
    1: { modulation: 'QPSK', points: [] },
    2: { modulation: 'QPSK', points: [] }
  };

  function initConstellation() {
    [1, 2].forEach(panelId => {
      const pState = constellationState[panelId];
      pState.points = [];
      const centers = MODULATION_CENTERS[pState.modulation] || MODULATION_CENTERS['QPSK'];
      const sigma = pState.modulation === '16-QAM' ? 0.06 : (pState.modulation === 'BPSK' ? 0.16 : 0.15);
      const count = 360;

      for (let i = 0; i < count; i++) {
        const c = centers[i % centers.length];
        const gI = gaussianRandom(0, sigma);
        const gQ = gaussianRandom(0, sigma);
        const dist = Math.hypot(gI, gQ);
        const alpha = Math.max(0.3, Math.min(0.98, 1.0 - dist * 2.0));

        pState.points.push({
          curI: c.i + gI,
          curQ: c.q + gQ,
          baseI: c.i,
          baseQ: c.q,
          sigma: sigma,
          alpha: alpha,
          size: Math.random() < 0.25 ? 2.2 : 1.6
        });
      }
    });
  }

  function drawConstellationPlot(canvasEl, panelId = 1) {
    if (!canvasEl) return;
    const { w, h, ctx } = resizeCanvas(canvasEl);
    if (!ctx || w <= 0 || h <= 0) return;

    ctx.clearRect(0, 0, w, h);

    // Deep black-blue background
    ctx.fillStyle = '#050811';
    ctx.fillRect(0, 0, w, h);

    // Margins for outer layout
    const padLeft = 20, padRight = 8, padTop = 8, padBottom = 16;
    const plotW = w - padLeft - padRight;
    const plotH = h - padTop - padBottom;
    const plotSize = Math.max(10, Math.min(plotW, plotH));
    const plotX = padLeft + Math.floor((plotW - plotSize) / 2);
    const plotY = padTop + Math.floor((plotH - plotSize) / 2);

    function toScreenX(iVal) {
      return plotX + ((iVal - (-2.0)) / 4.0) * plotSize;
    }
    function toScreenY(qVal) {
      return plotY + ((2.0 - qVal) / 4.0) * plotSize;
    }

    // Graph Area Fill
    ctx.fillStyle = '#04060a';
    ctx.fillRect(plotX, plotY, plotSize, plotSize);

    // Grid Lines & Ticks at [-2, -1, 0, 1, 2]
    const ticks = [-2, -1, 0, 1, 2];

    // Horizontal Grid Lines & Y Ticks
    ticks.forEach(t => {
      const y = toScreenY(t);
      ctx.beginPath();
      ctx.strokeStyle = t === 0 ? '#15416f' : '#0c2440';
      ctx.lineWidth = t === 0 ? 1.2 : 0.8;
      ctx.moveTo(plotX, y);
      ctx.lineTo(plotX + plotSize, y);
      ctx.stroke();

      // Y-axis tick label
      ctx.font = '8px "JetBrains Mono", monospace';
      ctx.fillStyle = '#5c7490';
      ctx.textAlign = 'right';
      ctx.textBaseline = 'middle';
      ctx.fillText(t.toString(), plotX - 3, y);
    });

    // Vertical Grid Lines & X Ticks
    ticks.forEach(t => {
      const x = toScreenX(t);
      ctx.beginPath();
      ctx.strokeStyle = t === 0 ? '#15416f' : '#0c2440';
      ctx.lineWidth = t === 0 ? 1.2 : 0.8;
      ctx.moveTo(x, plotY);
      ctx.lineTo(x, plotY + plotSize);
      ctx.stroke();

      // X-axis tick label
      ctx.font = '8px "JetBrains Mono", monospace';
      ctx.fillStyle = '#5c7490';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText(t.toString(), x, plotY + plotSize + 2);
    });

    // Outer Bounding Box
    ctx.strokeStyle = '#102e52';
    ctx.lineWidth = 1;
    ctx.strokeRect(plotX, plotY, plotSize, plotSize);

    // Scatter Points
    const pState = constellationState[panelId] || constellationState[1];
    const points = pState ? pState.points : [];

    ctx.save();
    // Clip within plot area
    ctx.beginPath();
    ctx.rect(plotX - 1, plotY - 1, plotSize + 2, plotSize + 2);
    ctx.clip();

    ctx.shadowBlur = 3.5;
    ctx.shadowColor = '#00e5ff';

    const isSimRunning = state.isRunning && !state.isPaused;

    for (let i = 0; i < points.length; i++) {
      const pt = points[i];
      let jitterX = 0, jitterY = 0;
      if (isSimRunning && Math.random() < 0.35) {
        jitterX = (Math.random() - 0.5) * 0.024;
        jitterY = (Math.random() - 0.5) * 0.024;
      }

      const screenX = toScreenX(pt.curI + jitterX);
      const screenY = toScreenY(pt.curQ + jitterY);

      // Cyan scatter dot
      ctx.fillStyle = `rgba(0, 229, 255, ${pt.alpha})`;
      ctx.fillRect(screenX - (pt.size / 2), screenY - (pt.size / 2), pt.size, pt.size);

      // Bright white/cyan core
      if (pt.alpha > 0.6) {
        ctx.fillStyle = `rgba(224, 248, 255, ${pt.alpha * 0.9})`;
        ctx.fillRect(screenX - 0.5, screenY - 0.5, 1, 1);
      }
    }
    ctx.restore();
  }

  // ==========================================================================
  // 8. RENDERERS FOR SECTION 2 (SCROLL-DOWN VIEW)
  // ==========================================================================

  // 8a. Section 2 Spectrogram (Time-Frequency View) - High Definition Radar Engine
  let sec2SpecBuffer = null;
  const specAnim = { phase: 0, pulseMod: 1.0, burstSeed: 0 };

  // Setup anime.js animation on the spectrogram waveform phase and pulse modulation
  if (typeof anime !== 'undefined') {
    anime({
      targets: specAnim,
      phase: Math.PI * 2,
      duration: 18000,
      easing: 'linear',
      loop: true
    });
    anime({
      targets: specAnim,
      pulseMod: [0.85, 1.15],
      duration: 1400,
      direction: 'alternate',
      easing: 'easeInOutSine',
      loop: true
    });
  }

  // Pre-generate deterministic scatter particles and chirp paths using Random.js
  const specParticles = [];
  const specBroadbandPulses = [
    0.025, 0.075, 0.14, 0.22, 0.29, 0.36, 0.44, 0.49, 0.58, 0.65, 0.72, 0.79, 0.85, 0.92, 0.97
  ];

  (function initSpectrogramParticles() {
    const pRng = (typeof rng !== 'undefined' && rng.real) ? rng : {
      real: (min, max) => min + Math.random() * (max - min)
    };

    // Background noise speckles (faint blue/cyan scatter dots across entire 0-20 GHz plane)
    for (let i = 0; i < 350; i++) {
      specParticles.push({
        type: 'bg_speckle',
        t: pRng.real(0.01, 0.99),
        f: pRng.real(0.5, 19.5),
        size: pRng.real(0.7, 1.4),
        alpha: pRng.real(0.2, 0.65),
        color: pRng.real(0, 1) > 0.4 ? 'rgba(0, 160, 255,' : 'rgba(0, 80, 200,'
      });
    }

    // Diagonal chirp trails (slanted frequency sweep lines in upper band)
    for (let chirp = 0; chirp < 8; chirp++) {
      const tStart = pRng.real(0.05, 0.85);
      const fStart = pRng.real(12.0, 17.5);
      const slope = pRng.real(-8.0, 8.0);
      const len = pRng.real(0.04, 0.10);
      const count = Math.floor(len * 200);

      for (let k = 0; k < count; k++) {
        const frac = k / count;
        specParticles.push({
          type: 'chirp_dot',
          t: tStart + frac * len,
          f: Math.max(1.0, Math.min(19.0, fStart + frac * len * slope + pRng.real(-0.15, 0.15))),
          size: pRng.real(1.0, 1.8),
          alpha: pRng.real(0.5, 0.9),
          color: 'rgba(0, 240, 255,'
        });
      }
    }
  })();

  function renderSec2Spectrogram() {
    const { w, h, ctx } = resizeCanvas(canvases.sec2Spectrogram);
    if (!ctx || w <= 0 || h <= 0) return;

    const padLeft = 34, padRight = 36, padTop = 8, padBottom = 22;
    const plotW = Math.max(10, w - padLeft - padRight);
    const plotH = Math.max(10, h - padTop - padBottom);
    ctx.clearRect(0, 0, w, h);

    const bufW = 480;
    const bufH = 200;

    if (!sec2SpecBuffer) {
      sec2SpecBuffer = document.createElement('canvas');
      sec2SpecBuffer.width = bufW;
      sec2SpecBuffer.height = bufH;
    }

    const bCtx = sec2SpecBuffer.getContext('2d');
    const imgData = bCtx.createImageData(bufW, bufH);
    const data = imgData.data;

    const phase = specAnim.phase;
    const pulseMod = specAnim.pulseMod;

    // Colormap lookup function (Power dB [-100 to -20] -> RGBA)
    function powerToRgb(p) {
      const v = Math.max(0, Math.min(1, (p - (-100)) / 80));
      let r = 0, g = 0, b = 0;

      if (v < 0.14) {
        // Deep Navy / Midnight Blue
        const f = v / 0.14;
        r = Math.round(1 + f * 4);
        g = Math.round(5 + f * 20);
        b = Math.round(28 + f * 90);
      } else if (v < 0.38) {
        // Cobalt / Electric Blue -> Deep Cyan
        const f = (v - 0.14) / 0.24;
        r = Math.round(5 + f * 0);
        g = Math.round(25 + f * 175);
        b = Math.round(118 + f * 137);
      } else if (v < 0.64) {
        // Cyan -> Vibrant Spring Green
        const f = (v - 0.38) / 0.26;
        r = Math.round(5 + f * 55);
        g = Math.round(200 + f * 55);
        b = Math.round(255 - f * 225);
      } else {
        // Bright Green -> Intense Solar Yellow -> Hot Core
        const f = (v - 0.64) / 0.36;
        r = Math.round(60 + f * 195);
        g = 255;
        b = f > 0.82 ? Math.round((f - 0.82) * 350) : Math.round(30 * (1 - f));
      }
      return [r, g, b];
    }

    // Step 1: Compute Continuous Heatmap Field
    for (let c = 0; c < bufW; c++) {
      const tNorm = c / bufW;

      // Vertical pulse streaks at specific pulse timestamps
      let pulseEnergy = 0;
      for (let i = 0; i < specBroadbandPulses.length; i++) {
        const d = Math.abs(tNorm - specBroadbandPulses[i]);
        if (d < 0.007) {
          pulseEnergy += Math.exp(-(d * d) / (2 * 0.0025 * 0.0025)) * 28;
        }
      }

      // Vertical micro-striae / FFT bin noise
      const striae = (Math.sin(c * 2.8 + 1.2) * Math.cos(c * 6.3) + Math.sin(c * 17.1)) * 3.8;

      // Signal center frequencies across time
      const f1 = 5.35 + 0.35 * Math.sin(tNorm * 14 + phase) + 0.18 * Math.cos(tNorm * 33) + 0.1 * Math.sin(tNorm * 62);
      const f2 = 8.85 + 0.7 * Math.sin(tNorm * 17 + 1.2 + phase) + 0.3 * Math.cos(tNorm * 36) + 0.15 * Math.sin(tNorm * 70);
      const f3a = 13.4 + 0.9 * Math.sin(tNorm * 11 + 0.5 + phase) + 0.45 * Math.sin(tNorm * 29);
      const f3b = 14.8 + 0.75 * Math.cos(tNorm * 15 + 2.1 + phase) + 0.35 * Math.sin(tNorm * 43);
      const f4 = 2.2 + 0.5 * Math.sin(tNorm * 15 + 3.0 + phase);
      const f5 = 17.6 + 0.6 * Math.sin(tNorm * 19 + 1.8 + phase);

      // Amplitude bursts
      const burst1 = Math.sin(tNorm * 24 + phase * 2) * 6 * pulseMod;
      const burst3 = Math.cos(tNorm * 20 + phase * 1.5) * 6 * pulseMod;

      for (let r = 0; r < bufH; r++) {
        const freq = 20 * (1 - r / bufH);

        // Base noise floor
        let p = -95 + striae + pulseEnergy * 0.45 + (Math.sin(r * 4.8 + c * 3.1) * 2.8);

        // Band 1: ~5.5 GHz (Yellow Intense Core)
        const d1 = freq - f1;
        p += Math.exp(-(d1 * d1) / (2 * 0.38 * 0.38)) * (78 + burst1);

        // Band 2: ~8.85 GHz (Cyan-Green Undulating Ribbon)
        const d2 = freq - f2;
        p += Math.exp(-(d2 * d2) / (2 * 0.52 * 0.52)) * 54;

        // Band 3: ~13.4 - 15.5 GHz (Multi-Ridge High Activity Radar)
        const d3a = freq - f3a;
        const d3b = freq - f3b;
        p += Math.exp(-(d3a * d3a) / (2 * 0.62 * 0.62)) * (70 + burst3);
        p += Math.exp(-(d3b * d3b) / (2 * 0.55 * 0.55)) * (62 + burst3);

        // Band 4: ~2.2 GHz (Subtle Low Whisper Ripple)
        const d4 = freq - f4;
        p += Math.exp(-(d4 * d4) / (2 * 0.46 * 0.46)) * 34;

        // Band 5: ~17.6 GHz (High Frequency Whisper Track)
        const d5 = freq - f5;
        p += Math.exp(-(d5 * d5) / (2 * 0.50 * 0.50)) * 32;

        // Pulse transient flare
        if (pulseEnergy > 4) {
          p += pulseEnergy * 0.55 * (0.85 + 0.35 * Math.sin(freq * 1.8));
        }

        // Sub-pixel organic noise
        p += (Math.random() - 0.5) * 4.2;

        const [red, green, blue] = powerToRgb(p);
        const idx = (r * bufW + c) * 4;
        data[idx] = red;
        data[idx + 1] = green;
        data[idx + 2] = blue;
        data[idx + 3] = 255;
      }
    }

    bCtx.putImageData(imgData, 0, 0);

    // Step 2: Draw Raster Heatmap to Display Canvas
    ctx.drawImage(sec2SpecBuffer, 0, 0, bufW, bufH, padLeft, padTop, plotW, plotH);

    // Step 3: Overlay Crisp Beaded Particle Chains along Signal Peaks
    ctx.save();
    ctx.beginPath();
    ctx.rect(padLeft, padTop, plotW, plotH);
    ctx.clip();

    // Helper coordinate converters
    function tToX(t) { return padLeft + t * plotW; }
    function fToY(f) { return padTop + (1 - f / 20) * plotH; }

    // Overlay deterministic background speckles and chirp trails
    specParticles.forEach(pt => {
      const x = tToX(pt.t);
      const y = fToY(pt.f);
      ctx.fillStyle = `${pt.color}${pt.alpha})`;
      ctx.fillRect(x - pt.size / 2, y - pt.size / 2, pt.size, pt.size);
    });

    // Overlay Band 1 Bead Chain (Bright Yellow & White Sparkles at ~5.5 GHz)
    ctx.shadowBlur = 4;
    ctx.shadowColor = '#ffff00';
    const numBeads1 = 120;
    for (let i = 0; i < numBeads1; i++) {
      const tNorm = i / numBeads1;
      const f = 5.35 + 0.35 * Math.sin(tNorm * 14 + phase) + 0.18 * Math.cos(tNorm * 33) + 0.1 * Math.sin(tNorm * 62);
      const x = tToX(tNorm);
      const y = fToY(f + (Math.sin(i * 3.7) * 0.12));

      // Intense yellow bead with white core
      ctx.fillStyle = (i % 3 === 0) ? '#ffffff' : '#ffff00';
      const bSize = (i % 4 === 0) ? 2.2 : 1.6;
      ctx.fillRect(x - bSize / 2, y - bSize / 2, bSize, bSize);

      // Cyan sparks above/below core
      if (i % 2 === 0) {
        ctx.fillStyle = 'rgba(0, 240, 255, 0.85)';
        ctx.fillRect(x - 0.7, y + (i % 4 === 0 ? 3 : -3), 1.4, 1.4);
      }
    }

    // Overlay Band 2 Bead Chain (Cyan-Green at ~8.85 GHz)
    ctx.shadowColor = '#00f0ff';
    ctx.shadowBlur = 3;
    const numBeads2 = 100;
    for (let i = 0; i < numBeads2; i++) {
      const tNorm = i / numBeads2;
      const f = 8.85 + 0.7 * Math.sin(tNorm * 17 + 1.2 + phase) + 0.3 * Math.cos(tNorm * 36) + 0.15 * Math.sin(tNorm * 70);
      const x = tToX(tNorm);
      const y = fToY(f + (Math.cos(i * 2.9) * 0.18));

      ctx.fillStyle = (i % 5 === 0) ? '#facc15' : (i % 2 === 0 ? '#00f0ff' : '#22c55e');
      const bSize = (i % 5 === 0) ? 2.0 : 1.4;
      ctx.fillRect(x - bSize / 2, y - bSize / 2, bSize, bSize);
    }

    // Overlay Band 3 Bead Chains (Multi-Ridge Yellow/Cyan at ~13.4 - 15.5 GHz)
    const numBeads3 = 140;
    for (let i = 0; i < numBeads3; i++) {
      const tNorm = i / numBeads3;
      const fA = 13.4 + 0.9 * Math.sin(tNorm * 11 + 0.5 + phase) + 0.45 * Math.sin(tNorm * 29);
      const fB = 14.8 + 0.75 * Math.cos(tNorm * 15 + 2.1 + phase) + 0.35 * Math.sin(tNorm * 43);

      const x = tToX(tNorm);
      const yA = fToY(fA + (Math.sin(i * 4.3) * 0.15));
      const yB = fToY(fB + (Math.cos(i * 3.1) * 0.15));

      // Ridge A beads
      ctx.shadowColor = '#ffff00';
      ctx.fillStyle = (i % 4 === 0) ? '#ffffff' : (i % 2 === 0 ? '#ffff00' : '#00f0ff');
      const sA = (i % 3 === 0) ? 2.0 : 1.5;
      ctx.fillRect(x - sA / 2, yA - sA / 2, sA, sA);

      // Ridge B beads
      ctx.shadowColor = '#00f0ff';
      ctx.fillStyle = (i % 3 === 0) ? '#22c55e' : '#00f0ff';
      const sB = (i % 4 === 0) ? 1.8 : 1.3;
      ctx.fillRect(x - sB / 2, yB - sB / 2, sB, sB);
    }

    // Overlay Vertical Pulse Transient Rays
    ctx.shadowBlur = 0;
    specBroadbandPulses.forEach(tPulse => {
      const x = tToX(tPulse);
      ctx.strokeStyle = 'rgba(0, 240, 255, 0.32)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();

      // Pulse spark nodes
      for (let k = 0; k < 6; k++) {
        const f = 2.0 + k * 3.2 + Math.sin(k * 2.3) * 0.8;
        const y = fToY(f);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
        ctx.fillRect(x - 1, y - 1, 2, 2);
      }
    });

    ctx.restore();

    // Step 4: Grid Frame & Bounding Box
    ctx.strokeStyle = '#1e222b';
    ctx.lineWidth = 1;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    // Y-Axis Title ("Frequency (GHz)")
    ctx.save();
    ctx.translate(10, padTop + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.font = '500 8px "Inter", -apple-system, sans-serif';
    ctx.fillStyle = '#71717a';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Frequency (GHz)', 0, 0);
    ctx.restore();

    // Y-Axis Ticks & Labels (0, 5, 10, 15, 20 GHz)
    ctx.font = '400 7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#a1a1aa';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const yTicks = [
      { val: 20, f: 0 },
      { val: 15, f: 0.25 },
      { val: 10, f: 0.50 },
      { val: 5, f: 0.75 },
      { val: 0, f: 1.0 }
    ];

    yTicks.forEach(tick => {
      const y = padTop + tick.f * plotH;
      // Tick notch
      ctx.strokeStyle = '#272b36';
      ctx.beginPath();
      ctx.moveTo(padLeft - 2, y);
      ctx.lineTo(padLeft, y);
      ctx.stroke();
      ctx.fillText(tick.val.toString(), padLeft - 4, y);
    });

    // X-Axis Ticks & Labels (0, 0.5M, 1.0M, 1.5M, 2.0M)
    ctx.font = '400 7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#a1a1aa';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';

    const xTicks = [
      { lbl: '0', f: 0 },
      { lbl: '0.5M', f: 0.25 },
      { lbl: '1.0M', f: 0.50 },
      { lbl: '1.5M', f: 0.75 },
      { lbl: '2.0M', f: 1.0 }
    ];

    xTicks.forEach(tick => {
      const x = padLeft + tick.f * plotW;
      // Tick notch
      ctx.strokeStyle = '#272b36';
      ctx.beginPath();
      ctx.moveTo(x, padTop + plotH);
      ctx.lineTo(x, padTop + plotH + 2);
      ctx.stroke();
      ctx.fillText(tick.lbl, x, padTop + plotH + 3);
    });

    // X-Axis Title ("Time of Arrival (µs)")
    ctx.font = '500 8px "Inter", -apple-system, sans-serif';
    ctx.fillStyle = '#71717a';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('Time of Arrival (µs)', padLeft + plotW / 2, h - 1);

    // Step 5: Colorbar (Right Side)
    const cbX = padLeft + plotW + 6;
    const cbY = padTop;
    const cbW = 7;
    const cbH = plotH;

    const cbGrad = ctx.createLinearGradient(cbX, cbY, cbX, cbY + cbH);
    cbGrad.addColorStop(0.00, '#facc15'); // Yellow (high power)
    cbGrad.addColorStop(0.22, '#10b981'); // Green
    cbGrad.addColorStop(0.48, '#38bdf8'); // Cyan
    cbGrad.addColorStop(0.73, '#0284c7'); // Blue
    cbGrad.addColorStop(1.00, '#030712'); // Dark Navy (low power)

    ctx.fillStyle = cbGrad;
    ctx.fillRect(cbX, cbY, cbW, cbH);
    ctx.strokeStyle = '#1e222b';
    ctx.lineWidth = 1;
    ctx.strokeRect(cbX, cbY, cbW, cbH);

    // Colorbar Ticks & Labels (-200, -40, -80, -100)
    ctx.font = '400 7px "JetBrains Mono", monospace';
    ctx.fillStyle = '#a1a1aa';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    const cbTicks = [
      { lbl: '-200', f: 0.00 },
      { lbl: '-40', f: 0.33 },
      { lbl: '-80', f: 0.66 },
      { lbl: '-100', f: 1.00 }
    ];

    cbTicks.forEach(tick => {
      const y = cbY + tick.f * cbH;
      // Tick notch
      ctx.strokeStyle = '#272b36';
      ctx.beginPath();
      ctx.moveTo(cbX + cbW, y);
      ctx.lineTo(cbX + cbW + 2, y);
      ctx.stroke();
      ctx.fillText(tick.lbl, cbX + cbW + 3, y);
    });
  }

  // 8b. Section 2 Spectrum (8.0 to 9.2 GHz)
  function renderSec2Spectrum() {
    const { w, h, ctx } = resizeCanvas(canvases.sec2Spectrum);
    if (!ctx || w <= 0 || h <= 0) return;

    const padLeft = 34, padRight = 10, padTop = 8, padBottom = 16;
    const plotW = w - padLeft - padRight, plotH = h - padTop - padBottom;
    ctx.clearRect(0, 0, w, h);

    function dbmToY(p) { return padTop + plotH * (1 - Math.max(0, Math.min(1, (p - (-120)) / 100))); }
    function freqToX(f) { return padLeft + ((f - 8.0) / 1.2) * plotW; }

    // Grid Horizontal Lines & Y-Axis Labels
    ctx.strokeStyle = '#181e29'; ctx.lineWidth = 1; ctx.font = '7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    [-20, -40, -60, -80, -100, -120].forEach(val => {
      const y = dbmToY(val);
      ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(padLeft + plotW, y); ctx.stroke();
      ctx.fillText(val.toString(), padLeft - 4, y);
    });

    // Vertical Cyan Peak Marker at 8.60 GHz
    const peakX = freqToX(8.60);
    ctx.save();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.2;
    ctx.shadowColor = 'rgba(56, 189, 248, 0.7)';
    ctx.shadowBlur = 4;
    ctx.beginPath();
    ctx.moveTo(peakX, padTop);
    ctx.lineTo(peakX, padTop + plotH);
    ctx.stroke();
    ctx.restore();

    // Spectrum Signal Curve with sharp central resonance spike at 8.60 GHz
    const pts = [];
    const numPoints = 260;
    for (let i = 0; i <= numPoints; i++) {
      const f = 8.0 + (i / numPoints) * 1.2;
      let p = -104 + gaussianRandom(0, 2.5);
      const diff = Math.abs(f - 8.60);
      if (diff < 0.28) {
        const peakPower = -20 - 85 * (Math.pow(diff / 0.035, 1.5) / (1 + Math.pow(diff / 0.035, 1.5)));
        p = Math.max(p, peakPower + gaussianRandom(0, 1.0));
      }
      pts.push({ x: freqToX(f), y: dbmToY(p) });
    }

    // Draw Spectrum Trace
    ctx.save();
    ctx.shadowBlur = 4;
    ctx.shadowColor = 'rgba(56, 189, 248, 0.7)';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    pts.forEach((pt, idx) => { idx === 0 ? ctx.moveTo(pt.x, pt.y) : ctx.lineTo(pt.x, pt.y); });
    ctx.stroke();
    ctx.restore();

    // X-Axis Labels (8.0 to 9.2 GHz)
    ctx.font = '7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    [8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2].forEach(f => {
      const x = freqToX(f);
      ctx.fillText(f.toFixed(1), x, padTop + plotH + 3);
    });

    // Outer Bounding Box
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);
  }

  // 8c. Section 2 I/Q Waveform
  function renderSec2Iq() {
    const { w, h, ctx } = resizeCanvas(canvases.sec2Iq);
    if (!ctx || w <= 0 || h <= 0) return;

    const padLeft = 34, padRight = 10, padTop = 6, padBottom = 16;
    const plotW = w - padLeft - padRight, plotH = h - padTop - padBottom;
    ctx.clearRect(0, 0, w, h);

    function iqToY(val) { return padTop + plotH * (1 - (val + 1) / 2); }

    // Grid Horizontal Lines & Y-Axis Labels (-1, 0, 1)
    ctx.strokeStyle = '#181e29'; ctx.lineWidth = 1; ctx.font = '7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
    [-1, 0, 1].forEach(val => {
      const y = iqToY(val);
      ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(padLeft + plotW, y); ctx.stroke();
      ctx.fillText(val.toString(), padLeft - 4, y);
    });

    const timeOffset = performance.now() * 0.004;
    const ptsI = [], ptsQ = [];
    const numPoints = 220;

    for (let i = 0; i <= numPoints; i++) {
      const t = (i / numPoints) * 10;
      const phase = t * 4.8 + timeOffset;
      const valI = 0.68 * Math.sin(phase) + gaussianRandom(0, 0.035);
      const valQ = 0.68 * Math.cos(phase) + gaussianRandom(0, 0.035);

      ptsI.push({ x: padLeft + (i / numPoints) * plotW, y: iqToY(valI) });
      ptsQ.push({ x: padLeft + (i / numPoints) * plotW, y: iqToY(valQ) });
    }

    // Draw In-Phase (I) Waveform (Cyan)
    ctx.save();
    ctx.shadowBlur = 3;
    ctx.shadowColor = 'rgba(56, 189, 248, 0.5)';
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ptsI.forEach((pt, idx) => { idx === 0 ? ctx.moveTo(pt.x, pt.y) : ctx.lineTo(pt.x, pt.y); });
    ctx.stroke();
    ctx.restore();

    // Draw Quadrature (Q) Waveform (Orange)
    ctx.save();
    ctx.shadowBlur = 3;
    ctx.shadowColor = 'rgba(251, 146, 60, 0.5)';
    ctx.strokeStyle = '#fb923c';
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    ptsQ.forEach((pt, idx) => { idx === 0 ? ctx.moveTo(pt.x, pt.y) : ctx.lineTo(pt.x, pt.y); });
    ctx.stroke();
    ctx.restore();

    // X-Axis Labels (0, 2, 4, 6, 8, 10 ms)
    ctx.font = '7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    [0, 2, 4, 6, 8, 10].forEach(t => {
      const x = padLeft + (t / 10) * plotW;
      ctx.fillText(t.toString(), x, padTop + plotH + 3);
    });

    // Outer Bounding Box
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 1;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);
  }

  // 8d. Section 2 Event Timeline
  const sec2TimelineEvents = [
    { ch: 0, t: 0.8, type: 'miss' }, { ch: 0, t: 1.2, type: 'miss' }, { ch: 0, t: 2.4, type: 'hit' },
    { ch: 0, t: 3.8, type: 'hit' }, { ch: 0, t: 5.2, type: 'switch' }, { ch: 0, t: 7.2, type: 'hit' },
    { ch: 0, t: 8.5, type: 'scenario' }, { ch: 0, t: 9.2, type: 'miss' },

    { ch: 1, t: 1.0, type: 'hit' }, { ch: 1, t: 2.1, type: 'miss' }, { ch: 1, t: 3.1, type: 'miss' },
    { ch: 1, t: 4.4, type: 'switch' }, { ch: 1, t: 4.7, type: 'hit' }, { ch: 1, t: 5.8, type: 'hit' },
    { ch: 1, t: 8.1, type: 'switch' }, { ch: 1, t: 9.0, type: 'miss' },

    { ch: 2, t: 0.5, type: 'hit' }, { ch: 2, t: 1.5, type: 'switch' }, { ch: 2, t: 2.6, type: 'hit' },
    { ch: 2, t: 3.0, type: 'miss' }, { ch: 2, t: 6.2, type: 'switch' }, { ch: 2, t: 7.5, type: 'switch' },
    { ch: 2, t: 9.8, type: 'switch' }
  ];

  function renderSec2Timeline() {
    const { w, h, ctx } = resizeCanvas(canvases.sec2Timeline);
    if (!ctx || w <= 0 || h <= 0) return;

    const padLeft = 8, padRight = 8, padTop = 4, padBottom = 16;
    const plotW = w - padLeft - padRight, plotH = h - padTop - padBottom;
    ctx.clearRect(0, 0, w, h);

    const rowH = plotH / 3;

    ctx.strokeStyle = '#181e29'; ctx.lineWidth = 1;
    for (let i = 0; i < 3; i++) {
      const y = padTop + (i + 0.5) * rowH;
      ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(padLeft + plotW, y); ctx.stroke();
    }

    sec2TimelineEvents.forEach(ev => {
      const x = padLeft + (ev.t / 10) * plotW;
      const yCenter = padTop + (ev.ch + 0.5) * rowH;
      const barH = rowH * 0.72;

      ctx.fillStyle = ev.type === 'hit' ? '#10b981' :
                      ev.type === 'miss' ? '#ef4444' :
                      ev.type === 'switch' ? '#f59e0b' : '#38bdf8';

      ctx.fillRect(x - 1, yCenter - barH / 2, 2.5, barH);
    });

    ctx.font = '8px "JetBrains Mono", monospace'; ctx.fillStyle = '#64748b'; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
    [0, 2, 4, 6, 8, 10].forEach(t => {
      const x = padLeft + (t / 10) * plotW;
      ctx.fillText(t.toString(), x, padTop + plotH + 2);
    });
  }

  // ==========================================================================
  // 8e. Section 2 Scanning View (Receiver) - Radar Polar Scanner
  // ==========================================================================
  const radarScanState = {
    angle: 0.85, // initial angle radians (~48 deg)
    blips: [
      { freqLbl: '8.50 GHz', angle: Math.PI / 2, radiusFrac: 0.82, color: '#facc15', size: 4.2 }, // Yellow (top)
      { freqLbl: '8.00 GHz', angle: (3 * Math.PI) / 4 + 0.05, radiusFrac: 0.76, color: '#ef4444', size: 3.8 }, // Red (top-left)
      { freqLbl: '8.75 GHz', angle: (5 * Math.PI) / 9, radiusFrac: 0.32, color: '#38bdf8', size: 3.2 }, // Blue (inner)
      { freqLbl: '7.50 GHz', angle: (4 * Math.PI) / 3 + 0.05, radiusFrac: 0.72, color: '#10b981', size: 3.8 }, // Green (bottom-left)
      { freqLbl: '9.25 GHz', angle: 0.04, radiusFrac: 0.85, color: '#c084fc', size: 4.0 } // Purple (right)
    ]
  };

  function renderScanningView() {
    const { w, h, ctx } = resizeCanvas(canvases.scanningView);
    if (!ctx || w <= 0 || h <= 0) return;

    ctx.clearRect(0, 0, w, h);

    const cx = w / 2;
    const cy = h / 2 + 2;
    const maxR = Math.min(w, h) * 0.325; // leave room for outer labels

    // 1. Draw Concentric Circles
    const rings = [0.25, 0.50, 0.75, 1.0];
    rings.forEach((frac, idx) => {
      ctx.strokeStyle = idx === rings.length - 1 ? '#23324d' : '#142033';
      ctx.lineWidth = idx === rings.length - 1 ? 1.4 : 1.0;
      ctx.beginPath();
      ctx.arc(cx, cy, maxR * frac, 0, Math.PI * 2);
      ctx.stroke();
    });

    // 2. Draw Crosshairs / Radial Spokes (0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°)
    ctx.strokeStyle = '#162338';
    ctx.lineWidth = 1;
    for (let i = 0; i < 8; i++) {
      const a = (i * Math.PI) / 4;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(a) * maxR, cy - Math.sin(a) * maxR);
      ctx.stroke();
    }

    // 3. Frequency Labels around Perimeter
    const labels = [
      { text: '8.50 GHz', angle: Math.PI / 2, align: 'center', baseline: 'bottom', dy: -5, dx: 0 },
      { text: '9.00 GHz', angle: Math.PI / 4, align: 'left', baseline: 'bottom', dy: -2, dx: 4 },
      { text: '9.25 GHz', angle: 0, align: 'left', baseline: 'middle', dy: 0, dx: 6 },
      { text: '9.50 GHz', angle: -Math.PI / 4, align: 'left', baseline: 'top', dy: 4, dx: 4 },
      { text: '7.50 GHz', angle: (5 * Math.PI) / 4, align: 'right', baseline: 'top', dy: 4, dx: -4 },
      { text: '7.75 GHz', angle: Math.PI, align: 'right', baseline: 'middle', dy: 0, dx: -6 },
      { text: '8.00 GHz', angle: (3 * Math.PI) / 4, align: 'right', baseline: 'bottom', dy: -2, dx: -4 }
    ];

    ctx.font = '500 7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';

    labels.forEach(lbl => {
      const lx = cx + Math.cos(lbl.angle) * maxR + lbl.dx;
      const ly = cy - Math.sin(lbl.angle) * maxR + lbl.dy;
      ctx.textAlign = lbl.align;
      ctx.textBaseline = lbl.baseline;
      ctx.fillText(lbl.text, lx, ly);
    });

    // 4. Advance Sweep Angle
    if (state.isRunning && !state.isPaused) {
      radarScanState.angle = (radarScanState.angle + 0.012) % (Math.PI * 2);
    }
    const curA = radarScanState.angle;
    const wedgeSpan = 0.44; // ~25 degrees

    // 5. Draw Sweeping Blue Wedge / Sector
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    // Draw arc backwards from leading edge
    ctx.arc(cx, cy, maxR, -curA, -(curA - wedgeSpan), true);
    ctx.closePath();

    // Gradient inside sector
    const startX = cx + Math.cos(curA) * maxR;
    const startY = cy - Math.sin(curA) * maxR;
    const wedgeGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, maxR);
    wedgeGrad.addColorStop(0, 'rgba(56, 189, 248, 0.45)');
    wedgeGrad.addColorStop(0.7, 'rgba(2, 132, 199, 0.22)');
    wedgeGrad.addColorStop(1, 'rgba(2, 132, 199, 0.06)');
    ctx.fillStyle = wedgeGrad;
    ctx.fill();

    // Bright Leading Edge Line
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(startX, startY);
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.6;
    ctx.shadowColor = '#38bdf8';
    ctx.shadowBlur = 8;
    ctx.stroke();
    ctx.restore();

    // 6. Draw Emitter Blips
    radarScanState.blips.forEach(blip => {
      const bx = cx + Math.cos(blip.angle) * (maxR * blip.radiusFrac);
      const by = cy - Math.sin(blip.angle) * (maxR * blip.radiusFrac);

      // Check distance from sweep angle to trigger flare
      let angleDiff = (curA - blip.angle) % (Math.PI * 2);
      if (angleDiff < 0) angleDiff += Math.PI * 2;
      const isSwept = angleDiff < 0.6;

      ctx.save();
      ctx.shadowBlur = isSwept ? 10 : 5;
      ctx.shadowColor = blip.color;
      ctx.fillStyle = blip.color;

      ctx.beginPath();
      ctx.arc(bx, by, blip.size * (isSwept ? 1.25 : 1.0), 0, Math.PI * 2);
      ctx.fill();

      // Bright white core when swept
      if (isSwept) {
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(bx, by, blip.size * 0.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    });
  }

  // ==========================================================================
  // 9. TAB SWITCHING & INTERACTIVE CONTROLS
  // ==========================================================================
  window.switchRfTab = function (tab) {
    state.activeRfTab = tab;
    const buttons = document.querySelectorAll('.rf-tab-btn');
    buttons.forEach(btn => {
      const btnText = btn.textContent.toLowerCase();
      if (
        (tab === 'spectrum' && btnText.includes('spectrum')) ||
        (tab === 'iq' && btnText.includes('i/q')) ||
        (tab === 'constellation' && btnText.includes('constellation'))
      ) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    const specBox = document.querySelector('.diag-spectrum-box');
    const iqBox = document.querySelector('.diag-iq-box');
    const constBox = document.getElementById('sec2ConstellationWrap');

    if (specBox && iqBox && constBox) {
      if (tab === 'spectrum') {
        specBox.style.display = 'flex';
        iqBox.style.display = 'flex';
        constBox.style.display = 'none';
      } else if (tab === 'iq') {
        specBox.style.display = 'none';
        iqBox.style.display = 'flex';
        iqBox.style.flex = '1';
        constBox.style.display = 'none';
      } else if (tab === 'constellation') {
        specBox.style.display = 'none';
        iqBox.style.display = 'none';
        constBox.style.display = 'flex';
        constBox.style.flex = '1';
      }
    }

    if (typeof anime !== 'undefined') {
      anime({
        targets: '.sec2-diag-content-wrapper',
        opacity: [0.5, 1],
        duration: 300,
        easing: 'easeOutQuad'
      });
    }
  };

  window.switchReceiver = function (rxNum) {
    if (state.activeReceiver === rxNum) return;
    state.activeReceiver = rxNum;

    [1, 2, 3].forEach(id => {
      const btn = document.getElementById('rxBtn' + id);
      if (btn) {
        if (id === rxNum) btn.classList.add('active');
        else btn.classList.remove('active');
      }
    });

    const rx = state.receivers[rxNum];
    if (!rx) return;

    if (typeof anime !== 'undefined') {
      anime({
        targets: '#valTunedFreq, #valIbw, #valDwellTime, #valSnrEst',
        opacity: [0.3, 1],
        translateY: [-4, 0],
        duration: 350,
        easing: 'easeOutQuad'
      });
    }

    const freqEl = document.getElementById('valTunedFreq');
    if (freqEl) freqEl.textContent = rx.freq.toFixed(3) + ' GHz';
    const ibwEl = document.getElementById('valIbw');
    if (ibwEl) ibwEl.textContent = rx.ibw + ' MHz';
    const dwellEl = document.getElementById('valDwellTime');
    if (dwellEl) dwellEl.textContent = rx.dwell + ' ms';
    const snrEl = document.getElementById('valSnrEst');
    if (snrEl) snrEl.textContent = rx.snr.toFixed(1) + ' dB';
    const countEl = document.getElementById('valDetectionCount');
    if (countEl) countEl.textContent = rx.detCount;
  };

  window.changeModulation = function (mod, panelId) {
    const id = panelId || 1;
    const pState = constellationState[id];
    if (!pState) return;

    pState.modulation = mod;

    const sel1 = document.getElementById('selModulation1');
    const sel2 = document.getElementById('selModulation2');
    if (id === 1 && sel1) sel1.value = mod;
    if (id === 2 && sel2) sel2.value = mod;

    const centers = MODULATION_CENTERS[mod] || MODULATION_CENTERS['QPSK'];
    const sigma = mod === '16-QAM' ? 0.06 : (mod === 'BPSK' ? 0.16 : 0.15);

    pState.points.forEach((pt, idx) => {
      const c = centers[idx % centers.length];
      const gI = gaussianRandom(0, sigma);
      const gQ = gaussianRandom(0, sigma);
      const targetI = c.i + gI;
      const targetQ = c.q + gQ;
      const dist = Math.hypot(gI, gQ);
      const targetAlpha = Math.max(0.3, Math.min(0.98, 1.0 - dist * 2.0));

      if (typeof anime !== 'undefined') {
        anime({
          targets: pt,
          curI: targetI,
          curQ: targetQ,
          alpha: targetAlpha,
          duration: 320 + (idx % 12) * 12,
          easing: 'easeOutCubic'
        });
      } else {
        pt.curI = targetI;
        pt.curQ = targetQ;
        pt.alpha = targetAlpha;
      }
      pt.baseI = c.i;
      pt.baseQ = c.q;
      pt.sigma = sigma;
    });

    const targetCanvas = id === 1 ? canvases.constellation : canvases.sec2Constellation;
    if (typeof anime !== 'undefined' && targetCanvas) {
      anime({
        targets: targetCanvas,
        opacity: [0.7, 1],
        duration: 250,
        easing: 'easeOutQuad'
      });
    }
  };

  function setupControls() {
    const btnStart = document.getElementById('btnStart');
    const btnPause = document.getElementById('btnPause');
    const btnStop = document.getElementById('btnStop');
    const chkShowReceiverWindow = document.getElementById('chkShowReceiverWindow');
    const btnClearLog = document.getElementById('btnClearLog');

    if (btnStart) {
      btnStart.addEventListener('click', () => {
        state.isRunning = true; state.isPaused = false;
        btnStart.classList.add('active'); btnPause.classList.remove('active'); btnStop.classList.remove('active');
      });
    }
    if (btnPause) {
      btnPause.addEventListener('click', () => {
        state.isPaused = !state.isPaused;
        if (state.isPaused) { btnPause.classList.add('active'); btnStart.classList.remove('active'); }
        else { btnPause.classList.remove('active'); btnStart.classList.add('active'); }
      });
    }
    if (btnStop) {
      btnStop.addEventListener('click', () => {
        state.isRunning = false; state.isPaused = false;
        btnStop.classList.add('active'); btnStart.classList.remove('active'); btnPause.classList.remove('active');
      });
    }
    if (chkShowReceiverWindow) {
      chkShowReceiverWindow.addEventListener('change', (e) => {
        state.showReceiverWindow = e.target.checked;
      });
    }
    if (btnClearLog) {
      btnClearLog.addEventListener('click', () => {
        const body = document.getElementById('sec2EventLogBody');
        if (body) body.innerHTML = '<tr><td colspan="3" style="text-align:center; color:#71717a;">Event Log Cleared</td></tr>';
      });
    }

    window.addEventListener('resize', () => {
      renderSpectrum();
      renderWaterfall();
      renderBandScan();
      renderTimeline();
      renderRxWindow();
      drawConstellationPlot(canvases.constellation, 1);
      renderSec2Spectrogram();
      renderSec2Spectrum();
      renderSec2Iq();
      renderSec2Timeline();
      renderScanningView();
    });
  }

  // ==========================================================================
  // 10. DOM TELEMETRY LOOP
  // ==========================================================================
  function updateTelemetryDom() {
    if (!state.isRunning || state.isPaused) return;

    state.simTime += 0.016;
    const simTimeEl = document.getElementById('valTimeSim');
    if (simTimeEl) simTimeEl.textContent = state.simTime.toFixed(3) + ' s';

    const utcClockEl = document.getElementById('utcClock');
    if (utcClockEl) {
      const now = new Date();
      const y = now.getUTCFullYear();
      const m = String(now.getUTCMonth() + 1).padStart(2, '0');
      const d = String(now.getUTCDate()).padStart(2, '0');
      const hh = String(now.getUTCHours()).padStart(2, '0');
      const mm = String(now.getUTCMinutes()).padStart(2, '0');
      const ss = String(now.getUTCSeconds()).padStart(2, '0');
      utcClockEl.textContent = `UTC: ${y}-${m}-${d} ${hh}:${mm}:${ss}`;
    }

    // Advance real-time RF Band Scan Engine
    updateBandScanEngine(0.016);
    // Advance Hit / Miss Timeline Engine
    updateTimelineEngine(0.016);
  }

  // ==========================================================================
  // 11. MAIN ANIMATION LOOP
  // ==========================================================================
  function animateLoop() {
    requestAnimationFrame(animateLoop);

    renderSpectrum();
    renderWaterfall();
    renderBandScan();
    renderTimeline();
    renderRxWindow();
    drawConstellationPlot(canvases.constellation, 1);

    renderSec2Spectrogram();
    renderSec2Spectrum();
    renderSec2Iq();
    renderSec2Timeline();
    renderScanningView();

    updateTelemetryDom();
  }

  function init() {
    initWaterfallHistory();
    initBandScanData();
    initTimeline();
    initConstellation();
    setupControls();
    requestAnimationFrame(animateLoop);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();

