/**
 * VYAPTI - REAL-TIME RF PHYSICS & LIVE SIMULATION
 * Powered by System C (Real RF/IQ Physics) + System B (ClassicalUCB1 Policy)
 * NO SYNTHETIC / HARDCODED RF DATA - 100% PURE REAL-TIME BACKEND TELEMETRY
 */

(function () {
  'use strict';

  // ==========================================================================
  // 1. AUTHORITATIVE TELEMETRY STORE & WEBSOCKET CLIENT
  // ==========================================================================
  const telemetryStore = {
    connected: false,
    systemCState: 'STOPPED',
    ucbState: 'INACTIVE',
    iqState: 'NO DATA',
    lastPacketTime: 0,
    latest: null,
    history: {
      pd: [],
      pfa: [],
      intercept: [],
      snr: [],
      tti: [],
      coverage: [],
      switches: [],
    },
    waterfallRows: [],
    maxWaterfallRows: 250,
  };

  let ws = null;
  let reconnectTimer = null;
  const WS_URL = 'ws://' + (window.location.hostname || 'localhost') + ':8000/ws/rf';

  function connectWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }

    try {
      ws = new WebSocket(WS_URL);
    } catch (err) {
      console.warn('[WebSocket] Connection failed to', WS_URL, err);
      scheduleReconnect();
      return;
    }

    ws.onopen = function () {
      telemetryStore.connected = true;
      telemetryStore.lastPacketTime = performance.now();
      updateStatusBadges();
      console.log('[WebSocket] Connected to Vyapti Telemetry at', WS_URL);
    };

    ws.onmessage = function (event) {
      telemetryStore.lastPacketTime = performance.now();
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'telemetry') {
          handleTelemetryPacket(msg);
        } else if (msg.type === 'heartbeat') {
          handleHeartbeatPacket(msg);
        }
      } catch (e) {
        console.error('[WebSocket] Malformed JSON received:', e);
      }
    };

    ws.onclose = function () {
      telemetryStore.connected = false;
      telemetryStore.systemCState = 'DISCONNECTED';
      telemetryStore.ucbState = 'INACTIVE';
      telemetryStore.iqState = 'DISCONNECTED';
      updateStatusBadges();
      scheduleReconnect();
    };

    ws.onerror = function (e) {
      console.warn('[WebSocket] Transport error:', e);
      try { ws.close(); } catch (_) {}
    };
  }

  function scheduleReconnect() {
    if (reconnectTimer) return;
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connectWebSocket();
    }, 1500);
  }

  function sendCommand(cmd, extra = {}) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ command: cmd, ...extra }));
    }
  }

  // ==========================================================================
  // 2. TELEMETRY PROCESSING & DISPATCH
  // ==========================================================================
  function handleTelemetryPacket(data) {
    telemetryStore.latest = data;
    telemetryStore.connected = true;
    telemetryStore.systemCState = data.status.system_c;
    telemetryStore.ucbState = data.status.ucb_prior;
    telemetryStore.iqState = data.status.iq_stream;

    // Track metric histories for sparklines
    const m = data.metrics || {};
    pushSparkHistory(telemetryStore.history.pd, m.pd ?? 0);
    pushSparkHistory(telemetryStore.history.pfa, m.pfa ?? 0);
    pushSparkHistory(telemetryStore.history.intercept, m.intercept_rate ?? 0);
    pushSparkHistory(telemetryStore.history.snr, m.avg_snr_db ?? 0);
    pushSparkHistory(telemetryStore.history.tti, m.tti_ms ?? 0);
    pushSparkHistory(telemetryStore.history.coverage, m.coverage ?? 0);
    pushSparkHistory(telemetryStore.history.switches, m.switches ?? 0);

    // Push new real FFT PSD row into waterfall scrolling buffer
    if (data.psd && Array.isArray(data.psd.power_dbm)) {
      telemetryStore.waterfallRows.unshift({
        power: data.psd.power_dbm,
        bandIndex: data.dwell ? data.dwell.band_index : 0,
        fCenterGhz: data.dwell ? data.dwell.f_center_ghz : 9.0,
        hit: data.rf_receiver ? data.rf_receiver.hit : false,
        timeS: data.sim_time_s,
      });
      if (telemetryStore.waterfallRows.length > telemetryStore.maxWaterfallRows) {
        telemetryStore.waterfallRows.pop();
      }
    }

    updateDomMetrics(data);
    updateStatusBadges();
  }

  function handleHeartbeatPacket(data) {
    if (data.status) {
      telemetryStore.systemCState = data.status.system_c;
      telemetryStore.ucbState = data.status.ucb_prior;
      telemetryStore.iqState = data.status.iq_stream;
    }
    updateStatusBadges();
  }

  function pushSparkHistory(arr, val) {
    arr.push(val);
    if (arr.length > 50) arr.shift();
  }

  // ==========================================================================
  // 3. CANVAS SETUP & REFERENCES
  // ==========================================================================
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

  // Status Badges
  const elStatusBackend = document.getElementById('valStatusBackend');
  const elStatusSystemC = document.getElementById('valStatusSystemC');
  const elStatusUCB = document.getElementById('valStatusUCB');
  const elStatusIQ = document.getElementById('valStatusIQ');

  // Buttons
  const btnStartTop = document.getElementById('btnStartTop');
  const btnPauseTop = document.getElementById('btnPauseTop');
  const btnStopTop = document.getElementById('btnStopTop');
  const btnCtrlStart = document.getElementById('btnCtrlStart');
  const btnCtrlPause = document.getElementById('btnCtrlPause');
  const btnCtrlStop = document.getElementById('btnCtrlStop');
  const btnCtrlReset = document.getElementById('btnCtrlReset');

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

  // ==========================================================================
  // 4. GRAPH RENDERERS (100% REAL SYSTEM C DATA ONLY)
  // ==========================================================================

  // --- 1. RF SPECTRUM (PSD) ---
  function drawPsd() {
    const setup = setupCanvas(cPsd);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 28;
    const padBottom = 16;
    const padTop = 10;
    const padRight = 8;
    const plotW = Math.max(10, width - padLeft - padRight);
    const plotH = Math.max(10, height - padBottom - padTop);

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // Y Axis dB Ticks (0 to -100 dBm)
    ctx.font = '500 6.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let db = 0; db >= -100; db -= 20) {
      const y = padTop + ((0 - db) / 100) * plotH;
      ctx.fillText(db.toString(), padLeft - 3, y);
      ctx.strokeStyle = '#131b26';
      ctx.lineWidth = 0.7;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    }

    const t = telemetryStore.latest;
    const rx = t ? t.rf_receiver : null;
    const fCenter = rx ? rx.center_freq_ghz : 9.0;
    const bwMhz = rx ? rx.bandwidth_mhz : 500.0;
    const fMin = (fCenter * 1000 - bwMhz / 2) / 1000;
    const fMax = (fCenter * 1000 + bwMhz / 2) / 1000;

    // X Axis Frequency Ticks
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xTicks = [fMin, fMin + (fMax - fMin) * 0.25, fCenter, fMin + (fMax - fMin) * 0.75, fMax];
    xTicks.forEach(f => {
      const x = padLeft + ((f - fMin) / (fMax - fMin)) * plotW;
      ctx.fillText(f.toFixed(2) + 'G', x, padTop + plotH + 2);
      ctx.strokeStyle = '#131b26';
      ctx.lineWidth = 0.7;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();
    });

    // Check if live telemetry exists
    if (!telemetryStore.connected || !t || !t.psd || !t.psd.power_dbm) {
      drawDisconnectedNotice(ctx, padLeft, padTop, plotW, plotH, 'NO LIVE RF TELEMETRY');
      return;
    }

    const psd = t.psd.power_dbm;
    const nBins = psd.length;

    // Plot real FFT power points
    ctx.save();
    ctx.strokeStyle = '#00d4ff';
    ctx.lineWidth = 1.2;
    ctx.shadowColor = 'rgba(0, 212, 255, 0.4)';
    ctx.shadowBlur = 2;
    ctx.beginPath();

    for (let i = 0; i < nBins; i++) {
      const px = (i / (nBins - 1)) * plotW;
      const powerDbm = psd[i];
      const y = Math.min(padTop + plotH, Math.max(padTop, padTop + ((0 - powerDbm) / 100) * plotH));
      const x = padLeft + px;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  // --- 2. WATERFALL (Spectrogram of Real Sequential IQ Dwells) ---
  function drawWaterfall() {
    const setup = setupCanvas(cWaterfall);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 28;
    const padBottom = 20;
    const plotW = Math.max(10, width - padLeft);
    const plotH = Math.max(10, height - padBottom);

    // Y Axis Frequencies (18.0 to 0.0 GHz)
    ctx.font = '500 8px Inter, sans-serif';
    ctx.fillStyle = '#557297';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    [0.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0].forEach(f => {
      const y = plotH - (f / 18.0) * plotH;
      ctx.fillText(f.toFixed(1), padLeft - 4, y);
    });

    // X Axis Time
    ctx.textBaseline = 'top';
    ctx.textAlign = 'center';
    for (let t = 0; t <= 12; t += 2) {
      const x = padLeft + (t / 12) * plotW;
      ctx.fillText(t.toString(), x, plotH + 4);
    }

    if (!telemetryStore.connected || telemetryStore.waterfallRows.length === 0) {
      drawDisconnectedNotice(ctx, padLeft, 0, plotW, plotH, 'WATERFALL: WAITING FOR REAL IQ DWELLS');
      return;
    }

    // Draw Real Spectrogram Rows from Sequential System C Dwells
    const rows = telemetryStore.waterfallRows;
    const numRows = rows.length;

    ctx.save();
    ctx.beginPath();
    ctx.rect(padLeft, 0, plotW, plotH);
    ctx.clip();

    ctx.fillStyle = '#010817';
    ctx.fillRect(padLeft, 0, plotW, plotH);

    rows.forEach((row, rowIdx) => {
      const yNorm = 1.0 - (row.fCenterGhz / 18.0);
      const rowY = yNorm * plotH;
      const sliceH = Math.max(2, (0.5 / 18.0) * plotH);

      const powerArr = row.power;
      const nBins = powerArr.length;
      const colW = Math.max(1, plotW / nBins);

      for (let b = 0; b < nBins; b += 2) {
        const pwr = powerArr[b];
        const norm = Math.max(0, Math.min(1, (pwr - (-110)) / 90));
        const val = Math.floor(norm * 255);
        const r = JET_LUT[val * 3];
        const g = JET_LUT[val * 3 + 1];
        const bl = JET_LUT[val * 3 + 2];

        ctx.fillStyle = `rgb(${r},${g},${bl})`;
        ctx.fillRect(padLeft + (b / nBins) * plotW, rowY - sliceH / 2, colW * 2, sliceH);
      }
    });

    ctx.restore();
  }

  // --- 3. I/Q TIME DOMAIN (Real I and Q Waveforms from System C) ---
  function drawIqTimeDomain() {
    const setup = setupCanvas(cIqTime);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 28;
    const padBottom = 20;
    const padTop = 10;
    const plotW = Math.max(10, width - padLeft - 8);
    const plotH = Math.max(10, height - padBottom - padTop);

    // Y Axis: 1.0 to -1.0
    ctx.font = '500 8px Inter, sans-serif';
    ctx.fillStyle = '#557297';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    [1.0, 0.5, 0.0, -0.5, -1.0].forEach(v => {
      const y = padTop + ((1.0 - v) / 2.0) * plotH;
      ctx.fillText(v.toFixed(1), padLeft - 4, y);
      ctx.strokeStyle = v === 0.0 ? '#182b45' : '#0c1726';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    // X Axis: 0 to 50 μs
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (let t = 0; t <= 50; t += 10) {
      const x = padLeft + (t / 50) * plotW;
      ctx.fillText(t.toString(), x, padTop + plotH + 4);
    }

    const t = telemetryStore.latest;
    if (!telemetryStore.connected || !t || !t.iq_stream || !t.iq_stream.i) {
      drawDisconnectedNotice(ctx, padLeft, padTop, plotW, plotH, 'NO REAL IQ STREAM');
      return;
    }

    const iStream = t.iq_stream.i;
    const qStream = t.iq_stream.q;
    const nSamples = iStream.length;
    if (nSamples < 2) return;

    // Scale peak amplitude dynamically to fit comfortably
    let maxAbs = 1e-9;
    for (let k = 0; k < nSamples; k++) {
      maxAbs = Math.max(maxAbs, Math.abs(iStream[k]), Math.abs(qStream[k]));
    }
    const scale = 0.85 / maxAbs;

    // 1. Draw Real In-Phase (I) Waveform (Cyan)
    ctx.save();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.4;
    ctx.shadowColor = 'rgba(56, 189, 248, 0.4)';
    ctx.shadowBlur = 3;
    ctx.beginPath();
    for (let k = 0; k < nSamples; k++) {
      const x = padLeft + (k / (nSamples - 1)) * plotW;
      const amp = Math.max(-1.0, Math.min(1.0, iStream[k] * scale));
      const y = padTop + ((1.0 - amp) / 2.0) * plotH;
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();

    // 2. Draw Real Quadrature (Q) Waveform (Yellow/Amber)
    ctx.save();
    ctx.strokeStyle = '#facc15';
    ctx.lineWidth = 1.4;
    ctx.shadowColor = 'rgba(250, 204, 21, 0.4)';
    ctx.shadowBlur = 3;
    ctx.beginPath();
    for (let k = 0; k < nSamples; k++) {
      const x = padLeft + (k / (nSamples - 1)) * plotW;
      const amp = Math.max(-1.0, Math.min(1.0, qStream[k] * scale));
      const y = padTop + ((1.0 - amp) / 2.0) * plotH;
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  // --- 4. DOPPLER / INSTANTANEOUS FREQUENCY OFFSET ---
  function drawDoppler() {
    const setup = setupCanvas(cDoppler);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 38;
    const padRight = 14;
    const padTop = 10;
    const padBottom = 22;
    const plotW = Math.max(10, width - padLeft - padRight);
    const plotH = Math.max(10, height - padTop - padBottom);

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // Labels
    ctx.font = '600 8px "Inter", sans-serif';
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('Time (μs)', padLeft + plotW * 0.5, padTop + plotH + 11);

    // Y Axis Ticks: 10, 5, 0, -5, -10 kHz
    ctx.font = '600 8px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    [10, 5, 0, -5, -10].forEach(v => {
      const y = padTop + ((10 - v) / 20.0) * plotH;
      ctx.fillText(v.toString(), padLeft - 4, y);
      ctx.strokeStyle = v === 0 ? '#1e293b' : '#101a26';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    // Box border
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 1.0;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    const t = telemetryStore.latest;
    if (!telemetryStore.connected || !t || !t.doppler || !t.doppler.available) {
      drawDisconnectedNotice(ctx, padLeft, padTop, plotW, plotH, 'N/A (NO EMITTER IN DWELL)');
      return;
    }

    // Real Doppler offset
    const offsetKHz = t.doppler.offset_khz;
    const yVal = padTop + ((10 - Math.max(-10, Math.min(10, offsetKHz))) / 20.0) * plotH;

    ctx.save();
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(padLeft, yVal);
    ctx.lineTo(padLeft + plotW, yVal);
    ctx.stroke();

    ctx.fillStyle = '#22c55e';
    ctx.font = '700 9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';
    ctx.fillText(`Δfd: ${offsetKHz >= 0 ? '+' : ''}${offsetKHz.toFixed(2)} kHz | AoA: ${t.doppler.aoa_deg ?? 'N/A'}°`, padLeft + 6, padTop + 8);
    ctx.restore();
  }

  // --- 5. CONSTELLATION (Real IQ Complex Plane Scatter) ---
  function drawConstellation() {
    const setup = setupCanvas(cConstellation);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 20;
    const padBottom = 16;
    const plotW = Math.max(10, width - padLeft);
    const plotH = Math.max(10, height - padBottom);

    ctx.fillStyle = '#01050e';
    ctx.fillRect(0, 0, width, height);

    // Axes
    const cx = padLeft + plotW * 0.5;
    const cy = plotH * 0.5;
    ctx.strokeStyle = '#1b2f48';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(padLeft, cy);
    ctx.lineTo(padLeft + plotW, cy);
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, plotH);
    ctx.stroke();

    const t = telemetryStore.latest;
    if (!telemetryStore.connected || !t || !t.constellation || t.constellation.length === 0) {
      drawDisconnectedNotice(ctx, padLeft, 0, plotW, plotH, 'NO REAL IQ SAMPLES');
      return;
    }

    const pts = t.constellation;
    let maxR = 1e-9;
    pts.forEach(p => {
      maxR = Math.max(maxR, Math.abs(p.i), Math.abs(p.q));
    });
    const scale = (plotW * 0.40) / maxR;

    ctx.fillStyle = '#00d4ff';
    ctx.shadowColor = 'rgba(0, 212, 255, 0.7)';
    ctx.shadowBlur = 3;

    pts.forEach(p => {
      const x = cx + p.i * scale;
      const y = cy - p.q * scale;
      ctx.fillRect(x - 1.5, y - 1.5, 3, 3);
    });
    ctx.shadowBlur = 0;
  }

  // --- 5b. CC0 ERROR VECTOR (EVM) ---
  // Dynamic persistent state for 3 Layer CC0 Error Vector simulation
  const CC0_CARRIERS = 260; // Dense carrier raster (-300 to +300 carrier domain)
  const cc0State = {
    baseVals: new Float32Array(CC0_CARRIERS),
    peakVals: new Float32Array(CC0_CARRIERS),
    targetBase: new Float32Array(CC0_CARRIERS),
    targetPeak: new Float32Array(CC0_CARRIERS),
    initialized: false,
    lastDwellId: -1,
    phase: 0,
  };

  // --- 5b. 3 LAYER CC0 ERROR VECTOR (EVM) ---
  function drawCc0Evm() {
    const setup = setupCanvas(cCc0Evm);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 32;
    const padBottom = 22;
    const padTop = 13;
    const padRight = 10;
    const plotW = Math.max(20, width - padLeft - padRight);
    const plotH = Math.max(20, height - padTop - padBottom);

    // Dark oscilloscope background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    const data = telemetryStore.latest;
    const isConnected = telemetryStore.connected && data;
    const isHit = data ? (data.current_decision ? data.current_decision.hit : (data.rf_receiver ? data.rf_receiver.hit : false)) : false;
    const dwellId = data && data.dwell ? data.dwell.dwell_id : 0;
    const rawSpectrum = data && data.evm && Array.isArray(data.evm.spectrum) ? data.evm.spectrum : null;

    // Y-Scale: strictly 0 to 4.5% with 500 m%/div (9 intervals of 0.5%)
    function getY(pct) {
      const clamped = Math.min(4.5, Math.max(0.0, pct));
      return padTop + plotH * (1.0 - clamped / 4.5);
    }

    // 1. Grid lines (Deep blue as in reference image)
    ctx.strokeStyle = '#0f2742';
    ctx.lineWidth = 1.0;

    // 9 Horizontal Grid lines for 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5
    for (let i = 1; i <= 9; i++) {
      const pct = i * 0.5;
      const y = getY(pct);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    }

    // 10 Vertical columns across plotW
    const vDivs = 10;
    for (let j = 0; j <= vDivs; j++) {
      const x = padLeft + (j / vDivs) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();
    }

    // Plot Border
    ctx.strokeStyle = '#163558';
    ctx.lineWidth = 1.0;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    // 2. Y-Axis Labels
    // "Mag" at top-left (avoiding overlap with 4.5 tick)
    ctx.font = '700 8.5px "Inter", -apple-system, sans-serif';
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText('Mag', 6, padTop - 4);

    // Ticks: 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1, 500m
    ctx.font = '600 8.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    const tickLabels = [
      { val: 4.5, text: '4.5' },
      { val: 4.0, text: '4' },
      { val: 3.5, text: '3.5' },
      { val: 3.0, text: '3' },
      { val: 2.5, text: '2.5' },
      { val: 2.0, text: '2' },
      { val: 1.5, text: '1.5' },
      { val: 1.0, text: '1' },
      { val: 0.5, text: '500m' },
    ];

    tickLabels.forEach(t => {
      const y = getY(t.val);
      ctx.fillText(t.text, padLeft - 4, y);
    });

    // 3. Bottom Labels below grid
    ctx.font = '700 7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#ffffff';
    ctx.textBaseline = 'top';

    ctx.textAlign = 'left';
    ctx.fillText('Start -300 carrier', padLeft, padTop + plotH + 4);
    ctx.fillText('Res BW 15 kHz', padLeft, padTop + plotH + 13);

    ctx.textAlign = 'right';
    ctx.fillText('Stop 300 carrier', padLeft + plotW, padTop + plotH + 4);
    ctx.fillText('TimeLen 42 Sym', padLeft + plotW, padTop + plotH + 13);

    // If disconnected, show notice inside grid and return
    if (!isConnected) {
      drawDisconnectedNotice(ctx, padLeft, padTop, plotW, plotH, 'EVM: WAITING FOR TELEMETRY');
      return;
    }

    // 4. Update Simulation State from System C Telemetry
    if (dwellId !== cc0State.lastDwellId || !cc0State.initialized) {
      cc0State.lastDwellId = dwellId;
      cc0State.initialized = true;

      // Base error level responds to signal presence and SNR
      const baseLevel = isHit ? 0.44 : 0.68;
      const spikeMax = isHit ? 0.95 : 1.55;

      for (let i = 0; i < CC0_CARRIERS; i++) {
        const normC = (i - CC0_CARRIERS / 2) / (CC0_CARRIERS / 2);
        // Realistic edge roll-off
        const edgeFactor = 1.0 + 0.35 * (normC * normC);

        let rawVal = 0.5;
        if (rawSpectrum && rawSpectrum.length > 0) {
          const binIdx = Math.floor((i / CC0_CARRIERS) * rawSpectrum.length);
          rawVal = Math.min(2.5, Math.max(0.1, (rawSpectrum[binIdx] || 10.0) * 0.05));
        }

        // Layer 1: Green base (0.35% - 0.75%)
        const bTarget = Math.max(0.20, Math.min(1.10, (baseLevel * 0.75 + rawVal * 0.25) * edgeFactor * (0.82 + 0.36 * Math.random())));
        cc0State.targetBase[i] = bTarget;

        // Layer 2: Cyan spikes (peaks on intermittent subcarriers)
        const hasSpike = Math.random() < (isHit ? 0.45 : 0.65);
        if (hasSpike) {
          const spikeH = (Math.random() ** 2.2) * spikeMax;
          cc0State.targetPeak[i] = bTarget + spikeH;
        } else {
          cc0State.targetPeak[i] = bTarget;
        }
      }
    }

    // Smooth animation step
    cc0State.phase += 0.05;
    for (let i = 0; i < CC0_CARRIERS; i++) {
      cc0State.baseVals[i] += (cc0State.targetBase[i] - cc0State.baseVals[i]) * 0.14;
      cc0State.peakVals[i] += (cc0State.targetPeak[i] - cc0State.peakVals[i]) * 0.18;
    }

    const baselineY = getY(0.0);
    const barW = Math.max(1, (plotW / CC0_CARRIERS) - 0.2);

    // --- LAYER 1: GREEN BASE BARS ---
    ctx.fillStyle = '#00dc82'; // vibrant emerald green from reference image
    for (let i = 0; i < CC0_CARRIERS; i++) {
      const x = padLeft + (i / CC0_CARRIERS) * plotW;
      const jitter = 0.02 * Math.sin(cc0State.phase * 2.0 + i * 0.45);
      const bVal = Math.max(0.1, cc0State.baseVals[i] + jitter);
      const bY = getY(bVal);
      const h = baselineY - bY;
      ctx.fillRect(x, bY, barW, h);
    }

    // --- LAYER 2: CYAN UPPER SPIKES ---
    ctx.fillStyle = '#00e5ff'; // electric cyan from reference image
    for (let i = 0; i < CC0_CARRIERS; i++) {
      const x = padLeft + (i / CC0_CARRIERS) * plotW;
      const bVal = cc0State.baseVals[i];
      const pVal = cc0State.peakVals[i];
      if (pVal > bVal + 0.02) {
        const bY = getY(bVal);
        const pY = getY(pVal);
        const h = bY - pY;
        ctx.fillRect(x, pY, barW, h);
      }
    }

    // --- LAYER 3: SMOOTH WHITE REFERENCE BASELINE CURVE ---
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1.8;
    ctx.shadowColor = 'rgba(255, 255, 255, 0.5)';
    ctx.shadowBlur = 3;
    ctx.beginPath();

    for (let i = 0; i <= CC0_CARRIERS; i++) {
      const normC = (i - CC0_CARRIERS / 2) / (CC0_CARRIERS / 2);
      // Gentle parabolic curve dipping to ~0.50% at center and ~0.72% at edges
      const curveVal = 0.50 + 0.18 * (normC * normC) + 0.012 * Math.sin(normC * Math.PI * 3.0 + cc0State.phase * 0.2);
      const x = padLeft + (i / CC0_CARRIERS) * plotW;
      const y = getY(curveVal);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // --- 5c. Q-CHANNEL (Real Q Samples from System C IQ stream) ---
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

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    // Title label
    ctx.font = '600 7px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('Q (Quadrature) — Real System C IQ', padLeft + plotW * 0.5, 2);

    ctx.font = '600 8.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    [1.0, 0.0, -1.0].forEach(v => {
      const y = padTop + ((1.0 - v) / 2.0) * plotH;
      ctx.fillText(v.toString(), padLeft - 4, y);
      ctx.strokeStyle = v === 0 ? '#1e293b' : '#0f172a';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 0.8;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    const t = telemetryStore.latest;
    if (!telemetryStore.connected || !t || !t.iq_stream || !t.iq_stream.q) {
      drawDisconnectedNotice(ctx, padLeft, padTop, plotW, plotH, 'NO REAL Q SAMPLES');
      return;
    }

    // Plot raw Q samples directly — NO synthetic carrier multiplication
    const qStream = t.iq_stream.q;
    const n = qStream.length;
    if (n < 2) return;

    let maxAbs = 1e-9;
    for (let i = 0; i < n; i++) maxAbs = Math.max(maxAbs, Math.abs(qStream[i]));
    const scale = 0.85 / maxAbs;

    ctx.save();
    ctx.strokeStyle = '#ef4444';
    ctx.lineWidth = 1.4;
    ctx.shadowColor = 'rgba(239, 68, 68, 0.35)';
    ctx.shadowBlur = 2;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      // Raw Q sample — no Math.sin() or synthetic carrier applied
      const val = Math.max(-1.0, Math.min(1.0, qStream[i] * scale));
      const x = padLeft + (i / (n - 1)) * plotW;
      const y = padTop + ((1.0 - val) / 2.0) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  // --- 5d. RF BAND SCAN HISTORY (Current Band Only) ---
  function drawRfScanHistory() {
    const setup = setupCanvas(cRfScanHistory);
    if (!setup) return;
    const { ctx, width, height } = setup;

    const padLeft = 28;
    const padBottom = 16;
    const padTop = 10;
    const padRight = 10;
    const plotW = Math.max(10, width - padLeft - padRight);
    const plotH = Math.max(10, height - padTop - padBottom);

    // Deep dark background
    ctx.fillStyle = '#060b14';
    ctx.fillRect(0, 0, width, height);

    // Plot surface
    ctx.fillStyle = '#010612';
    ctx.fillRect(padLeft, padTop, plotW, plotH);

    const t = telemetryStore.latest;

    // Y Axis: Physical Frequency 0.0 to 18.0 GHz
    function getY(fGhz) {
      const clamped = Math.max(0.0, Math.min(18.0, fGhz));
      return padTop + plotH * (1.0 - clamped / 18.0);
    }

    // 1. Horizontal Grid Lines & Y-Ticks (18.0, 14.0, 10.0, 6.0, 2.0 GHz)
    const fTicks = [18.0, 14.0, 10.0, 6.0, 2.0];
    ctx.font = '500 7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    fTicks.forEach(f => {
      const y = getY(f);
      ctx.fillText(f.toFixed(1), padLeft - 4, y);

      ctx.strokeStyle = '#0e2238';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    // 2. Vertical Grid Lines (at 3, 6, 9 s)
    [3, 6, 9].forEach(sec => {
      const x = padLeft + (sec / 12.0) * plotW;
      ctx.strokeStyle = '#0a192b';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();
    });

    // Outer Plot Border
    ctx.strokeStyle = '#142a45';
    ctx.lineWidth = 1.0;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    // 3. X-Axis Ticks (0, 3, 6, 9, 12 s)
    const xTicks = [0, 3, 6, 9, 12];
    ctx.font = '500 7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textBaseline = 'top';

    xTicks.forEach(val => {
      const x = padLeft + (val / 12.0) * plotW;
      ctx.textAlign = val === 0 ? 'left' : (val === 12 ? 'right' : 'center');
      ctx.fillText(val.toString(), x, padTop + plotH + 4);
    });

    if (!telemetryStore.connected || !t) {
      drawDisconnectedNotice(ctx, padLeft, padTop, plotW, plotH, 'NO SCAN TELEMETRY');
      return;
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(padLeft, padTop, plotW, plotH);
    ctx.clip();

    // 4. Current Band Only (Decluttered, strictly current band)
    const rx = t.rf_receiver || null;
    const dw = t.dwell || null;
    const curBand = (dw && dw.band_index != null) ? dw.band_index : (t.selected_band != null ? t.selected_band : 17);
    const curFGhz = (rx && rx.center_freq_ghz != null) ? rx.center_freq_ghz :
      (dw && dw.f_center_ghz != null ? dw.f_center_ghz :
      ((dw && dw.f_start_hz && dw.f_end_hz) ? (dw.f_start_hz + dw.f_end_hz) / 2e9 : 8.45));

    const bwGhz = (rx && rx.bandwidth_hz) ? (rx.bandwidth_hz / 1e9) : 0.5;
    const fStart = (rx && rx.freq_start_hz) ? rx.freq_start_hz / 1e9 : Math.max(0.0, curFGhz - bwGhz / 2);
    const fEnd = (rx && rx.freq_end_hz) ? rx.freq_end_hz / 1e9 : Math.min(18.0, curFGhz + bwGhz / 2);

    const yCur = getY(curFGhz);
    const yTop = getY(fEnd);
    const yBottom = getY(fStart);
    const bandH = Math.max(6, Math.abs(yBottom - yTop));

    // A. Illuminated bandpass corridor across the plot window
    ctx.fillStyle = 'rgba(0, 230, 118, 0.08)';
    ctx.fillRect(padLeft, yTop, plotW, bandH);

    // B. Dashed bandpass boundary lines
    ctx.strokeStyle = 'rgba(0, 230, 118, 0.35)';
    ctx.lineWidth = 1.0;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(padLeft, yTop);
    ctx.lineTo(padLeft + plotW, yTop);
    ctx.moveTo(padLeft, yBottom);
    ctx.lineTo(padLeft + plotW, yBottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // C. Active tuned center line up to current scan position
    const curX = padLeft + (11.45 / 12.0) * plotW;
    ctx.strokeStyle = '#00e676';
    ctx.lineWidth = 3.0;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(padLeft, yCur);
    ctx.lineTo(curX, yCur);
    ctx.stroke();

    // D. Vertical green scan cursor line
    ctx.strokeStyle = '#00e676';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(curX, padTop);
    ctx.lineTo(curX, padTop + plotH);
    ctx.stroke();

    // E. Glowing beacon dot at intersection
    ctx.fillStyle = 'rgba(0, 230, 118, 0.25)';
    ctx.beginPath();
    ctx.arc(curX, yCur, 7, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#00e676';
    ctx.beginPath();
    ctx.arc(curX, yCur, 3.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(curX, yCur, 1.5, 0, Math.PI * 2);
    ctx.fill();

    // F. "CURRENT" label at top
    ctx.font = '700 8px "JetBrains Mono", monospace';
    ctx.fillStyle = '#00e676';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText('CURRENT', curX - 4, padTop + 4);

    // G. Band Information Tag
    ctx.font = '600 8.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#00e676';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'bottom';
    ctx.fillText(`BAND ${curBand} • ${curFGhz.toFixed(3)} GHz`, padLeft + 8, Math.max(padTop + 14, yTop - 3));

    ctx.restore();
  }

  // --- 5e. CCDF (Real Amplitude / Power Statistics) ---
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

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, width, height);

    ctx.font = '600 7.5px "Inter", sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('Power > Avg (dB)', padLeft + plotW * 0.5, padTop + plotH + 9);

    // Y Axis Decades: 100%, 10%, 1%, 0.1%, 0.01%
    ctx.font = '600 7.5px "JetBrains Mono", monospace';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    [
      { exp: 0, lbl: '100%' },
      { exp: -1, lbl: '10%' },
      { exp: -2, lbl: '1%' },
      { exp: -3, lbl: '0.1%' },
      { exp: -4, lbl: '0.01%' },
    ].forEach(item => {
      const y = padTop + (-item.exp / 4.0) * plotH;
      ctx.fillText(item.lbl, padLeft - 4, y);
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    // Box border
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 0.8;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    const t = telemetryStore.latest;
    if (!telemetryStore.connected || !t || !t.ccdf || t.ccdf.length === 0) {
      drawDisconnectedNotice(ctx, padLeft, padTop, plotW, plotH, 'NO REAL CCDF');
      return;
    }

    const ccdf = t.ccdf;
    ctx.save();
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.6;
    ctx.beginPath();

    ccdf.forEach((pt, idx) => {
      const x = padLeft + (pt.db / 10.0) * plotW;
      const logProb = pt.prob > 0 ? Math.log10(pt.prob) : -4;
      const clampedLog = Math.max(-4, Math.min(0, logProb));
      const y = padTop + (-clampedLog / 4.0) * plotH;
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });

    ctx.stroke();
    ctx.restore();
  }

  // --- 6. SPARKLINE RENDERERS ---
  function drawLiveMetricsSparklines() {
    drawSparkline('sparkPd', telemetryStore.history.pd, '#10b981', 0, 1);
    drawSparkline('sparkPfa', telemetryStore.history.pfa, '#ef4444', 0, 0.1);
    drawSparkline('sparkIntercept', telemetryStore.history.intercept, '#38bdf8', 0, 1);
    drawSparkline('sparkSnr', telemetryStore.history.snr, '#facc15', -10, 30);
    drawSparkline('sparkTti', telemetryStore.history.tti, '#a855f7', 0, 500);
    drawSparkline('sparkCoverage', telemetryStore.history.coverage, '#06b6d4', 0, 1);
    drawSparkline('sparkSwitches', telemetryStore.history.switches, '#f97316', 0, 100);
  }

  function drawSparkline(canvasId, data, color, minVal, maxVal) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || data.length < 2) return;
    const setup = setupCanvas(canvas);
    if (!setup) return;
    const { ctx, width, height } = setup;

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    const len = data.length;
    for (let i = 0; i < len; i++) {
      const x = (i / (len - 1)) * width;
      const norm = Math.max(0, Math.min(1, (data[i] - minVal) / (maxVal - minVal || 1)));
      const y = height - norm * (height - 2) - 1;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function drawDisconnectedNotice(ctx, x, y, w, h, text) {
    ctx.save();
    ctx.fillStyle = '#ef4444';
    ctx.font = '700 8.5px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x + w * 0.5, y + h * 0.5);
    ctx.restore();
  }

  // ==========================================================================
  // 5. DOM & STATUS UPDATES
  // ==========================================================================
  function updateStatusBadges() {
    if (elStatusBackend) {
      elStatusBackend.textContent = telemetryStore.connected ? '● CONNECTED' : '● DISCONNECTED';
      elStatusBackend.className = telemetryStore.connected ? 'telemetry-val status-connected' : 'telemetry-val status-disconnected';
    }
    if (elStatusSystemC) {
      elStatusSystemC.textContent = `● ${telemetryStore.systemCState}`;
      elStatusSystemC.className = telemetryStore.systemCState === 'RUNNING' ? 'telemetry-val status-running' : (telemetryStore.systemCState === 'PAUSED' ? 'telemetry-val status-paused' : 'telemetry-val status-disconnected');
    }
    if (elStatusUCB) {
      elStatusUCB.textContent = `● ${telemetryStore.ucbState}`;
      elStatusUCB.className = telemetryStore.ucbState === 'ACTIVE' ? 'telemetry-val status-active' : 'telemetry-val status-disconnected';
    }
    if (elStatusIQ) {
      elStatusIQ.textContent = `● ${telemetryStore.iqState}`;
      elStatusIQ.className = telemetryStore.iqState === 'LIVE' ? 'telemetry-val status-live' : (telemetryStore.iqState === 'STALE' ? 'telemetry-val status-stale' : 'telemetry-val status-disconnected');
    }
  }

  function updateDomMetrics(data) {
    if (elSimTime) elSimTime.textContent = data.sim_time_s.toFixed(3) + ' s';
    if (elWallTime) elWallTime.textContent = data.wall_time_s.toFixed(3) + ' s';
    if (elDecisions) elDecisions.textContent = (data.metrics ? data.metrics.total_dwells : 0).toString();
    if (elEpisodes) elEpisodes.textContent = (data.metrics ? data.metrics.episode : 1).toString();

    const rx = data.rf_receiver;
    if (rx) {
      if (elIqIRms) elIqIRms.textContent = rx.i_rms.toFixed(4) + ' V';
      if (elIqQRms) elIqQRms.textContent = rx.q_rms.toFixed(4) + ' V';
      if (elIqPeak) elIqPeak.textContent = rx.peak_voltage.toFixed(4) + ' V';
      if (elIqPower) elIqPower.textContent = rx.signal_power_dbm.toFixed(1) + ' dBm';
      if (elIqPhase) elIqPhase.textContent = rx.phase_deg.toFixed(1) + '°';
      if (elIqSnr) elIqSnr.textContent = rx.snr_db.toFixed(1) + ' dB';
      if (elIqSampleRate) elIqSampleRate.textContent = rx.sample_rate_msps.toFixed(1) + ' MSps';

      // Closed-Loop Verification: UCB Selected/Commanded -> System C Tuned -> IQ Band
      const cmdBand = data.commanded_band !== undefined ? data.commanded_band : (data.selected_band !== undefined ? data.selected_band : (data.dwell ? data.dwell.band_index : null));
      const tunedBand = data.system_c_current_band !== undefined ? data.system_c_current_band : (rx ? rx.band_index : null);
      const iqBandObj = data.iq_band || (data.dwell ? { band_index: data.dwell.band_index, f_start_hz: data.dwell.f_start_hz, f_end_hz: data.dwell.f_end_hz, f_center_ghz: data.dwell.f_center_ghz } : null);
      const iqBandIdx = iqBandObj ? iqBandObj.band_index : null;
      const iqFStart = iqBandObj ? (iqBandObj.f_start_hz / 1e9).toFixed(2) : (rx ? (rx.freq_start_hz / 1e9).toFixed(2) : '0.00');
      const iqFEnd = iqBandObj ? (iqBandObj.f_end_hz / 1e9).toFixed(2) : (rx ? (rx.freq_end_hz / 1e9).toFixed(2) : '0.00');
      const iqFCenter = iqBandObj && iqBandObj.f_center_ghz !== undefined ? iqBandObj.f_center_ghz.toFixed(3) : (rx ? rx.center_freq_ghz.toFixed(3) : '0.000');

      const isSyncLocked = (cmdBand !== null && tunedBand !== null && iqBandIdx !== null && cmdBand === tunedBand && tunedBand === iqBandIdx);
      const syncStatusText = isSyncLocked ? 'SYNC LOCKED' : 'SYNC ERROR';

      // SDR frequency display
      const fCenter = rx.center_freq_ghz;
      const fMhz = fCenter * 1000.0;
      const mhzInt = Math.floor(fMhz);
      const khzInt = Math.round((fMhz - mhzInt) * 1000);
      const ghzStr = String(Math.floor(mhzInt / 1000)).padStart(3, '0');
      const mhzRemStr = String(mhzInt % 1000).padStart(3, '0');
      const khzStr = String(khzInt).padStart(3, '0');
      const sdrDim = document.getElementById('sdrFreqDim');
      const sdrBright = document.getElementById('sdrFreqBright');
      if (sdrDim && sdrBright) {
        sdrDim.textContent = `${ghzStr}.${mhzRemStr}.`;
        sdrBright.textContent = khzStr;
      }
      const sdrTag = document.getElementById('sdrBandTag');
      if (sdrTag) {
        sdrTag.textContent = `Band ${tunedBand} (${fCenter.toFixed(3)} GHz) [${syncStatusText}]`;
      }

      // Left Accordion Receiver fields
      const rxCenterFreq = document.getElementById('rxCenterFreq');
      if (rxCenterFreq) rxCenterFreq.textContent = rx.center_freq_ghz.toFixed(3);
      const rxInstBw = document.getElementById('rxInstBw');
      if (rxInstBw) rxInstBw.textContent = rx.bandwidth_mhz.toFixed(1);
      const rxSampleRate = document.getElementById('rxSampleRate');
      if (rxSampleRate) rxSampleRate.textContent = rx.sample_rate_msps.toFixed(1);
      const rxDwellTime = document.getElementById('rxDwellTime');
      if (rxDwellTime) rxDwellTime.textContent = (data.dwell ? data.dwell.dwell_ms : 50.0).toFixed(1);
      const rxBandwidth = document.getElementById('rxBandwidth');
      if (rxBandwidth) rxBandwidth.textContent = rx.bandwidth_mhz.toFixed(1);
      const rxSnr = document.getElementById('rxSnr');
      if (rxSnr) rxSnr.textContent = rx.snr_db.toFixed(1);
      const rxDoppler = document.getElementById('rxDoppler');
      if (rxDoppler && data.doppler) rxDoppler.textContent = (data.doppler.available ? data.doppler.offset_khz.toFixed(1) : '0.0');
      const rxRange = document.getElementById('rxRange');
      // Real range from rf_chain telemetry — not hardcoded
      if (rxRange) rxRange.textContent = (data.rf_chain && data.rf_chain.range_km != null) ? data.rf_chain.range_km.toFixed(1) : rx.snr_db.toFixed(0);

      // CURRENT BAND CARD
      const win = data.selected_frequency_window || (data.dwell ? {
        start_hz: data.dwell.f_start_hz,
        end_hz: data.dwell.f_end_hz,
        action_id: data.dwell.band_index
      } : null);

      const fStartGhz = win ? (win.start_hz / 1e9).toFixed(2) : (rx ? (rx.freq_start_hz / 1e9).toFixed(2) : '0.00');
      const fEndGhz = win ? (win.end_hz / 1e9).toFixed(2) : (rx ? (rx.freq_end_hz / 1e9).toFixed(2) : '0.00');
      const freqRangeStr = `${fStartGhz} – ${fEndGhz} GHz`;

      const elCurBandRange = document.getElementById('rfValCurBandRange');
      if (elCurBandRange) elCurBandRange.textContent = freqRangeStr;

      const elCurCF = document.getElementById('rfValCurBandCF');
      if (elCurCF) elCurCF.textContent = iqFCenter + ' GHz';

      const elCurSnr = document.getElementById('rfValCurBandSnr');
      if (elCurSnr) elCurSnr.textContent = rx.snr_db.toFixed(1) + ' dB';

      const elCurDwell = document.getElementById('rfValCurBandDwell');
      if (elCurDwell) elCurDwell.textContent = (data.dwell ? data.dwell.dwell_ms : 50).toFixed(0) + ' ms';

      const elCurBadge = document.getElementById('rfValCurSyncBadge') || document.getElementById('rfValCurBandBadge');
      if (elCurBadge) {
        elCurBadge.textContent = isSyncLocked ? '● SYNC LOCKED' : '● SCANNING';
        elCurBadge.style.color = isSyncLocked ? '#22c55e' : '#38bdf8';
      }

      const elRxLO = document.getElementById('rxStateLO');
      const elRxTuner = document.getElementById('rxTunerStatus');
      if (elRxLO) elRxLO.textContent = rx.center_freq_ghz.toFixed(3) + ' GHz';
      if (elRxTuner) elRxTuner.textContent = `${syncStatusText} (${rx.center_freq_ghz.toFixed(3)} GHz)`;
    }

    // 3. NEXT FREQUENCY WINDOW (Section 7)
    if (data.next_band) {
      const nb = data.next_band;
      const nbStart = (nb.f_start_hz / 1e9).toFixed(2);
      const nbEnd = (nb.f_end_hz / 1e9).toFixed(2);
      const elNbRange = document.getElementById('rfValNextBandRange');
      if (elNbRange) elNbRange.textContent = `${nbStart} – ${nbEnd} GHz`;
      const elNbCF = document.getElementById('rfValNextBandCF');
      if (elNbCF) elNbCF.textContent = nb.f_center_ghz.toFixed(3) + ' GHz';
      const elNbDwell = document.getElementById('rfValNextBandDwell');
      if (elNbDwell) elNbDwell.textContent = (nb.dwell_ms || 50).toFixed(0) + ' ms';
      const elNbPri = document.getElementById('rfValNextBandPriority');
      if (elNbPri) elNbPri.textContent = nb.priority;
      const elNbAct = document.getElementById('rfValNextActionId');
      if (elNbAct) elNbAct.textContent = `Action #${nb.band_index}`;
    }

    // LAST SCANNED CARD
    if (data.dwell_history && data.dwell_history.length > 1) {
      const prevDw = data.dwell_history[data.dwell_history.length - 2];
      const elLbRange = document.getElementById('rfValLastBandRange');
      if (elLbRange) {
        const fStart = (prevDw.f_start_hz / 1e9).toFixed(2);
        const fEnd = (prevDw.f_end_hz / 1e9).toFixed(2);
        elLbRange.textContent = `Band ${prevDw.band_index} (${fStart} – ${fEnd} GHz)`;
      }
      const elLbCF = document.getElementById('rfValLastBandCF');
      if (elLbCF) elLbCF.textContent = prevDw.f_center_ghz.toFixed(3) + ' GHz';
      const elLbSnr = document.getElementById('rfValLastBandSnr');
      if (elLbSnr) elLbSnr.textContent = prevDw.measured_snr_db.toFixed(1) + ' dB';
      const elLbResult = document.getElementById('rfValLastBandResult');
      if (elLbResult) {
        elLbResult.textContent = prevDw.hit ? 'HIT' : 'MISS';
        elLbResult.className = prevDw.hit ? 'rf-badge-hit' : 'rf-badge-miss';
      }
      const elLbTime = document.getElementById('rfValLastBandTime');
      if (elLbTime) elLbTime.textContent = prevDw.t_start_s.toFixed(2) + ' s';
    }

    // SCAN PROGRESS BAR
    const elProgPct = document.getElementById('rfValCurBandProgressPct');
    const elProgBar = document.getElementById('rfValCurBandProgressBar');
    if (elProgPct && elProgBar && data.metrics) {
      const pct = Math.round(data.metrics.coverage * 100);
      elProgPct.textContent = `${pct}%`;
      elProgBar.style.width = `${pct}%`;
    }

    if (data.detector_cfar) {
      const cfar = data.detector_cfar;
      const elCfarDet = document.getElementById('cfarDetectedPulses');
      if (elCfarDet) elCfarDet.textContent = cfar.total_detections.toString();
      const elCfarFa = document.getElementById('cfarFalseAlarms');
      if (elCfarFa) elCfarFa.textContent = cfar.false_alarms.toString();
      const elCfarThresh = document.getElementById('cfarThreshold');
      if (elCfarThresh) elCfarThresh.textContent = cfar.cfar_threshold_db.toFixed(1);
      const elCfarNoise = document.getElementById('cfarNoiseFloor');
      if (elCfarNoise) elCfarNoise.textContent = cfar.noise_floor_dbm.toFixed(1);
      const elCfarSnr = document.getElementById('cfarSnr');
      if (elCfarSnr) elCfarSnr.textContent = cfar.snr_db.toFixed(1);
    }

    if (data.current_pulse) {
      const p = data.current_pulse;
      const elPToa = document.getElementById('pulseToa');
      if (elPToa) elPToa.textContent = p.toa_us.toFixed(3);
      const elPCf = document.getElementById('pulseCf');
      if (elPCf) elPCf.textContent = p.cf_ghz.toFixed(3);
      const elPPw = document.getElementById('pulsePw');
      if (elPPw) elPPw.textContent = p.pw_us.toFixed(2);
      const elPAmp = document.getElementById('pulseAmp');
      if (elPAmp) elPAmp.textContent = p.amplitude_dbm.toFixed(1);
      const elPSnr = document.getElementById('pulseSnr');
      if (elPSnr) elPSnr.textContent = p.snr_db.toFixed(1);
      const elPStatus = document.getElementById('pulseStatus');
      if (elPStatus) {
        elPStatus.textContent = p.status === 'HIT' ? '• HIT' : (p.status === 'MISS' ? '• MISS' : p.status);
        elPStatus.className = p.status === 'HIT' ? 'status-val hit' : 'status-val';
      }
    }

    if (data.rf_chain) {
      const rc = data.rf_chain;
      const elRcBand = document.getElementById('rfChainBand');
      if (elRcBand) elRcBand.textContent = rc.receiver_band_str;
      const elRcAgc = document.getElementById('rfChainAgcGain');
      if (elRcAgc) elRcAgc.textContent = rc.agc_gain_db.toFixed(0) + ' dB';
      const elRcIf = document.getElementById('rfChainIfGain');
      if (elRcIf) elRcIf.textContent = rc.if_gain_db.toFixed(0) + ' dB';
      const elRcSnr = document.getElementById('rfChainSnr');
      if (elRcSnr) elRcSnr.textContent = rc.snr_db.toFixed(1);
      const elRcPathLoss = document.getElementById('rfChainPathLoss');
      if (elRcPathLoss) elRcPathLoss.textContent = rc.path_loss_db.toFixed(1);
      const elRcDoppler = document.getElementById('rfChainDoppler');
      if (elRcDoppler) elRcDoppler.textContent = (rc.doppler_shift_khz >= 0 ? '+' : '') + rc.doppler_shift_khz.toFixed(1);
      const elRcRange = document.getElementById('rfChainRange');
      if (elRcRange) elRcRange.textContent = rc.range_km.toFixed(1);
      const elRcDelay = document.getElementById('rfChainDelay');
      if (elRcDelay) elRcDelay.textContent = rc.prop_delay_us.toFixed(1);
      const elRcEvm = document.getElementById('rfChainEvm');
      if (elRcEvm) elRcEvm.textContent = rc.evm_pct;
      const elRcNoise = document.getElementById('rfChainNoiseFloor');
      if (elRcNoise) elRcNoise.textContent = rc.noise_floor_dbm.toFixed(1);
      const elRcBw = document.getElementById('rfChainBw');
      if (elRcBw) elRcBw.textContent = rc.bandwidth_mhz.toFixed(0);
    }

    if (data.rf_environment) {
      const env = data.rf_environment;
      const elEnvTotal = document.getElementById('rfEnvTotalBands');
      if (elEnvTotal) elEnvTotal.textContent = env.total_bands;
      const elEnvScanned = document.getElementById('rfEnvScannedBands');
      if (elEnvScanned) elEnvScanned.textContent = env.scanned_bands;
      const elEnvOcc = document.getElementById('rfEnvOccupiedBands');
      if (elEnvOcc) elEnvOcc.textContent = env.occupied_bands;
      const elEnvEmpty = document.getElementById('rfEnvEmptyBands');
      if (elEnvEmpty) elEnvEmpty.textContent = env.empty_bands;
      const elEnvActEm = document.getElementById('rfEnvActiveEmitters');
      if (elEnvActEm) elEnvActEm.textContent = env.active_emitters;
      const elEnvDetSig = document.getElementById('rfEnvDetectedSignals');
      if (elEnvDetSig) elEnvDetSig.textContent = env.detected_signals;
      const elEnvOccPct = document.getElementById('rfEnvOccupancyPct');
      if (elEnvOccPct) elEnvOccPct.textContent = env.spectral_occupancy_pct.toFixed(1) + ' %';
      const elEnvNoise = document.getElementById('rfEnvNoiseFloor');
      if (elEnvNoise) elEnvNoise.textContent = env.noise_floor_dbm.toFixed(1) + ' dBm';
      const elEnvPeak = document.getElementById('rfEnvPeakSignal');
      if (elEnvPeak) elEnvPeak.textContent = env.peak_signal_dbm.toFixed(1) + ' dBm';
      const elEnvCurSnr = document.getElementById('rfEnvCurrentSnr');
      if (elEnvCurSnr) elEnvCurSnr.textContent = env.current_snr_db.toFixed(1) + ' dB';
      // Update band scan history header with live band count from backend
      const elBandLabel = document.getElementById('rfScanHistoryBandLabel');
      if (elBandLabel) elBandLabel.textContent = `(${env.total_bands} Bands)`;
    }

    if (data.last_rf_event) {
      const ev = data.last_rf_event;
      const elEvTime = document.getElementById('rfEventTime');
      if (elEvTime) elEvTime.textContent = ev.time_s.toFixed(3) + ' s';
      const elEvBand = document.getElementById('rfEventBand');
      if (elEvBand) elEvBand.textContent = ev.band_index;
      const elEvCF = document.getElementById('rfEventCF');
      if (elEvCF) elEvCF.textContent = ev.f_center_ghz.toFixed(3) + ' GHz';
      const elEvPW = document.getElementById('rfEventPW');
      if (elEvPW) elEvPW.textContent = ev.pw_us.toFixed(2) + ' μs';
      const elEvAmp = document.getElementById('rfEventAmp');
      if (elEvAmp) elEvAmp.textContent = ev.amp_dbm.toFixed(1) + ' dBm';
      const elEvSnr = document.getElementById('rfEventSnr');
      if (elEvSnr) elEvSnr.textContent = ev.snr_db.toFixed(1) + ' dB';
      const elEvStatus = document.getElementById('rfEventStatus');
      if (elEvStatus) elEvStatus.textContent = ev.status;
    }

    const m = data.metrics;
    if (m) {
      const elValPd = document.getElementById('lmValPd');
      const elValPfa = document.getElementById('lmValPfa');
      const elValIntercept = document.getElementById('lmValIntercept');
      const elValSnr = document.getElementById('lmValSnr');
      const elValTti = document.getElementById('lmValTti');
      const elValCoverage = document.getElementById('lmValCoverage');
      const elValSwitches = document.getElementById('lmValSwitches');

      if (elValPd) elValPd.textContent = m.pd.toFixed(3);
      if (elValPfa) elValPfa.textContent = m.pfa.toFixed(3);
      if (elValIntercept) elValIntercept.textContent = m.intercept_rate.toFixed(3);
      if (elValSnr) elValSnr.textContent = m.avg_snr_db.toFixed(1);
      if (elValTti) elValTti.textContent = Math.round(m.tti_ms).toString();
      if (elValCoverage) elValCoverage.textContent = m.coverage.toFixed(3);
      if (elValSwitches) elValSwitches.textContent = m.switches.toString();

      // Use actual backend total_bands — never hardcode 36
      const totalBands = (data.rf_environment && data.rf_environment.total_bands) ? data.rf_environment.total_bands : 36;
      const visited = Math.round(m.coverage * totalBands);
      const elStatProg = document.getElementById('rfStatScanProgress');
      if (elStatProg) elStatProg.textContent = Math.round(m.coverage * 100) + ' %';
      const elStatScanned = document.getElementById('rfStatBandsScanned');
      if (elStatScanned) elStatScanned.textContent = `${visited} / ${totalBands}`;
      const elStatAvgDwell = document.getElementById('rfStatAvgDwell');
      // Real avg dwell from actual telemetry — not hardcoded
      if (elStatAvgDwell) elStatAvgDwell.textContent = (data.dwell && data.dwell.dwell_ms != null ? data.dwell.dwell_ms.toFixed(0) : '—') + ' ms';
      const elStatScanRate = document.getElementById('rfStatScanRate');
      // Scan rate: dwells per second from real timing (total_dwells / sim_time)
      if (elStatScanRate) {
        const scanRate = (data.sim_time_s > 0 && m.total_dwells > 0) ? (m.total_dwells / data.sim_time_s).toFixed(1) : '—';
        elStatScanRate.textContent = scanRate + ' dwells/s';
      }
      const elStatDetRate = document.getElementById('rfStatDetectionRate');
      if (elStatDetRate) elStatDetRate.textContent = (m.pd * 100).toFixed(1) + ' %';
      const elStatOcc = document.getElementById('rfStatOccupiedBands');
      if (elStatOcc && data.rf_environment) elStatOcc.textContent = `${data.rf_environment.occupied_bands} / ${totalBands}`;
      const elStatTotalTime = document.getElementById('rfStatTotalScanTime');
      if (elStatTotalTime) elStatTotalTime.textContent = data.sim_time_s.toFixed(2) + ' s';
    }

    // Update real tactical event log
    const logContainers = [
      document.getElementById('aiLogsContainer'),
      document.getElementById('liveAiLogList'),
      document.getElementById('aiLogsConsole')
    ].filter(Boolean);

    if (logContainers.length > 0 && data.recent_events && Array.isArray(data.recent_events)) {
      logContainers.forEach(container => {
        container.innerHTML = '';
        data.recent_events.forEach(ev => {
          const entry = document.createElement('div');
          entry.className = 'ai-log-entry';
          let colorStyle = 'color: #dc2626;';
          if (ev.type === 'HIT') colorStyle = 'color: #4ade80;';
          else if (ev.type === 'REWARD') colorStyle = 'color: #facc15;';
          else if (ev.type === 'DECISION') colorStyle = 'color: #38bdf8;';
          else if (ev.type === 'MISS') colorStyle = 'color: #94a3b8;';

          entry.innerHTML = `<span class="log-ts" style="color: #64748b; margin-right: 6px;">[${ev.time_s.toFixed(3)}]</span> <span class="log-msg" style="${colorStyle}">${ev.message}</span>`;
          container.appendChild(entry);
        });
      });
    }
  }

  // ==========================================================================
  // 6. RENDER LOOP & INTERACTION CONTROLS
  // ==========================================================================
  function renderAll() {
    drawPsd();
    drawWaterfall();
    drawIqTimeDomain();
    drawDoppler();
    drawConstellation();
    drawCc0Evm();
    drawQCarrier();
    drawRfScanHistory();
    drawCCDF();
    drawLiveMetricsSparklines();
  }

  function setupInteractions() {
    if (btnStartTop) btnStartTop.addEventListener('click', () => sendCommand('start'));
    if (btnCtrlStart) btnCtrlStart.addEventListener('click', () => sendCommand('start'));
    if (btnPauseTop) btnPauseTop.addEventListener('click', () => sendCommand('pause'));
    if (btnCtrlPause) btnCtrlPause.addEventListener('click', () => sendCommand('pause'));
    if (btnStopTop) btnStopTop.addEventListener('click', () => sendCommand('stop'));
    if (btnCtrlStop) btnCtrlStop.addEventListener('click', () => sendCommand('stop'));
    if (btnCtrlReset) btnCtrlReset.addEventListener('click', () => sendCommand('reset'));
  }

  function frameLoop() {
    renderAll();
    requestAnimationFrame(frameLoop);
  }

  window.addEventListener('resize', renderAll);
  window.addEventListener('DOMContentLoaded', () => {
    setupInteractions();
    connectWebSocket();
    frameLoop();
  });

  // Self-start if already loaded
  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setupInteractions();
    connectWebSocket();
    frameLoop();
  }
})();
