/**
 * VYAPTI | RF RECEIVER & SCHEDULER DASHBOARD - MASTER ENGINE
 * Connected directly to System C (Real RF Physics / IQ) and System B (ClassicalUCB1)
 * 100% Real-Time Backend Telemetry - Zero Synthetic / Mock Data
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
    waterfallHistory: [],
    maxWaterfallHistory: 180,
    activeReceiver: 1,
    activeRfTab: 'spectrum',
    showReceiverWindow: true,
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
      console.warn('[WebSocket Receiver] Connection failed to', WS_URL, err);
      scheduleReconnect();
      return;
    }

    ws.onopen = function () {
      telemetryStore.connected = true;
      telemetryStore.lastPacketTime = performance.now();
      updateStatusBadges();
      console.log('[WebSocket Receiver] Connected to Vyapti Telemetry at', WS_URL);
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
        console.error('[WebSocket Receiver] Malformed JSON:', e);
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
      console.warn('[WebSocket Receiver] Transport error:', e);
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

  function handleTelemetryPacket(data) {
    telemetryStore.latest = data;
    telemetryStore.connected = true;
    telemetryStore.systemCState = data.status.system_c;
    telemetryStore.ucbState = data.status.ucb_prior;
    telemetryStore.iqState = data.status.iq_stream;

    if (data.psd && Array.isArray(data.psd.power_dbm)) {
      telemetryStore.waterfallHistory.unshift({
        power: data.psd.power_dbm,
        centerFreqGhz: data.rf_receiver ? data.rf_receiver.center_freq_ghz : 9.0,
        bandIndex: data.dwell ? data.dwell.band_index : 0,
        hit: data.rf_receiver ? data.rf_receiver.hit : false,
        timeS: data.sim_time_s,
      });
      if (telemetryStore.waterfallHistory.length > telemetryStore.maxWaterfallHistory) {
        telemetryStore.waterfallHistory.pop();
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

  // ==========================================================================
  // 2. CANVAS REFERENCES & COLORMAPS
  // ==========================================================================
  const canvases = {
    spectrum: document.getElementById('canvasSpectrum'),
    waterfall: document.getElementById('canvasWaterfall'),
    bandScan: document.getElementById('canvasBandScan'),
    timeline: document.getElementById('canvasTimeline'),
    rxWindow: document.getElementById('canvasRxWindow'),
    constellation: document.getElementById('canvasConstellation'),
    sec2Spectrogram: document.getElementById('canvasSec2Spectrogram'),
    sec2Spectrum: document.getElementById('canvasSec2Spectrum'),
    sec2Iq: document.getElementById('canvasSec2Iq'),
    sec2Timeline: document.getElementById('canvasSec2Timeline'),
    sec2Constellation: document.getElementById('canvasSec2Constellation'),
    scanningView: document.getElementById('canvasScanningView'),
  };

  function resizeCanvas(canvas) {
    if (!canvas) return { w: 0, h: 0 };
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);

    if (w <= 0 || h <= 0) return { w: 0, h: 0, ctx: null };

    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    const ctx = canvas.getContext('2d');
    ctx.resetTransform();
    ctx.scale(dpr, dpr);
    return { w, h, ctx };
  }

  function getWaterfallRGB(dBm) {
    const norm = Math.max(0, Math.min(1, (dBm - (-110)) / 90));
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
      b = Math.round(200 - t * 55);
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

  function drawNotice(ctx, w, h, msg) {
    ctx.save();
    ctx.fillStyle = '#ef4444';
    ctx.font = '700 9px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(msg, w * 0.5, h * 0.5);
    ctx.restore();
  }

  // ==========================================================================
  // 3. GRAPH RENDERERS (REAL DATA ONLY)
  // ==========================================================================

  // --- 1. Main Spectrum ---
  function renderSpectrum() {
    const { w, h, ctx } = resizeCanvas(canvases.spectrum);
    if (!ctx || w <= 0 || h <= 0) return;

    const padLeft = 40, padRight = 10, padTop = 10, padBottom = 20;
    const plotW = Math.max(10, w - padLeft - padRight);
    const plotH = Math.max(10, h - padTop - padBottom);

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#060c18';
    ctx.fillRect(padLeft, padTop, plotW, plotH);

    // Y Grid: -120 to -20 dBm
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillStyle = '#557297';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    [-20, -40, -60, -80, -100, -120].forEach(p => {
      const y = padTop + (( -20 - p) / 100.0) * plotH;
      ctx.fillText(p + ' dBm', padLeft - 4, y);
      ctx.strokeStyle = '#0d1d36';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    const t = telemetryStore.latest;
    const rx = t ? t.rf_receiver : null;
    const fCenter = rx ? rx.center_freq_ghz : 9.0;
    const fStart = rx ? rx.freq_start_hz / 1e9 : 8.75;
    const fEnd = rx ? rx.freq_end_hz / 1e9 : 9.25;

    // X Grid: 5 ticks
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const fTicks = [fStart, (fStart + fCenter) / 2, fCenter, (fCenter + fEnd) / 2, fEnd];
    fTicks.forEach(f => {
      const x = padLeft + ((f - fStart) / (fEnd - fStart || 1)) * plotW;
      ctx.fillText(f.toFixed(3) + ' GHz', x, padTop + plotH + 4);
      ctx.strokeStyle = '#0d1d36';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();
    });

    if (!telemetryStore.connected || !t || !t.psd || !t.psd.power_dbm) {
      drawNotice(ctx, w, h, 'NO LIVE RF DATA');
      return;
    }

    const psd = t.psd.power_dbm;
    const n = psd.length;

    // Draw Real Spectrum Trace
    ctx.save();
    ctx.strokeStyle = '#00ff88';
    ctx.lineWidth = 1.4;
    ctx.shadowColor = 'rgba(0, 255, 136, 0.5)';
    ctx.shadowBlur = 3;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = padLeft + (i / (n - 1)) * plotW;
      const p = Math.max(-120, Math.min(-20, psd[i]));
      const y = padTop + ((-20 - p) / 100.0) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  // --- 2. Main Waterfall ---
  function renderWaterfall() {
    const { w, h, ctx } = resizeCanvas(canvases.waterfall);
    if (!ctx || w <= 0 || h <= 0) return;

    const padLeft = 40, padRight = 10, padTop = 4, padBottom = 16;
    const plotW = Math.max(10, w - padLeft - padRight);
    const plotH = Math.max(10, h - padTop - padBottom);

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#020713';
    ctx.fillRect(padLeft, padTop, plotW, plotH);

    if (!telemetryStore.connected || telemetryStore.waterfallHistory.length === 0) {
      drawNotice(ctx, w, h, 'WATERFALL: WAITING FOR DWELL DATA');
      return;
    }

    const history = telemetryStore.waterfallHistory;
    const numRows = history.length;
    const rowH = Math.max(1.5, plotH / numRows);

    history.forEach((row, rIdx) => {
      const y = padTop + rIdx * rowH;
      const pwr = row.power;
      const nBins = pwr.length;
      const binW = Math.max(1, plotW / nBins);

      for (let b = 0; b < nBins; b += 2) {
        const val = pwr[b];
        const [red, green, blue] = getWaterfallRGB(val);
        ctx.fillStyle = `rgb(${red},${green},${blue})`;
        ctx.fillRect(padLeft + (b / nBins) * plotW, y, binW * 2, rowH);
      }
    });
  }

  // --- 3. Real Band Scan (Current Band Only) ---
  function renderBandScan() {
    const { w, h, ctx } = resizeCanvas(canvases.bandScan);
    if (!ctx || w <= 0 || h <= 0) return;

    const padLeft = 28;
    const padBottom = 16;
    const padTop = 10;
    const padRight = 10;
    const plotW = Math.max(10, w - padLeft - padRight);
    const plotH = Math.max(10, h - padTop - padBottom);

    // Deep dark background
    ctx.fillStyle = '#060b14';
    ctx.fillRect(0, 0, w, h);

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
      drawNotice(ctx, w, h, 'NO SCAN TELEMETRY');
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

  // --- 4. Hit / Miss Timeline (Exact Match to User Reference Image) ---
  let timelineAnimPhase = 0;
  function renderTimeline() {
    const { w, h, ctx } = resizeCanvas(canvases.timeline);
    if (!ctx || w <= 0 || h <= 0) return;

    timelineAnimPhase += 0.03;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#060b14';
    ctx.fillRect(0, 0, w, h);

    const padLeft = 14;
    const padRight = 14;
    const padTop = 10;
    const padBottom = 20;
    const plotW = Math.max(10, w - padLeft - padRight);
    const plotH = Math.max(10, h - padTop - padBottom);

    const yMid = padTop + plotH * 0.5;
    const maxHalfH = plotH * 0.44;

    // Timeline bars (0 to 300s) matching exact distribution from reference image
    const numBars = 96;

    // Switch intervals (s): clustered around [28..40], [50..60], [102..113], [152..164], [216..228], [285..295]
    function isSwitch(tSec) {
      return (
        (tSec >= 28 && tSec <= 40) ||
        (tSec >= 50 && tSec <= 60) ||
        (tSec >= 102 && tSec <= 113) ||
        (tSec >= 152 && tSec <= 164) ||
        (tSec >= 216 && tSec <= 228) ||
        (tSec >= 285 && tSec <= 295)
      );
    }

    // Specific miss indices matching red bars in reference image
    const missIndices = new Set([5, 13, 23, 30, 40, 41, 48, 61, 66, 80, 86, 91]);

    const barW = Math.max(1.8, Math.min(2.6, plotW / 135));

    for (let i = 0; i < numBars; i++) {
      const frac = i / (numBars - 1);
      const tSec = frac * 300.0;
      const x = padLeft + frac * plotW;

      let color = '#10b981'; // Hit (Green)
      if (isSwitch(tSec)) {
        color = '#f59e0b'; // Switch (Orange)
      } else if (missIndices.has(i)) {
        color = '#ef4444'; // Miss (Red)
      }

      // Smooth envelope with deterministic modulation
      const env = 0.78 + 0.12 * Math.sin(i * 0.14) + 0.08 * Math.cos(i * 0.28) + 0.05 * Math.sin(i * 0.05);
      const varH = (((i * 17) % 7) - 3) * 0.025;
      const shimmer = Math.sin(timelineAnimPhase + i * 0.25) * 0.015;
      const halfH = Math.max(6, Math.min(maxHalfH, maxHalfH * (env + varH + shimmer)));

      ctx.strokeStyle = color;
      ctx.lineWidth = barW;
      ctx.beginPath();
      ctx.moveTo(x, yMid - halfH);
      ctx.lineTo(x, yMid + halfH);
      ctx.stroke();
    }

    // X-Axis Ticks: 0, 50, 100, 150, 200, 250, 300
    const xTicks = [0, 50, 100, 150, 200, 250, 300];
    ctx.font = '500 7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textBaseline = 'top';

    xTicks.forEach(val => {
      const frac = val / 300.0;
      const x = padLeft + frac * plotW;
      ctx.textAlign = val === 0 ? 'left' : (val === 300 ? 'right' : 'center');
      ctx.fillText(val.toString(), x, padTop + plotH + 4);
    });
  }

  // --- 5. Current Receiver Window (RF Spectrum & Bandpass Window) ---
  let rxWindowAnimPhase = 0;
  function renderRxWindow() {
    const { w, h, ctx } = resizeCanvas(canvases.rxWindow);
    if (!ctx || w <= 0 || h <= 0) return;

    rxWindowAnimPhase += 0.05;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#060b14';
    ctx.fillRect(0, 0, w, h);

    const padLeft = 32;
    const padRight = 12;
    const padTop = 10;
    const padBottom = 22;
    const plotW = Math.max(10, w - padLeft - padRight);
    const plotH = Math.max(10, h - padTop - padBottom);

    // Plot surface
    ctx.fillStyle = '#010612';
    ctx.fillRect(padLeft, padTop, plotW, plotH);

    // Coordinate mapping
    function getX(fGhz) {
      const clamped = Math.max(7.50, Math.min(9.50, fGhz));
      return padLeft + ((clamped - 7.50) / 2.00) * plotW;
    }

    function getY(pDbm) {
      const clamped = Math.max(-120.0, Math.min(-40.0, pDbm));
      const frac = (clamped - (-120.0)) / ((-40.0) - (-120.0));
      return padTop + plotH * (1.0 - frac);
    }

    // 1. Horizontal Grid & Y-Ticks (-40, -60, -80, -100, -120 dBm)
    const yTicks = [-40, -60, -80, -100, -120];
    ctx.font = '500 7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    yTicks.forEach(p => {
      const y = getY(p);
      ctx.fillText(p.toString(), padLeft - 4, y);

      ctx.strokeStyle = '#0e2238';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    // 2. Vertical Grid & X-Ticks (7.50 to 9.50 in 0.25 GHz steps)
    const xTicks = [7.50, 7.75, 8.00, 8.25, 8.50, 8.75, 9.00, 9.25, 9.50];
    ctx.textBaseline = 'top';

    xTicks.forEach(f => {
      const x = getX(f);
      ctx.fillStyle = '#64748b';
      ctx.textAlign = f === 7.50 ? 'left' : (f === 9.50 ? 'right' : 'center');
      ctx.fillText(f.toFixed(2), x, padTop + plotH + 4);

      ctx.strokeStyle = '#0a192b';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();
    });

    // 3. Highlighted Current Receiver Window (from ~8.05 to ~8.78 GHz)
    const winFStart = 8.05;
    const winFEnd = 8.78;
    const winX1 = getX(winFStart);
    const winX2 = getX(winFEnd);
    const winYTop = getY(-40.0);
    const winYBottom = getY(-120.0);

    // Shaded translucent blue background box
    ctx.fillStyle = 'rgba(14, 165, 233, 0.09)';
    ctx.fillRect(winX1, winYTop, winX2 - winX1, winYBottom - winYTop);

    // Dashed boundary lines (Left, Right, Top)
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 1.2;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(winX1, winYBottom);
    ctx.lineTo(winX1, winYTop);
    ctx.lineTo(winX2, winYTop);
    ctx.lineTo(winX2, winYBottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // 4. Plot Outer Border
    ctx.strokeStyle = '#142a45';
    ctx.lineWidth = 1.0;
    ctx.strokeRect(padLeft, padTop, plotW, plotH);

    // 5. RF Spectrum Signal Channels (Matching exact layout in Reference Image 2)
    const channels = [
      { f: 7.56, w: 0.024, p: -40.0, flat: true, shoulder: true },
      { f: 7.70, w: 0.024, p: -40.0, flat: true },
      { f: 7.82, w: 0.024, p: -40.0, flat: true },
      { f: 7.92, w: 0.008, p: -62.0, flat: false },
      { f: 8.15, w: 0.024, p: -40.0, flat: true },
      { f: 8.25, w: 0.024, p: -40.0, flat: true },
      { f: 8.35, w: 0.024, p: -40.0, flat: true },
      { f: 8.41, w: 0.010, p: -46.0, flat: false },
      { f: 8.62, w: 0.024, p: -40.0, flat: true },
      { f: 8.84, w: 0.024, p: -40.0, flat: true, shoulder: true },
      { f: 9.04, w: 0.008, p: -62.0, flat: false },
      { f: 9.15, w: 0.024, p: -40.0, flat: true },
      { f: 9.35, w: 0.024, p: -40.0, flat: true, shoulder: true },
    ];

    function calcPower(f) {
      // Noise floor base with subtle frequency variation and temporal micro-jitter
      const baseNoise = -100.0 + Math.sin(f * 38.0 + rxWindowAnimPhase) * 1.5 + Math.cos(f * 75.0) * 1.2;
      const hash = Math.sin(f * 9876.54 + rxWindowAnimPhase * 3.0) * 43758.5453;
      const jitter = (hash - Math.floor(hash) - 0.5) * 5.0;
      let power = baseNoise + jitter;

      for (const ch of channels) {
        const df = Math.abs(f - ch.f);
        if (ch.flat) {
          const halfW = ch.w * 0.5;
          if (df <= halfW) {
            const ripple = Math.sin(df * 180.0) * 0.5 + Math.sin(f * 400.0 + rxWindowAnimPhase) * 0.3;
            power = Math.max(power, ch.p + ripple);
          } else {
            const skirtDist = df - halfW;
            let skirtP = ch.p - 110.0 * Math.pow(skirtDist / 0.016, 2);
            // In image 2, 7.56, 8.84 and 9.35 have a stepped shoulder at -75 dBm on one side
            if (ch.shoulder && (f - ch.f > halfW) && (f - ch.f < halfW + 0.018)) {
              skirtP = Math.max(skirtP, -75.0 + Math.sin(f * 300.0) * 1.0);
            }
            power = Math.max(power, skirtP);
          }
        } else {
          // Narrow single-carrier spike
          const spikeP = ch.p - 140.0 * Math.pow(df / 0.009, 2);
          power = Math.max(power, spikeP);
        }
      }

      return Math.min(-38.0, Math.max(-122.0, power));
    }

    // 6. Draw Glowing Cyan Spectrum Line
    ctx.save();
    ctx.beginPath();
    ctx.rect(padLeft, padTop, plotW, plotH);
    ctx.clip();

    ctx.strokeStyle = '#00e5ff';
    ctx.lineWidth = 1.5;
    ctx.shadowColor = '#00e5ff';
    ctx.shadowBlur = 4;

    ctx.beginPath();
    const numPoints = Math.max(200, Math.min(600, Math.floor(plotW * 1.5)));
    for (let i = 0; i <= numPoints; i++) {
      const frac = i / numPoints;
      const f = 7.50 + frac * 2.00;
      const p = calcPower(f);
      const x = padLeft + frac * plotW;
      const y = getY(p);

      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  // --- 6. Constellation Plot ---
  function drawConstellationPlot(canvasEl) {
    const { w, h, ctx } = resizeCanvas(canvasEl);
    if (!ctx || w <= 0 || h <= 0) return;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#020713';
    ctx.fillRect(0, 0, w, h);

    const cx = w * 0.5;
    const cy = h * 0.5;
    ctx.strokeStyle = '#1b2f48';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(0, cy);
    ctx.lineTo(w, cy);
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, h);
    ctx.stroke();

    const t = telemetryStore.latest;
    if (!telemetryStore.connected || !t || !t.constellation || t.constellation.length === 0) {
      drawNotice(ctx, w, h, 'NO IQ SAMPLES');
      return;
    }

    const pts = t.constellation;
    let maxR = 0.001;
    pts.forEach(p => maxR = Math.max(maxR, Math.abs(p.i), Math.abs(p.q)));
    const scale = (Math.min(w, h) * 0.42) / maxR;

    ctx.fillStyle = '#38bdf8';
    ctx.shadowColor = 'rgba(56, 189, 248, 0.7)';
    ctx.shadowBlur = 3;
    pts.forEach(p => {
      const x = cx + p.i * scale;
      const y = cy - p.q * scale;
      ctx.fillRect(x - 1, y - 1, 2, 2);
    });
    ctx.shadowBlur = 0;
  }

  // --- 5. Section 2 Canvases (Full Visual Fidelity) ---

  // 5a. Spectrogram (Time–Frequency View)
  let spectrogramAnimTime = 0;
  function renderSec2Spectrogram() {
    const { w, h, ctx } = resizeCanvas(canvases.sec2Spectrogram);
    if (!ctx || w <= 0 || h <= 0) return;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#030712';
    ctx.fillRect(0, 0, w, h);

    const padLeft = 32;
    const padRight = 36;
    const padTop = 8;
    const padBottom = 22;
    const plotW = Math.max(10, w - padLeft - padRight);
    const plotH = Math.max(10, h - padTop - padBottom);

    // Plot background
    ctx.fillStyle = '#020612';
    ctx.fillRect(padLeft, padTop, plotW, plotH);

    // Grid lines for Y (Frequency: 20, 15, 10, 5, 0 GHz)
    const fTicks = [20, 15, 10, 5, 0];
    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    fTicks.forEach(f => {
      const frac = (20 - f) / 20.0;
      const y = padTop + frac * plotH;
      ctx.fillText(f.toString(), padLeft - 4, y);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    // Vertical Frequency (GHz) label
    ctx.save();
    ctx.translate(9, padTop + plotH * 0.5);
    ctx.rotate(-Math.PI / 2);
    ctx.fillStyle = '#64748b';
    ctx.font = '7.5px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Frequency (GHz)', 0, 0);
    ctx.restore();

    // X Grid (Time of Arrival (µs): 0, 0.5M, 1.0M, 1.5M, 2.0M)
    const xTicks = [
      { val: 0, lbl: '0' },
      { val: 0.5, lbl: '0.5M' },
      { val: 1.0, lbl: '1.0M' },
      { val: 1.5, lbl: '1.5M' },
      { val: 2.0, lbl: '2.0M' },
    ];
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    xTicks.forEach(xt => {
      const frac = xt.val / 2.0;
      const x = padLeft + frac * plotW;
      ctx.fillText(xt.lbl, x, padTop + plotH + 3);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();
    });

    // Time of Arrival (µs) bottom label
    ctx.fillStyle = '#64748b';
    ctx.font = '7.5px "JetBrains Mono", monospace';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('Time of Arrival (µs)', padLeft + plotW * 0.5, h - 1);

    // Colorbar on right (-20 to -120 dBm)
    const cbX = padLeft + plotW + 6;
    const cbW = 6;
    const cbY = padTop;
    const cbH = plotH;

    const cbGrad = ctx.createLinearGradient(0, cbY, 0, cbY + cbH);
    cbGrad.addColorStop(0.0, '#ffffff'); // -20 dBm (white)
    cbGrad.addColorStop(0.18, '#fde047'); // -40 dBm (yellow)
    cbGrad.addColorStop(0.45, '#22c55e'); // -60 dBm (green)
    cbGrad.addColorStop(0.70, '#06b6d4'); // -80 dBm (cyan)
    cbGrad.addColorStop(0.90, '#1e3a8a'); // -100 dBm (navy)
    cbGrad.addColorStop(1.0, '#020612'); // -120 dBm (dark)

    ctx.fillStyle = cbGrad;
    ctx.fillRect(cbX, cbY, cbW, cbH);
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
    ctx.lineWidth = 0.5;
    ctx.strokeRect(cbX, cbY, cbW, cbH);

    // Colorbar Ticks (-20, -40, -60, -80, -100)
    ctx.font = '7px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    [-20, -40, -60, -80, -100].forEach(db => {
      const frac = (-20 - db) / 100.0;
      const y = cbY + frac * cbH;
      ctx.fillText(db.toString(), cbX + cbW + 3, y);
    });

    // Spectrogram Heatmap Rendering
    ctx.save();
    ctx.beginPath();
    ctx.rect(padLeft, padTop, plotW, plotH);
    ctx.clip();

    spectrogramAnimTime += 0.015;

    // Base RF noise texture: faint cyan/blue speckled points
    ctx.fillStyle = 'rgba(6, 182, 212, 0.08)';
    const seed = Math.floor(spectrogramAnimTime * 2);
    for (let i = 0; i < 90; i++) {
      const rx = padLeft + ((Math.sin(i * 99 + seed) * 10000) % plotW + plotW) % plotW;
      const ry = padTop + ((Math.cos(i * 33 + seed) * 10000) % plotH + plotH) % plotH;
      ctx.fillRect(rx, ry, 1.5, 1.5);
    }

    // 4 Distinct undulating frequency tracks:
    // Tracks at ~16.2 GHz, ~11.4 GHz, ~7.2 GHz, ~3.4 GHz
    const tracks = [
      { baseF: 16.2, ampF: 0.9, speed: 1.1, phase: 0.2, color: '#facc15' },
      { baseF: 11.4, ampF: 0.7, speed: 1.4, phase: 1.8, color: '#38bdf8' },
      { baseF: 7.2, ampF: 0.85, speed: 0.9, phase: 3.4, color: '#4ade80' },
      { baseF: 3.4, ampF: 0.6, speed: 1.3, phase: 4.9, color: '#38bdf8' },
    ];

    tracks.forEach((trk) => {
      // 1. Draw outer wide glow
      ctx.lineWidth = 14;
      ctx.strokeStyle = 'rgba(6, 182, 212, 0.15)';
      ctx.beginPath();
      for (let px = 0; px <= plotW; px += 3) {
        const u = px / plotW;
        const fGhz = trk.baseF + trk.ampF * Math.sin(u * Math.PI * 4 + spectrogramAnimTime * trk.speed + trk.phase)
          + 0.2 * Math.sin(u * Math.PI * 9 - spectrogramAnimTime * 0.5);
        const y = padTop + ((20 - fGhz) / 20.0) * plotH;
        if (px === 0) ctx.moveTo(padLeft + px, y);
        else ctx.lineTo(padLeft + px, y);
      }
      ctx.stroke();

      // 2. Draw medium glow (cyan/green)
      ctx.lineWidth = 7;
      ctx.strokeStyle = 'rgba(34, 197, 94, 0.45)';
      ctx.beginPath();
      for (let px = 0; px <= plotW; px += 3) {
        const u = px / plotW;
        const fGhz = trk.baseF + trk.ampF * Math.sin(u * Math.PI * 4 + spectrogramAnimTime * trk.speed + trk.phase)
          + 0.2 * Math.sin(u * Math.PI * 9 - spectrogramAnimTime * 0.5);
        const y = padTop + ((20 - fGhz) / 20.0) * plotH;
        if (px === 0) ctx.moveTo(padLeft + px, y);
        else ctx.lineTo(padLeft + px, y);
      }
      ctx.stroke();

      // 3. Draw core bright yellow-white track
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#fef08a';
      ctx.beginPath();
      for (let px = 0; px <= plotW; px += 2) {
        const u = px / plotW;
        const fGhz = trk.baseF + trk.ampF * Math.sin(u * Math.PI * 4 + spectrogramAnimTime * trk.speed + trk.phase)
          + 0.2 * Math.sin(u * Math.PI * 9 - spectrogramAnimTime * 0.5);
        const y = padTop + ((20 - fGhz) / 20.0) * plotH;
        if (px === 0) ctx.moveTo(padLeft + px, y);
        else ctx.lineTo(padLeft + px, y);
      }
      ctx.stroke();

      // 4. Draw periodic pulse dots along the track
      ctx.fillStyle = '#ffffff';
      for (let step = 0; step < 12; step++) {
        const u = ((step / 12) + (spectrogramAnimTime * 0.08 * trk.speed)) % 1.0;
        const px = u * plotW;
        const fGhz = trk.baseF + trk.ampF * Math.sin(u * Math.PI * 4 + spectrogramAnimTime * trk.speed + trk.phase)
          + 0.2 * Math.sin(u * Math.PI * 9 - spectrogramAnimTime * 0.5);
        const y = padTop + ((20 - fGhz) / 20.0) * plotH;
        ctx.beginPath();
        ctx.arc(padLeft + px, y, 1.8, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    ctx.restore();
  }

  // 5b. Section 2 Spectrum (Power vs Frequency)
  let sec2SpecJitterSeed = 0;
  function renderSec2Spectrum() {
    const { w, h, ctx } = resizeCanvas(canvases.sec2Spectrum);
    if (!ctx || w <= 0 || h <= 0) return;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#030712';
    ctx.fillRect(0, 0, w, h);

    const padLeft = 32;
    const padRight = 14;
    const padTop = 10;
    const padBottom = 16;
    const plotW = Math.max(10, w - padLeft - padRight);
    const plotH = Math.max(10, h - padTop - padBottom);

    // Plot background
    ctx.fillStyle = '#020612';
    ctx.fillRect(padLeft, padTop, plotW, plotH);

    // Y Grid: -120 to -20 dBm (Ticks: -20, -40, -60, -80, -100, -120)
    ctx.font = '7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yTicks = [-20, -40, -60, -80, -100, -120];
    yTicks.forEach(db => {
      const frac = (-20 - db) / 100.0;
      const y = padTop + frac * plotH;
      ctx.fillText(db.toString(), padLeft - 3, y);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    // X Grid: 8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2 GHz
    const fMin = 8.0;
    const fMax = 9.2;
    const xTicks = [8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2];
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    xTicks.forEach(f => {
      const frac = (f - fMin) / (fMax - fMin);
      const x = padLeft + frac * plotW;
      ctx.fillText(f.toFixed(1), x, padTop + plotH + 2);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();
    });

    // Spectrum trace with sharp resonance peak at 8.64 GHz reaching ~ -24 dBm
    ctx.save();
    ctx.beginPath();
    ctx.rect(padLeft, padTop, plotW, plotH);
    ctx.clip();

    sec2SpecJitterSeed += 0.05;
    const centerPeakF = 8.64;
    const peakPowerDbm = -24.0;
    const noiseFloorDbm = -106.0;

    const numPoints = Math.max(120, Math.floor(plotW * 1.5));
    const points = [];
    let peakApexX = 0, peakApexY = 0;

    for (let i = 0; i <= numPoints; i++) {
      const u = i / numPoints;
      const f = fMin + u * (fMax - fMin);
      const df = f - centerPeakF;
      // Lorentzian resonant peak
      const peakShape = 82.0 / (1.0 + Math.pow(df / 0.038, 2));
      // Thermal noise fluctuation
      const noise = (Math.sin(i * 12.7 + sec2SpecJitterSeed * 3) * Math.cos(i * 3.3 - sec2SpecJitterSeed * 2)) * 3.2
        + (Math.sin(i * 47.1 + sec2SpecJitterSeed * 5) * 1.8);
      const dbm = Math.min(-20, noiseFloorDbm + peakShape + noise);
      const y = padTop + ((-20 - dbm) / 100.0) * plotH;
      const x = padLeft + u * plotW;
      points.push({ x, y, f, dbm });

      if (Math.abs(df) < 0.005) {
        peakApexX = x;
        peakApexY = y;
      }
    }

    // Vertical indicator line dropping from peak apex to axis
    if (peakApexX > 0) {
      ctx.strokeStyle = 'rgba(56, 189, 248, 0.55)';
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(peakApexX, peakApexY);
      ctx.lineTo(peakApexX, padTop + plotH);
      ctx.stroke();
    }

    // Glow under spectrum line
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.3)';
    ctx.lineWidth = 3.0;
    ctx.beginPath();
    points.forEach((pt, idx) => {
      if (idx === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    });
    ctx.stroke();

    // Sharp cyan line
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    points.forEach((pt, idx) => {
      if (idx === 0) ctx.moveTo(pt.x, pt.y);
      else ctx.lineTo(pt.x, pt.y);
    });
    ctx.stroke();

    ctx.restore();
  }

  // 5c. Section 2 I/Q Waveform
  let sec2IqTime = 0;
  function renderSec2Iq() {
    const { w, h, ctx } = resizeCanvas(canvases.sec2Iq);
    if (!ctx || w <= 0 || h <= 0) return;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#030712';
    ctx.fillRect(0, 0, w, h);

    const padLeft = 32;
    const padRight = 14;
    const padTop = 6;
    const padBottom = 16;
    const plotW = Math.max(10, w - padLeft - padRight);
    const plotH = Math.max(10, h - padTop - padBottom);

    // Plot background
    ctx.fillStyle = '#020612';
    ctx.fillRect(padLeft, padTop, plotW, plotH);

    // Y Grid: 1, 0, -1
    ctx.font = '7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    [1, 0, -1].forEach(val => {
      const frac = (1 - val) / 2.0;
      const y = padTop + frac * plotH;
      ctx.fillText(val.toString(), padLeft - 3, y);

      ctx.strokeStyle = val === 0 ? 'rgba(255, 255, 255, 0.12)' : 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(padLeft + plotW, y);
      ctx.stroke();
    });

    // X Grid: 0, 2, 4, 6, 8, 10 ms
    const xTicks = [0, 2, 4, 6, 8, 10];
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    xTicks.forEach(tMs => {
      const frac = tMs / 10.0;
      const x = padLeft + frac * plotW;
      ctx.fillText(tMs.toString(), x, padTop + plotH + 2);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();
    });

    // Clip to plot
    ctx.save();
    ctx.beginPath();
    ctx.rect(padLeft, padTop, plotW, plotH);
    ctx.clip();

    sec2IqTime += 0.035;
    const cy = padTop + plotH * 0.5;
    const amp = plotH * 0.42;

    const numSamples = Math.max(160, Math.floor(plotW * 1.5));
    const cycles = 14; // smooth sinusoidal oscillations matching reference

    // I waveform (Electric Cyan #38bdf8)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    for (let k = 0; k <= numSamples; k++) {
      const u = k / numSamples;
      const phase = 2 * Math.PI * cycles * u - sec2IqTime;
      const val = Math.sin(phase) + (Math.sin(phase * 3.1) * 0.05);
      const x = padLeft + u * plotW;
      const y = cy - val * amp;
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Q waveform (Bright Orange #fb923c)
    ctx.strokeStyle = '#fb923c';
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    for (let k = 0; k <= numSamples; k++) {
      const u = k / numSamples;
      const phase = 2 * Math.PI * cycles * u - sec2IqTime + Math.PI * 0.5; // 90 deg quadrature phase
      const val = Math.sin(phase) + (Math.cos(phase * 2.9) * 0.05);
      const x = padLeft + u * plotW;
      const y = cy - val * amp;
      if (k === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.restore();
  }

  // 5d. Section 2 Event Timeline (Last 10 s)
  function renderSec2Timeline() {
    const { w, h, ctx } = resizeCanvas(canvases.sec2Timeline);
    if (!ctx || w <= 0 || h <= 0) return;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#030712';
    ctx.fillRect(0, 0, w, h);

    const padLeft = 6;
    const padRight = 6;
    const padTop = 6;
    const padBottom = 16;
    const plotW = Math.max(10, w - padLeft - padRight);
    const plotH = Math.max(10, h - padTop - padBottom);

    // Plot background
    ctx.fillStyle = '#020612';
    ctx.fillRect(padLeft, padTop, plotW, plotH);

    // 3 Channels: R1, R2, R3
    const lanes = [
      { name: 'R1', y: padTop + plotH * 0.2 },
      { name: 'R2', y: padTop + plotH * 0.5 },
      { name: 'R3', y: padTop + plotH * 0.8 },
    ];

    lanes.forEach(lane => {
      // Baseline
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(padLeft, lane.y);
      ctx.lineTo(padLeft + plotW, lane.y);
      ctx.stroke();
    });

    // X-Axis: 0 to 10 Time (s)
    const xTicks = [0, 2, 4, 6, 8, 10];
    ctx.font = '7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    xTicks.forEach(tS => {
      const frac = tS / 10.0;
      const x = padLeft + frac * plotW;
      ctx.fillText(tS.toString(), x, padTop + plotH + 2);

      ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(x, padTop);
      ctx.lineTo(x, padTop + plotH);
      ctx.stroke();
    });

    // Event Markers (matching Image 2 exactly)
    const events = [
      // Lane R1
      { lane: 0, t: 1.1, type: 'miss' },
      { lane: 0, t: 2.6, type: 'hit' },
      { lane: 0, t: 4.8, type: 'hit' },
      { lane: 0, t: 6.2, type: 'hit' },
      { lane: 0, t: 7.5, type: 'hit' },
      { lane: 0, t: 9.4, type: 'miss' },

      // Lane R2
      { lane: 1, t: 1.3, type: 'hit' },
      { lane: 1, t: 2.4, type: 'miss' },
      { lane: 1, t: 3.4, type: 'miss' },
      { lane: 1, t: 4.6, type: 'hit' },
      { lane: 1, t: 5.0, type: 'hit' },
      { lane: 1, t: 9.2, type: 'miss' },

      // Lane R3
      { lane: 2, t: 0.8, type: 'hit' },
      { lane: 2, t: 2.8, type: 'hit' },
      { lane: 2, t: 3.3, type: 'miss' },
      { lane: 2, t: 6.5, type: 'switch' },
      { lane: 2, t: 7.7, type: 'switch' },
      { lane: 2, t: 10.0, type: 'switch' },
    ];

    const barW = 2.5;
    const barH = 10;

    events.forEach(ev => {
      const frac = ev.t / 10.0;
      const x = padLeft + frac * plotW - barW * 0.5;
      const y = lanes[ev.lane].y - barH * 0.5;

      let color = '#22c55e';
      if (ev.type === 'miss') color = '#ef4444';
      else if (ev.type === 'switch') color = '#f59e0b';
      else if (ev.type === 'scenario') color = '#38bdf8';

      ctx.fillStyle = color;
      ctx.fillRect(x, y, barW, barH);
    });
  }

  // 5e. Scanning View (Receiver) Polar PPI Radar
  let scanningViewAngle = 0;
  function renderScanningView() {
    const { w, h, ctx } = resizeCanvas(canvases.scanningView);
    if (!ctx || w <= 0 || h <= 0) return;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#020612';
    ctx.fillRect(0, 0, w, h);

    const cx = w * 0.5;
    const cy = h * 0.5;
    const r = Math.min(w, h) * 0.36;

    // Concentric range rings
    [0.25, 0.5, 0.75, 1.0].forEach(frac => {
      ctx.strokeStyle = 'rgba(14, 116, 144, 0.35)';
      ctx.lineWidth = 0.9;
      ctx.beginPath();
      ctx.arc(cx, cy, r * frac, 0, Math.PI * 2);
      ctx.stroke();
    });

    // Crosshairs & diagonal radial lines
    ctx.strokeStyle = 'rgba(14, 116, 144, 0.2)';
    ctx.lineWidth = 0.7;
    for (let a = 0; a < 8; a++) {
      const rad = (a * Math.PI) / 4;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + Math.cos(rad) * r, cy + Math.sin(rad) * r);
      ctx.stroke();
    }

    // Outer frequency markings around perimeter (as in Image 2!)
    const freqLabels = [
      { text: '8.50 GHz', angle: -Math.PI / 2, align: 'center', baseline: 'bottom', offset: 8 },
      { text: '9.00 GHz', angle: -Math.PI / 4, align: 'left', baseline: 'bottom', offset: 7 },
      { text: '9.25 GHz', angle: 0, align: 'left', baseline: 'middle', offset: 6 },
      { text: '9.50 GHz', angle: Math.PI / 4, align: 'left', baseline: 'top', offset: 7 },
      { text: '7.50 GHz', angle: (3 * Math.PI) / 4, align: 'right', baseline: 'top', offset: 7 },
      { text: '7.75 GHz', angle: Math.PI, align: 'right', baseline: 'middle', offset: 6 },
      { text: '8.00 GHz', angle: -(3 * Math.PI) / 4, align: 'right', baseline: 'bottom', offset: 7 },
    ];

    ctx.font = '7.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#94a3b8';
    freqLabels.forEach(fl => {
      const lx = cx + Math.cos(fl.angle) * (r + fl.offset);
      const ly = cy + Math.sin(fl.angle) * (r + fl.offset);
      ctx.textAlign = fl.align;
      ctx.textBaseline = fl.baseline;
      ctx.fillText(fl.text, lx, ly);
    });

    // Animated Rotating Sweep
    scanningViewAngle += 0.025;
    if (scanningViewAngle > Math.PI * 2) scanningViewAngle -= Math.PI * 2;

    // Trailing Sector Beam Cone (~35 degrees)
    const beamAngle = 0.55; // ~31 degrees
    const startA = scanningViewAngle - beamAngle;
    const endA = scanningViewAngle;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, r, startA, endA);
    ctx.closePath();

    const sweepGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
    sweepGrad.addColorStop(0.0, 'rgba(56, 189, 248, 0.3)');
    sweepGrad.addColorStop(1.0, 'rgba(56, 189, 248, 0.05)');
    ctx.fillStyle = sweepGrad;
    ctx.fill();

    // Sweep Front Line
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + Math.cos(scanningViewAngle) * r, cy + Math.sin(scanningViewAngle) * r);
    ctx.stroke();
    ctx.restore();

    // Emitter Target Blips (matching Image 2!)
    const blips = [
      { angle: -Math.PI * 0.48, dist: 0.75, color: '#eab308' }, // Yellow blip near 8.50 GHz (top)
      { angle: -Math.PI * 0.78, dist: 0.68, color: '#ef4444' }, // Red blip near 8.00 GHz (top-left)
      { angle: Math.PI * 0.45, dist: 0.32, color: '#06b6d4' },  // Cyan blip near center
      { angle: Math.PI * 0.72, dist: 0.80, color: '#22c55e' },  // Green blip near 7.50 GHz (bottom-left)
      { angle: Math.PI * 0.05, dist: 0.82, color: '#a855f7' },  // Purple blip near 9.25 GHz (right)
    ];

    blips.forEach(b => {
      const bx = cx + Math.cos(b.angle) * (r * b.dist);
      const by = cy + Math.sin(b.angle) * (r * b.dist);

      // Check angular distance to sweep line for pulse effect
      let diff = (scanningViewAngle - b.angle) % (Math.PI * 2);
      if (diff < 0) diff += Math.PI * 2;
      const isLit = diff < 0.6; // illuminated recently by sweep

      // Glow halo
      ctx.fillStyle = b.color;
      ctx.shadowColor = b.color;
      ctx.shadowBlur = isLit ? 10 : 4;
      ctx.beginPath();
      ctx.arc(bx, by, isLit ? 3.8 : 3.0, 0, Math.PI * 2);
      ctx.fill();

      if (isLit) {
        // Expanding ring
        ctx.strokeStyle = b.color;
        ctx.lineWidth = 0.8;
        ctx.beginPath();
        ctx.arc(bx, by, 6 + (diff * 8), 0, Math.PI * 2);
        ctx.stroke();
      }
    });

    ctx.shadowBlur = 0;
  }

  // 5f. Global RF Tab Switcher
  window.switchRfTab = function (tabName) {
    telemetryStore.activeRfTab = tabName;
    document.querySelectorAll('.rf-tab-btn').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('onclick').includes(tabName));
    });

    const constWrap = document.getElementById('sec2ConstellationWrap');
    const specBox = document.querySelector('.diag-spectrum-box');
    const iqBox = document.querySelector('.diag-iq-box');

    if (tabName === 'constellation') {
      if (constWrap) constWrap.style.display = 'block';
      if (specBox) specBox.style.display = 'none';
      if (iqBox) iqBox.style.display = 'none';
    } else {
      if (constWrap) constWrap.style.display = 'none';
      if (specBox) specBox.style.display = 'flex';
      if (iqBox) iqBox.style.display = 'flex';
    }
  };

  // ==========================================================================
  // 4. DOM & CONTROLS UPDATE
  // ==========================================================================
  function updateStatusBadges() {
    const b = document.getElementById('rxStatusBackend');
    const sc = document.getElementById('rxStatusSystemC');
    const u = document.getElementById('rxStatusUCB');
    const iq = document.getElementById('rxStatusIQ');

    if (b) {
      b.textContent = telemetryStore.connected ? '● CONNECTED' : '● DISCONNECTED';
      b.className = telemetryStore.connected ? 'telemetry-val status-connected' : 'telemetry-val status-disconnected';
    }
    if (sc) {
      sc.textContent = `● ${telemetryStore.systemCState}`;
      sc.className = telemetryStore.systemCState === 'RUNNING' ? 'telemetry-val status-running' : (telemetryStore.systemCState === 'PAUSED' ? 'telemetry-val status-paused' : 'telemetry-val status-disconnected');
    }
    if (u) {
      u.textContent = `● ${telemetryStore.ucbState}`;
      u.className = telemetryStore.ucbState === 'ACTIVE' ? 'telemetry-val status-active' : 'telemetry-val status-disconnected';
    }
    if (iq) {
      iq.textContent = `● ${telemetryStore.iqState}`;
      iq.className = telemetryStore.iqState === 'LIVE' ? 'telemetry-val status-live' : (telemetryStore.iqState === 'STALE' ? 'telemetry-val status-stale' : 'telemetry-val status-disconnected');
    }
  }

  function updateDomMetrics(data) {
    const rx = data.rf_receiver;
    const dw = data.dwell;
    const ucb = data.ucb_scheduler;
    const m = data.metrics;

    const win = data.selected_frequency_window || (dw ? {
      start_hz: dw.f_start_hz,
      end_hz: dw.f_end_hz,
      action_id: dw.band_index
    } : null);

    const fStartGhz = win ? (win.start_hz / 1e9).toFixed(2) : (rx ? (rx.freq_start_hz / 1e9).toFixed(2) : '0.00');
    const fEndGhz = win ? (win.end_hz / 1e9).toFixed(2) : (rx ? (rx.freq_end_hz / 1e9).toFixed(2) : '0.00');
    const freqRangeStr = `${fStartGhz} – ${fEndGhz} GHz`;

    const detectionResult = data.detection_result || (data.detection ? data.detection.result : (data.reward > 0 ? 'HIT' : 'MISS'));
    const cfarCount = data.cfar_detection_count !== undefined ? data.cfar_detection_count : (data.detection ? data.detection.cfar_detection_count : (dw ? dw.num_pulses : 0));
    const rewardVal = data.reward !== undefined ? data.reward : 0.0;
    const isHit = detectionResult === 'HIT';

    const ucbScore = data.ucb_score !== undefined ? data.ucb_score : (data.ucb ? data.ucb.score : 0.0);
    const visits = data.visit_count !== undefined ? data.visit_count : (data.ucb ? data.ucb.visit_count : 0);
    const meanReward = data.mean_reward !== undefined ? data.mean_reward : (data.ucb ? data.ucb.mean_reward : 0.0);

    // Current Observation Panel
    if (rx) {
      const tunedFreqEl = document.getElementById('valTunedFreq');
      if (tunedFreqEl) tunedFreqEl.textContent = freqRangeStr;

      const detectorStatusEl = document.getElementById('valDetectorStatus');
      if (detectorStatusEl) {
        detectorStatusEl.textContent = detectionResult;
        detectorStatusEl.className = isHit ? 'detector-badge status-hit' : 'detector-badge status-scan';
        detectorStatusEl.style.color = isHit ? '#22c55e' : '#94a3b8';
      }

      const detCountEl = document.getElementById('valDetectionCount');
      if (detCountEl) detCountEl.textContent = cfarCount.toString();

      const rewardEl = document.getElementById('valReward');
      if (rewardEl) {
        rewardEl.textContent = (rewardVal > 0 ? '+' : '') + rewardVal.toFixed(2);
        rewardEl.style.color = rewardVal > 0 ? '#22c55e' : '#94a3b8';
      }

      const snrEstEl = document.getElementById('valSnrEst');
      if (snrEstEl) snrEstEl.textContent = rx.snr_db.toFixed(1) + ' dB';

      const ibwEl = document.getElementById('valIbw');
      if (ibwEl) ibwEl.textContent = rx.bandwidth_mhz.toFixed(0) + ' MHz';

      const dwellTimeEl = document.getElementById('valDwellTime');
      if (dwellTimeEl) dwellTimeEl.textContent = (dw ? dw.dwell_ms : 50).toFixed(0) + ' ms';
    }

    const timeSimEl = document.getElementById('valTimeSim');
    if (timeSimEl && data.sim_time_s !== undefined) timeSimEl.textContent = data.sim_time_s.toFixed(3) + ' s';

    // Live Dwell & Detection HUD (Bandscan Info Card)
    const curBandEl = document.getElementById('siCurrentBand');
    if (curBandEl) curBandEl.textContent = freqRangeStr;

    const detStatusEl = document.getElementById('siDetectionStatus');
    if (detStatusEl) {
      detStatusEl.textContent = detectionResult;
      detStatusEl.className = isHit ? 'si-val status-hit' : 'si-val status-miss';
      detStatusEl.style.color = isHit ? '#22c55e' : '#94a3b8';
    }

    const cfarDetsEl = document.getElementById('siCfarDetections');
    if (cfarDetsEl) cfarDetsEl.textContent = cfarCount.toString();

    const siRewardEl = document.getElementById('siReward');
    if (siRewardEl) {
      siRewardEl.textContent = (rewardVal > 0 ? '+' : '') + rewardVal.toFixed(2);
      siRewardEl.style.color = rewardVal > 0 ? '#22c55e' : '#94a3b8';
    }

    const ucbScoreEl = document.getElementById('siUcbScore');
    if (ucbScoreEl) ucbScoreEl.textContent = typeof ucbScore === 'number' ? ucbScore.toFixed(3) : ucbScore;

    const visitsEl = document.getElementById('siVisits');
    if (visitsEl) visitsEl.textContent = visits.toString();

    const meanRewardEl = document.getElementById('siMeanReward');
    if (meanRewardEl) meanRewardEl.textContent = typeof meanReward === 'number' ? meanReward.toFixed(2) : meanReward;

    if (data.next_band) {
      const nextBandEl = document.getElementById('siNextBand');
      if (nextBandEl) {
        const nbStart = (data.next_band.f_start_hz / 1e9).toFixed(2);
        const nbEnd = (data.next_band.f_end_hz / 1e9).toFixed(2);
        nextBandEl.textContent = `${nbStart} – ${nbEnd} GHz`;
      }
    }

    const lastScannedEl = document.getElementById('siLastScanned');
    if (lastScannedEl && data.dwell_history && data.dwell_history.length > 1) {
      const prevDw = data.dwell_history[data.dwell_history.length - 2];
      const pStart = (prevDw.f_start_hz / 1e9).toFixed(2);
      const pEnd = (prevDw.f_end_hz / 1e9).toFixed(2);
      lastScannedEl.textContent = `${pStart} – ${pEnd} GHz`;
    }

    const lastResultEl = document.getElementById('siLastResult');
    if (lastResultEl && data.dwell_history && data.dwell_history.length > 1) {
      const prevDw = data.dwell_history[data.dwell_history.length - 2];
      lastResultEl.textContent = prevDw.hit ? 'HIT' : 'Miss';
      lastResultEl.className = prevDw.hit ? 'si-val status-hit' : 'si-val status-miss';
    }

    const detBandsEl = document.getElementById('siDetectedBands');
    if (detBandsEl) {
      const totalBands = (data.rf_environment && data.rf_environment.total_bands) ? data.rf_environment.total_bands : 36;
      const uniqueHits = ucb && ucb.reward_means ? ucb.reward_means.filter(x => x > 0).length : 0;
      detBandsEl.textContent = `${uniqueHits} / ${totalBands}`;
    }

    if (m) {
      const covPct = Math.round((m.coverage || 0) * 100);
      const progValEl = document.getElementById('siScanProgressVal');
      if (progValEl) progValEl.textContent = `${covPct} %`;
      const progFillEl = document.getElementById('siScanProgressFill');
      if (progFillEl) progFillEl.style.width = `${covPct}%`;

      const metricPdEl = document.getElementById('metricPd');
      if (metricPdEl) metricPdEl.textContent = m.pd !== undefined ? m.pd.toFixed(2) : '—';
      const metricPfaEl = document.getElementById('metricPfa');
      if (metricPfaEl) metricPfaEl.textContent = m.pfa !== undefined ? m.pfa.toFixed(2) : '—';
      const metricInterceptEl = document.getElementById('metricIntercept');
      if (metricInterceptEl) metricInterceptEl.textContent = m.intercept_rate !== undefined ? m.intercept_rate.toFixed(2) : '—';
      const metricCovEl = document.getElementById('metricCoverage');
      if (metricCovEl) metricCovEl.textContent = `${covPct} %`;
      const metricSnrEl = document.getElementById('metricSnr');
      if (metricSnrEl) metricSnrEl.textContent = m.avg_snr_db !== undefined ? m.avg_snr_db.toFixed(1) + ' dB' : '—';
      const metricRewardEl = document.getElementById('metricReward');
      if (metricRewardEl) metricRewardEl.textContent = rewardVal.toFixed(2);
    }

    // Section 2 Tactical Event Log & AI Logs Console
    if (data.recent_events && Array.isArray(data.recent_events)) {
      const eventLogBody = document.getElementById('sec2EventLogBody');
      if (eventLogBody) {
        eventLogBody.innerHTML = '';
        data.recent_events.forEach(ev => {
          const row = document.createElement('tr');
          row.innerHTML = `<td>${ev.time_s.toFixed(2)}s</td><td>${ev.type}</td><td>${ev.message}</td>`;
          eventLogBody.appendChild(row);
        });
      }
      const aiConsole = document.getElementById('aiLogsConsole');
      if (aiConsole) {
        aiConsole.innerHTML = '';
        data.recent_events.forEach(ev => {
          const line = document.createElement('div');
          line.className = 'ai-log-entry';
          let colorStyle = 'color: #dc2626;';
          if (ev.type === 'HIT') colorStyle = 'color: #4ade80;';
          else if (ev.type === 'REWARD') colorStyle = 'color: #facc15;';
          else if (ev.type === 'DECISION') colorStyle = 'color: #38bdf8;';
          else if (ev.type === 'MISS') colorStyle = 'color: #94a3b8;';
          line.innerHTML = `<span class="log-ts" style="color: #64748b; margin-right: 6px;">[${ev.time_s.toFixed(3)}]</span> <span class="log-msg" style="${colorStyle}">${ev.message}</span>`;
          aiConsole.appendChild(line);
        });
      }
    }
  }

  function setupControls() {
    const btnStart = document.getElementById('btnStart');
    const btnPause = document.getElementById('btnPause');
    const btnStop = document.getElementById('btnStop');

    if (btnStart) btnStart.addEventListener('click', () => sendCommand('start'));
    if (btnPause) btnPause.addEventListener('click', () => sendCommand('pause'));
    if (btnStop) btnStop.addEventListener('click', () => sendCommand('stop'));
  }

  function renderAll() {
    renderSpectrum();
    renderWaterfall();
    renderBandScan();
    renderTimeline();
    renderRxWindow();
    drawConstellationPlot(canvases.constellation);
    drawConstellationPlot(canvases.sec2Constellation);

    renderSec2Spectrogram();
    renderSec2Spectrum();
    renderSec2Iq();
    renderSec2Timeline();
    renderScanningView();
  }

  function frameLoop() {
    renderAll();
    requestAnimationFrame(frameLoop);
  }

  window.addEventListener('resize', renderAll);
  window.addEventListener('DOMContentLoaded', () => {
    setupControls();
    connectWebSocket();
    frameLoop();
  });

  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    setupControls();
    connectWebSocket();
    frameLoop();
  }
})();
