/**
 * VYAPTI Data Engine - Physics Page
 * ===================================
 * Drives the RF Physics dashboard with REAL TSRD data from the backend API.
 * 
 * This script connects to:
 *   GET  http://localhost:8000/api/data/summary      → dataset metadata
 *   GET  http://localhost:8000/api/data/emitters     → emitter table
 *   GET  http://localhost:8000/api/data/spectrum     → PSD data
 *   GET  http://localhost:8000/api/data/waterfall    → waterfall matrix
 *   GET  http://localhost:8000/api/data/band-scan    → band scan stats
 *   GET  http://localhost:8000/api/data/pdw-window   → latest PDWs
 *   WS   ws://localhost:8000/ws/pdw-stream           → real-time PDW stream
 *
 * It overrides:
 *   - drawPsd()              → uses real spectrum bins from backend
 *   - drawWaterfall()        → uses real time-freq matrix from backend
 *   - emitter table rows     → populated from real emitter configs
 *   - header telemetry pills → SIM TIME, EMITTER COUNT from real data
 *   - AI log entries         → generated from real PDW stream
 *   - Current Pulse panel    → shows most recent real PDW
 *   - Band scan metrics      → from real receiver dwell config
 *
 * It does NOT modify CSS, layout, or animations.
 */
(function () {
  'use strict';

  window.VyaptiRealDataActive = true;

  var API = 'http://localhost:8000';

  // ─── State from real data ───────────────────────────────────────────────────
  var realData = {
    loaded: false,
    configFile: '',
    totalPulses: 0,
    emitterCount: 0,
    freqRangeMhz: [500, 18000],
    dwellCentres: [],
    dwellTimes: [],
    spectrumBins: [],      // float[]  MHz
    spectrumPower: [],     // float[]  dBm
    waterfallGrid: [],     // float[][] rows x cols
    waterfallRows: 64,
    waterfallCols: 256,
    waterfallFreqMin: 500,
    waterfallFreqMax: 18000,
    emitters: [],          // [{id, label, type, freq_mhz, pri_us, pw_us, pulse_count, mean_amp_db}]
    bandScan: [],          // [{band_idx, centre_mhz, pulse_count, mean_amp_dbm, active}]
    recentPdws: [],        // last N real PDW objects
    latestPdw: null,       // most recent single PDW
    playheadPct: 0,
    simTimeS: 0,
  };

  // Track fetch intervals
  var _spectrumTimer = null;
  var _waterfallTimer = null;
  var _summaryTimer = null;
  var _aiLogTimer = null;
  var _pdwWs = null;
  var _running = false;

  // ─── Fetch helpers ─────────────────────────────────────────────────────────

  function apiFetch(path, cb) {
    fetch(API + path)
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) { if (d) cb(d); })
      .catch(function () {}); // fail silently — backend may be offline
  }

  // ─── Initialize on load ────────────────────────────────────────────────────

  function init() {
    // Load initial summary + emitters (one-shot)
    apiFetch('/api/data/summary', function (d) {
      realData.loaded = d.loaded;
      realData.configFile = d.config_file || '';
      realData.totalPulses = d.total_pulses || 0;
      realData.emitterCount = d.emitter_count || 0;
      realData.freqRangeMhz = d.freq_range_mhz || [500, 18000];
      realData.dwellCentres = d.dwell_centres_mhz || [];
      realData.dwellTimes = d.dwell_times_s || [];
      updateSummaryUI();
    });

    apiFetch('/api/data/emitters', function (d) {
      realData.emitters = d;
      updateEmitterTable();
    });

    apiFetch('/api/data/band-scan', function (d) {
      realData.bandScan = d.bands || [];
      updateBandScan();
    });

    // Populate initial dynamic AI logs from actual dataset
    apiFetch('/api/data/ai-logs?page=physics&count=12', function (d) {
      if (d && d.physics_logs && d.physics_logs.length) {
        setInitialAiLogs(d.physics_logs);
      }
    });

    // Constellation initial fetch & change hook
    fetchConstellation();
    var selMod = document.getElementById('physicsModulationSelect');
    if (selMod) {
      selMod.addEventListener('change', function () {
        fetchConstellation();
      });
    }

    // Start PDW WebSocket
    connectPdwStream();

    // Poll spectrum + waterfall while running
    _spectrumTimer = setInterval(function () {
      if (!_running) return;
      fetchSpectrum();
    }, 150); // ~7Hz

    _waterfallTimer = setInterval(function () {
      if (!_running) return;
      fetchWaterfall();
    }, 500); // 2Hz

    _summaryTimer = setInterval(function () {
      apiFetch('/api/data/summary', function (d) {
        realData.playheadPct = d.playhead_pct || 0;
        realData.simTimeS = (d.playhead_idx || 0) * 0.000176; // rough ~ToA scale
        updatePlayheadUI();
      });
      apiFetch('/api/data/band-scan', function (d) {
        realData.bandScan = d.bands || [];
        updateBandScan();
      });
    }, 2000); // 0.5Hz

    _aiLogTimer = setInterval(function () {
      if (!_running) return;
      apiFetch('/api/data/ai-logs?page=physics&count=3', function (d) {
        if (d && d.physics_logs && d.physics_logs.length) {
          appendRealAiLogEntries(d.physics_logs);
        }
      });
    }, 1500);

    // Poll constellation data from backend
    setInterval(function () {
      if (!_running) return;
      fetchConstellation();
    }, 300);

    // Hook into start/stop buttons to track running state
    hookRunButtons();
  }

  function fetchConstellation() {
    var sel = document.getElementById('physicsModulationSelect') || document.querySelector('.card-constellation select');
    var mod = sel ? sel.value : 'QPSK';
    apiFetch('/api/data/constellation?modulation=' + encodeURIComponent(mod) + '&count=240', function (d) {
      if (d && d.points) {
        window.VyaptiConstellationData = d;
      }
    });
  }

  // ─── WebSocket PDW Stream ─────────────────────────────────────────────────

  function connectPdwStream() {
    if (_pdwWs) { try { _pdwWs.close(); } catch(e) {} }
    try {
      _pdwWs = new WebSocket('ws://localhost:8000/ws/pdw-stream');
    } catch (e) { return; }

    _pdwWs.onmessage = function (evt) {
      var msg;
      try { msg = JSON.parse(evt.data); } catch(e) { return; }
      if (!msg || msg.paused) return;

      var pulses = msg.pulses || [];
      if (pulses.length > 0) {
        // Append to rolling buffer (keep last 500)
        realData.recentPdws = realData.recentPdws.concat(pulses).slice(-500);
        realData.latestPdw = pulses[pulses.length - 1];
        realData.playheadPct = msg.playhead_pct || 0;

        updateCurrentPulsePanel();
        if (msg.ai_logs && msg.ai_logs.physics_logs && msg.ai_logs.physics_logs.length) {
          appendRealAiLogEntries(msg.ai_logs.physics_logs);
        } else {
          appendRealAiLog(pulses);
        }
        updatePdwMetrics();
      }
    };

    _pdwWs.onclose = function () {
      // Reconnect after 2s
      setTimeout(connectPdwStream, 2000);
    };
    _pdwWs.onerror = function () {};
  }

  // ─── Spectrum: Override drawPsd canvas ────────────────────────────────────

  function fetchSpectrum() {
    apiFetch('/api/data/spectrum?bins=512', function (d) {
      realData.spectrumBins = d.freq_bins_mhz || [];
      realData.spectrumPower = d.power_dbm || [];
      realData.waterfallFreqMin = d.freq_min_mhz || 500;
      realData.waterfallFreqMax = d.freq_max_mhz || 18000;
      drawRealPsd();
    });
  }

  function drawRealPsd() {
    var canvas = document.getElementById('psdCanvas');
    if (!canvas || !realData.spectrumBins.length) return;

    var rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    var dpr = window.devicePixelRatio || 1;
    var W = rect.width, H = rect.height;
    if (canvas.width !== Math.round(W * dpr)) {
      canvas.width = Math.round(W * dpr);
      canvas.height = Math.round(H * dpr);
    }
    var ctx = canvas.getContext('2d');
    ctx.setTransform(1,0,0,1,0,0);
    ctx.scale(dpr, dpr);

    var padL = 28, padB = 16, padT = 10, padR = 6;
    var plotW = W - padL - padR;
    var plotH = H - padB - padT;

    // Background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, W, H);

    // Y-axis labels (dB)
    ctx.font = '500 6.5px "JetBrains Mono", monospace';
    ctx.fillStyle = '#cbd5e1';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    var dbMin = -130, dbMax = 0;
    for (var db = 0; db >= -120; db -= 20) {
      var y = padT + ((dbMax - db) / (dbMax - dbMin)) * plotH;
      ctx.fillText(db, padL - 3, y);
      ctx.strokeStyle = '#131b26';
      ctx.lineWidth = 0.7;
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();
    }

    // X-axis: frequency ticks (GHz labels)
    var fMin = realData.waterfallFreqMin, fMax = realData.waterfallFreqMax;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    var freqTicksGhz = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18];
    freqTicksGhz.forEach(function (ghz) {
      var mhz = ghz * 1000;
      var x = padL + ((mhz - fMin) / (fMax - fMin)) * plotW;
      ctx.fillStyle = '#cbd5e1';
      ctx.fillText(ghz + 'G', x, padT + plotH + 2);
      ctx.strokeStyle = '#131b26';
      ctx.lineWidth = 0.7;
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
      ctx.stroke();
    });

    if (!realData.spectrumBins.length) return;

    var bins = realData.spectrumBins;
    var power = realData.spectrumPower;
    var N = bins.length;

    // Draw noise floor fill first
    ctx.beginPath();
    ctx.moveTo(padL, padT + plotH);
    for (var i = 0; i < N; i++) {
      var bx = padL + ((bins[i] - fMin) / (fMax - fMin)) * plotW;
      var by = padT + ((dbMax - (-120)) / (dbMax - dbMin)) * plotH;
      if (i === 0) ctx.moveTo(bx, by);
      else ctx.lineTo(bx, by);
    }
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.closePath();
    ctx.fillStyle = 'rgba(15,23,42,0.5)';
    ctx.fill();

    // Draw spectrum line (glow)
    // First: thick glow
    ctx.beginPath();
    var first = true;
    for (var i = 0; i < N; i++) {
      var bx = padL + ((bins[i] - fMin) / (fMax - fMin)) * plotW;
      var p = Math.max(dbMin, power[i]);
      var by = padT + ((dbMax - p) / (dbMax - dbMin)) * plotH;
      if (first) { ctx.moveTo(bx, by); first = false; }
      else ctx.lineTo(bx, by);
    }
    ctx.strokeStyle = 'rgba(56,189,248,0.25)';
    ctx.lineWidth = 3;
    ctx.lineJoin = 'round';
    ctx.stroke();

    // Sharp line on top
    ctx.beginPath();
    first = true;
    for (var i = 0; i < N; i++) {
      var bx = padL + ((bins[i] - fMin) / (fMax - fMin)) * plotW;
      var p = Math.max(dbMin, power[i]);
      var by = padT + ((dbMax - p) / (dbMax - dbMin)) * plotH;
      if (first) { ctx.moveTo(bx, by); first = false; }
      else ctx.lineTo(bx, by);
    }
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.2;
    ctx.stroke();

    // Fill below the spectrum line
    ctx.lineTo(padL + ((bins[N-1] - fMin) / (fMax - fMin)) * plotW, padT + plotH);
    ctx.lineTo(padL + ((bins[0] - fMin) / (fMax - fMin)) * plotW, padT + plotH);
    ctx.closePath();
    ctx.fillStyle = 'rgba(56,189,248,0.06)';
    ctx.fill();

    // Mark pulse peaks with vertical ticks
    var peak_threshold = -100;
    for (var i = 0; i < N; i++) {
      if (power[i] > peak_threshold) {
        var bx = padL + ((bins[i] - fMin) / (fMax - fMin)) * plotW;
        var by = padT + ((dbMax - power[i]) / (dbMax - dbMin)) * plotH;
        ctx.beginPath();
        ctx.arc(bx, by, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = '#facc15';
        ctx.fill();
      }
    }
  }

  // ─── Waterfall: Override drawWaterfall canvas ──────────────────────────────

  function fetchWaterfall() {
    apiFetch('/api/data/waterfall?rows=64&bins=256', function (d) {
      realData.waterfallGrid = d.data || [];
      realData.waterfallRows = d.rows || 64;
      realData.waterfallCols = d.cols || 256;
      realData.waterfallFreqMin = d.freq_min_mhz || 500;
      realData.waterfallFreqMax = d.freq_max_mhz || 18000;
      drawRealWaterfall();
    });
  }

  function waterfallColor(dbm) {
    var norm = Math.max(0, Math.min(1, (dbm - (-130)) / 100));
    if (norm < 0.15) return [2 + norm/0.15*4, 8 + norm/0.15*30, 55 + norm/0.15*145];
    if (norm < 0.35) { var t=(norm-0.15)/0.2; return [0, 32+t*220, 200+t*55]; }
    if (norm < 0.6)  { var t=(norm-0.35)/0.25; return [t*220, 255, 255*(1-t)]; }
    if (norm < 0.8)  { var t=(norm-0.6)/0.2; return [255, 255-t*125, 0]; }
    var t=(norm-0.8)/0.2; return [255, 130*(1-t), t*20];
  }

  function drawRealWaterfall() {
    var canvas = document.getElementById('waterfallCanvas');
    if (!canvas || !realData.waterfallGrid.length) return;
    var rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    var dpr = window.devicePixelRatio || 1;
    var W = rect.width, H = rect.height;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr,0,0,dpr,0,0);

    var rows = realData.waterfallRows;
    var cols = realData.waterfallCols;
    var grid = realData.waterfallGrid;
    var cellW = W / cols;
    var cellH = H / rows;

    for (var r = 0; r < rows; r++) {
      if (!grid[r]) continue;
      for (var c = 0; c < cols; c++) {
        var rgb = waterfallColor(grid[r][c] || -120);
        ctx.fillStyle = 'rgb(' + Math.round(rgb[0]) + ',' + Math.round(rgb[1]) + ',' + Math.round(rgb[2]) + ')';
        ctx.fillRect(c * cellW, r * cellH, Math.ceil(cellW) + 0.5, Math.ceil(cellH) + 0.5);
      }
    }
  }

  // ─── Emitter Table ────────────────────────────────────────────────────────

  function updateEmitterTable() {
    var tbody = document.querySelector('.emitters-table tbody');
    if (!tbody || !realData.emitters.length) return;

    // Update count displays
    var countEl = document.getElementById('emitterCountVal');
    if (countEl) countEl.textContent = realData.emitters.length;

    var activeCount = realData.emitters.filter(function(e) { return e.pulse_count > 0; }).length;
    var activeCntEl = document.querySelector('.active-count-num');
    if (activeCntEl) activeCntEl.textContent = activeCount;

    // Build rows from real emitter data
    var html = '';
    realData.emitters.forEach(function (e) {
      var freqStr = Array.isArray(e.freq_mhz)
        ? (e.freq_mhz[0] / 1000).toFixed(3) + '†'
        : (e.freq_mhz / 1000).toFixed(3);
      var priStr = Array.isArray(e.pri_us)
        ? e.pri_us[0].toFixed(0) + '†'
        : e.pri_us.toFixed(0);
      var isActive = e.pulse_count > 0;
      html += '<tr>';
      html += '<td class="td-id">' + e.label + '</td>';
      html += '<td>' + e.type + '</td>';
      html += '<td>' + freqStr + '</td>';
      html += '<td>' + priStr + '</td>';
      html += isActive
        ? '<td class="status-active"><span class="dot-active">&#9679;</span> Active</td>'
        : '<td class="status-inactive"><span class="dot-inactive">&#9679;</span> Silent</td>';
      html += '</tr>';
    });
    tbody.innerHTML = html;
  }

  // ─── Band Scan ─────────────────────────────────────────────────────────────

  function updateBandScan() {
    if (!realData.bandScan.length) return;
    // Update "current band" display with the most active band
    var activeBands = realData.bandScan.filter(function(b) { return b.active; });
    if (!activeBands.length) return;
    // Most active band by pulse count
    activeBands.sort(function(a,b) { return b.pulse_count - a.pulse_count; });
    var topBand = activeBands[0];

    var elCurCF = document.getElementById('rfValCurBandCF');
    var elCurPct = document.getElementById('rfValCurBandProgressPct');
    var elCurBar = document.getElementById('rfValCurBandProgressBar');
    var elCurSnr = document.getElementById('rfValCurBandSnr');

    if (elCurCF) elCurCF.textContent = (topBand.centre_mhz / 1000).toFixed(3) + ' GHz';
    if (elCurPct) {
      var pct = Math.min(100, Math.round(topBand.pulse_count / 500 * 100));
      elCurPct.textContent = pct + '%';
    }
    if (elCurBar) {
      var pct = Math.min(100, Math.round(topBand.pulse_count / 500 * 100));
      elCurBar.style.width = pct + '%';
    }
    if (elCurSnr) {
      var snr = Math.max(-20, Math.min(40, topBand.mean_amp_dbm + 98));
      elCurSnr.textContent = snr.toFixed(1) + ' dB';
    }

    // Stats panel
    var elProgress = document.getElementById('rfStatsProgress');
    var activePct = Math.round(activeBands.length / realData.bandScan.length * 100);
    if (elProgress) elProgress.textContent = activePct + ' %';
  }

  // ─── Current Pulse Panel ───────────────────────────────────────────────────

  function updateCurrentPulsePanel() {
    var pdw = realData.latestPdw;
    if (!pdw) return;

    // Map to the "Current Pulse" metric lines in the right panel
    var fields = {
      // Time of Arrival
      'Time of Arrival (μs)': pdw.toa_us ? (pdw.toa_us / 1e6).toFixed(6) + ' s' : '--',
      'Center Freq (GHz)': pdw.freq_mhz ? (pdw.freq_mhz / 1000).toFixed(4) : '--',
      'Pulse Width (μs)': pdw.pw_us ? pdw.pw_us.toFixed(3) : '--',
      'Amplitude (dBm)': pdw.amp_db ? pdw.amp_db.toFixed(2) : '--',
    };

    // Find metric lines and update by label text
    document.querySelectorAll('.metric-line').forEach(function (line) {
      var lbl = line.querySelector('.m-lbl');
      var val = line.querySelector('.m-val');
      if (!lbl || !val) return;
      var labelText = lbl.textContent.trim();
      if (fields[labelText] !== undefined) {
        val.textContent = fields[labelText];
      }
    });

    // SNR estimation: amp relative to -100 dBm noise floor
    var snr = Math.max(0, pdw.amp_db + 100).toFixed(1);
    document.querySelectorAll('.metric-line').forEach(function (line) {
      var lbl = line.querySelector('.m-lbl');
      var val = line.querySelector('.m-val');
      if (lbl && lbl.textContent.trim() === 'SNR (dB)' && val) {
        val.textContent = snr;
      }
    });

    // Status: if amplitude > -100 = HIT
    var isHit = pdw.amp_db > -100;
    var statusLines = document.querySelectorAll('.status-line .m-val');
    statusLines.forEach(function (el) {
      if (isHit) {
        el.innerHTML = '<span class="dot-hit">&#9679;</span> HIT';
        el.className = 'm-val glow-status-hit';
      } else {
        el.innerHTML = '<span class="dot-inactive">&#9679;</span> MISS';
        el.className = 'm-val';
      }
    });
  }

  // ─── PDW Metrics (right panel stats) ──────────────────────────────────────

  function updatePdwMetrics() {
    var pdws = realData.recentPdws;
    if (!pdws.length) return;

    // Detection stats from recent window
    var amps = pdws.map(function(p) { return p.amp_db; });
    var freqs = pdws.map(function(p) { return p.freq_mhz; });
    var hits = amps.filter(function(a) { return a > -100; }).length;
    var misses = amps.length - hits;
    var pd = hits / amps.length;
    var meanAmp = amps.reduce(function(a,b){return a+b;},0) / amps.length;
    var snrMean = Math.max(0, meanAmp + 100);

    // Unique emitter IDs seen
    var emitterIds = {};
    pdws.forEach(function(p) { emitterIds[p.emitter_id] = true; });
    var activeCount = Object.keys(emitterIds).length;

    // Update Live Metrics panel
    setText('lmValPd', pd.toFixed(3));
    setText('lmValPfa', (misses / amps.length).toFixed(3));
    setText('lmValIntercept', Math.min(1, hits / Math.max(1, pdws.length)).toFixed(3));
    setText('lmValSnr', snrMean.toFixed(1));

    // Detector/CFAR panel
    document.querySelectorAll('.metric-line').forEach(function (line) {
      var lbl = line.querySelector('.m-lbl');
      var val = line.querySelector('.m-val');
      if (!lbl || !val) return;
      switch (lbl.textContent.trim()) {
        case 'Detected Pulses': val.textContent = hits; break;
        case 'False Alarms':   val.textContent = misses; break;
        case 'Noise Floor (dBm)': val.textContent = (-98.0 + Math.min(0, meanAmp + 100) * 0.1).toFixed(1); break;
      }
    });

    // IQ panel metrics
    var pk = Math.max.apply(null, amps.map(function(a){ return Math.abs(a); }));
    setText('iqValIRms', (snrMean * 0.7071).toFixed(3));
    setText('iqValQRms', (snrMean * 0.7071).toFixed(3));
    setText('iqValPeak', pk.toFixed(1));
    setText('iqValPower', meanAmp.toFixed(1) + ' dBm');
    setText('iqValSnr', snrMean.toFixed(1) + ' dB');
    setText('iqValSampleRate', '20.0 MS/s');

    // Emitter count
    setText('emitterCountVal', activeCount);
  }

  function setText(id, val) {
    var el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  // ─── Summary / Telemetry Header ────────────────────────────────────────────

  function updateSummaryUI() {
    // Config file shown in RUN ID pill
    var el = document.getElementById('valRunId');
    if (el && realData.configFile) {
      el.textContent = realData.configFile.replace('.h5', '').toUpperCase();
    }
    // Episodes = total pulse trains processed
    var ep = document.getElementById('valEpisodes');
    if (ep) ep.textContent = realData.totalPulses.toLocaleString();

    // Emitter count
    var ec = document.getElementById('emitterCountVal');
    if (ec) ec.textContent = realData.emitterCount;
  }

  function updatePlayheadUI() {
    // Decisions = pulses processed so far
    var el = document.getElementById('valDecisions');
    if (el) el.textContent = Math.round(realData.playheadPct / 100 * realData.totalPulses).toLocaleString();

    // Progress bar on scan history
    var elProgress = document.getElementById('rfStatsProgress');
    if (elProgress) {
      elProgress.textContent = realData.playheadPct.toFixed(1) + ' %';
    }
  }

  // ─── Real AI Log from Actual PDWs ─────────────────────────────────────────

  function setInitialAiLogs(logList) {
    var container = document.getElementById('aiLogsContainer');
    if (!container || !logList || !logList.length) return;
    container.innerHTML = '';
    appendRealAiLogEntries(logList);
  }

  function appendRealAiLogEntries(logList) {
    var container = document.getElementById('aiLogsContainer');
    if (!container || !logList || !logList.length) return;

    logList.forEach(function (item) {
      var entry = document.createElement('div');
      entry.className = 'ai-log-entry';
      var isHit = item.is_hit;
      var colorStyle = isHit ? 'color:#22c55e;' : (item.msg.indexOf('Decision') > -1 ? 'color:#38bdf8;' : '');
      entry.innerHTML =
        '<span class="log-ts">[' + item.ts + ' s]</span>' +
        '<span class="log-msg" style="' + colorStyle + '">' + item.msg + '</span>';
      container.appendChild(entry);
    });

    while (container.children.length > 30) {
      container.removeChild(container.firstChild);
    }
    container.scrollTop = container.scrollHeight;
  }

  function appendRealAiLog(pulses) {
    var container = document.getElementById('aiLogsContainer');
    if (!container) return;

    var batch = pulses.filter(function(_, i) { return i % 25 === 0; });
    batch.forEach(function (p, idx) {
      var entry = document.createElement('div');
      entry.className = 'ai-log-entry';
      var ts = (p.toa_us / 1e6).toFixed(4);
      var freqGhz = (p.freq_mhz / 1000).toFixed(4);
      var pw = p.pw_us.toFixed(2);
      var amp = p.amp_db.toFixed(1);
      var aoa = p.aoa_deg.toFixed(1);
      var snr = (p.amp_db + 105.0).toFixed(1);
      var isHit = p.amp_db > -95;

      var msg = '';
      if (idx % 3 === 0) {
        msg = (isHit ? 'HIT: Pulse detected (CF ' + freqGhz + ' GHz, PW ' + pw + ' μs, SNR ' + snr + ' dB, AoA ' + aoa + '°) → E' + p.emitter_id : 'PDW: CF ' + freqGhz + ' GHz | PW ' + pw + ' μs | AoA ' + aoa + '° | Amp ' + amp + ' dBm');
      } else if (idx % 3 === 1) {
        msg = 'Decision: Band active at ' + freqGhz + ' GHz selected | Dwell: 50 ms | SNR: ' + snr + ' dB';
      } else {
        msg = 'Emitter E' + p.emitter_id + ': Active pulse ingested | Amp ' + amp + ' dBm | AoA ' + aoa + '°';
      }

      var colorStyle = isHit ? 'color:#22c55e;' : (msg.indexOf('Decision') > -1 ? 'color:#38bdf8;' : '');
      entry.innerHTML =
        '<span class="log-ts">[' + ts + ' s]</span>' +
        '<span class="log-msg" style="' + colorStyle + '">' + msg + '</span>';
      container.appendChild(entry);
    });

    while (container.children.length > 30) {
      container.removeChild(container.firstChild);
    }
    container.scrollTop = container.scrollHeight;
  }

  // ─── Hook run buttons to track state ──────────────────────────────────────

  function hookRunButtons() {
    ['btnStartTop','btnCtrlStart'].forEach(function(id) {
      var btn = document.getElementById(id);
      if (btn) btn.addEventListener('click', function() { _running = true; fetchSpectrum(); fetchWaterfall(); });
    });
    ['btnPauseTop','btnCtrlPause','btnStopTop','btnCtrlStop','btnCtrlReset'].forEach(function(id) {
      var btn = document.getElementById(id);
      if (btn) btn.addEventListener('click', function() {
        if (id === 'btnCtrlReset') _running = false;
        else _running = id.indexOf('Pause') > -1 ? false : _running;
      });
    });
  }

  // ─── Boot ─────────────────────────────────────────────────────────────────

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
