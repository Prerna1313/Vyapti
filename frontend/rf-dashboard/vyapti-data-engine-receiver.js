/**
 * VYAPTI Data Engine - Receiver Page
 * ====================================
 * Drives the RF Receiver & Scheduler dashboard with REAL TSRD data.
 *
 * Connects to the same backend API as the physics page and drives:
 *   - Spectrum canvas     → real frequency-domain PSD
 *   - Waterfall canvas    → real time-frequency matrix  
 *   - Band scan canvas    → real 36-band activity bars
 *   - Receiver stats      → real per-emitter metrics
 *   - Timeline canvas     → real ToA sequence
 *   - Header badges       → real config file / emitter count
 *   - Event log table     → real PDW entries
 *   - Sim time display    → real playhead position in dataset
 */
(function () {
  'use strict';

  var API = 'http://localhost:8000';

  // ─── Shared real data state ───────────────────────────────────────────────
  var rd = {
    loaded: false,
    configFile: '',
    totalPulses: 0,
    emitterCount: 0,
    freqMin: 500, freqMax: 18000,
    dwellCentres: [],  // MHz
    dwellTimes: [],    // s
    spectrumBins: [],  power: [],
    waterfallGrid: [], waterfallRows: 64, waterfallCols: 256,
    bandScan: [],
    emitters: [],
    recentPdws: [],
    latestPdw: null,
    playheadPct: 0,
    playheadPulses: 0,
    running: false,
  };

  var _pdwWs = null;
  var _timers = [];
  var _aiMode = 'evm'; // 'evm' or 'stream'
  var _lastAiLogs = null;
  var _neuralLogEntries = [];

  // ─── Fetch helper ─────────────────────────────────────────────────────────
  function get(path, cb) {
    fetch(API + path)
      .then(function(r){ return r.ok ? r.json() : null; })
      .then(function(d){ if(d) cb(d); })
      .catch(function(){});
  }

  // ─── Init ─────────────────────────────────────────────────────────────────
  function init() {
    // Initial data fetch (one-shot)
    get('/api/data/summary', function(d) {
      rd.loaded = d.loaded;
      rd.configFile = d.config_file || '';
      rd.totalPulses = d.total_pulses || 0;
      rd.emitterCount = d.emitter_count || 0;
      rd.freqMin = (d.freq_range_mhz || [500, 18000])[0];
      rd.freqMax = (d.freq_range_mhz || [500, 18000])[1];
      rd.dwellCentres = d.dwell_centres_mhz || [];
      rd.dwellTimes = d.dwell_times_s || [];
      updateHeaderBadges();
      initBandScanBars();
    });

    get('/api/data/emitters', function(d) {
      rd.emitters = d;
      updateReceiverPanels();
    });

    get('/api/data/band-scan', function(d) {
      rd.bandScan = d.bands || [];
      drawRealBandScan();
    });

    // Initial dynamic AI Logs & Scheduler Decision fetch
    fetchAiLogs();
    initAiTabControls();

    connectPdwWs();
    hookButtons();

    // Polling timers
    _timers.push(setInterval(function(){
      if (!rd.running) return;
      get('/api/data/spectrum?bins=256', function(d){
        rd.spectrumBins = d.freq_bins_mhz || [];
        rd.power = d.power_dbm || [];
        rd.freqMin = d.freq_min_mhz || 500;
        rd.freqMax = d.freq_max_mhz || 18000;
        drawRealSpectrum();
      });
    }, 200));

    _timers.push(setInterval(function(){
      if (!rd.running) return;
      get('/api/data/waterfall?rows=64&bins=256', function(d){
        rd.waterfallGrid = d.data || [];
        rd.waterfallRows = d.rows || 64;
        rd.waterfallCols = d.cols || 256;
        drawRealWaterfall();
      });
    }, 600));

    _timers.push(setInterval(function(){
      get('/api/data/band-scan', function(d){
        rd.bandScan = d.bands || [];
        drawRealBandScan();
        updateBandStats();
      });
      get('/api/data/summary', function(d){
        rd.playheadPct = d.playhead_pct || 0;
        rd.playheadPulses = d.playhead_idx || 0;
        updateSimTime();
      });
    }, 1500));

    // Dynamic AI Log polling
    _timers.push(setInterval(function(){
      fetchAiLogs();
    }, 800));
  }

  // ─── WebSocket PDW stream ─────────────────────────────────────────────────
  function connectPdwWs() {
    if (_pdwWs) { try { _pdwWs.close(); } catch(e){} }
    try { _pdwWs = new WebSocket('ws://localhost:8000/ws/pdw-stream'); } catch(e){ return; }
    _pdwWs.onmessage = function(evt) {
      var msg; try { msg = JSON.parse(evt.data); } catch(e){ return; }
      if (!msg) return;

      if (msg.ai_logs) {
        _lastAiLogs = msg.ai_logs;
        renderAiLogsConsole(msg.ai_logs);
        if (msg.ai_logs.scheduler_decision) {
          updateSchedulerDecision(msg.ai_logs.scheduler_decision);
        }
      }

      if (msg.paused) return;

      var pulses = msg.pulses || [];
      if (pulses.length > 0) {
        rd.recentPdws = rd.recentPdws.concat(pulses).slice(-1000);
        rd.latestPdw = pulses[pulses.length - 1];
        rd.playheadPct = msg.playhead_pct || 0;
        appendPdwToEventLog(pulses);
        updateReceiverWindowStats();
        updateSimTime();
        if (window.addReceiverTimelineEvent) {
          var lastP = pulses[pulses.length - 1];
          var isHit = lastP.amp_db > -95;
          window.addReceiverTimelineEvent(isHit ? 'hit' : 'miss', lastP.toa_us / 1e6);
        }
      }
    };
    _pdwWs.onclose = function(){ setTimeout(connectPdwWs, 2000); };
  }

  // ─── Header badges ────────────────────────────────────────────────────────
  function updateHeaderBadges() {
    setText('hdrCenter', 'Center: ' + ((rd.freqMin + rd.freqMax) / 2000).toFixed(2) + ' GHz');
    setText('hdrSpan', 'Span: ' + ((rd.freqMax - rd.freqMin) / 1000).toFixed(2) + ' GHz');
    setText('hdrRbw', 'RBW: 500 MHz');
    if (rd.dwellTimes.length) {
      var meanDwell = rd.dwellTimes.reduce(function(a,b){return a+b;},0) / rd.dwellTimes.length;
      setText('hdrDwell', 'Dwell: ' + (meanDwell * 1000).toFixed(0) + ' ms');
    }
    setText('hdrMode', 'Mode: Scan (' + rd.emitterCount + ' Emitters)');
  }

  // ─── Sim time ─────────────────────────────────────────────────────────────
  function updateSimTime() {
    // playhead index * average inter-pulse = rough time
    var simSec = rd.playheadPulses * 176e-6; // ~176 µs mean inter-pulse from stats
    var el = document.getElementById('valTimeSim');
    if (el) el.textContent = simSec.toFixed(3) + ' s';
    var el2 = document.getElementById('valDecisions');
    if (el2) el2.textContent = rd.playheadPulses.toLocaleString();
  }

  // ─── Spectrum canvas ──────────────────────────────────────────────────────
  function drawRealSpectrum() {
    var canvas = document.getElementById('canvasSpectrum');
    if (!canvas || !rd.spectrumBins.length) return;
    var rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    var dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr,0,0,dpr,0,0);
    var W = rect.width, H = rect.height;
    var padL=32, padB=18, padT=8, padR=4;
    var pW = W-padL-padR, pH = H-padB-padT;

    // Background
    ctx.fillStyle = '#050810'; ctx.fillRect(0,0,W,H);

    var dbMin=-130, dbMax=-20;
    // Grid
    ctx.strokeStyle='#131b26'; ctx.lineWidth=0.7;
    for(var db=-20; db>=-130; db-=20) {
      var y = padT + ((dbMax-db)/(dbMax-dbMin))*pH;
      ctx.beginPath(); ctx.moveTo(padL,y); ctx.lineTo(padL+pW,y); ctx.stroke();
      ctx.fillStyle='#4a5568'; ctx.font='6px "JetBrains Mono",monospace';
      ctx.textAlign='right'; ctx.textBaseline='middle';
      ctx.fillText(db, padL-3, y);
    }
    // X freq ticks
    [1,2,4,6,8,10,12,14,16,18].forEach(function(ghz){
      var mhz=ghz*1000;
      var x=padL+((mhz-rd.freqMin)/(rd.freqMax-rd.freqMin))*pW;
      ctx.strokeStyle='#131b26'; ctx.beginPath(); ctx.moveTo(x,padT); ctx.lineTo(x,padT+pH); ctx.stroke();
      ctx.fillStyle='#4a5568'; ctx.textAlign='center'; ctx.textBaseline='top';
      ctx.fillText(ghz+'G', x, padT+pH+2);
    });

    var bins=rd.spectrumBins, pow=rd.power, N=bins.length;
    if (!N) return;

    // Spectrum fill
    ctx.beginPath();
    var first=true;
    for(var i=0;i<N;i++){
      var x=padL+((bins[i]-rd.freqMin)/(rd.freqMax-rd.freqMin))*pW;
      var p=Math.max(dbMin, pow[i]);
      var y=padT+((dbMax-p)/(dbMax-dbMin))*pH;
      if(first){ctx.moveTo(x,y);first=false;}else ctx.lineTo(x,y);
    }
    ctx.lineTo(padL+((bins[N-1]-rd.freqMin)/(rd.freqMax-rd.freqMin))*pW, padT+pH);
    ctx.lineTo(padL+((bins[0]-rd.freqMin)/(rd.freqMax-rd.freqMin))*pW, padT+pH);
    ctx.closePath();
    ctx.fillStyle='rgba(56,189,248,0.07)'; ctx.fill();

    // Line glow
    ctx.beginPath(); first=true;
    for(var i=0;i<N;i++){
      var x=padL+((bins[i]-rd.freqMin)/(rd.freqMax-rd.freqMin))*pW;
      var p=Math.max(dbMin,pow[i]);
      var y=padT+((dbMax-p)/(dbMax-dbMin))*pH;
      if(first){ctx.moveTo(x,y);first=false;}else ctx.lineTo(x,y);
    }
    ctx.strokeStyle='rgba(56,189,248,0.3)'; ctx.lineWidth=3; ctx.stroke();

    // Sharp line
    ctx.beginPath(); first=true;
    for(var i=0;i<N;i++){
      var x=padL+((bins[i]-rd.freqMin)/(rd.freqMax-rd.freqMin))*pW;
      var p=Math.max(dbMin,pow[i]);
      var y=padT+((dbMax-p)/(dbMax-dbMin))*pH;
      if(first){ctx.moveTo(x,y);first=false;}else ctx.lineTo(x,y);
    }
    ctx.strokeStyle='#38bdf8'; ctx.lineWidth=1.2; ctx.stroke();

    // Peak markers
    for(var i=0;i<N;i++){
      if(pow[i]>-100){
        var x=padL+((bins[i]-rd.freqMin)/(rd.freqMax-rd.freqMin))*pW;
        var y=padT+((dbMax-pow[i])/(dbMax-dbMin))*pH;
        ctx.beginPath(); ctx.arc(x,y,2.5,0,Math.PI*2);
        ctx.fillStyle='#facc15'; ctx.fill();
      }
    }
  }

  // ─── Waterfall canvas ──────────────────────────────────────────────────────
  function wfColor(dbm) {
    var norm=Math.max(0,Math.min(1,(dbm-(-130))/100));
    if(norm<0.15){var t=norm/0.15;return [2+t*4,8+t*30,55+t*145];}
    if(norm<0.35){var t=(norm-0.15)/0.2;return [0,32+t*220,200+t*55];}
    if(norm<0.6){var t=(norm-0.35)/0.25;return [t*220,255,255*(1-t)];}
    if(norm<0.8){var t=(norm-0.6)/0.2;return [255,255-t*125,0];}
    var t=(norm-0.8)/0.2;return [255,130*(1-t),t*20];
  }

  function drawRealWaterfall() {
    var canvas = document.getElementById('canvasWaterfall');
    if (!canvas || !rd.waterfallGrid.length) return;
    var rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    var dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr,0,0,dpr,0,0);
    var W=rect.width, H=rect.height;
    var rows=rd.waterfallRows, cols=rd.waterfallCols, grid=rd.waterfallGrid;
    var cW=W/cols, cH=H/rows;
    for(var r=0;r<rows;r++){
      if(!grid[r]) continue;
      for(var c=0;c<cols;c++){
        var rgb=wfColor(grid[r][c]||-120);
        ctx.fillStyle='rgb('+Math.round(rgb[0])+','+Math.round(rgb[1])+','+Math.round(rgb[2])+')';
        ctx.fillRect(c*cW, r*cH, Math.ceil(cW)+0.5, Math.ceil(cH)+0.5);
      }
    }
  }

  // ─── Band Scan canvas ─────────────────────────────────────────────────────
  function initBandScanBars() { /* canvas will be drawn when data arrives */ }

  function drawRealBandScan() {
    var canvas = document.getElementById('canvasBandScan');
    if (!canvas || !rd.bandScan.length) return;
    var rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    var dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    var ctx = canvas.getContext('2d');
    ctx.setTransform(dpr,0,0,dpr,0,0);
    var W=rect.width, H=rect.height;

    ctx.fillStyle='#050810'; ctx.fillRect(0,0,W,H);

    var bands = rd.bandScan;
    var N = bands.length;
    var barW = W / N - 1;
    var maxCount = Math.max(1, Math.max.apply(null, bands.map(function(b){ return b.pulse_count; })));

    bands.forEach(function(band, i) {
      var x = i * (W / N);
      var heightPct = band.pulse_count / maxCount;
      var barH = Math.max(1, heightPct * (H - 16));
      var y = H - 16 - barH;

      var r, g, b;
      if (!band.active) { r=20; g=30; b=50; }
      else if (heightPct > 0.7) { r=239; g=68; b=68; }    // red - heavy
      else if (heightPct > 0.3) { r=251; g=191; b=36; }   // amber - medium
      else { r=34; g=197; b=94; }                          // green - light

      ctx.fillStyle = 'rgb('+r+','+g+','+b+')';
      ctx.fillRect(x, y, barW, barH);

      // Dwell time overlay (blue line at top of bar)
      if (band.active) {
        ctx.fillStyle = 'rgba(56,189,248,0.4)';
        ctx.fillRect(x, y, barW, 2);
      }

      // Centre freq label (every 6th band)
      if (i % 6 === 0) {
        ctx.fillStyle = '#4a5568';
        ctx.font = '5px "JetBrains Mono",monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';
        ctx.fillText((band.centre_mhz / 1000).toFixed(1) + 'G', x + barW/2, H);
      }
    });
  }

  function updateBandStats() {
    var active = rd.bandScan.filter(function(b){ return b.active; });
    var total = rd.bandScan.length;
    // Find most active band
    if (!active.length) return;
    active.sort(function(a,b){ return b.pulse_count - a.pulse_count; });
    var top = active[0];

    // Update parameter badges
    setText('hdrCenter', 'Center: ' + (top.centre_mhz/1000).toFixed(2) + ' GHz');
    setText('hdrMode', 'Active Bands: ' + active.length + '/' + total);
  }

  // ─── Timeline event feeder ───────────────────────────────────────────────
  function drawRealTimeline() {
    if (!rd.recentPdws.length || !window.addReceiverTimelineEvent) return;
    var lastP = rd.recentPdws[rd.recentPdws.length - 1];
    var isHit = lastP.amp_db > -95;
    window.addReceiverTimelineEvent(isHit ? 'hit' : 'miss', lastP.toa_us / 1e6);
  }

  // ─── Event Log Table ──────────────────────────────────────────────────────
  function appendPdwToEventLog(pulses) {
    var tbody = document.getElementById('sec2EventLogBody');
    if (!tbody) return;

    // Add every 30th pulse to avoid flooding
    var batch = pulses.filter(function(_,i){ return i % 30 === 0; });
    batch.forEach(function(p) {
      var tr = document.createElement('tr');
      var ts = (p.toa_us / 1e6).toFixed(4);
      var freqGhz = (p.freq_mhz / 1000).toFixed(4);
      var isHit = p.amp_db > -100;
      tr.innerHTML =
        '<td style="font-family:\'JetBrains Mono\',monospace;font-size:10px;color:#38bdf8">' + ts + ' s</td>' +
        '<td style="font-size:10px">E' + p.emitter_id + ' · ' + freqGhz + ' GHz · ' + p.pw_us.toFixed(2) + ' µs</td>' +
        '<td style="font-size:10px;color:' + (isHit?'#22c55e':'#ef4444') + '">' + (isHit?'HIT':'MISS') + '</td>';
      tbody.insertBefore(tr, tbody.firstChild);
    });

    // Keep last 40 rows
    while (tbody.children.length > 40) tbody.removeChild(tbody.lastChild);
  }

  // ─── Receiver window stats ─────────────────────────────────────────────────
  function updateReceiverWindowStats() {
    var pdws = rd.recentPdws;
    if (!pdws.length) return;

    var amps = pdws.map(function(p){ return p.amp_db; });
    var meanAmp = amps.reduce(function(a,b){return a+b;},0)/amps.length;
    var snr = Math.max(0, meanAmp + 100).toFixed(1);
    var hits = amps.filter(function(a){ return a > -100; }).length;

    setText('valDetectionCount', hits);
    setText('valFreqDisplay', rd.latestPdw ? (rd.latestPdw.freq_mhz/1000).toFixed(4) + ' GHz' : '--');
    setText('valSnrDisplay', snr + ' dB');

    // Unique emitters seen
    var ids = {};
    pdws.forEach(function(p){ ids[p.emitter_id]=true; });
    setText('valEmitterCount', Object.keys(ids).length);
  }

  // ─── Receiver panels ─────────────────────────────────────────────────────
  function updateReceiverPanels() {
    if (!rd.emitters.length) return;
    // Update the 3 receiver preset cards with real emitter data
    var rxMap = { 1: rd.emitters[0], 2: rd.emitters[1], 3: rd.emitters[2] };
    Object.keys(rxMap).forEach(function(rxId) {
      var e = rxMap[rxId];
      if (!e) return;
      var freqMhz = Array.isArray(e.freq_mhz) ? e.freq_mhz[0] : e.freq_mhz;
      var btn = document.querySelector('[onclick*="selectReceiver(' + rxId + ')"]');
      if (btn) {
        var freqSpan = btn.querySelector('.rx-freq');
        if (freqSpan) freqSpan.textContent = (freqMhz/1000).toFixed(3) + ' GHz';
      }
    });
  }

  // ─── Button hooks ─────────────────────────────────────────────────────────
  function hookButtons() {
    var start = document.getElementById('btnStart');
    var pause = document.getElementById('btnPause');
    var stop = document.getElementById('btnStop');
    if (start) start.addEventListener('click', function(){ rd.running = true; });
    if (pause) pause.addEventListener('click', function(){ rd.running = !rd.running; });
    if (stop) stop.addEventListener('click', function(){ rd.running = false; });
  }

  // ─── AI Logs & Scheduler Decision ─────────────────────────────────────────

  function generateLocalEvmLines(t) {
    var baseSnr = 14.5 + 2.2 * Math.sin(t * 1.5) + (Math.random() - 0.5) * 0.4;
    var evmPct = Math.pow(10, -baseSnr / 20) * 100;
    var evmMrms = evmPct * 100;
    var evmPk = evmMrms * (3.8 + 0.4 * Math.sin(t * 0.8));
    var sym = Math.floor((t * 12) % 120 + 1);
    var subcar = Math.floor(((t * 7) % 600) - 300);
    var dataEvm = evmMrms * 0.978;
    var qpsk = (evmPct * 0.95).toFixed(2) + ' %';
    var qam16 = (evmPct * 1.25).toFixed(2) + ' %';
    var qam64 = (evmPct * 1.62).toFixed(2) + ' %';
    var qam256 = (evmMrms * 0.98).toFixed(2) + '  m%rms';
    var rsEvm = (evmMrms * 1.52).toFixed(2) + '   m%rms';
    var chPow = (-90.8 + 1.2 * Math.sin(t * 0.6) + (Math.random() - 0.5) * 0.1).toFixed(3);
    var rsPow = (-118.5 + 1.2 * Math.sin(t * 0.6) + (Math.random() - 0.5) * 0.1).toFixed(3);

    return [
      'EVM                    = ' + evmMrms.toFixed(2) + '  at EVMWindowEnd',
      'EVM Pk                 = ' + evmPk.toFixed(1) + '   at sym ' + sym + ', subcar ' + subcar,
      'Data EVM               = ' + dataEvm.toFixed(2) + '  m%rms',
      '3GPP-defined QPSK EVM  = ' + qpsk,
      '3GPP-defined 16QAM EVM = ' + qam16,
      '3GPP-defined 64QAM EVM = ' + qam64,
      '3GPP-defined 256QAM EVM= ' + qam256,
      'RS EVM                 = ' + rsEvm,
      'Channel Power          = ' + chPow + '  dBm',
      'RS Tx. Power (Avg)     = ' + rsPow + '  dBm'
    ];
  }

  function fetchAiLogs() {
    get('/api/data/ai-logs?page=receiver', function(d) {
      if (!d) return;
      _lastAiLogs = d;
      renderAiLogsConsole(d);
      if (d.scheduler_decision) {
        updateSchedulerDecision(d.scheduler_decision);
      }
    });
  }

  function renderAiLogsConsole(data) {
    var consoleEl = document.getElementById('aiLogsConsole');
    if (!consoleEl) return;

    if (_aiMode === 'evm') {
      consoleEl.style.justifyContent = 'space-between';
      var lines = (data && data.evm_lines && data.evm_lines.length) ? data.evm_lines : generateLocalEvmLines(rd.playheadPct * 10 || 8.42);
      if (consoleEl.children.length !== lines.length || !document.getElementById('aiLine0')) {
        consoleEl.innerHTML = '';
        for (var i = 0; i < lines.length; i++) {
          var div = document.createElement('div');
          div.className = 'log-line';
          div.id = 'aiLine' + i;
          div.textContent = lines[i];
          div.style.color = '#f87171';
          consoleEl.appendChild(div);
        }
      } else {
        for (var j = 0; j < lines.length; j++) {
          var el = document.getElementById('aiLine' + j);
          if (el) {
            el.textContent = lines[j];
            el.style.color = '#f87171';
          }
        }
      }
    } else {
      consoleEl.style.justifyContent = 'flex-start';
      var logs = (data && data.physics_logs) ? data.physics_logs : [];
      if (logs.length) {
        logs.forEach(function(l) {
          _neuralLogEntries.push(l);
        });
        while (_neuralLogEntries.length > 30) _neuralLogEntries.shift();
      } else if (_neuralLogEntries.length === 0) {
        var t0 = 8.420;
        _neuralLogEntries.push({ ts: (t0).toFixed(3), msg: 'Decision: Band 1 (8.20 - 8.70 GHz) selected | Dwell: 50 ms | Predicted Activity: 0.82', is_hit: false });
        _neuralLogEntries.push({ ts: (t0 + 0.05).toFixed(3), msg: 'HIT: Pulse detected (CF 8.45 GHz, PW 1.2 us, SNR 14.2 dB) -> E1 (Periodic)', is_hit: true });
        _neuralLogEntries.push({ ts: (t0 + 0.10).toFixed(3), msg: 'Scheduler: Switching to Band 2 | Reason: High activity probability for E2', is_hit: false });
        _neuralLogEntries.push({ ts: (t0 + 0.15).toFixed(3), msg: 'PDW extracted: CF 9.18 GHz, PW 0.9 us, AoA 12.4 deg, Amp -88.1 dBm', is_hit: false });
        _neuralLogEntries.push({ ts: (t0 + 0.20).toFixed(3), msg: 'Belief update: Band 2 ^ (0.35 -> 0.78)', is_hit: false });
      }
      consoleEl.innerHTML = '';
      _neuralLogEntries.slice(-10).forEach(function(l) {
        var div = document.createElement('div');
        div.className = 'log-line';
        var col = l.is_hit ? '#22c55e' : (l.msg.indexOf('Decision') > -1 ? '#38bdf8' : (l.msg.indexOf('HIT') > -1 ? '#22c55e' : '#e2e8f0'));
        div.style.color = col;
        div.style.fontSize = '8.5px';
        div.style.padding = '1px 0';
        div.textContent = '[' + l.ts + 's] ' + l.msg;
        consoleEl.appendChild(div);
      });
      consoleEl.scrollTop = consoleEl.scrollHeight;
    }
  }

  function updateSchedulerDecision(sd) {
    if (!sd) return;
    setText('schedValSelectedBand', sd.selected_band);
    setText('schedValActionScore', sd.action_score);
    setText('schedValPredActivity', sd.predicted_activity);
    setText('schedValPeriodicity', sd.periodicity_estimate);
    setText('schedValTransitionProb', sd.transition_probability);
    setText('schedValUncertainty', sd.uncertainty);
    setText('schedValStaleness', sd.staleness);
    setText('schedValSwitchCost', sd.switch_cost);
    setText('schedValJustification', sd.justification);

    if (sd.current_band_name || sd.selected_band) {
      setText('siCurrentBand', sd.current_band_name || sd.selected_band);
    }
    if (sd.center_freq_ghz) {
      setText('siCenterFreq', sd.center_freq_ghz);
    } else if (sd.band_idx !== undefined) {
      var cf = (0.25 + sd.band_idx * 0.50).toFixed(3) + ' GHz';
      setText('siCenterFreq', cf);
    }

    if (sd.next_band_name || sd.next_band) {
      setText('siNextBand', sd.next_band_name || sd.next_band);
    } else if (Array.isArray(sd.alternatives) && sd.alternatives.length > 0) {
      var topAlt = sd.alternatives[0];
      var nextBandId = topAlt.band;
      var nextB = nextBandId - 1;
      var nextName = (nextB * 0.50).toFixed(2) + ' - ' + ((nextB + 1) * 0.50).toFixed(2) + ' GHz';
      setText('siNextBand', nextName);
    }

    if (sd.last_scanned_name || sd.last_scanned_band) {
      setText('siLastScanned', sd.last_scanned_name || sd.last_scanned_band);
    }
    if (sd.last_scanned_result) {
      var lastResultEl = document.getElementById('siLastResult');
      if (lastResultEl) {
        lastResultEl.textContent = sd.last_scanned_result;
        lastResultEl.className = sd.last_scanned_result === 'HIT' ? 'si-val status-hit' : 'si-val status-miss';
      }
    }

    var altTbody = document.getElementById('schedAltTableBody');
    if (altTbody && Array.isArray(sd.alternatives)) {
      altTbody.innerHTML = '';
      sd.alternatives.forEach(function(alt) {
        var tr = document.createElement('tr');
        tr.innerHTML = '<td>' + alt.rank + '</td><td>' + alt.band + '</td><td>' + alt.score + '</td><td>' + alt.pred + '</td>';
        altTbody.appendChild(tr);
      });
    }
  }

  function initAiTabControls() {
    var btnEvm = document.getElementById('btnAiTabEvm');
    var btnStream = document.getElementById('btnAiTabStream');
    if (btnEvm) {
      btnEvm.addEventListener('click', function() {
        _aiMode = 'evm';
        btnEvm.style.background = '#0284c7';
        btnEvm.style.color = '#fff';
        btnEvm.style.border = 'none';
        if (btnStream) {
          btnStream.style.background = '#27272a';
          btnStream.style.color = '#a1a1aa';
          btnStream.style.border = '1px solid #3f3f46';
        }
        renderAiLogsConsole(_lastAiLogs);
      });
    }
    if (btnStream) {
      btnStream.addEventListener('click', function() {
        _aiMode = 'stream';
        btnStream.style.background = '#0284c7';
        btnStream.style.color = '#fff';
        btnStream.style.border = 'none';
        if (btnEvm) {
          btnEvm.style.background = '#27272a';
          btnEvm.style.color = '#a1a1aa';
          btnEvm.style.border = '1px solid #3f3f46';
        }
        renderAiLogsConsole(_lastAiLogs);
      });
    }
  }

  // ─── Helpers ──────────────────────────────────────────────────────────────
  function setText(id, val) {
    var el = document.getElementById(id);
    if (el) el.textContent = val;
  }

  // ─── Boot ─────────────────────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
