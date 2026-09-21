'use client';

import { useEffect, useRef, useState, useCallback } from 'react';

const FREQ_START = 100; // MHz
const FREQ_END = 1000;  // MHz
const BINS = 800;
const BANDS = 36;
const BAND_FREQ = (FREQ_END - FREQ_START) / BANDS; // 25 MHz per band
const WATERFALL_ROWS = 120;
const THRESHOLD_DBM = -77;

function dbToColor(dbm: number): [number, number, number] {
  const t = Math.max(0, Math.min(1, (dbm + 97) / 60));
  if (t < 0.2) return [2, 7, 11];
  if (t < 0.4) return [0, 40, 80];
  if (t < 0.6) return [0, 120, 180];
  if (t < 0.8) return [50, 220, 200];
  return [63, 232, 239];
}

export default function LiveRFPage() {
  const spectrumRef = useRef<HTMLCanvasElement>(null);
  const waterfallRef = useRef<HTMLCanvasElement>(null);
  const beliefRef = useRef<HTMLCanvasElement>(null);
  const wfBuf = useRef<ImageData | null>(null);
  const psdRef = useRef<Float32Array>(new Float32Array(BINS).fill(-97));
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [nextBand, setNextBand] = useState<number | null>(null);
  const [alloc, setAlloc] = useState<number[]>(new Array(BANDS).fill(100 / BANDS));
  const [hitCount, setHitCount] = useState(0);
  const [slotCount, setSlotCount] = useState(0);
  const wsRef = useRef<WebSocket | null>(null);
  const hitRef = useRef(0);
  const slotRef = useRef(0);

  // Draw spectrum canvas
  const drawSpectrum = useCallback(() => {
    const canvas = spectrumRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    const pL = 42, pR = 8, pT = 8, pB = 20;
    const plotW = w - pL - pR, plotH = h - pT - pB;
    ctx.fillStyle = '#050c13';
    ctx.fillRect(0, 0, w, h);

    // Grid
    ctx.strokeStyle = 'rgba(63,232,239,0.07)';
    ctx.lineWidth = 1;
    ctx.font = '9px IBM Plex Mono, monospace';
    ctx.fillStyle = '#4a7080';
    const ySteps = 6;
    for (let s = 0; s <= ySteps; s++) {
      const dbm = -97 + (s / ySteps) * 77;
      const y = pT + (1 - (dbm + 97) / 77) * plotH;
      ctx.beginPath(); ctx.moveTo(pL, y); ctx.lineTo(w - pR, y); ctx.stroke();
      ctx.fillText(dbm.toFixed(0), 2, y + 3);
    }
    const xSteps = 9;
    for (let s = 0; s <= xSteps; s++) {
      const f = FREQ_START + (FREQ_END - FREQ_START) * (s / xSteps);
      const x = pL + plotW * (s / xSteps);
      ctx.beginPath(); ctx.moveTo(x, pT); ctx.lineTo(x, h - pB); ctx.stroke();
      ctx.fillText(f.toFixed(0), x - 12, h - 3);
    }

    // Threshold line
    const ty = pT + (1 - (THRESHOLD_DBM + 97) / 77) * plotH;
    ctx.strokeStyle = 'rgba(255,85,119,0.6)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(pL, ty); ctx.lineTo(w - pR, ty); ctx.stroke();
    ctx.setLineDash([]);

    // Next band highlight
    if (nextBand !== null) {
      const bx = pL + plotW * (nextBand / BANDS);
      const bw = plotW / BANDS;
      ctx.fillStyle = 'rgba(63,232,239,0.08)';
      ctx.fillRect(bx, pT, bw, plotH);
      ctx.strokeStyle = 'rgba(63,232,239,0.4)';
      ctx.lineWidth = 1;
      ctx.strokeRect(bx, pT, bw, plotH);
    }

    // PSD line
    const psd = psdRef.current;
    ctx.strokeStyle = '#3fe8ef';
    ctx.lineWidth = 1.5;
    ctx.shadowColor = 'rgba(63,232,239,0.4)';
    ctx.shadowBlur = 3;
    ctx.beginPath();
    for (let i = 0; i < BINS; i++) {
      const x = pL + (i / (BINS - 1)) * plotW;
      const y = pT + (1 - Math.max(0, Math.min(1, (psd[i] + 97) / 77))) * plotH;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }, [nextBand]);

  // Push waterfall row
  const pushWaterfall = useCallback(() => {
    const canvas = waterfallRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;

    if (!wfBuf.current || wfBuf.current.width !== w) {
      wfBuf.current = ctx.createImageData(w, h);
      // fill with dark
      for (let i = 0; i < wfBuf.current.data.length; i += 4) {
        wfBuf.current.data[i] = 2; wfBuf.current.data[i+1] = 7; wfBuf.current.data[i+2] = 11; wfBuf.current.data[i+3] = 255;
      }
    }

    const buf = wfBuf.current;
    // Scroll up
    buf.data.copyWithin(0, w * 4);

    // Write new bottom row
    const psd = psdRef.current;
    const rowOffset = (h - 1) * w * 4;
    for (let x = 0; x < w; x++) {
      const binIdx = Math.floor((x / w) * BINS);
      const [r, g, b] = dbToColor(psd[binIdx]);
      buf.data[rowOffset + x * 4 + 0] = r;
      buf.data[rowOffset + x * 4 + 1] = g;
      buf.data[rowOffset + x * 4 + 2] = b;
      buf.data[rowOffset + x * 4 + 3] = 255;
    }
    ctx.putImageData(buf, 0, 0);
  }, []);

  // Draw belief bar chart
  const drawBelief = useCallback((allocArr: number[], nextB: number | null) => {
    const canvas = beliefRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width, h = canvas.height;
    ctx.fillStyle = '#050c13';
    ctx.fillRect(0, 0, w, h);
    const barW = w / BANDS;
    const maxA = Math.max(...allocArr, 1);
    for (let i = 0; i < BANDS; i++) {
      const barH = (allocArr[i] / maxA) * (h - 14);
      const isNext = i === nextB;
      ctx.fillStyle = isNext ? '#3fe8ef' : 'rgba(63,232,239,0.3)';
      ctx.fillRect(i * barW + 1, h - barH - 12, barW - 2, barH);
      if (isNext) {
        ctx.fillStyle = '#3fe8ef';
        ctx.font = '7px IBM Plex Mono, monospace';
        ctx.fillText('B' + String(i + 1).padStart(2, '0'), i * barW, h - 1);
      }
    }
  }, []);

  // Animation loop
  const animRef = useRef<number>(0);
  const animate = useCallback(() => {
    drawSpectrum();
    pushWaterfall();
    animRef.current = requestAnimationFrame(animate);
  }, [drawSpectrum, pushWaterfall]);

  // Start/stop
  const startEngine = useCallback(async () => {
    try {
      await fetch('http://localhost:8000/api/system-c/start', { method: 'POST' });
      setRunning(true);
    } catch (e) { console.error(e); }
  }, []);

  const stopEngine = useCallback(async () => {
    try {
      await fetch('http://localhost:8000/api/system-c/stop', { method: 'POST' });
      setRunning(false);
    } catch (e) { console.error(e); }
  }, []);

  useEffect(() => {
    // Auto-start engine and connect WebSocket
    startEngine();

    const ws = new WebSocket('ws://localhost:8000/ws/system-c');
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === 'fft_frame' && data.psd) {
          const psd = psdRef.current;
          for (let i = 0; i < BINS && i < data.psd.length; i++) psd[i] = data.psd[i];

          if (data.next_band !== undefined) {
            setNextBand(data.next_band);
            slotRef.current++;
            setSlotCount(slotRef.current);

            // check hit in scheduled band
            const bStart = Math.floor(data.next_band * BINS / BANDS);
            const bEnd = bStart + Math.floor(BINS / BANDS);
            let peak = -120;
            for (let i = bStart; i < bEnd; i++) peak = Math.max(peak, psd[i]);
            if (peak > THRESHOLD_DBM) {
              hitRef.current++;
              setHitCount(hitRef.current);
            }
          }
          if (data.alloc) setAlloc(data.alloc);
        }
      } catch (_) {}
    };

    animRef.current = requestAnimationFrame(animate);

    return () => {
      ws.close();
      cancelAnimationFrame(animRef.current);
    };
  }, []);

  useEffect(() => {
    drawBelief(alloc, nextBand);
  }, [alloc, nextBand, drawBelief]);

  const hitRate = slotCount > 0 ? ((hitCount / slotCount) * 100).toFixed(2) : '0.00';
  const nextBandFreqLo = nextBand !== null ? (FREQ_START + nextBand * BAND_FREQ).toFixed(0) : '--';
  const nextBandFreqHi = nextBand !== null ? (FREQ_START + (nextBand + 1) * BAND_FREQ).toFixed(0) : '--';

  return (
    <div style={{ background: '#050c13', color: '#cfeaf0', fontFamily: "'IBM Plex Sans', system-ui, sans-serif", minHeight: '100%', padding: '16px 20px', fontSize: 13 }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14, borderBottom: '1px solid #163040', paddingBottom: 10 }}>
        <div>
          <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 14, letterSpacing: '0.08em', fontWeight: 600, color: '#3fe8ef' }}>
            LIVE RF ENGINE — SYSTEM C
          </div>
          <div style={{ fontSize: 11, color: '#6d93a0', marginTop: 2 }}>
            Real-time I/Q Physics Simulation · 4 Emitters · 100–1000 MHz · HybridMetaScheduler
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ width: 7, height: 7, borderRadius: '50%', background: connected ? '#3ee9a6' : '#ff5577', display: 'inline-block', boxShadow: connected ? '0 0 6px #3ee9a6' : 'none' }} />
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, color: connected ? '#3ee9a6' : '#ff5577' }}>
              {connected ? 'WS CONNECTED' : 'WS OFFLINE'}
            </span>
          </div>
          <button
            onClick={running ? stopEngine : startEngine}
            style={{ background: running ? 'rgba(255,85,119,0.15)' : 'rgba(63,232,239,0.12)', border: `1px solid ${running ? '#ff5577' : '#3fe8ef'}`, color: running ? '#ff5577' : '#3fe8ef', borderRadius: 3, padding: '4px 14px', fontFamily: "'IBM Plex Mono', monospace", fontSize: 11, cursor: 'pointer', letterSpacing: '0.06em' }}
          >
            {running ? 'STOP ENGINE' : 'START ENGINE'}
          </button>
        </div>
      </div>

      {/* Top stats row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: 10, marginBottom: 14 }}>
        {[
          { label: 'NEXT SCAN BAND', value: nextBand !== null ? `B${String(nextBand + 1).padStart(2, '0')}` : '--', sub: `${nextBandFreqLo}–${nextBandFreqHi} MHz`, color: '#3fe8ef' },
          { label: 'LIVE HIT RATE', value: `${hitRate}%`, sub: `${hitCount} hits`, color: '#3ee9a6' },
          { label: 'SLOTS PROCESSED', value: slotCount.toString(), sub: 'HybridMetaScheduler', color: '#cfeaf0' },
          { label: 'FREQ RANGE', value: '100–1000 MHz', sub: '900 MHz span', color: '#cfeaf0' },
          { label: 'EMITTERS ACTIVE', value: '4', sub: '275 / 400 / 520–695 / 955 MHz', color: '#ffcf6b' },
        ].map((s) => (
          <div key={s.label} style={{ background: '#0a161f', border: '1px solid #163040', borderRadius: 3, padding: '10px 12px' }}>
            <div style={{ fontSize: 9, fontFamily: "'IBM Plex Mono', monospace", letterSpacing: '0.07em', color: '#4a7080', marginBottom: 4 }}>{s.label}</div>
            <div style={{ fontSize: 22, fontFamily: "'IBM Plex Mono', monospace", fontWeight: 700, color: s.color }}>{s.value}</div>
            <div style={{ fontSize: 10, color: '#6d93a0', marginTop: 2 }}>{s.sub}</div>
          </div>
        ))}
      </div>

      {/* Spectrum canvas */}
      <div style={{ background: '#0a161f', border: '1px solid #163040', borderRadius: 3, marginBottom: 10, padding: '8px 10px' }}>
        <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: '#4a7080', letterSpacing: '0.07em', marginBottom: 6 }}>
          POWER SPECTRAL DENSITY (dBm) — LIVE FFT · 800 bins · 100–1000 MHz
        </div>
        <canvas ref={spectrumRef} width={1200} height={200} style={{ width: '100%', height: 200, display: 'block' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#3f5c66', marginTop: 3, padding: '0 42px 0 42px' }}>
          {[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000].map(f => <span key={f}>{f}</span>)}
        </div>
      </div>

      {/* Waterfall */}
      <div style={{ background: '#0a161f', border: '1px solid #163040', borderRadius: 3, marginBottom: 10, padding: '8px 10px' }}>
        <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: '#4a7080', letterSpacing: '0.07em', marginBottom: 6 }}>
          STFT WATERFALL — FREQ (MHz) × TIME · each row = 50ms dwell
        </div>
        <canvas ref={waterfallRef} width={1200} height={120} style={{ width: '100%', height: 120, display: 'block', imageRendering: 'pixelated' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 9, color: '#3f5c66', marginTop: 3, padding: '0 0 0 0' }}>
          {[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000].map(f => <span key={f}>{f} MHz</span>)}
        </div>
      </div>

      {/* HybridMetaScheduler band allocation */}
      <div style={{ background: '#0a161f', border: '1px solid #163040', borderRadius: 3, marginBottom: 10, padding: '8px 10px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
          <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: '#4a7080', letterSpacing: '0.07em' }}>
            HYBRIDMETASCHEDULER — 36-BAND BELIEF STATE (P_ACTIVE) · NEXT SCAN = {nextBand !== null ? `B${String(nextBand+1).padStart(2,'0')} (${nextBandFreqLo}–${nextBandFreqHi} MHz)` : '--'}
          </div>
          <div style={{ fontSize: 10, color: '#3fe8ef', fontFamily: "'IBM Plex Mono', monospace" }}>
            Layer-4 Meta-UCB Bandit Combiner
          </div>
        </div>
        <canvas ref={beliefRef} width={1200} height={70} style={{ width: '100%', height: 70, display: 'block' }} />
        <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 8, color: '#3f5c66', marginTop: 2 }}>
          {Array.from({ length: 9 }, (_, i) => (FREQ_START + (i + 1) * 100)).map(f => <span key={f}>{f}</span>)}
        </div>
      </div>

      {/* Emitter table */}
      <div style={{ background: '#0a161f', border: '1px solid #163040', borderRadius: 3, padding: '8px 10px' }}>
        <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 10, color: '#4a7080', letterSpacing: '0.07em', marginBottom: 8 }}>EMITTER REGISTRY — LIVE I/Q SOURCES</div>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 11, fontFamily: "'IBM Plex Mono', monospace" }}>
          <thead>
            <tr style={{ borderBottom: '1px solid #163040', color: '#4a7080', fontSize: 10 }}>
              {['ID', 'TYPE', 'CENTER FREQ', 'AOA', 'SNR', 'STATUS'].map(h => (
                <th key={h} style={{ padding: '4px 8px', textAlign: 'left', fontWeight: 400 }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {[
              { id: 'E-00', type: 'Fixed Continuous', freq: '400 MHz', aoa: '12.0°', snr: '40 dB', active: true },
              { id: 'E-01', type: 'Frequency Agile', freq: '520 / 635 / 695 MHz', aoa: '58.0°', snr: '38 dB', active: true },
              { id: 'E-02', type: 'Fixed Continuous', freq: '275 MHz', aoa: '-30.0°', snr: '45 dB', active: true },
              { id: 'E-03', type: 'Fixed Continuous', freq: '955 MHz', aoa: '100.0°', snr: '30 dB', active: true },
            ].map(e => (
              <tr key={e.id} style={{ borderBottom: '1px solid #0f2230' }}>
                <td style={{ padding: '5px 8px', color: '#3fe8ef' }}>{e.id}</td>
                <td style={{ padding: '5px 8px', color: '#cfeaf0' }}>{e.type}</td>
                <td style={{ padding: '5px 8px', color: '#ffcf6b' }}>{e.freq}</td>
                <td style={{ padding: '5px 8px' }}>{e.aoa}</td>
                <td style={{ padding: '5px 8px' }}>{e.snr}</td>
                <td style={{ padding: '5px 8px' }}>
                  <span style={{ background: 'rgba(62,233,166,0.12)', border: '1px solid #3ee9a6', color: '#3ee9a6', borderRadius: 2, padding: '1px 6px', fontSize: 9 }}>ACTIVE</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
