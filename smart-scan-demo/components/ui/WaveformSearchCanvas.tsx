"use client";

import React, { useRef, useEffect } from "react";

export interface WaveformStep {
  step: number;
  band: number;
  hit?: boolean;
  occupied?: boolean;
  dwell_ms?: number;
  snr_db?: number;
}

export interface WaveBandConfig {
  band: number; // 0-indexed (0 to 35)
  label: string; // e.g. "B36"
  freqGhz: number; // e.g. 19.5
  color: string; // high-contrast luminous tactical color
  name: string; // e.g. "Band 36 (High Freq)"
}

// 7 Continuous Reference Frequency Bands with Soft Light Pastel Tones (Light and not too bright)
export const REFERENCE_WAVE_BANDS: WaveBandConfig[] = [
  { band: 35, label: "B36", freqGhz: 19.5, color: "#fef08a", name: "Band 36 (High Freq)" }, // Soft Pale Lemon Cream
  { band: 29, label: "B30", freqGhz: 16.5, color: "#fca5a5", name: "Band 30" },             // Soft Pale Coral Rose
  { band: 23, label: "B24", freqGhz: 13.5, color: "#86efac", name: "Band 24" },             // Soft Pale Mint Green
  { band: 17, label: "B18", freqGhz: 10.5, color: "#f0abfc", name: "Band 18" },             // Soft Pale Orchid Pink
  { band: 11, label: "B12", freqGhz: 7.5,  color: "#c4b5fd", name: "Band 12" },             // Soft Pale Lavender
  { band: 5,  label: "B06", freqGhz: 4.5,  color: "#93c5fd", name: "Band 06" },             // Soft Pale Sky Blue
  { band: 0,  label: "B01", freqGhz: 2.0,  color: "#6ee7b7", name: "Band 01" },             // Soft Pale Seafoam
];

export interface WaveformSearchCanvasProps {
  width?: number;
  height?: number;
  bands?: number;
  timeSlots?: number;
  matrix?: number[][];
  scanPath?: WaveformStep[];
  activeStep?: number;
  activeBand?: number;
  showGroundTruth?: boolean;
  showScanPath?: boolean;
  showMisses?: boolean;
}

export const WaveformSearchCanvas = React.memo(function WaveformSearchCanvas({
  width = 1100,
  height = 380,
  bands = 36,
  timeSlots = 50,
  matrix = [],
  scanPath = [],
  activeStep = 0,
  activeBand = 0,
  showGroundTruth = true,
  showScanPath = true,
  showMisses = true,
}: WaveformSearchCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  // Per-band phase accumulators — indexed by WaveBandConfig position in REFERENCE_WAVE_BANDS
  const bandPhasesRef = useRef<Float64Array>(new Float64Array(REFERENCE_WAVE_BANDS.length));

  // Synchronize latest props into ref for 60 FPS animation loop
  const propsRef = useRef({
    width,
    height,
    bands,
    timeSlots,
    matrix,
    scanPath,
    activeStep,
    activeBand,
    showGroundTruth,
    showScanPath,
    showMisses,
  });

  useEffect(() => {
    propsRef.current = {
      width,
      height,
      bands,
      timeSlots,
      matrix,
      scanPath,
      activeStep,
      activeBand,
      showGroundTruth,
      showScanPath,
      showMisses,
    };
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;

    const render = () => {
      const p = propsRef.current;
      const dpr = typeof window !== "undefined" ? window.devicePixelRatio || 1 : 1;

      if (canvas.width !== p.width * dpr || canvas.height !== p.height * dpr) {
        canvas.width = p.width * dpr;
        canvas.height = p.height * dpr;
      }

      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const padL = 54;
      const padR = 14;
      const padT = 26;
      const padB = 32;
      const plotW = p.width - padL - padR;
      const plotH = p.height - padT - padB;
      const slotW = plotW / p.timeSlots;

      // Deep dark pitch-black tactical canvas background
      ctx.fillStyle = "#040810";
      ctx.fillRect(0, 0, p.width, p.height);

      ctx.fillStyle = "#020408";
      ctx.fillRect(padL, padT, plotW, plotH);

      // Coordinate mapping helpers
      const getSlotCenterX = (t: number) => padL + t * slotW + slotW / 2;
      const getBandCenterY = (b: number) => padT + (p.bands - 1 - b) * (plotH / (p.bands - 1));

      // Fast O(1) hash map for scanPath
      const scanMap = new Map<number, WaveformStep>();
      for (let i = 0; i < p.scanPath.length; i++) {
        const s = p.scanPath[i];
        scanMap.set(s.band * 1000 + s.step, s);
      }

      // 1. Subtle Background Grid
      // Vertical time grid lines
      ctx.lineWidth = 0.5;
      for (let t = 0; t <= p.timeSlots; t++) {
        const x = padL + t * slotW;
        const isMajor = t % 10 === 0;
        ctx.strokeStyle = isMajor ? "rgba(35, 60, 95, 0.45)" : "rgba(20, 36, 60, 0.25)";
        ctx.beginPath();
        ctx.moveTo(x, padT);
        ctx.lineTo(x, padT + plotH);
        ctx.stroke();
      }

      // Horizontal baseline guide lines for each reference wave band
      REFERENCE_WAVE_BANDS.forEach((wb) => {
        const y = getBandCenterY(wb.band);
        ctx.strokeStyle = "rgba(35, 60, 95, 0.35)";
        ctx.lineWidth = 0.5;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(padL, y);
        ctx.lineTo(padL + plotW, y);
        ctx.stroke();
        ctx.setLineDash([]);
      });

      // 2. Continuous Clean Waves — frequency-scaled traveling waves
      // Each band advances its own phase proportional to its physical GHz frequency.
      // freqGhz range: 2.0 GHz (B01, slow) → 19.5 GHz (B36, fast)
      // phaseSpeed maps: 2.0 GHz → ~0.018 rad/frame, 19.5 GHz → ~0.175 rad/frame
      const freqMin = REFERENCE_WAVE_BANDS[REFERENCE_WAVE_BANDS.length - 1].freqGhz; // 2.0
      const freqMax = REFERENCE_WAVE_BANDS[0].freqGhz;                               // 19.5

      if (p.showGroundTruth) {
        REFERENCE_WAVE_BANDS.forEach((wb, wbIdx) => {
          const b = wb.band;
          const baseY = getBandCenterY(b);

          // Advance per-band phase this frame, proportional to actual physical frequency
          // Speed range: 2 GHz → 0.018 rad/frame, 19.5 GHz → 0.175 rad/frame
          const normFreq = (wb.freqGhz - freqMin) / (freqMax - freqMin); // 0 = 2 GHz, 1 = 19.5 GHz
          const phaseSpeed = 0.018 + normFreq * 0.157; // rad per animation frame
          bandPhasesRef.current[wbIdx] += phaseSpeed;
          const bandPhase = bandPhasesRef.current[wbIdx];

          // Aggregate pulse activity across this band and adjacent band cluster (±2 bands)
          const clusterBands = [b - 2, b - 1, b, b + 1, b + 2].filter((cb) => cb >= 0 && cb < p.bands);

          const hasPulseAtSlot = (t: number) => {
            for (let i = 0; i < clusterBands.length; i++) {
              const cb = clusterBands[i];
              if (p.matrix[cb]?.[t]) return true;
            }
            return false;
          };

          // Spatial frequency (wavelength) also scales with physical frequency:
          // High freq (19.5 GHz) → short wavelength (dense oscillations)
          // Low freq  (2.0 GHz)  → long wavelength  (wide oscillations)
          const spatialOmega = 0.030 + normFreq * 0.160; // 0.030 rad/px → 0.190 rad/px
          const baseAmp = 6.0 + (1 - normFreq) * 5.0;    // 6px (high freq) → 11px (low freq)
          const pulseBoostAmp = 9.0 + (1 - normFreq) * 6.0;

          ctx.save();
          ctx.beginPath();

          let started = false;
          const stepPx = 2; // high-resolution smooth sampling

          for (let px = 0; px <= plotW; px += stepPx) {
            const x = padL + px;
            const tIdx = Math.min(p.timeSlots - 1, Math.max(0, Math.floor(px / slotW)));

            // Smooth Gaussian envelope around active pulse intervals
            let activeWeight = 0;
            for (let dt = -1; dt <= 1; dt++) {
              const checkT = tIdx + dt;
              if (checkT >= 0 && checkT < p.timeSlots && hasPulseAtSlot(checkT)) {
                const pulseCenterPx = (checkT + 0.5) * slotW;
                const dist = Math.abs(px - pulseCenterPx);
                const w = Math.exp(-0.5 * Math.pow(dist / (slotW * 0.48), 2));
                if (w > activeWeight) activeWeight = w;
              }
            }

            const amp = baseAmp + pulseBoostAmp * activeWeight;
            // Continuous traveling wave propagation: sin(k*x - omega*t)
            const travelingPhase = px * spatialOmega - bandPhase;
            const y = baseY + Math.sin(travelingPhase) * amp;

            if (!started) {
              ctx.moveTo(x, y);
              started = true;
            } else {
              ctx.lineTo(x, y);
            }
          }

          // Soft, clean, light wave line — low opacity
          ctx.strokeStyle = wb.color;
          ctx.lineWidth = 1.0;
          ctx.globalAlpha = 0.32;
          ctx.stroke();
          ctx.restore();

          // Soft subtle energy aura on active pulse intervals
          for (let t = 0; t < p.timeSlots; t++) {
            if (hasPulseAtSlot(t)) {
              const cx = getSlotCenterX(t);
              const cy = baseY;

              const grad = ctx.createRadialGradient(cx, cy, 1, cx, cy, slotW * 0.55);
              grad.addColorStop(0, `${wb.color}08`);
              grad.addColorStop(1, "transparent");
              ctx.fillStyle = grad;
              ctx.beginPath();
              ctx.arc(cx, cy, slotW * 0.55, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        });
      }

      // 3. Bold Receiver Scan Path Trajectory Connecting Hit Dots
      if (p.showScanPath && p.scanPath.length > 1) {
        ctx.save();
        ctx.strokeStyle = "#00e5ff";
        ctx.lineWidth = 2.5;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.globalAlpha = 0.95;
        ctx.shadowColor = "rgba(0, 229, 255, 0.35)";
        ctx.shadowBlur = 4;
        ctx.beginPath();

        let first = true;
        for (let i = 0; i < p.scanPath.length; i++) {
          const pt = p.scanPath[i];
          if (pt.step < p.timeSlots && pt.band < p.bands) {
            const cx = getSlotCenterX(pt.step);
            const cy = getBandCenterY(pt.band);
            if (first) {
              ctx.moveTo(cx, cy);
              first = false;
            } else {
              ctx.lineTo(cx, cy);
            }
          }
        }
        ctx.stroke();
        ctx.restore();
      }

      // 4. Soft Highlighted Hits (⊕) and Dwells
      for (let i = 0; i < p.scanPath.length; i++) {
        const pt = p.scanPath[i];
        if (pt.step >= p.timeSlots || pt.band >= p.bands) continue;

        const cx = getSlotCenterX(pt.step);
        const cy = getBandCenterY(pt.band);
        const isOccupied = p.matrix[pt.band] ? Boolean(p.matrix[pt.band][pt.step]) : false;
        const isHit = Boolean(pt.hit) || isOccupied;

        if (isHit) {
          // Confirmed Intercept (Hit ⊕) — Soft Emerald Green Target
          ctx.save();
          const aura = ctx.createRadialGradient(cx, cy, 1, cx, cy, 9);
          aura.addColorStop(0, "rgba(110, 231, 183, 0.25)");
          aura.addColorStop(1, "rgba(110, 231, 183, 0.0)");
          ctx.fillStyle = aura;
          ctx.beginPath();
          ctx.arc(cx, cy, 9, 0, Math.PI * 2);
          ctx.fill();

          // Soft green ring
          ctx.strokeStyle = "#6ee7b7";
          ctx.lineWidth = 1.3;
          ctx.beginPath();
          ctx.arc(cx, cy, 5.5, 0, Math.PI * 2);
          ctx.stroke();

          // Crosshair ⊕
          ctx.beginPath();
          ctx.moveTo(cx - 3.8, cy);
          ctx.lineTo(cx + 3.8, cy);
          ctx.moveTo(cx, cy - 3.8);
          ctx.lineTo(cx, cy + 3.8);
          ctx.stroke();

          // Solid core
          ctx.fillStyle = "#6ee7b7";
          ctx.beginPath();
          ctx.arc(cx, cy, 1.8, 0, Math.PI * 2);
          ctx.fill();
          ctx.restore();
        } else {
          // Quiet Receiver Dwell (Empty Spectrum / Quiet Cell)
          ctx.strokeStyle = "rgba(125, 211, 252, 0.60)";
          ctx.lineWidth = 1.0;
          ctx.setLineDash([2, 2]);
          ctx.beginPath();
          ctx.arc(cx, cy, 4.0, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }

      // 5. Soft Highlighted Missed Opportunities (⊙)
      if (p.showMisses) {
        REFERENCE_WAVE_BANDS.forEach((wb) => {
          const b = wb.band;
          const cy = getBandCenterY(b);
          const clusterBands = [b - 2, b - 1, b, b + 1, b + 2].filter((cb) => cb >= 0 && cb < p.bands);

          for (let t = 0; t <= Math.min(p.activeStep, p.timeSlots - 1); t++) {
            let hasPulse = false;
            for (let i = 0; i < clusterBands.length; i++) {
              if (p.matrix[clusterBands[i]]?.[t]) {
                hasPulse = true;
                break;
              }
            }

            if (hasPulse) {
              const scanned = scanMap.get(b * 1000 + t);
              // Missed opportunity: active pulse present, but receiver dwelt elsewhere
              if (!scanned) {
                const cx = getSlotCenterX(t);

                ctx.save();
                // Soft coral halo
                const missAura = ctx.createRadialGradient(cx, cy, 1, cx, cy, 7.5);
                missAura.addColorStop(0, "rgba(248, 113, 113, 0.18)");
                missAura.addColorStop(1, "rgba(248, 113, 113, 0.0)");
                ctx.fillStyle = missAura;
                ctx.beginPath();
                ctx.arc(cx, cy, 7.5, 0, Math.PI * 2);
                ctx.fill();

                // Dashed warning ring
                ctx.strokeStyle = "#f87171";
                ctx.lineWidth = 1.1;
                ctx.setLineDash([2, 2]);
                ctx.beginPath();
                ctx.arc(cx, cy, 4.8, 0, Math.PI * 2);
                ctx.stroke();
                ctx.setLineDash([]);

                // Soft coral core dot
                ctx.fillStyle = "#f87171";
                ctx.beginPath();
                ctx.arc(cx, cy, 1.5, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
              }
            }
          }
        });
      }

      // 6. Active Scanner Cursor & Live Dwell Window
      if (p.activeStep < p.timeSlots) {
        const curX = getSlotCenterX(p.activeStep);
        const curY = getBandCenterY(p.activeBand);

        // Vertical scanner cursor line
        ctx.save();
        ctx.strokeStyle = "rgba(125, 211, 252, 0.70)";
        ctx.lineWidth = 1.1;
        ctx.setLineDash([3, 2]);
        ctx.beginPath();
        ctx.moveTo(curX, padT);
        ctx.lineTo(curX, padT + plotH);
        ctx.stroke();
        ctx.setLineDash([]);

        // Live dwell reticle (Soft Sky Blue)
        ctx.strokeStyle = "#7dd3fc";
        ctx.lineWidth = 1.5;
        ctx.strokeRect(curX - slotW * 0.55, curY - 8, slotW * 1.1, 16);

        // Reticle center dot
        ctx.beginPath();
        ctx.arc(curX, curY, 2.2, 0, Math.PI * 2);
        ctx.fillStyle = "#7dd3fc";
        ctx.fill();
        ctx.restore();
      }

      // 7. Y-Axis Frequency Labels (GHz + Band tag)
      ctx.font = "8px monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";

      REFERENCE_WAVE_BANDS.forEach((wb) => {
        const y = getBandCenterY(wb.band);
        ctx.fillStyle = wb.color;
        ctx.font = "bold 8.5px monospace";
        ctx.fillText(wb.label, padL - 6, y);

        ctx.fillStyle = "#8fa3b7";
        ctx.font = "7.5px monospace";
        ctx.fillText(`${wb.freqGhz.toFixed(1)}G`, padL - 25, y);
      });

      // 8. X-Axis Time Labels (t=0, t=10, ..., and ms values)
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let t = 0; t < p.timeSlots; t += 10) {
        const x = getSlotCenterX(t);

        // Tick mark
        ctx.strokeStyle = "rgba(43, 67, 98, 0.7)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, padT + plotH);
        ctx.lineTo(x, padT + plotH + 4);
        ctx.stroke();

        // Step and time labels
        ctx.fillStyle = "#8fa3b7";
        ctx.font = "8px monospace";
        ctx.fillText(`t=${t}`, x, p.height - 13);

        ctx.fillStyle = "#5d7289";
        ctx.font = "7px monospace";
        ctx.fillText(`${t * 50}ms`, x, p.height - 3);
      }

      ctx.restore();

      // Continue animation loop
      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => {
      cancelAnimationFrame(animId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: "100%",
        height: `${height}px`,
        display: "block",
        borderRadius: "4px",
        background: "#040810",
      }}
    />
  );
});
