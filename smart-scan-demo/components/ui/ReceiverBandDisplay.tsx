"use client";

import React, { useRef, useEffect } from "react";

export interface ReceiverBandDisplayProps {
  bands?: number;
  activeBand?: number;
  lastHitBand?: number;
  hitTimestamp?: number;
  bandActivities?: number[];
  onSelectBand?: (band: number) => void;
}


// Single green for all hit blinks
const HIT_GREEN_RGB = "74, 222, 128";

export const ReceiverBandDisplay = React.memo(function ReceiverBandDisplay({
  bands = 36,
  activeBand = 18,
  lastHitBand = -1,
  hitTimestamp = 0,
  bandActivities = [],
  onSelectBand,
}: ReceiverBandDisplayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const animIdRef = useRef<number>(0);
  const blinkRef = useRef<{ band: number; ts: number }>({ band: -1, ts: 0 });

  useEffect(() => {
    if (lastHitBand >= 0 && hitTimestamp > 0 && hitTimestamp !== blinkRef.current.ts) {
      blinkRef.current = { band: lastHitBand, ts: hitTimestamp };
    }
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const render = () => {
      const W = container.clientWidth || 1100;
      const H = container.clientHeight || 120;
      const dpr = window.devicePixelRatio || 1;

      if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
        canvas.width = Math.round(W * dpr);
        canvas.height = Math.round(H * dpr);
      }

      ctx.save();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      // Clean flat dark background — no waves
      ctx.fillStyle = "#060c14";
      ctx.fillRect(0, 0, W, H);

      ctx.restore();
      animIdRef.current = requestAnimationFrame(render);
    };

    animIdRef.current = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animIdRef.current);
  }, []);

  const [tick, setTick] = React.useState(0);
  useEffect(() => {
    const id = setInterval(() => setTick((t) => t + 1), 80);
    return () => clearInterval(id);
  }, []);

  const now = Date.now();
  const blinkBand = blinkRef.current.band;
  const blinkAge = now - blinkRef.current.ts;
  const isBlinkActive = blinkBand >= 0 && blinkAge < 1800;
  const blinkAlpha = isBlinkActive ? 0.5 + 0.5 * Math.sin((blinkAge / 1000) * Math.PI * 5) : 0;

  void tick;

  return (
    <div ref={containerRef} style={{ position: "relative", width: "100%", minHeight: 112 }}>
      <canvas
        ref={canvasRef}
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          borderRadius: 4,
          display: "block",
          pointerEvents: "none",
        }}
      />
      <div
        style={{
          position: "relative",
          zIndex: 1,
          display: "grid",
          gridTemplateColumns: "repeat(12, minmax(0, 1fr))",
          gap: 3,
          padding: "10px 12px",
        }}
      >
        {Array.from({ length: bands }, (_, idx) => {
          const isScanned = idx === activeBand;
          const isHit = isBlinkActive && blinkBand === idx;
          const activity = bandActivities[idx] ?? 0.05;
          const hitBg = "rgba(" + HIT_GREEN_RGB + ", " + (0.10 + blinkAlpha * 0.48) + ")";
          const hitBorder = "rgba(" + HIT_GREEN_RGB + ", " + (0.6 + blinkAlpha * 0.4) + ")";
          const hitShadow = "0 0 " + (8 + Math.round(blinkAlpha * 14)) + "px rgba(" + HIT_GREEN_RGB + ", " + (0.45 + blinkAlpha * 0.45) + ")";

          return (
            <div
              key={idx}
              title={"Band " + (idx + 1) + " (" + (2.0 + idx * 0.5).toFixed(1) + "-" + (2.5 + idx * 0.5).toFixed(1) + " GHz) Activity " + Math.round(activity * 100) + "%"}
              onClick={() => onSelectBand && onSelectBand(idx)}
              style={{
                aspectRatio: "1",
                border: isHit
                  ? "1.5px solid " + hitBorder
                  : isScanned
                  ? "1.5px solid #00e5ff"
                  : "1px solid rgba(55,80,110,0.45)",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                cursor: "pointer",
                borderRadius: 3,
                background: isHit
                  ? hitBg
                  : isScanned
                  ? "rgba(0,229,255,0.20)"
                  : "rgba(77,216,232," + Math.max(0.04, activity * 0.80) + ")",
                boxShadow: isHit
                  ? hitShadow
                  : isScanned
                  ? "0 0 6px rgba(0,229,255,0.50)"
                  : "none",
                color: isScanned ? "#050810" : "rgba(143,163,183,0.9)",
                fontWeight: isScanned || isHit ? "bold" : "normal",
              }}
            >
              <div style={{ fontSize: 7.5, fontFamily: "monospace", lineHeight: 1 }}>
                {"B" + String(idx + 1).padStart(2, "0")}
              </div>
              <div style={{ fontSize: 7, color: isScanned ? "#050810" : "rgba(77,216,232,0.80)", marginTop: 1 }}>
                {Math.round(activity * 100) + "%"}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
});