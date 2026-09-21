import React, { useRef, useEffect, useState } from 'react';
import {
  FREQ_MIN,
  FREQ_MAX,
  NUM_BINS,
  generateSpectrumFrame,
  detectPeaks,
  getVerticalCursorBins,
  dbmToThermalColor
} from '../utils/rfEngine';

export default function SpectrumMonitor({
  scenario,
  thresholdDbm,
  onThresholdChange,
  isScanning
}) {
  const psdCanvasRef = useRef(null);
  const stftCanvasRef = useRef(null);
  const isDraggingRef = useRef(false);
  const containerRef = useRef(null);

  const [dimensions, setDimensions] = useState({ width: 800, height: 400 });
  const waterfallHistoryRef = useRef([]);
  const maxHistoryRows = 160;

  // Handle Resize
  useEffect(() => {
    const handleResize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight
        });
      }
    };
    
    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Animation loop
  useEffect(() => {
    let animId;
    let time = 0;
    
    // Exact sizing proportions
    const psdHeight = Math.floor(dimensions.height * 0.58);
    const stftHeight = dimensions.height - psdHeight - 3; // Space for divider

    const render = () => {
      time += 0.04;

      const { psd, activeEmitters } = generateSpectrumFrame(scenario, time);
      const peaks = detectPeaks(psd, thresholdDbm, activeEmitters);
      const verticalCursors = getVerticalCursorBins(activeEmitters);

      // --- COLORS FROM REFERENCE ---
      const bgDark = '#0b0e14';
      const gridColor = 'rgba(255, 255, 255, 0.04)';
      const axisLineColor = '#283142';
      const textColor = '#5a6a7a';
      const labelColor = '#c0d0e0';
      const traceColor = '#56b6c2';
      const cursorColor = '#d03a58';
      const triangleColor = '#4db88d';

      // --- 1. RENDER PSD TOP CANVAS ---
      const psdCanvas = psdCanvasRef.current;
      if (psdCanvas) {
        psdCanvas.width = dimensions.width;
        psdCanvas.height = psdHeight;
        const ctx = psdCanvas.getContext('2d');
        const width = psdCanvas.width;
        const height = psdCanvas.height;

        const paddingLeft = 50;
        const paddingRight = 20;
        const paddingTop = 25;
        const paddingBottom = 0; // STFT will have padding bottom

        const graphWidth = width - paddingLeft - paddingRight;
        const graphHeight = height - paddingTop - paddingBottom;

        // Background
        ctx.fillStyle = bgDark;
        ctx.fillRect(0, 0, width, height);

        // Grid (Horizontal)
        const dbmLevels = [-20, -40, -60, -80, -100, -120];
        ctx.strokeStyle = gridColor;
        ctx.lineWidth = 1;
        dbmLevels.forEach((val) => {
          const y = paddingTop + ((val - (-20)) / (-120 - (-20))) * graphHeight;
          ctx.beginPath();
          ctx.moveTo(paddingLeft, y);
          ctx.lineTo(width - paddingRight, y);
          ctx.stroke();
        });

        // Grid (Vertical)
        const freqSteps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        freqSteps.forEach((f) => {
          const x = paddingLeft + ((f - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)) * graphWidth;
          ctx.beginPath();
          ctx.moveTo(x, paddingTop);
          ctx.lineTo(x, height);
          ctx.stroke();
        });

        // Axes Lines
        ctx.strokeStyle = axisLineColor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        // Left Y Axis
        ctx.moveTo(paddingLeft, paddingTop);
        ctx.lineTo(paddingLeft, height);
        // Right bound (subtle)
        ctx.moveTo(width - paddingRight, paddingTop);
        ctx.lineTo(width - paddingRight, height);
        // Top bound
        ctx.moveTo(paddingLeft, paddingTop);
        ctx.lineTo(width - paddingRight, paddingTop);
        ctx.stroke();

        // Y-Axis Tick marks & Labels
        ctx.fillStyle = textColor;
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        
        dbmLevels.forEach((val) => {
          const y = paddingTop + ((val - (-20)) / (-120 - (-20))) * graphHeight;
          // Tick mark pointing inwards
          ctx.beginPath();
          ctx.moveTo(paddingLeft, y);
          ctx.lineTo(paddingLeft + 3, y);
          ctx.stroke();
          // Text label
          ctx.fillText(`${val}`, paddingLeft - 5, y);
        });

        // Vertical Red Dashed Cursors
        ctx.save();
        ctx.strokeStyle = cursorColor;
        ctx.lineWidth = 1.0;
        ctx.setLineDash([3, 3]);
        verticalCursors.forEach((cursor) => {
          const x = paddingLeft + (cursor.binIndex / NUM_BINS) * graphWidth;
          ctx.beginPath();
          ctx.moveTo(x, paddingTop);
          ctx.lineTo(x, height);
          ctx.stroke();
        });
        ctx.restore();

        // PSD Trace Line
        ctx.save();
        ctx.strokeStyle = traceColor;
        ctx.lineWidth = 1.5; // Thicker trace
        ctx.shadowColor = traceColor; // Subtle glow
        ctx.shadowBlur = 4;
        ctx.beginPath();

        for (let i = 0; i < NUM_BINS; i++) {
          const x = paddingLeft + (i / NUM_BINS) * graphWidth;
          const dbm = Math.max(-125, Math.min(-15, psd[i]));
          const y = paddingTop + ((dbm - (-20)) / (-120 - (-20))) * graphHeight;

          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();

        // Horizontal Threshold Line
        ctx.save();
        const thresholdY = paddingTop + ((thresholdDbm - (-20)) / (-120 - (-20))) * graphHeight;
        ctx.strokeStyle = cursorColor;
        ctx.lineWidth = 1.0;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(paddingLeft, thresholdY);
        ctx.lineTo(width - paddingRight, thresholdY);
        ctx.stroke();
        ctx.restore();

        // Green Triangles & Text Labels
        peaks.forEach((peak) => {
          const x = paddingLeft + (peak.binIndex / NUM_BINS) * graphWidth;
          const dbm = Math.max(-125, Math.min(-15, peak.power));
          const y = paddingTop + ((dbm - (-20)) / (-120 - (-20))) * graphHeight;

          // Tiny Mint Green Triangle pointing UP ▲
          ctx.fillStyle = triangleColor;
          ctx.beginPath();
          ctx.moveTo(x, y);
          ctx.lineTo(x - 2, y + 4);
          ctx.lineTo(x + 2, y + 4);
          ctx.closePath();
          ctx.fill();

          // Crisp Label Text
          ctx.fillStyle = labelColor;
          ctx.font = '7.5px sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText(peak.labelLine1, x, y - 11);
          ctx.fillText(peak.labelLine2, x, y - 4);
        });

        // Y-Axis Title
        ctx.save();
        ctx.translate(12, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillStyle = textColor;
        ctx.font = '9px sans-serif';
        ctx.fillText('Power Spectral Density (dBm)', 0, 0);
        ctx.restore();
      }

      // --- 2. RENDER STFT WATERFALL BOTTOM CANVAS ---
      if (isScanning) {
        const rowData = new Uint8ClampedArray(NUM_BINS * 4);
        for (let i = 0; i < NUM_BINS; i++) {
          const [r, g, b, a] = dbmToThermalColor(psd[i]);
          const idx = i * 4;
          rowData[idx] = r;
          rowData[idx + 1] = g;
          rowData[idx + 2] = b;
          rowData[idx + 3] = a;
        }

        waterfallHistoryRef.current.unshift(rowData);
        if (waterfallHistoryRef.current.length > maxHistoryRows) {
          waterfallHistoryRef.current.pop();
        }
      }

      const stftCanvas = stftCanvasRef.current;
      if (stftCanvas) {
        stftCanvas.width = dimensions.width;
        stftCanvas.height = stftHeight;
        const ctx = stftCanvas.getContext('2d');
        const width = stftCanvas.width;
        const height = stftCanvas.height;

        const paddingLeft = 50;
        const paddingRight = 20;
        const paddingTop = 0;
        const paddingBottom = 25; // Space for X axis

        const graphWidth = width - paddingLeft - paddingRight;
        const graphHeight = height - paddingTop - paddingBottom;

        // Background
        ctx.fillStyle = bgDark;
        ctx.fillRect(0, 0, width, height);

        // Vertical Grid (continues from top)
        ctx.strokeStyle = gridColor;
        ctx.lineWidth = 1;
        const freqSteps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
        freqSteps.forEach((f) => {
          const x = paddingLeft + ((f - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)) * graphWidth;
          ctx.beginPath();
          ctx.moveTo(x, paddingTop);
          ctx.lineTo(x, height - paddingBottom);
          ctx.stroke();
        });

        // Draw Waterfall Heatmap Image
        const history = waterfallHistoryRef.current;
        if (history.length > 0) {
          const tempCanvas = document.createElement('canvas');
          tempCanvas.width = NUM_BINS;
          tempCanvas.height = history.length;
          const tempCtx = tempCanvas.getContext('2d');
          const imgData = tempCtx.createImageData(NUM_BINS, history.length);

          for (let r = 0; r < history.length; r++) {
            const row = history[r];
            const destOffset = r * NUM_BINS * 4;
            imgData.data.set(row, destOffset);
          }
          tempCtx.putImageData(imgData, 0, 0);

          ctx.drawImage(tempCanvas, paddingLeft, paddingTop, graphWidth, graphHeight);
        }

        // Axes Lines
        ctx.strokeStyle = axisLineColor;
        ctx.lineWidth = 1;
        ctx.beginPath();
        // Left Y Axis
        ctx.moveTo(paddingLeft, paddingTop);
        ctx.lineTo(paddingLeft, height - paddingBottom);
        // Right bound
        ctx.moveTo(width - paddingRight, paddingTop);
        ctx.lineTo(width - paddingRight, height - paddingBottom);
        // Bottom X Axis
        ctx.moveTo(paddingLeft, height - paddingBottom);
        ctx.lineTo(width - paddingRight, height - paddingBottom);
        ctx.stroke();

        // Left Tick marks
        ctx.beginPath();
        for (let i = 1; i < 4; i++) {
           const y = paddingTop + (i / 4) * graphHeight;
           ctx.moveTo(paddingLeft, y);
           ctx.lineTo(paddingLeft + 3, y);
        }
        ctx.stroke();

        // Vertical Red Dashed Cursors (continuing through STFT)
        ctx.save();
        ctx.strokeStyle = cursorColor;
        ctx.lineWidth = 1.0;
        ctx.setLineDash([3, 3]);
        verticalCursors.forEach((cursor) => {
          const x = paddingLeft + (cursor.binIndex / NUM_BINS) * graphWidth;
          ctx.beginPath();
          ctx.moveTo(x, paddingTop);
          ctx.lineTo(x, paddingTop + graphHeight);
          ctx.stroke();
        });
        ctx.restore();

        // X Axis Frequency Labels & Tick Marks
        ctx.fillStyle = textColor;
        ctx.font = '9px sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        
        ctx.beginPath();
        ctx.strokeStyle = axisLineColor;

        freqSteps.forEach((f) => {
          const x = paddingLeft + ((f - FREQ_MIN) / (FREQ_MAX - FREQ_MIN)) * graphWidth;
          // tick mark pointing UPWARDS
          ctx.moveTo(x, height - paddingBottom);
          ctx.lineTo(x, height - paddingBottom - 3);
          // Label
          ctx.fillText(f.toFixed(1), x, height - paddingBottom + 4);
        });
        ctx.stroke();

        // X Axis Title
        ctx.fillStyle = '#6b7b8c';
        ctx.font = '8px sans-serif';
        // The image has a typo: Frequency (kMHz)
        ctx.fillText('Frequency (kMHz)', paddingLeft + graphWidth / 2, height - 10);

        // Y Axis Title
        ctx.save();
        ctx.translate(12, height / 2 - 10);
        ctx.rotate(-Math.PI / 2);
        ctx.textAlign = 'center';
        ctx.fillStyle = textColor;
        ctx.font = '9px sans-serif';
        ctx.fillText('STFT History (Rows)', 0, 0);
        ctx.restore();
      }

      animId = requestAnimationFrame(render);
    };

    render();

    return () => cancelAnimationFrame(animId);
  }, [scenario, thresholdDbm, isScanning, dimensions]);

  // Drag threshold line handler
  const handleMouseDown = (e) => {
    isDraggingRef.current = true;
    updateThresholdFromMouse(e);
  };

  const handleMouseMove = (e) => {
    if (isDraggingRef.current) {
      updateThresholdFromMouse(e);
    }
  };

  const handleMouseUp = () => {
    isDraggingRef.current = false;
  };

  const updateThresholdFromMouse = (e) => {
    const canvas = psdCanvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const mouseY = e.clientY - rect.top;

    const paddingTop = 25;
    const paddingBottom = 0;
    const graphHeight = canvas.height - paddingTop - paddingBottom;

    const clampedY = Math.max(paddingTop, Math.min(canvas.height - paddingBottom, mouseY));
    const ratio = (clampedY - paddingTop) / graphHeight;

    const newThreshold = -20 + ratio * (-120 - (-20));
    onThresholdChange(Math.round(newThreshold * 10) / 10);
  };

  return (
    <div ref={containerRef} className="flex-1 flex flex-col bg-[#0b0e14] select-none min-h-[300px]">
      <div className="relative cursor-ns-resize" style={{ height: '58%' }}>
        <canvas
          ref={psdCanvasRef}
          className="block absolute top-0 left-0"
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        />
      </div>

      {/* Divider */}
      <div className="h-[3px] bg-[#1a1f2c] w-full border-y border-[#06080a]"></div>

      <div className="relative flex-1">
        <canvas
          ref={stftCanvasRef}
          className="block absolute top-0 left-0"
        />
      </div>
    </div>
  );
}
