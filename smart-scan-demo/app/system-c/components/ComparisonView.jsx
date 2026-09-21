import React from 'react';
import SpectrumMonitor from './SpectrumMonitor';

export default function ComparisonView({
  scenario,
  normalThreshold,
  increasedThreshold,
  onNormalThresholdChange,
  onIncreasedThresholdChange,
  isScanning
}) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      {/* Left Panel: Standard Baseline Threshold */}
      <div className="relative">
        <SpectrumMonitor
          scenario={scenario}
          thresholdDbm={normalThreshold}
          onThresholdChange={onNormalThresholdChange}
          isScanning={isScanning}
          title="RF MONITOR CONTROL (STANDARD THRESHOLD)"
          subtitle={`Threshold: ${normalThreshold.toFixed(1)} dBm`}
        />
        <div className="absolute top-3 right-3 bg-slate-900/90 text-cyan-400 font-mono text-xs px-2.5 py-1 rounded border border-cyan-500/30">
          Standard Mode
        </div>
      </div>

      {/* Right Panel: Increased Threshold (Matching User Screenshot!) */}
      <div className="relative">
        <SpectrumMonitor
          scenario={scenario}
          thresholdDbm={increasedThreshold}
          onThresholdChange={onIncreasedThresholdChange}
          isScanning={isScanning}
          title="RF MONITOR CONTROL (INCREASED THRESHOLD)"
          subtitle="FILTERED NOISE FLOOR"
        />
        <div className="absolute top-3 right-3 bg-rose-950/90 text-rose-300 font-mono text-xs font-bold px-3 py-1 rounded border border-rose-500/40 shadow-lg flex items-center space-x-1">
          <span className="h-2 w-2 rounded-full bg-rose-400 animate-pulse"></span>
          <span>with increased threshold</span>
        </div>
      </div>
    </div>
  );
}
