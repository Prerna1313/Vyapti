import React from 'react';
import { SCENARIOS } from '../utils/rfEngine';
import { Play, Square, RotateCcw, Sliders, Activity, Cpu, Split } from 'lucide-react';

export default function ControlPanel({
  scenario,
  onScenarioChange,
  thresholdDbm,
  onThresholdChange,
  isScanning,
  onToggleScan,
  onResetProfile,
  isComparisonMode,
  onToggleComparisonMode
}) {
  return (
    <div className="bg-[#0b0f17] border border-slate-800 rounded-lg p-4 shadow-xl mb-4 text-slate-200">
      <div className="flex flex-wrap items-center justify-between gap-4">
        {/* Scenario Selection */}
        <div className="flex items-center space-x-2">
          <Activity className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-bold text-slate-300 uppercase tracking-wider">Emitter Scenario:</span>
          <select
            value={scenario}
            onChange={(e) => onScenarioChange(e.target.value)}
            className="bg-slate-900 border border-slate-700 text-cyan-300 text-xs rounded px-3 py-1.5 font-mono focus:outline-none focus:border-cyan-500"
          >
            {Object.values(SCENARIOS).map((sc) => (
              <option key={sc} value={sc}>
                {sc}
              </option>
            ))}
          </select>
        </div>

        {/* Detection Threshold Slider */}
        <div className="flex items-center space-x-3 bg-slate-900 px-3 py-1.5 rounded border border-slate-800">
          <Sliders className="w-4 h-4 text-rose-400" />
          <span className="text-xs font-bold text-slate-300 uppercase">Detection Threshold:</span>
          <input
            type="range"
            min="-110"
            max="-40"
            step="0.5"
            value={thresholdDbm}
            onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
            className="w-32 accent-rose-500 cursor-pointer"
          />
          <span className="font-mono text-xs font-bold text-rose-400 w-16 text-right">
            {thresholdDbm.toFixed(1)} dBm
          </span>
        </div>

        {/* Action Controls */}
        <div className="flex items-center space-x-2">
          <button
            onClick={onToggleScan}
            className={`flex items-center space-x-1.5 px-4 py-1.5 rounded text-xs font-bold transition-all shadow-lg ${
              isScanning
                ? 'bg-rose-600 hover:bg-rose-500 text-white border border-rose-500'
                : 'bg-emerald-600 hover:bg-emerald-500 text-white border border-emerald-500'
            }`}
          >
            {isScanning ? (
              <>
                <Square className="w-3.5 h-3.5 fill-current" />
                <span>STOP SCAN</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>START SCAN</span>
              </>
            )}
          </button>

          <button
            onClick={onResetProfile}
            className="flex items-center space-x-1 px-3 py-1.5 rounded text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            <span>Reset Profile</span>
          </button>

          <button
            onClick={onToggleComparisonMode}
            className={`flex items-center space-x-1.5 px-3 py-1.5 rounded text-xs font-semibold border transition-all ${
              isComparisonMode
                ? 'bg-cyan-500/20 text-cyan-300 border-cyan-500'
                : 'bg-slate-900 text-slate-400 border-slate-800 hover:text-slate-200'
            }`}
          >
            <Split className="w-3.5 h-3.5" />
            <span>{isComparisonMode ? 'Split Comparison (ON)' : 'Side-by-Side View'}</span>
          </button>
        </div>
      </div>

      {/* Receiver Hardware Status Ribbon */}
      <div className="mt-3 pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400 font-mono">
        <div className="flex items-center space-x-4">
          <span className="flex items-center space-x-1.5">
            <Cpu className="w-3.5 h-3.5 text-cyan-400" />
            <span>Receiver Interface: <strong className="text-slate-200">Synthetic SDR Engine (GNU Radio Ready)</strong></span>
          </span>
          <span className="text-slate-600">|</span>
          <span>Sampling Rate: <strong className="text-slate-200">2.4 MSps</strong></span>
          <span className="text-slate-600">|</span>
          <span>Bandwidth: <strong className="text-slate-200">0.1 - 1.0 kHz</strong></span>
        </div>
        <div className="flex items-center space-x-2">
          <span className="inline-block w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
          <span className="text-emerald-400 font-bold">SYSTEM ACTIVE</span>
        </div>
      </div>
    </div>
  );
}
