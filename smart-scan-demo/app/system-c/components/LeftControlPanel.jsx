import React from 'react';
import { SCENARIOS } from '../utils/rfEngine';

export default function LeftControlPanel({
  scenario,
  onScenarioChange,
  thresholdDbm,
  onThresholdChange,
  isScanning,
  onToggleScan,
  onResetProfile
}) {
  return (
    <div className="w-64 bg-[#090c10] border-r border-[#1e2532] h-full flex flex-col flex-shrink-0 text-[#8b9bb4]">
      {/* Header */}
      <div className="px-4 py-3 border-b border-[#1e2532]">
        <h2 className="text-[10px] font-bold text-[#43b0d8] tracking-widest uppercase">RF MONITOR CONTROL</h2>
      </div>

      {/* Scrollable Content Area */}
      <div className="flex-1 overflow-y-auto custom-scrollbar p-3 space-y-4">
        
        {/* RECEIVER / DATA SOURCE */}
        <div>
          <label className="block text-[9px] font-bold text-[#5a6a7a] uppercase mb-1.5 tracking-wider">RECEIVER / DATA SOURCE</label>
          <select className="w-full bg-[#111720] border border-[#1e2532] text-[#c0d0e0] text-[10px] rounded px-2 py-1.5 focus:outline-none mb-2">
            <option>Local Simulation</option>
          </select>
          <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[9px]">
            <span className="text-[#5a6a7a]">Seed:</span>
            <span className="text-[#a0aec0] text-right">42</span>
            <span className="text-[#5a6a7a]">Freq Range:</span>
            <span className="text-[#a0aec0] text-right">100.0 - 1000.0 MHz</span>
            <span className="text-[#5a6a7a]">Emitter Count:</span>
            <span className="text-[#a0aec0] text-right">16 emitters</span>
          </div>
        </div>

        {/* Scenario Preset */}
        <div>
          <label className="block text-[9px] font-bold text-[#5a6a7a] uppercase mb-1.5 tracking-wider">Scenario Preset:</label>
          <select 
            value={scenario}
            onChange={(e) => onScenarioChange(e.target.value)}
            className="w-full bg-[#111720] border border-[#1e2532] text-[#c0d0e0] text-[10px] rounded px-2 py-1.5 focus:outline-none"
          >
            {Object.values(SCENARIOS).map((sc) => (
              <option key={sc} value={sc}>{sc}</option>
            ))}
          </select>
        </div>

        {/* Engine Playback State */}
        <div>
          <label className="block text-[9px] font-bold text-[#5a6a7a] uppercase mb-1.5 tracking-wider">Engine Playback State:</label>
          <div className="grid grid-cols-2 gap-2">
            <button 
              onClick={() => { if(!isScanning) onToggleScan(); }}
              className={`py-1.5 rounded text-[9px] font-bold border transition-colors ${
                isScanning ? 'bg-[#0f241a] text-[#4ade80] border-[#225035]' : 'bg-[#111720] text-[#5a6a7a] border-[#1e2532] hover:bg-[#151c27]'
              }`}
            >
              START
            </button>
            <button 
               onClick={() => { if(isScanning) onToggleScan(); }}
              className={`py-1.5 rounded text-[9px] font-bold border transition-colors ${
                !isScanning ? 'bg-[#2a2210] text-[#fcd34d] border-[#5e4b20]' : 'bg-[#111720] text-[#5a6a7a] border-[#1e2532] hover:bg-[#151c27]'
              }`}
            >
              PAUSE
            </button>
            <button className="py-1.5 rounded text-[9px] font-bold border bg-[#111720] text-[#5a6a7a] border-[#1e2532] hover:bg-[#151c27]">
              STOP
            </button>
            <button 
              onClick={onResetProfile}
              className="py-1.5 rounded text-[9px] font-bold border bg-[#111720] text-[#5a6a7a] border-[#1e2532] hover:bg-[#151c27]"
            >
              RESET
            </button>
          </div>
        </div>

        {/* Simulation Speed */}
        <div>
          <label className="block text-[9px] font-bold text-[#5a6a7a] uppercase mb-1.5 tracking-wider">Simulation Speed:</label>
          <select className="w-full bg-[#111720] border border-[#1e2532] text-[#c0d0e0] text-[10px] rounded px-2 py-1.5 focus:outline-none">
            <option>1.00x</option>
          </select>
        </div>

        <hr className="border-[#1e2532]" />

        {/* Frequencies */}
        <div>
          <label className="block text-[9px] font-bold text-[#5a6a7a] uppercase mb-1.5 tracking-wider">Start Frequency (MHz):</label>
          <div className="bg-[#111720] border border-[#1e2532] text-[#c0d0e0] text-[10px] rounded px-2 py-1.5 flex justify-between items-center mb-3">
             <span>100.00</span>
             <div className="flex flex-col opacity-50"><span className="text-[6px]">▲</span><span className="text-[6px]">▼</span></div>
          </div>
          
          <label className="block text-[9px] font-bold text-[#5a6a7a] uppercase mb-1.5 tracking-wider">Stop Frequency (MHz):</label>
          <div className="bg-[#111720] border border-[#1e2532] text-[#c0d0e0] text-[10px] rounded px-2 py-1.5 flex justify-between items-center">
             <span>1000.00</span>
             <div className="flex flex-col opacity-50"><span className="text-[6px]">▲</span><span className="text-[6px]">▼</span></div>
          </div>
        </div>

        {/* Detection Threshold */}
        <div>
          <label className="block text-[9px] font-bold text-[#5a6a7a] uppercase mb-1.5 tracking-wider">Detection Threshold (dBm):</label>
          <input
            type="range"
            min="-110"
            max="-40"
            step="0.5"
            value={thresholdDbm}
            onChange={(e) => onThresholdChange(parseFloat(e.target.value))}
            className="w-full accent-[#e05252] cursor-pointer h-1 bg-[#1e2532] rounded outline-none appearance-none"
            style={{
               background: `linear-gradient(to right, #e05252 0%, #e05252 ${(thresholdDbm - -110) / (-40 - -110) * 100}%, #1e2532 ${(thresholdDbm - -110) / (-40 - -110) * 100}%, #1e2532 100%)`
            }}
          />
          <div className="text-[10px] font-bold text-[#43b0d8] mt-1">{thresholdDbm.toFixed(1)} dBm</div>
        </div>

        <hr className="border-[#1e2532]" />

        {/* Power Scale */}
        <div>
          <label className="block text-[9px] font-bold text-[#5a6a7a] uppercase mb-1.5 tracking-wider">Power Scale (dBm Range):</label>
          <div className="grid grid-cols-2 gap-2 mb-2">
             <div className="bg-[#111720] border border-[#1e2532] text-[#c0d0e0] text-[10px] rounded px-2 py-1 flex justify-between items-center">
                <span>-120.00 Min</span>
                <span className="text-[6px] opacity-50">▼</span>
             </div>
             <div className="bg-[#111720] border border-[#1e2532] text-[#c0d0e0] text-[10px] rounded px-2 py-1 flex justify-between items-center">
                <span>-20.00 Max</span>
                <span className="text-[6px] opacity-50">▼</span>
             </div>
          </div>
          <button className="w-full py-1.5 rounded text-[9px] font-bold border bg-[#111720] text-[#5a6a7a] border-[#1e2532] hover:bg-[#151c27] mb-2 uppercase">
             PAUSE VISUALIZATION
          </button>
          <button className="w-full py-1.5 rounded text-[9px] font-bold border bg-[#111720] text-[#5a6a7a] border-[#1e2532] hover:bg-[#151c27] uppercase">
             RESET PLOT VIEW
          </button>
        </div>

      </div>

      {/* Footer / Diagnostics */}
      <div className="px-4 py-2 border-t border-[#1e2532] bg-[#07090c]">
        <h2 className="text-[8px] font-bold text-[#1f3a47] tracking-widest uppercase">SYSTEM DIAGNOSTICS</h2>
      </div>
    </div>
  );
}
