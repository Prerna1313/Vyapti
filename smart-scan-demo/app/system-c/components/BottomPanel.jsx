import React, { useState, useEffect } from 'react';
import { BASE_EMITTERS } from '../utils/rfEngine';

export default function BottomPanel({ thresholdDbm, isScanning }) {
  const [channels, setChannels] = useState([]);
  const [activeTab, setActiveTab] = useState('STATEFUL EMITTER REGISTRY');

  const tabs = [
    'STATEFUL EMITTER REGISTRY',
    'PRIORITY SCAN SCHEDULER',
    'ADAPTIVE SCAN STRATEGY',
    'FREQ + TIME HEATMAP',
    'LEARNING CURVE'
  ];

  // Initialize channels
  useEffect(() => {
    const initial = BASE_EMITTERS.filter(e => !e.markerOnly).map((e, idx) => ({
      id: e.id,
      freq: e.freq,
      bw: e.bw,
      peakPower: e.basePower,
      signalType: e.type,
      status: 'ACTIVE',
      hits: Math.floor(Math.random() * 50) + 10,
      misses: 0,
      confidence: `${(40 + Math.random() * 50).toFixed(1)}%`
    }));

    initial.push(
      { id: '3', freq: 462.9883, bw: 4066.9, peakPower: -70.3, signalType: 'Multi-Tone', status: 'ACTIVE', hits: 187, misses: 0, confidence: '55.2%' },
      { id: '11', freq: 280.1758, bw: 5273.4, peakPower: -65.9, signalType: 'Multi-Tone', status: 'INACTIVE', hits: 42, misses: 8, confidence: '38.0%' }
    );

    setChannels(initial.sort((a, b) => a.id - b.id));
  }, []);

  // Dynamic Hit / Miss update loop when scanning
  useEffect(() => {
    if (!isScanning) return;

    const interval = setInterval(() => {
      setChannels(prev =>
        prev.map(ch => {
          const isAboveThreshold = ch.peakPower >= thresholdDbm;
          let hits = ch.hits;
          let misses = ch.misses;
          let status = ch.status;

          if (isAboveThreshold) {
            hits += 1;
            status = 'ACTIVE';
            if (misses > 0 && Math.random() > 0.5) misses -= 1;
          } else {
            misses += 1;
            if (misses >= 3) status = 'INACTIVE';
          }

          return { ...ch, hits, misses, status };
        })
      );
    }, 1200);

    return () => clearInterval(interval);
  }, [isScanning, thresholdDbm]);

  return (
    <div className="flex-1 min-h-[220px] bg-[#090c10] border-t border-[#1e2532] flex flex-col font-sans">
      {/* Tabs */}
      <div className="flex border-b border-[#1e2532] bg-[#0b0e14]">
        {tabs.map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 text-[9px] font-bold tracking-wider uppercase transition-colors ${
              activeTab === tab 
                ? 'text-[#43b0d8] border-b-2 border-[#43b0d8] bg-[#090c10]' 
                : 'text-[#5a6a7a] border-b-2 border-transparent hover:text-[#8b9bb4]'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Table Area */}
        <div className="flex-[3] overflow-auto custom-scrollbar p-2">
          {activeTab === 'STATEFUL EMITTER REGISTRY' && (
            <table className="w-full text-left text-[10px] text-[#8b9bb4] font-mono border-collapse">
              <thead className="text-[#5a6a7a] border-b border-[#1e2532]">
                <tr>
                  <th className="py-2 px-3 font-normal">ID</th>
                  <th className="py-2 px-3 font-normal">Frequency (MHz)</th>
                  <th className="py-2 px-3 font-normal">Bandwidth (kHz)</th>
                  <th className="py-2 px-3 font-normal">Peak Power (dBm)</th>
                  <th className="py-2 px-3 font-normal">Signal Type</th>
                  <th className="py-2 px-3 font-normal">Confidence (%)</th>
                  <th className="py-2 px-3 font-normal">Status</th>
                  <th className="py-2 px-3 font-normal text-right">Hits / Misses</th>
                </tr>
              </thead>
              <tbody>
                {channels.map((ch, idx) => (
                  <tr key={idx} className="border-b border-[#141a24] hover:bg-[#111720] transition-colors cursor-pointer">
                    <td className="py-1.5 px-3">{ch.id.replace('ID ', '')}</td>
                    <td className="py-1.5 px-3 text-[#c0d0e0]">{(ch.freq * 1000).toFixed(4)}</td>
                    <td className="py-1.5 px-3">{ch.bw * 1000}</td>
                    <td className="py-1.5 px-3">{ch.peakPower.toFixed(1)}</td>
                    <td className="py-1.5 px-3">{ch.signalType}</td>
                    <td className="py-1.5 px-3">{ch.confidence}</td>
                    <td className="py-1.5 px-3">
                      <span className={ch.status === 'ACTIVE' ? 'text-[#4ade80]' : 'text-[#f87171]'}>{ch.status}</span>
                    </td>
                    <td className="py-1.5 px-3 text-right">
                       <span className="text-[#4ade80]">{ch.hits}</span>
                       <span className="text-[#5a6a7a] mx-1">/</span>
                       <span className={ch.misses === 0 ? 'text-[#4ade80]' : ch.misses < 3 ? 'text-[#fcd34d]' : 'text-[#f87171]'}>{ch.misses}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {activeTab !== 'STATEFUL EMITTER REGISTRY' && (
             <div className="flex items-center justify-center h-full text-[10px] text-[#5a6a7a]">
                Select an active signal track to view ML reasoning.
             </div>
          )}
        </div>

        {/* Reasoning Panel */}
        <div className="flex-1 bg-[#05070a] border-l border-[#1e2532] p-4 flex flex-col justify-end text-[9px] text-[#6b7b8c] font-mono relative">
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-20">
             {/* Decorative reasoning placeholder text in middle if empty */}
             {activeTab !== 'STATEFUL EMITTER REGISTRY' && "Select an active scan task to view Priority reasoning."}
          </div>
          {activeTab === 'STATEFUL EMITTER REGISTRY' && (
             <>
              <div className="mb-4">
                <div className="flex justify-between border-b border-[#1e2532] pb-1 mb-1">
                   <span>Duration:</span><span className="text-[#c0d0e0]">38.00 s</span>
                </div>
                <div className="flex justify-between border-b border-[#1e2532] pb-1 mb-1">
                   <span>Pulse Width:</span><span className="text-[#c0d0e0]">N/A</span>
                </div>
                <div className="flex justify-between border-b border-[#1e2532] pb-1 mb-1">
                   <span>Duty Cycle:</span><span className="text-[#c0d0e0]">1.000 (100%)</span>
                </div>
              </div>

              <h4 className="text-[#43b0d8] font-bold uppercase mb-2 tracking-wider">REASONING</h4>
              <ul className="space-y-1.5 leading-tight list-disc pl-3">
                <li>Multiple distinct carrier sub-peaks detected (peaks: 5)</li>
                <li>Flat-topped multi-carrier occupied bandwidth</li>
                <li>High temporal consistency across all tones</li>
              </ul>
              
              <div className="mt-4 pt-3 border-t border-[#1e2532] space-y-1">
                <div className="flex items-center space-x-2">
                   <div className="w-2 h-2 rounded-full bg-[#4ade80]"></div>
                   <span>0 misses <span className="text-[#4ade80]">&rarr;</span> signal locked</span>
                </div>
                <div className="flex items-center space-x-2">
                   <div className="w-2 h-2 rounded-full bg-[#fcd34d]"></div>
                   <span>1-2 misses <span className="text-[#fcd34d]">&rarr;</span> signal fading</span>
                </div>
                <div className="flex items-center space-x-2">
                   <div className="w-2 h-2 rounded-full bg-[#f87171]"></div>
                   <span>3+ misses <span className="text-[#f87171]">&rarr;</span> signal likely lost</span>
                </div>
              </div>
             </>
          )}
        </div>
      </div>
    </div>
  );
}
