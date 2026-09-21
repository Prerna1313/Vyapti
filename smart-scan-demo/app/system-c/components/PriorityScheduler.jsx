import React, { useState, useEffect } from 'react';
import { BASE_EMITTERS } from '../utils/rfEngine';

export default function PriorityScheduler({ scenario, thresholdDbm, isScanning }) {
  const [channels, setChannels] = useState([]);

  // Initialize channels
  useEffect(() => {
    const initial = BASE_EMITTERS.map((e, idx) => ({
      id: e.id,
      freq: e.freq,
      bw: e.bw,
      peakPower: e.basePower,
      signalType: e.type,
      status: 'Active',
      hits: Math.floor(Math.random() * 15) + 10,
      misses: 0,
      scanAction: idx % 2 === 0 ? 'Fine Scan' : 'Coarse Scan',
      priorityScore: Math.round((85 - idx * 5) * 10) / 10
    }));

    // Add extra inactive/noise channels
    initial.push(
      { id: 'ID 14', freq: 0.150, bw: 0.010, peakPower: -95, signalType: 'Noise Floor', status: 'Inactive', hits: 2, misses: 5, scanAction: 'Coarse Scan', priorityScore: 12.5 },
      { id: 'ID 15', freq: 0.770, bw: 0.012, peakPower: -92, signalType: 'Transient', status: 'Inactive', hits: 0, misses: 8, scanAction: 'Coarse Scan', priorityScore: 5.0 }
    );

    setChannels(initial);
  }, [scenario]);

  // Dynamic Hit / Miss update loop when scanning
  useEffect(() => {
    if (!isScanning) return;

    const interval = setInterval(() => {
      setChannels(prev =>
        prev.map(ch => {
          // If channel peak power is above thresholdDbm, register a HIT, else register a MISS
          const isAboveThreshold = ch.peakPower >= thresholdDbm;
          
          let hits = ch.hits;
          let misses = ch.misses;
          let status = ch.status;
          let priorityScore = ch.priorityScore;

          if (isAboveThreshold) {
            hits += 1;
            status = 'Active';
            priorityScore = Math.min(100, priorityScore + 2.5);
            // Slowly recover misses on hits
            if (misses > 0 && Math.random() > 0.5) misses -= 1;
          } else {
            misses += 1;
            priorityScore = Math.max(0, priorityScore - 3.0);
            if (misses >= 3) status = 'Inactive';
          }

          const scanAction = priorityScore > 40 ? 'Fine Scan' : 'Coarse Scan';

          return {
            ...ch,
            hits,
            misses,
            status,
            priorityScore: Math.round(priorityScore * 10) / 10,
            scanAction
          };
        })
      );
    }, 1200);

    return () => clearInterval(interval);
  }, [isScanning, thresholdDbm]);

  const getHealthBadge = (misses) => {
    if (misses === 0) {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5"></span>
          Green (0 Miss)
        </span>
      );
    } else if (misses <= 2) {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-amber-500/20 text-amber-400 border border-amber-500/30">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-400 mr-1.5"></span>
          Yellow ({misses} Miss)
        </span>
      );
    } else {
      return (
        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-semibold bg-rose-500/20 text-rose-400 border border-rose-500/30">
          <span className="w-1.5 h-1.5 rounded-full bg-rose-400 mr-1.5"></span>
          Red ({misses}+ Miss)
        </span>
      );
    }
  };

  return (
    <div className="bg-[#0b0f17] border border-slate-800 rounded-lg p-4 shadow-xl">
      <div className="flex items-center justify-between mb-3">
        <div>
          <h3 className="text-sm font-bold text-slate-200 tracking-wide uppercase flex items-center">
            <span className="h-2 w-2 rounded-full bg-cyan-400 mr-2"></span>
            WaveForge Priority Scheduler & Learning Database
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Observes scan feedback, updates hit/miss state, and re-prioritizes scan action.
          </p>
        </div>
        <div className="text-xs font-mono text-slate-400 bg-slate-900 px-3 py-1.5 rounded border border-slate-800">
          Scanned Channels: <strong className="text-cyan-400">{channels.length}</strong>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs text-slate-300 font-mono">
          <thead className="bg-slate-900/80 text-slate-400 uppercase text-[11px] tracking-wider border-b border-slate-800">
            <tr>
              <th className="py-2.5 px-3">Emitter ID</th>
              <th className="py-2.5 px-3">Freq (kHz)</th>
              <th className="py-2.5 px-3">Peak Power</th>
              <th className="py-2.5 px-3">Signal Type</th>
              <th className="py-2.5 px-3">Status</th>
              <th className="py-2.5 px-3">Scan Action</th>
              <th className="py-2.5 px-3">Hits</th>
              <th className="py-2.5 px-3">Misses</th>
              <th className="py-2.5 px-3">Health Status</th>
              <th className="py-2.5 px-3 text-right">Priority Score</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {channels.map((ch) => (
              <tr key={ch.id} className="hover:bg-slate-800/40 transition-colors">
                <td className="py-2.5 px-3 font-bold text-cyan-400">{ch.id}</td>
                <td className="py-2.5 px-3 text-slate-200">{ch.freq.toFixed(3)}</td>
                <td className="py-2.5 px-3 text-slate-300">{ch.peakPower} dBm</td>
                <td className="py-2.5 px-3 text-slate-400">{ch.signalType}</td>
                <td className="py-2.5 px-3">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    ch.status === 'Active' ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-slate-700/30 text-slate-400'
                  }`}>
                    {ch.status}
                  </span>
                </td>
                <td className="py-2.5 px-3">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                    ch.scanAction === 'Fine Scan' ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30' : 'bg-amber-500/10 text-amber-300'
                  }`}>
                    {ch.scanAction}
                  </span>
                </td>
                <td className="py-2.5 px-3 text-emerald-400 font-bold">{ch.hits}</td>
                <td className="py-2.5 px-3 text-rose-400 font-bold">{ch.misses}</td>
                <td className="py-2.5 px-3">{getHealthBadge(ch.misses)}</td>
                <td className="py-2.5 px-3 text-right font-bold text-cyan-400">
                  <div className="flex items-center justify-end space-x-2">
                    <div className="w-16 bg-slate-800 h-1.5 rounded-full overflow-hidden">
                      <div
                        className="bg-cyan-400 h-full rounded-full"
                        style={{ width: `${Math.min(100, ch.priorityScore)}%` }}
                      />
                    </div>
                    <span>{ch.priorityScore.toFixed(1)}</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
