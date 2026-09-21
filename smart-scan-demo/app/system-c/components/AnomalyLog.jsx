import React, { useState, useEffect } from 'react';
import { AlertTriangle, ShieldAlert } from 'lucide-react';

export default function AnomalyLog({ scenario, isScanning }) {
  const [anomalies, setAnomalies] = useState([]);

  useEffect(() => {
    if (!isScanning || scenario !== 'Anomalous Activity') return;

    const interval = setInterval(() => {
      const now = new Date().toLocaleTimeString();
      const freq = (0.700 + Math.random() * 0.150).toFixed(3);
      const newAnomaly = {
        id: `ANOM-0${Math.floor(Math.random() * 900) + 100}`,
        timestamp: now,
        freq: `${freq} kHz`,
        type: 'Hopping Spread Spectrum Burst',
        power: `${(-45 + Math.random() * 10).toFixed(1)} dBm`,
        confidence: `${Math.floor(88 + Math.random() * 10)}%`
      };

      setAnomalies(prev => [newAnomaly, ...prev.slice(0, 9)]);
    }, 4000);

    return () => clearInterval(interval);
  }, [scenario, isScanning]);

  if (scenario !== 'Anomalous Activity') return null;

  return (
    <div className="bg-[#180d14] border border-rose-900/50 rounded-lg p-4 shadow-xl mb-4 text-slate-200">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center space-x-2 text-rose-400 font-bold text-xs uppercase tracking-wider">
          <ShieldAlert className="w-4 h-4 animate-bounce" />
          <span>Anomalous Activity Detector & Threat Database</span>
        </div>
        <span className="text-[10px] font-mono text-rose-300 bg-rose-950 px-2 py-0.5 rounded border border-rose-800">
          Live Monitoring
        </span>
      </div>

      {anomalies.length === 0 ? (
        <p className="text-xs text-rose-300/60 font-mono italic">Scanning spectrum for anomalous emitters...</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-2">
          {anomalies.map((a) => (
            <div key={a.id} className="bg-rose-950/40 border border-rose-800/40 rounded p-2 text-xs font-mono">
              <div className="flex justify-between font-bold text-rose-300">
                <span>{a.id}</span>
                <span>{a.timestamp}</span>
              </div>
              <div className="text-slate-300 mt-1">Freq: <span className="text-cyan-400">{a.freq}</span></div>
              <div className="text-slate-400">Type: {a.type}</div>
              <div className="flex justify-between text-slate-400 mt-1 text-[11px]">
                <span>Peak: {a.power}</span>
                <span>Conf: {a.confidence}</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
