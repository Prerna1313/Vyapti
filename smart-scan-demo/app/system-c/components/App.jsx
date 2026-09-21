import React, { useState } from 'react';
import Sidebar from './Sidebar';
import LeftControlPanel from './LeftControlPanel';
import SpectrumMonitor from './SpectrumMonitor';
import BottomPanel from './BottomPanel';
import { SCENARIOS } from '../utils/rfEngine';

export default function App() {
  const [scenario, setScenario] = useState(SCENARIOS.DYNAMIC);
  const [thresholdDbm, setThresholdDbm] = useState(-80.0);
  const [isScanning, setIsScanning] = useState(true);

  const handleResetProfile = () => {
    setScenario(SCENARIOS.DYNAMIC);
    setThresholdDbm(-80.0);
    setIsScanning(true);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#06080a] text-[#a0aec0] font-sans selection:bg-[#43b0d8]/30">
      
      {/* 1. Left Navigation Sidebar */}
      <Sidebar />

      {/* 2. RF Monitor Control Panel */}
      <LeftControlPanel
        scenario={scenario}
        onScenarioChange={setScenario}
        thresholdDbm={thresholdDbm}
        onThresholdChange={setThresholdDbm}
        isScanning={isScanning}
        onToggleScan={() => setIsScanning(!isScanning)}
        onResetProfile={handleResetProfile}
      />

      {/* 3. Main Content Area (Graphs & Tables) */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        
        {/* Top: Spectrum & Waterfall Graphs */}
        <div className="flex-[3] flex flex-col min-h-0 relative">
          <SpectrumMonitor
            scenario={scenario}
            thresholdDbm={thresholdDbm}
            onThresholdChange={setThresholdDbm}
            isScanning={isScanning}
          />
        </div>

        {/* Bottom: Tabbed Table & Reasoning Panel */}
        <BottomPanel 
          thresholdDbm={thresholdDbm} 
          isScanning={isScanning} 
        />
        
      </div>

    </div>
  );
}
