import React from 'react';
import { LayoutDashboard, Radio, Activity, ShieldAlert, Cpu } from 'lucide-react';

export default function Sidebar() {
  const menuItems = [
    { name: 'Dashboard', icon: LayoutDashboard },
    { name: 'SIH Demo Mode', icon: Cpu },
    { name: 'Spectrum Monitor', icon: Radio, active: true },
    { name: 'Detections', icon: Activity },
    { name: 'Anomalies', icon: ShieldAlert },
    { name: 'Strategy Engine', icon: Cpu },
  ];

  return (
    <div className="w-48 bg-[#090b0e] border-r border-[#1e2532] flex flex-col h-full flex-shrink-0">
      {/* Logo Area */}
      <div className="h-20 flex flex-col items-center justify-center border-b border-[#1e2532] pt-2">
        <div className="flex items-center space-x-1">
           <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="text-[#cda434]">
             <path d="M2 12L12 2L22 12L12 22L2 12Z" stroke="currentColor" strokeWidth="2"/>
             <path d="M7 12L12 7L17 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
           </svg>
        </div>
        <div className="mt-1 flex flex-col items-center">
            <span className="text-[#cda434] text-xs font-bold tracking-widest leading-none">WAVEFORGE</span>
            <span className="text-[#5a6a7a] text-[8px] font-bold tracking-widest mt-0.5">V1.0</span>
        </div>
      </div>

      {/* Menu Items */}
      <div className="flex-1 py-4 flex flex-col gap-1 px-2">
        {menuItems.map((item) => (
          <button
            key={item.name}
            className={`flex items-center space-x-3 w-full px-3 py-2.5 rounded text-[11px] font-semibold transition-colors text-left ${
              item.active 
                ? 'bg-[#cda434] text-[#090b0e] shadow-[0_0_10px_rgba(205,164,52,0.2)]' 
                : 'text-[#6b7b8c] hover:bg-[#121720] hover:text-[#a0aec0]'
            }`}
          >
            {/* We can omit the icon in the sidebar if we want exact video match, but let's keep them very subtle or remove them if the video doesn't have them. Looking closely at the screenshot, there are no icons next to menu items, just text. Let's adjust. */}
            <span className="truncate w-full text-center">{item.name}</span>
          </button>
        ))}
      </div>
      
      {/* Optional: Add a subtle user/settings area at the bottom if needed, video doesn't show it clearly */}
    </div>
  );
}
