"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const PAGES = [
  { id: 'mission', num: '01', label: 'MISSION', href: '/' },
  { id: 'spectrum', num: '02', label: 'SPECTRUM', href: '/spectrum' },
  { id: 'emitters', num: '03', label: 'EMITTERS', href: '/emitters' },
  { id: 'simulation', num: '04', label: 'SIMULATION', href: '/simulation' },
  { id: 'strategy', num: '05', label: 'STRATEGIES', href: '/strategies' },
  { id: 'comparison', num: '06', label: 'COMPARISON', href: '/comparison' },
  { id: 'tsrd', num: '07', label: 'TSRD', href: '/tsrd' },
  { id: 'pdw', num: '08', label: 'PDW', href: '/pdw' },
  { id: 'detection', num: '09', label: 'DETECTION', href: '/detection' },
  { id: 'experiments', num: '10', label: 'EXPERIMENTS', href: '/experiments' },
  { id: 'validation', num: '11', label: 'VALIDATION', href: '/validation' },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <nav className="w-[196px] shrink-0 bg-[#071019] border-r border-[#1B324A] flex flex-col">
      <div className="px-3.5 pt-4 pb-3 border-b border-[#1B324A]">
        <div className="font-mono text-[15px] font-bold text-[#4DD8E8] tracking-wide">VYAPTI</div>
        <div className="text-[10px] text-[#5E7B96] mt-1 leading-relaxed">
          Cognitive Smart Scan<br />for Electronic Support (ES)
        </div>
      </div>
      <div className="flex-1 overflow-y-auto py-2">
        {PAGES.map((p) => {
          const isActive = pathname === p.href || (pathname === '/' && p.id === 'mission');
          return (
            <Link key={p.id} href={p.href} className={`flex items-center gap-2.5 px-3.5 py-2 text-[11.5px] font-medium tracking-wide border-l-2 select-none transition-colors ${
              isActive 
                ? 'bg-[#0E1A2E] text-[#4DD8E8] border-[#4DD8E8]' 
                : 'text-[#5E7B96] border-transparent hover:bg-[#0A1220] hover:text-[#D8E4EE]'
            }`}>
              <span className={`font-mono text-[9px] w-3.5 ${isActive ? 'text-[#4DD8E8]' : 'text-[#3D5468]'}`}>{p.num}</span>
              <span>{p.label}</span>
            </Link>
          );
        })}
      </div>
      <div className="px-3.5 py-2.5 border-t border-[#1B324A] font-mono text-[9.5px] text-[#3D5468] leading-relaxed">
        <span className="inline-block w-1.5 h-1.5 rounded-full bg-[#4DE87F] mr-1.5 shadow-[0_0_4px_#4DE87F]"></span>
        SIM ENGINE ONLINE<br />
        BUILD 0.9.3-research
      </div>
    </nav>
  );
}
