import React from "react";

export function Panel({ title, unit, children, tags = [] }: { title: string, unit?: string, children: React.ReactNode, tags?: {label: string, type: 'live'|'sim'|'default'}[] }) {
  return (
    <div className="bg-[#0A1220] border border-[#1B324A] mb-3.5 flex flex-col">
      <div className="flex items-center justify-between px-3.5 py-2.5 border-b border-[#1B324A] bg-[#0E1A2E]">
        <span className="text-[11px] font-semibold tracking-wide text-[#D8E4EE] uppercase">
          {title} {unit && <span className="text-[#5E7B96] font-normal ml-1.5 font-mono text-[9.5px] lowercase">{unit}</span>}
        </span>
        {tags.length > 0 && (
          <div className="flex gap-2">
            {tags.map((t, i) => (
              <span key={i} className={`font-mono text-[9px] px-1.5 py-0.5 border tracking-wide uppercase ${
                t.type === 'live' ? 'text-[#4DE87F] border-[#4DE87F]' :
                t.type === 'sim' ? 'text-[#E8B84D] border-[#E8B84D]' :
                'text-[#5E7B96] border-[#2D5578]'
              }`}>
                {t.label}
              </span>
            ))}
          </div>
        )}
      </div>
      <div className="p-3.5 flex-1">
        {children}
      </div>
    </div>
  );
}
