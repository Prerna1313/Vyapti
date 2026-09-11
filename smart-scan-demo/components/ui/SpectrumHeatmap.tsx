"use client";

import { BandActivity } from "@/lib/types";
import clsx from "clsx";

export function SpectrumHeatmap({ bands, activeBand }: { bands: BandActivity[]; activeBand?: number }) {
  return (
    <div>
      <div className="grid grid-cols-12 gap-1 sm:grid-cols-[repeat(18,minmax(0,1fr))] md:grid-cols-[repeat(36,minmax(0,1fr))]">
        {bands.map((b) => {
          const intensity = Math.round(b.activityLevel * 100);
          return (
            <div
              key={b.band}
              title={`Band ${b.band} — ${b.freqMhz} MHz — activity ${intensity}%`}
              className={clsx(
                "aspect-square border",
                b.band === activeBand ? "border-cyan-bright" : "border-steel"
              )}
              style={{
                backgroundColor: `rgba(77, 216, 232, ${Math.max(0.05, b.activityLevel * 0.85)})`,
              }}
            />
          );
        })}
      </div>
      <div className="mt-2.5 flex items-center gap-3 text-[10px] text-dim">
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 border border-steel" style={{ backgroundColor: "rgba(77,216,232,0.08)" }} />
          quiet
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 border border-steel" style={{ backgroundColor: "rgba(77,216,232,0.85)" }} />
          high activity
        </span>
        <span className="flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 border border-cyan-bright" />
          currently scanned band
        </span>
      </div>
    </div>
  );
}
