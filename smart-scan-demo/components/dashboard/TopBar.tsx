"use client";

import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { useBackendMode } from "@/lib/useBackendMode";
import { Tag } from "@/components/ui/Tag";

const TITLES: Record<string, string> = {
  "/mission": "Mission Dashboard",
  "/environment": "RF Simulation / Environment",
  "/scheduler": "Scheduler",
  "/live": "Live Experiment",
  "/results": "Results & Analysis",
  "/learning": "Emitter Pattern Learning",
  "/validation": "Validation",
  "/documentation": "Documentation",
};

function useClock() {
  const [now, setNow] = useState<Date | null>(null);
  useEffect(() => {
    setNow(new Date());
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);
  return now;
}

export default function TopBar() {
  const pathname = usePathname() || "/mission";
  const { mode, connected } = useBackendMode();
  const now = useClock();
  const title = TITLES[pathname] ?? "Vyapti Console";

  return (
    <div className="flex h-11 flex-shrink-0 items-center justify-between border-b border-steel bg-panel-deep px-[18px]">
      <div className="flex items-center gap-2.5">
        <span className="text-[12.5px] font-semibold text-primary">{title}</span>
        <span className="font-mono text-[10px] text-faint">/ VYAPTI / EW CONSOLE</span>
      </div>
      <div className="flex items-center gap-4 font-mono text-[10.5px] text-dim">
        <Tag variant={mode === "LIVE" ? "live" : "demo"}>{mode === "LIVE" ? "LIVE — PYTHON CONNECTED" : "DEMO MODE — REPLAYING SAVED RESULTS"}</Tag>
        <div className="flex items-center gap-1.5">
          SYSTEM
          <span className={connected ? "font-semibold text-confirm" : "font-semibold text-amber"}>
            {connected ? "ONLINE" : "OFFLINE (adapter not reachable)"}
          </span>
        </div>
        <div>
          T+ <span className="font-semibold text-cyan">{now ? now.toLocaleTimeString("en-GB", { hour12: false }) : "--:--:--"}</span>
        </div>
      </div>
    </div>
  );
}
