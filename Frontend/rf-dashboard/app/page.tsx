"use client";

import { useEffect, useRef, useState } from "react";

type Dashboard = "physics" | "receiver";
const details: Record<Dashboard, { label: string; sub: string; viewBox: string; path: string }> = {
  physics: { label: "RF Physics", sub: "Simulation", viewBox: "0 0 24 24", path: "M3 12h3l3-8 4 16 3-8 3 4 2-4" },
  receiver: { label: "RF Receiver", sub: "Scheduler", viewBox: "0 0 28 18", path: "M2 9H7L10 2L14 16L18 6L21 12L24 9H28" },
};

function WaveIcon({ className, viewBox, path }: { className: string; viewBox: string; path: string }) {
  return <svg className={className} viewBox={viewBox} fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"><path d={path} /></svg>;
}

export default function Home() {
  const [selected, setSelected] = useState<Dashboard>("physics");
  const [open, setOpen] = useState(false);
  const [receiverLoaded, setReceiverLoaded] = useState(false);
  const dropdown = useRef<HTMLDivElement>(null);
  const current = details[selected];

  useEffect(() => {
    const onPointerDown = (event: MouseEvent) => {
      if (open && !dropdown.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("click", onPointerDown);
    return () => document.removeEventListener("click", onPointerDown);
  }, [open]);

  function selectPage(page: Dashboard) {
    setSelected(page);
    setOpen(false);
    if (page === "receiver") setReceiverLoaded(true);
  }

  return <>
    <div className="page-switcher-bar" id="pageSwitcherBar">
      <div className="ps-brand">
        <img src="/legacy/ultraHD-logo.png" alt="VYAPTI Logo" className="ps-logo-img" />
        <span className="ps-brand-name">VYAPTI</span>
      </div>
      <span className="ps-divider" />
      <div className="ps-dropdown" id="psDropdown" ref={dropdown}>
        <button className={`ps-dropdown-btn${open ? " open" : ""}`} id="psDropdownBtn" onClick={() => setOpen(value => !value)} aria-haspopup="listbox" aria-expanded={open}>
          <WaveIcon className="ps-dd-icon" viewBox={current.viewBox} path={current.path} />
          <div className="ps-dd-labels"><span className="ps-dd-label">{current.label}</span><span className="ps-dd-sub">{current.sub}</span></div>
          <svg className="ps-dd-chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="6 9 12 15 18 9" /></svg>
        </button>
        <div className={`ps-dropdown-menu${open ? " open" : ""}`} id="psDropdownMenu" role="listbox">
          {(["physics", "receiver"] as Dashboard[]).map(page => <div key={page} className={`ps-menu-item${selected === page ? " active" : ""}`} role="option" aria-selected={selected === page} onClick={() => selectPage(page)}>
            <WaveIcon className="ps-item-icon" viewBox={details[page].viewBox} path={details[page].path} />
            <div className="ps-item-labels"><span className="ps-item-label">{details[page].label}</span><span className="ps-item-sub">{details[page].sub}</span></div>
            <svg className="ps-item-check" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round"><polyline points="20 6 9 17 4 12" /></svg>
          </div>)}
          <a className="ps-menu-item" role="option" href="/tsrd/index.html" aria-label="Open TSRD dashboard">
            <WaveIcon className="ps-item-icon" viewBox="0 0 24 24" path="M3 12h3l3-8 4 16 3-8 3 4 2-4" />
            <div className="ps-item-labels"><span className="ps-item-label">TSRD</span><span className="ps-item-sub">Dashboard</span></div>
          </a>
        </div>
      </div>
    </div>
    <div className={`page-iframe-wrap${selected === "physics" ? "" : " page-iframe-wrap--hidden"}`} id="pagePhysics">
      <iframe id="framePhysics" src="/legacy/page-physics.html" className="page-iframe" title="RF Physics Simulation" />
    </div>
    <div className={`page-iframe-wrap${selected === "receiver" ? "" : " page-iframe-wrap--hidden"}`} id="pageReceiver">
      <iframe id="frameReceiver" className="page-iframe" src={receiverLoaded ? "/legacy/page-receiver.html" : undefined} title="RF Receiver Scheduler" />
    </div>
  </>;
}
