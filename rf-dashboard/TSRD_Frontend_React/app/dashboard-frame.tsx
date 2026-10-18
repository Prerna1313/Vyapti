"use client";

import { useCallback, useRef, useState } from "react";

export function DashboardFrame() {
  const frameRef = useRef<HTMLIFrameElement>(null);
  const [ready, setReady] = useState(false);

  const toggleFullscreen = useCallback(async () => {
    const frame = frameRef.current;
    if (!frame) return;

    if (document.fullscreenElement) {
      await document.exitFullscreen();
    } else {
      await frame.requestFullscreen();
    }
  }, []);

  return (
    <main className="shell">
      <header className="app-bar">
        <div>
          <p className="eyebrow">Next.js / React migration</p>
          <h1>VYAPTI TSRD Environment</h1>
        </div>
        <button type="button" onClick={toggleFullscreen} className="fullscreen-button">
          Full screen
        </button>
      </header>
      <iframe
        ref={frameRef}
        className="dashboard"
        src="/dashboard.html?v=fast_local"
        title="VYAPTI TSRD tactical dashboard"
        onLoad={() => setReady(true)}
        allowFullScreen
      />
    </main>
  );
}
