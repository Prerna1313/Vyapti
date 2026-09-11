import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VYAPTI :: Smart Scan Strategy — EW Simulation Console",
  description: "Vyapti Intelligent EW/Radar Receiver Scheduler — Mission Dashboard (DRDO PS26055)",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>
        {children}
      </body>
    </html>
  );
}
