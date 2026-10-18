import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "VYAPTI TSRD Environment",
  description: "Tactical spectrum receiver dashboard",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
