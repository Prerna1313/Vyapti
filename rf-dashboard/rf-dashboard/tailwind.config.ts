import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{js,ts,jsx,tsx,mdx}", "./components/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: { extend: { colors: { "vyapti-bg": "#0a0a0c", "vyapti-cyan": "#38bdf8" }, fontFamily: { sans: ["Inter", "sans-serif"], display: ["Chakra Petch", "sans-serif"], mono: ["JetBrains Mono", "monospace"] } } },
  plugins: [],
};
export default config;
