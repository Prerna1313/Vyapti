import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        ui: ["IBM Plex Sans", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      colors: {
        void: "#050810",
        panel: {
          DEFAULT: "#0A1220",
          raised: "#0E1A2E",
          deep: "#071019",
        },
        steel: {
          DEFAULT: "#1B324A",
          bright: "#2D5578",
        },
        primary: "#D8E4EE",
        dim: "#5E7B96",
        faint: "#3D5468",
        cyan: {
          signal: "#4DD8E8",
          bright: "#8FEFFA",
          dim: "#164654",
        },
        amber: "#E8B84D",
        critical: "#E8604D",
        confirm: "#4DE87F",
      },
    },
  },
  plugins: [],
};
export default config;
