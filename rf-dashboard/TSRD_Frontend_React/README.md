# VYAPTI TSRD Environment — Next.js / React

This package wraps the supplied tactical-dashboard HTML in a Next.js React client component while preserving its existing canvas simulation, controls, styles, and interactions.

## Requirements

- Node.js 18.18 or newer (Node.js 20 LTS recommended)
- npm 9 or newer

## Run locally

```powershell
npm install
npm run dev
```

Open `http://localhost:3000` in a browser.

## Production build

```powershell
npm run build
npm run start
```

## Refreshing the supplied dashboard asset

If `C:/Users/cs768/Downloads/index (2).html` is updated, run:

```powershell
npm run sync:dashboard
```

This replaces `public/dashboard.html` with that source file.
