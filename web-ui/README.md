# Energy Fault Detector UI

A sleek React dashboard for managing Energy Fault Detector prediction runs. Upload telemetry batches, observe real-time webhook activity, visualize anomalies, and request natural language narratives once processing completes.

## ✨ Features

- 🚀 **Batch upload workflow** – Drag-and-drop interface to trigger predictions instantly.
- 🔔 **Real-time webhook stream** – Live status pills and timeline feed for anomaly, critical anomaly, and RCA events (SSE with polling fallback).
- 📈 **Insightful charts** – Gradient area chart mirroring the analytics in the visualization module.
- 🧠 **Narrative intelligence** – On-demand summary button that stays disabled until all webhook data is received.
- 📄 **Report handling** – Quick link for downloading generated prediction reports.
- 🎛️ **Configurable endpoints** – Point the UI at any backend using environment variables.

## 📁 Project structure

```
web-ui/
├── src/
│   ├── components/      # UI building blocks
│   ├── hooks/           # Custom React hooks (e.g. webhook stream listener)
│   ├── services/        # API helpers
│   ├── styles/          # Global styling
│   ├── App.jsx          # Main layout
│   └── main.jsx         # Application entry point
├── index.html
├── package.json
├── vite.config.js
└── README.md
```

## 🛠️ Getting started

1. **Install dependencies**

   ```bash
   cd web-ui
   npm install
   ```

2. **Configure environment (optional)**

   Create a `.env` file if you need to override defaults:

   ```bash
   VITE_API_BASE_URL=https://your-api.example.com
   ```

   The UI expects the backend to expose:

   - `POST /api/predictions` – accepts multipart file upload, returns `{ prediction_id }`.
   - `GET /webhooks/anomalies|critical|rca` – Server-Sent Events (SSE) streams emitting JSON payloads.
   - `POST /api/narratives` – generates the NLP summary when invoked with `{ prediction_id }`.
   - `GET /api/predictions/{id}/report` – returns the downloadable report (PDF/CSV).

3. **Run locally**

   ```bash
   npm run dev
   ```

   Vite starts the development server on [http://localhost:5173](http://localhost:5173).

4. **Build for production**

   ```bash
   npm run build
   ```

   Serve the generated files from `dist/` with your preferred static host.

## 🔄 Webhook streaming

The dashboard listens to SSE endpoints defined in `src/config.js`. Each endpoint should publish events as JSON messages; a typical payload could look like:

```json
{
  "timestamp": "2024-03-18T10:21:00Z",
  "metric": "compressor_vibration",
  "anomaly_score": 0.83,
  "severity": "high",
  "message": "Spike detected across channel #4"
}
```

If SSE is unavailable, the hook automatically falls back to 5-second interval polling.

## 🧠 Narrative workflow

- Narrative button activates only when all anomaly, critical, and RCA streams report `ready`.
- Clicking the button triggers the dedicated narrative endpoint and renders the summary in a glassmorphism card.
- A floating action button offers quick access to the narrative action anywhere on the page.

## 🧪 Linting

Optional, but recommended:

```bash
npm run lint
```

## 🎨 Design inspiration

The interface combines Ant Design components with glassmorphism styling and gradient accents for a modern, operations-ready feel. Feel free to adapt the theme tokens in `App.jsx` to match your branding.

---

Need more enhancements? Consider augmenting the dashboard with live KPI comparisons, alerting controls, or embedded RCA notebook previews.
