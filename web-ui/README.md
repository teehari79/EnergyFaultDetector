# Energy Fault Detector UI

A sleek React dashboard for managing Energy Fault Detector prediction runs. Authenticate against the asynchronous prediction API, upload telemetry batches, and track each pipeline stage through rich visualisations and narratives built from the trained asset models.

## ✨ Features

- 🚀 **Batch upload workflow** – Drag-and-drop interface that stages datasets and launches the async prediction pipeline.
- 🔄 **Pipeline awareness** – Live status pills and event feeds sourced from the asynchronous job state (no more placeholder heuristics).
- 📈 **Insightful charts** – Gradient area chart mirroring the analytics returned by the trained detector.
- 🧠 **Narrative intelligence** – Auto-generated summaries once anomaly, criticality, and RCA stages complete.
- 📄 **Report handling** – Download concise CSV summaries compiled from the real prediction output.
- 🎛️ **Configurable endpoints** – Point the UI at your dataset staging API and asynchronous prediction service via environment variables.

## 📁 Project structure

```
web-ui/
├── src/
│   ├── components/      # UI building blocks
│   ├── hooks/           # Custom React hooks (e.g. async job poller)
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
   VITE_API_BASE_URL=https://your-node-service.example.com
   ```

   The UI now talks to a single aggregation layer:

   - **Energy Fault Detector Node service** (`VITE_API_BASE_URL`) – proxies authentication and prediction requests to the FastAPI backend, stores job metadata/results in MongoDB, and exposes `/api/jobs` for querying historical runs.

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

## 🔄 Prediction pipeline

1. **Authenticate** – Supply organisation id, username, password, and the tenant seed token. The Node service performs the encryption, exchanges credentials with FastAPI `/auth`, and returns the auth token to the browser.
2. **Upload dataset** – The browser posts multipart uploads to the Node layer; it forwards the payload to FastAPI for staging and records the submission against the authenticated user.
3. **Launch prediction** – The Node layer encrypts the prediction payload, calls `/predict`, and stores the issued job id together with metadata in MongoDB.
4. **Monitor progress** – The dashboard polls `/api/jobs/{id}` on the Node service, which synchronises upstream status, persists results, and serves enriched job snapshots for the UI grid and detail views.

All steps and payloads are visible in the dashboard, with CSV exports generated from the returned prediction result.

## 🧠 Narrative workflow

- Narratives are generated automatically when the job reaches the `narrative_generation` stage.
- The panel lists individual event narratives and highlights the combined summary once available.

## 🧪 Linting

Optional, but recommended:

```bash
npm run lint
```

## 🎨 Design inspiration

The interface combines Ant Design components with glassmorphism styling and gradient accents for a modern, operations-ready feel. Feel free to adapt the theme tokens in `App.jsx` to match your branding.

---

Need more enhancements? Consider augmenting the dashboard with historical job comparisons, webhook delivery diagnostics, or embedded RCA notebook previews.
