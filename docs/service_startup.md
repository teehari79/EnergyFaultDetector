# Service start-up guide

This guide explains how to start all of the Energy Fault Detector services that are
commonly used during local development or demonstrations.

## API processes

Use the helper scripts in the `scripts` directory to launch both FastAPI
applications at once:

```bash
./scripts/start_services.sh [path/to/service_config.yaml] [additional uvicorn args]
```

```powershell
scripts\start_services.ps1 [-ConfigPath path\to\service_config.yaml]
```

Both scripts perform the following steps:

1. Validate the configuration file (defaults to `energy_fault_detector/api/service_config.yaml`).
2. Export the `EFD_SERVICE_CONFIG` environment variable so that the APIs load the
   correct settings.
3. Start `energy_fault_detector.api.app:app` on port `8000`.
4. Start `energy_fault_detector.api.prediction_api:app` on port `8001`.

You can override the bind addresses and ports by setting the `API_HOST`,
`API_PORT`, `PREDICTION_API_HOST`, or `PREDICTION_API_PORT` environment variables
before invoking the shell script, or by passing `-ApiHost`, `-ApiPort`,
`-PredictionHost`, or `-PredictionPort` to the PowerShell script. Any additional
arguments supplied after the configuration path are forwarded to both Uvicorn
processes (for example, `--reload`). The scripts remain attached to the console
so that pressing `Ctrl+C` gracefully stops the services.

## Webhook receivers

To capture webhook payloads emitted by the asynchronous prediction API, use the
companion scripts:

```bash
./scripts/start_webhooks.sh [webhook options]
```

```powershell
scripts\start_webhooks.ps1 [-AdditionalArgs '--host 127.0.0.1']
```

Each helper launches the five sample webhook receivers in parallel:

- `scripts/webhook_anomalies.py` (port 9010)
- `scripts/webhook_events.py` (port 9011)
- `scripts/webhook_criticality.py` (port 9012)
- `scripts/webhook_root_cause.py` (port 9013)
- `scripts/webhook_narrative.py` (port 9014)

Arguments that follow the command are forwarded to every receiver, allowing you
to override the host, port, or log level if required. The scripts keep running
until you terminate them with `Ctrl+C`, ensuring all background processes are
shut down cleanly.
