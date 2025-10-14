# API Quickstart

This guide walks through running the Energy Fault Detector REST API locally and
executing a prediction for a Farm C asset.

## 1. Prepare the model artefacts

The API looks for trained models under the directory configured by
`model_store.root_directory` in
[`energy_fault_detector/api/service_config.yaml`](energy_fault_detector/api/service_config.yaml).
By default this path resolves to `energy_fault_detector/api/models` inside the
repository. Arrange your Farm C model artefacts using the following structure:

```
energy_fault_detector/
└── api/
    └── models/
        └── farm-c/
            └── 1.0.0/
                ├── config.yaml
                ├── data_preprocessor/
                ├── autoencoder/
                ├── threshold_selector/
                └── anomaly_score/
```

* Replace `farm-c` with the logical model name you intend to send in requests.
* Replace `1.0.0` with the version identifier you want to expose. The API will
  automatically select the latest version when you omit `model_version` from the
  payload. If you only have a single version you may omit the intermediate
  directory and place the artefacts directly in `models/farm-c`.

## 2. Launch the API server

Use the helper script to start the FastAPI application with Uvicorn:

```bash
scripts/run_api.sh
```

On Windows, launch the PowerShell equivalent:

```powershell
pwsh scripts/run_api.ps1
```

The script accepts an optional first argument pointing to a custom service
configuration file. Any additional arguments are forwarded to Uvicorn, allowing
you to change the listening address or port, for example:

```bash
scripts/run_api.sh energy_fault_detector/api/service_config.yaml --host 0.0.0.0 --port 8080
```

```powershell
pwsh scripts/run_api.ps1 energy_fault_detector/api/service_config.yaml --host 0.0.0.0 --port 8080
```

## 3. Create a prediction payload

A ready-to-edit payload for Farm C is available at
[`docs/examples/farm_c_prediction.json`](examples/farm_c_prediction.json). The
sample payload now targets asset `34` and expects the CSV file to live at
`docs/examples/data/farm_c_asset_34.csv`. Create the `data` directory if it does
not already exist and drop your asset CSV there. Relative paths in the payload
are resolved against the JSON file automatically by the sample client, so you
can relocate the CSV without rewriting the payload—simply update the relative
path if you choose a different file name.

The `asset_name` field controls how outputs are named on disk, and `model_name`
should match the directory name chosen in step 1. The ignore feature list is
pre-populated with the Farm C defaults from
[`energy_fault_detector/base_config.yaml`](../energy_fault_detector/base_config.yaml);
feel free to adjust it for your use-case.

## 4. Submit the prediction request

With the API running you can trigger a prediction using `curl`
(the helper script pipes the response through `jq` for readability):

```bash
scripts/predict_farm_c.sh
```

On Windows, use the PowerShell helper, which returns formatted JSON via
`Invoke-RestMethod`:

```powershell
pwsh scripts/predict_farm_c.ps1
```

Pass a different payload file or endpoint URL if required:

```bash
scripts/predict_farm_c.sh http://localhost:8000/predict /path/to/custom_payload.json
```

```powershell
pwsh scripts/predict_farm_c.ps1 http://localhost:8000/predict /path/to/custom_payload.json
```

The script prints the JSON response from the `/predict` endpoint, which includes
summary counts and per-event details. Any generated artefacts are written to the
`prediction_output/<asset_name>` directory adjacent to the CSV file unless you
override the `prediction.output_directory` configuration.

## 5. Use the sample Python client

If you prefer a Python example that handles authentication and payload
encryption, use [`scripts/sample_api_client.py`](../scripts/sample_api_client.py):

```bash
python scripts/sample_api_client.py --base-url http://localhost:8000 \
    --payload docs/examples/farm_c_prediction.json --poll
```

The client performs the following steps:

1. Encrypts the configured username and password with the tenant seed token and
   authenticates against `/auth` to obtain an auth token.
2. Reads the payload JSON, optionally adds webhook URLs supplied with
   `--webhook name=url` overrides, and encrypts the payload using the derived
   hash from the auth token.
3. Submits the encrypted request to `/predict` and either prints the job
   identifier or, when `--poll` is enabled, waits for completion and displays
   the final job status.

Narrative generation is disabled for the sample client to keep the example
lightweight. Pass `--enable-narrative` to opt back into LLM narrative synthesis
once your environment is ready to handle the additional workload.

The script defaults to the credentials shipped in the sample service
configuration. Override `--username`, `--password`, `--organization` or
`--seed-token` to match your environment. You can provide multiple webhook
overrides; for example:

```bash
python scripts/sample_api_client.py --webhook anomalies=http://localhost:9010 \
    --webhook events=http://localhost:9011
```

Lightweight CLI webhook receivers are available in the `scripts` directory
(`webhook_anomalies.py`, `webhook_events.py`, `webhook_criticality.py`,
`webhook_root_cause.py` and `webhook_narrative.py`). Each script prints incoming
requests, making it easy to verify webhook deliveries during local testing.

## 6. Passing the model name in API calls

The `model_name` and optional `model_version` fields in the JSON request are used
by the API to locate artefacts. For the directory layout above the payload should
contain:

```json
{
  "model_name": "farm-c",
  "model_version": "1.0.0",
  "data_path": "/absolute/path/to/data/farm_c_asset.csv"
}
```

Omit `model_version` to automatically select the most recent version in the
model directory. Ensure the process executing the API has read access to both
the model artefacts and the CSV input file.
