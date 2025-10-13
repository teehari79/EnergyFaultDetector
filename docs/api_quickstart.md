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

The script accepts an optional first argument pointing to a custom service
configuration file. Any additional arguments are forwarded to Uvicorn, allowing
you to change the listening address or port, for example:

```bash
scripts/run_api.sh energy_fault_detector/api/service_config.yaml --host 0.0.0.0 --port 8080
```

## 3. Create a prediction payload

A ready-to-edit payload for Farm C is available at
[`docs/examples/farm_c_prediction.json`](examples/farm_c_prediction.json). Update
`data_path` so it points to the CSV file for the asset you want to analyse. The
`asset_name` field controls how outputs are named on disk, and `model_name`
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

Pass a different payload file or endpoint URL if required:

```bash
scripts/predict_farm_c.sh http://localhost:8000/predict /path/to/custom_payload.json
```

The script prints the JSON response from the `/predict` endpoint, which includes
summary counts and per-event details. Any generated artefacts are written to the
`prediction_output/<asset_name>` directory adjacent to the CSV file unless you
override the `prediction.output_directory` configuration.

## 5. Passing the model name in API calls

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
