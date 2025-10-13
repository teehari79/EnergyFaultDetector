
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/AEFDI/EnergyFaultDetector/blob/main/img/2025_Logo_Energy-Fault-Detector_white-green.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/AEFDI/EnergyFaultDetector/blob/main/img/Logo_Energy-Fault-Detector.png">
  <img alt="EnergyFaultDetector Logo" src="https://github.com/AEFDI/EnergyFaultDetector/blob/main/img/Logo_Energy-Fault-Detector.png" height="100">
</picture>


# Energy Fault Detector - Autoencoder-based Fault Detection for the Future Energy System

**Energy Fault Detector** is an open-source Python package designed for the automated detection of anomalies in
operational data from renewable energy systems as well as power grids. It uses autoencoder-based normal behaviour
models to identify irregularities in operational data. In addition to the classic anomaly detection, the package 
includes the unique “ARCANA” approach for root cause analysis and thus allows interpretable early fault detection. 
In addition to the pure ML models, the package also contains a range of preprocessing methods, which are particularly 
useful for analyzing systems in the energy sector. A holistic `EnergyFaultDetector` framework is provided for easy use of all 
these methods, which can be adapted to the respective use case via a single configuration file.

The software is particularly valuable in the context of the future energy system, optimizing the monitoring and enabling
predictive maintenance of renewable energy assets.

<img src="https://github.com/AEFDI/EnergyFaultDetector/blob/main/img/OSS-Grafical_abstract2.png" alt="drawing" width="600" style="display: block; margin: 0 auto" />

## Main Features
- **User-friendly interface**: Easy to use and quick to demo using the [command line interface](#Quick-fault-detection).
- **Data Preprocessing Module**: Prepares numerical operational data for analysis with the `EnergyFaultDetector`, 
  with many options such as data clipping, imputation, signal hangers and column selection based on variance and
  missing values. 
- **Fault Detection**: Uses autoencoder architectures to model normal operational behavior and identify deviations.
- **Root Cause Analysis**: Pinpoints the specific sensor values responsible for detected anomalies using [ARCANA](https://doi.org/10.1016/j.egyai.2021.100065).
- **Scalability**: Algorithms can easily be adapted to various datasets and trained models can be transferred to and
   fine-tuned on similar datasets. Quickly evaluate many different model configurations

## Installation
To install the `energy-fault-detector` package, run: `pip install energy-fault-detector`


## Quick fault detection
The `quick_fault_detector` CLI now supports dedicated training and prediction workflows:

- **Train and evaluate a model** (default mode). This trains a new autoencoder, evaluates it on the provided
  test slice, and reports where the model artefacts were stored.

  ```bash
  quick_fault_detector <path_to_training_data.csv> --mode train [--options options.yaml]
  ```

- **Run predictions with an existing model**. Supply the dataset to score alongside the directory that contains the
  saved model files returned from a previous training run.

  ```bash
  quick_fault_detector <path_to_evaluation_data.csv> --mode predict --model_path <path_to_saved_model> [--options options.yaml]
  ```

  The `--model_path` argument is mandatory in predict mode.

Prediction artefacts (anomaly scores, reconstructions, and detected events) are written to the directory specified by
`--results_dir` (defaults to `./results`). For an example using one of the CARE2Compare datasets, run:

```bash
quick_fault_detector <path_to_c2c_dataset.csv> --c2c_example
```

For more information, have a look at the notebook [Quick Failure Detection](./notebooks/Example%20-%20Quick%20Failure%20Detection.ipynb)


## REST prediction API

The project ships with a lightweight FastAPI application that exposes the prediction workflow via HTTP. The service
resolves models by name and version using the directory structure described in
[`energy_fault_detector/api/service_config.yaml`](energy_fault_detector/api/service_config.yaml).

Start the API with:

```bash
uvicorn energy_fault_detector.api.app:app --reload
```

By default the service reads its configuration from the bundled `service_config.yaml`. Provide the
`EFD_SERVICE_CONFIG` environment variable to point to a custom YAML file when you want to adapt the model root
directory, override default ignore patterns, or tweak other runtime parameters. Predictions are triggered with a `POST`
request to `/predict` and expect a JSON payload containing at least the `model_name` and `data_path` fields. Optional
fields such as `model_version`, `ignore_features`, and `asset_name` refine which artefacts are used and how the results
are stored.

### LLM powered narrative endpoint

The API now exposes `/narrative`, an asynchronous endpoint that executes the full fault detection pipeline and then
uses LangChain agents to craft a story-like report for each detected event. The workflow orchestrates specialist agents
that reason about anomaly counts, criticality, likely sensor faults, configuration context, and (optionally) root-cause
insights fetched from the Perplexity API before a final "narrator" agent assembles the final prose.

```json
POST /narrative
{
  "model_name": "demo_wind_turbine",
  "data_path": "~/datasets/wt/powercurve.csv",
  "llm": {
    "provider": "ollama",
    "model": "phi3:mini",
    "temperature": 0.1,
    "base_url": "http://localhost:11434"
  },
  "enable_web_search": true,
  "perplexity_model": "sonar-small-chat"
}
```

Key capabilities:

- **Input validation first** – requests are validated before long-running work starts, and informative HTTP errors are
  returned for missing data or models.
- **Parallel specialist agents** – counts, criticality, glitch detection, sensor attribution, and turbine configuration
  are analysed in parallel before narrative synthesis.
- **Optional web enrichment** – when `enable_web_search` is true, the service queries Perplexity using a ReAct-style
  prompt to augment each event with external failure mode knowledge.
- **Streaming-friendly design** – results are grouped per event in the response so API clients can stream narratives to
  end users as soon as each event is available.

API keys required:

- `OPENAI_API_KEY` (or a value supplied through `llm.api_key`) when using OpenAI hosted models.
- AWS credentials with Bedrock permissions when selecting a Bedrock model.
- `PERPLEXITY_API_KEY` to enable the optional web-search augmentation.

For laptop or edge deployments you can run a local Ollama model instead of calling a hosted API. The endpoint uses
LangChain so it can also be packaged inside AWS Lambda or other serverless runtimes; the heavy prediction work executes
in a worker thread to avoid blocking the FastAPI event loop.


## Fault detection in 5 lines of code

```python
from energy_fault_detector.fault_detector import FaultDetector
from energy_fault_detector.config import Config

fault_detector = FaultDetector(config=Config('base_config.yaml'))
model_data = fault_detector.train(sensor_data=sensor_data, normal_index=normal_index)
results = fault_detector.predict(sensor_data=test_sensor_data)
```

The pandas `DataFrame` `sensor_data` contains the operational data in wide format with the timestamp as index, the
pandas `Series` `normal_index` indicates which timestamps are considered 'normal' operation and can be used to create
a normal behaviour model. The [`base_config.yaml`](energy_fault_detector/base_config.yaml) file contains all model 
settings, an example is found [here](energy_fault_detector/base_config.yaml).


## Background
This project was initially developed in the research project ADWENTURE, to create a software for early fault detection
in wind turbines. The software was developed in such a way that the algorithms do not depend on a specific data source
and can be applied to other use cases as well.

## Documentation
Comprehensive documentation is available [here](https://aefdi.github.io/EnergyFaultDetector/).

## Contributing
Contributions are welcome! Please feel free to open issues or submit pull requests.
All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

### Planned updates and features
1. More autoencoder types:
   1. Variational autoencoders
   2. CNN- and LSTM-based autoencoders with time-series support.

2. Unification, standardisation and generic improvements
   1. Additional options for all autoencoders (e.g. drop out, regularization)
   2. Data preparation (e.g. extend imputation strategies).
   3. Download method for the Care2Compare class.
   3. Unify default value settings. 
   4. No or low configuration

3. Conditions and dependency updates
   1. Conditional autoencoder 
   2. Upgrade to Keras 3.0

4. Root cause analysis expansion: integrate SHAP and possibly other XAI-methods.

## License
This project is licensed under the [MIT License](./LICENSE).

## References
If you use this work, please cite us:

**ARCANA Algorithm**:
Autoencoder-based anomaly root cause analysis for wind turbines. Energy and AI. 2021;4:100065. https://doi.org/10.1016/j.egyai.2021.100065

**CARE to Compare dataset and CARE-Score**:
- Paper: CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine Data. Data. 2024; 9(12):138. https://doi.org/10.3390/data9120138 
- Dataset: Wind Turbine SCADA Data For Early Fault Detection. Zenodo, Mar. 2025, https://doi.org/10.5281/ZENODO.14958989.

**Transfer learning methods**:
Transfer learning applications for autoencoder-based anomaly detection in wind turbines. Energy and AI. 2024;17:100373. https://doi.org/10.1016/j.egyai.2024.100373

**Autoencoder-based anomaly detection**:
Evaluation of Anomaly Detection of an Autoencoder Based on Maintenance Information and Scada-Data. Energies. 2020; 13(5):1063., https://doi.org/10.3390/en13051063.
