import pytest

pd = pytest.importorskip('pandas')

from energy_fault_detector.config import Config
from energy_fault_detector.fault_detector import FaultDetector


class _IdentityPreprocessor:
    def transform(self, data):
        return data


def _build_config(ignore_features=None):
    config_dict = {
        'train': {
            'anomaly_score': {'name': 'rmse', 'params': {}},
            'autoencoder': {'name': 'default', 'params': {'layers': [1]}},
            'data_preprocessor': {'params': {}},
            'threshold_selector': {'name': 'quantile', 'params': {}},
        },
        'root_cause_analysis': {},
    }
    if ignore_features is not None:
        config_dict['root_cause_analysis']['ignore_features'] = ignore_features
    return Config(config_dict=config_dict)


def _build_fault_detector(config):
    detector = FaultDetector.__new__(FaultDetector)
    detector.config = config
    detector.data_preprocessor = _IdentityPreprocessor()
    detector.autoencoder = object()
    return detector


def test_run_root_cause_analysis_uses_configured_ignore_features(monkeypatch):
    config = _build_config(ignore_features=['temp_*'])
    detector = _build_fault_detector(config)

    captured_kwargs = {}

    class DummyArcana:
        def __init__(self, model, **kwargs):
            captured_kwargs.update(kwargs)

        def find_arcana_bias(self, x, track_losses, track_bias):
            return x.copy(), pd.DataFrame(), []

    monkeypatch.setattr('energy_fault_detector.fault_detector.Arcana', DummyArcana)

    sensor_data = pd.DataFrame({'temp_a': [1.0], 'temp_b': [2.0]})
    detector.run_root_cause_analysis(sensor_data=sensor_data)

    assert captured_kwargs.get('ignore_features') == ['temp_*']


def test_run_root_cause_analysis_allows_ignore_feature_override(monkeypatch):
    config = _build_config(ignore_features=['temp_*'])
    detector = _build_fault_detector(config)

    captured_kwargs = {}

    class DummyArcana:
        def __init__(self, model, **kwargs):
            captured_kwargs.update(kwargs)

        def find_arcana_bias(self, x, track_losses, track_bias):
            return x.copy(), pd.DataFrame(), []

    monkeypatch.setattr('energy_fault_detector.fault_detector.Arcana', DummyArcana)

    sensor_data = pd.DataFrame({'temp_a': [1.0], 'temp_b': [2.0]})
    detector.run_root_cause_analysis(sensor_data=sensor_data, ignore_features=['pressure_*'])

    assert captured_kwargs.get('ignore_features') == ['pressure_*']
