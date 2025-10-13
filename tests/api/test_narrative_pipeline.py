"""Tests for the narrative generation helpers."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

import pytest

pd = pytest.importorskip("pandas")
fake_llm_module = pytest.importorskip("langchain_community.llms.fake")
FakeListLLM = fake_llm_module.FakeListLLM

from energy_fault_detector.api.narrative_pipeline import (
    NarrativeContext,
    NarrativePipeline,
    classify_event,
    derive_event_insights,
    gather_root_cause_hypotheses,
    summarise_configuration,
)


class DummyResult:
    def __init__(self) -> None:
        index = pd.date_range(datetime(2024, 1, 1), periods=4, freq="T")
        self.predicted_anomalies = pd.DataFrame(
            {
                "anomaly": [True, True, False, False],
                "critical_event": [True, True, False, False],
                "event_id": [1, 1, None, None],
            },
            index=index,
        )
        self.reconstruction = pd.DataFrame(
            {
                "power_output": [1000.0, 1010.0, 990.0, 995.0],
                "wind_speed": [12.1, 12.4, 11.9, 12.0],
            },
            index=index,
        )
        self.anomaly_score = pd.DataFrame({"score": [0.1, 0.25, 0.05, 0.07]}, index=index)


def test_classify_event_tmp_glitch():
    meta = pd.Series({"duration": timedelta(minutes=2)})
    severity, reason = classify_event(meta, {"sensor_a": 0.1})
    assert severity == "temporary_glitch"
    assert "glitch" in reason.lower()


def test_pipeline_generates_responses():
    event_meta = pd.DataFrame(
        {
            "start": [datetime(2024, 1, 1, 0, 0)],
            "end": [datetime(2024, 1, 1, 0, 5)],
            "duration": [timedelta(minutes=5)],
        }
    )
    event_analysis = [
        {"arcana_mean_importances": pd.Series({"power_output": 0.8, "wind_speed": 0.2})}
    ]
    result = DummyResult()
    insights = derive_event_insights(event_meta, event_analysis, result.predicted_anomalies)

    assert insights[0].sensors[0] == "power_output"

    llm = FakeListLLM(
        responses=[
            "Thought: analyse counts\nAnswer: overall",
            "Thought: config\nAnswer: config summary",
            "Thought: event\nAnswer: event summary",
        ]
    )
    pipeline = NarrativePipeline(llm)
    context = NarrativeContext(
        total_anomalies=2,
        critical_events=1,
        glitch_events=0,
        event_insights=insights,
        configuration_context=summarise_configuration(result),
        metadata={"asset_name": "WT-01", "model_name": "demo", "model_version": "1.0"},
    )

    narrative_result = asyncio_run(pipeline.arun(context))

    assert "overall" in narrative_result.global_summary
    assert narrative_result.event_narratives[0].endswith("event summary")


def test_gather_root_cause_hypotheses_handles_callback():
    queries = ["query one", "query two"]
    responses: List[Optional[str]] = ["result 1", None]

    def fake_search(query: str) -> Optional[str]:
        return responses.pop(0)

    gathered = asyncio_run(gather_root_cause_hypotheses(queries, fake_search))
    assert gathered == ["result 1", None]


def asyncio_run(coro):
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
