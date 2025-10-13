"""Narrative generation pipeline that orchestrates multiple LangChain agents."""

from __future__ import annotations

import asyncio
import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import logging

import pandas as pd

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence


logger = logging.getLogger(__name__)


@dataclass
class EventInsight:
    """Structured representation of a detected anomaly event."""

    event_id: int
    start: Optional[pd.Timestamp]
    end: Optional[pd.Timestamp]
    duration_minutes: float
    anomaly_count: int
    severity: str
    severity_reason: str
    sensors: List[str]
    sensor_scores: Dict[str, float]
    potential_root_cause: Optional[str]

    def serialisable(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "start": self.start.isoformat() if isinstance(self.start, pd.Timestamp) else str(self.start),
            "end": self.end.isoformat() if isinstance(self.end, pd.Timestamp) else str(self.end),
            "duration_minutes": self.duration_minutes,
            "anomaly_count": self.anomaly_count,
            "severity": self.severity,
            "severity_reason": self.severity_reason,
            "sensors": self.sensors,
            "sensor_scores": self.sensor_scores,
            "potential_root_cause": self.potential_root_cause,
        }


@dataclass
class NarrativeContext:
    """Aggregate information required to build the story."""

    total_anomalies: int
    critical_events: int
    glitch_events: int
    event_insights: Sequence[EventInsight]
    configuration_context: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass
class NarrativeResult:
    """Outputs produced by :class:`NarrativePipeline`."""

    global_summary: str
    configuration_summary: str
    event_narratives: List[str]


class NarrativePipeline:
    """Coordinates individual LLM agents and assembles the final narrative."""

    def __init__(self, llm: BaseChatModel) -> None:
        self.llm = llm
        self._parser = StrOutputParser()
        self._global_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an analytics specialist generating executive summaries. "
                    "Follow the ReAct pattern and always respond with sections labelled Thought and Answer."
                    " Use the provided structured context only; do not fabricate data.",
                ),
                (
                    "human",
                    "Context:\n{context}\n\n"
                    "Detail the high-level anomaly statistics and how many events are critical versus temporary glitches."
                    " Provide the answer as two paragraphs: the first containing the counts, the second highlighting trends.",
                ),
            ]
        )
        self._configuration_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You translate operational data into configuration insights."
                    " Use ReAct reasoning (Thought -> Answer).",
                ),
                (
                    "human",
                    "Configuration data:\n{context}\n\n"
                    "Summarise the wind turbine operating configuration, including optimal power ranges and notable sensor states."
                    " Keep the answer under 120 words.",
                ),
            ]
        )
        self._event_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You narrate anomaly events for reliability engineers."
                    " Follow the ReAct reasoning template with explicit Thought and Answer sections.",
                ),
                (
                    "human",
                    "Event context:\n{context}\n\n"
                    "Craft a concise story (max 100 words) explaining what happened during this date range,"
                    " why it is classified as {severity}, and which sensors or turbine subcomponents are implicated."
                    " If a root cause hypothesis is provided include it as justification.",
                ),
            ]
        )

    async def _ainvoke(self, prompt: ChatPromptTemplate, data: Dict[str, Any]) -> str:
        chain: RunnableSequence = prompt | self.llm | self._parser
        if hasattr(chain, "ainvoke"):
            return await chain.ainvoke(data)
        return await asyncio.to_thread(chain.invoke, data)

    async def arun(self, context: NarrativeContext) -> NarrativeResult:
        serialisable_events = [insight.serialisable() for insight in context.event_insights]
        global_context = {
            "total_anomalies": context.total_anomalies,
            "critical_events": context.critical_events,
            "glitch_events": context.glitch_events,
            "event_count": len(serialisable_events),
            "events": serialisable_events,
            "metadata": context.metadata,
        }
        configuration_context = {
            "configuration": context.configuration_context,
            "event_overview": serialisable_events,
        }

        global_task = self._ainvoke(self._global_prompt, {"context": global_context})
        configuration_task = self._ainvoke(self._configuration_prompt, {"context": configuration_context})

        event_tasks = [
            self._ainvoke(
                self._event_prompt,
                {
                    "context": {**event.serialisable(), "metadata": context.metadata},
                    "severity": event.severity,
                },
            )
            for event in context.event_insights
        ]

        global_summary, configuration_summary, *event_narratives = await asyncio.gather(
            global_task, configuration_task, *event_tasks
        )

        return NarrativeResult(
            global_summary=global_summary,
            configuration_summary=configuration_summary,
            event_narratives=list(event_narratives),
        )


def _normalise_importance(series: Optional[pd.Series]) -> Dict[str, float]:
    if series is None:
        return {}
    cleaned = series.dropna()
    if cleaned.empty:
        return {}
    normalised = cleaned.abs() / cleaned.abs().max()
    return {str(idx): float(value) for idx, value in normalised.sort_values(ascending=False).items()}


def _select_top_sensors(scores: Dict[str, float], limit: int = 5) -> List[str]:
    return [sensor for sensor, _ in list(scores.items())[:limit]]


def classify_event(event_meta: pd.Series, importances: Dict[str, float]) -> Tuple[str, str]:
    duration_minutes = float(event_meta.get("duration", pd.Timedelta(0)) / pd.Timedelta(minutes=1))
    if not importances:
        importances = {}
    max_score = max(importances.values()) if importances else 0.0
    if duration_minutes < 5 and max_score < 0.35:
        return (
            "temporary_glitch",
            "Short-lived deviation with low sensor bias intensity; likely a transient sensor glitch.",
        )
    if duration_minutes < 10 and max_score < 0.5:
        return (
            "monitor",
            "Moderate-duration anomaly with subdued bias. Track closely but escalation is not yet required.",
        )
    return (
        "critical",
        "Sustained anomaly with strong feature bias, suggesting a material impact on the turbine.",
    )


def derive_event_insights(
    event_metadata: pd.DataFrame,
    event_analysis: Sequence[Dict[str, Any]],
    predicted_anomalies: pd.DataFrame,
    root_cause_hypotheses: Optional[Iterable[Optional[str]]] = None,
) -> List[EventInsight]:
    insights: List[EventInsight] = []
    hypotheses = list(root_cause_hypotheses or [])
    for index, (event_id, meta_row) in enumerate(zip(range(1, len(event_metadata) + 1), event_metadata.itertuples())):
        analysis = event_analysis[index] if index < len(event_analysis) else {}
        importances_series = analysis.get("arcana_mean_importances") if analysis else None
        scores = _normalise_importance(importances_series)
        sensors = _select_top_sensors(scores)
        severity, reason = classify_event(pd.Series(meta_row._asdict()), scores)
        matching_rows = predicted_anomalies[predicted_anomalies.get("event_id") == event_id]
        duration_raw = getattr(meta_row, "duration", pd.Timedelta(0))
        try:
            duration = pd.to_timedelta(duration_raw)
        except (TypeError, ValueError):  # pragma: no cover - defensive casting
            duration = pd.Timedelta(0)
        anomaly_count = int(len(matching_rows))
        potential_root_cause = hypotheses[index] if index < len(hypotheses) else None
        insights.append(
            EventInsight(
                event_id=event_id,
                start=getattr(meta_row, "start", None),
                end=getattr(meta_row, "end", None),
                duration_minutes=float(duration / pd.Timedelta(minutes=1)) if duration else 0.0,
                anomaly_count=anomaly_count,
                severity=severity,
                severity_reason=reason,
                sensors=sensors,
                sensor_scores=scores,
                potential_root_cause=potential_root_cause,
            )
        )
    return insights


def summarise_configuration(prediction_results: Any) -> Dict[str, Any]:
    reconstruction: Optional[pd.DataFrame] = getattr(prediction_results, "reconstruction", None)
    anomaly_scores: Optional[pd.DataFrame] = getattr(prediction_results, "anomaly_score", None)
    context: Dict[str, Any] = {}
    if isinstance(reconstruction, pd.DataFrame) and not reconstruction.empty:
        numeric_columns = reconstruction.select_dtypes(include=["number"])
        if not numeric_columns.empty:
            power_columns = [col for col in numeric_columns.columns if "power" in col.lower()]
            stats: Dict[str, Any] = {}
            for column in power_columns or numeric_columns.columns[:3]:
                series = numeric_columns[column]
                stats[column] = {
                    "mean": float(series.mean()),
                    "p95": float(series.quantile(0.95)),
                    "max": float(series.max()),
                }
            context["sensor_statistics"] = stats
    if isinstance(anomaly_scores, pd.DataFrame) and not anomaly_scores.empty:
        context["anomaly_score_range"] = {
            "min": float(anomaly_scores.min(axis=1).min()),
            "max": float(anomaly_scores.max(axis=1).max()),
        }
    return context


async def gather_root_cause_hypotheses(
    queries: Sequence[str],
    search_callback: Optional[Callable[[str], Optional[str]]],
) -> List[Optional[str]]:
    if not search_callback:
        return [None] * len(queries)

    loop = asyncio.get_running_loop()
    tasks = [loop.run_in_executor(None, search_callback, query) for query in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    hypotheses: List[Optional[str]] = []
    for result in results:
        if isinstance(result, Exception):  # pragma: no cover - defensive logging
            logger.warning("Root cause search failed: %s", result)
            hypotheses.append(None)
        else:
            hypotheses.append(result)
    return hypotheses
