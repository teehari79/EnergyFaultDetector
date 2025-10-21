export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
export const ASYNC_API_BASE_URL =
  import.meta.env.VITE_ASYNC_API_BASE_URL || 'http://localhost:8001';

export const PIPELINE_STEPS = [
  {
    key: 'anomaly_detection',
    label: 'Anomaly Detection'
  },
  {
    key: 'event_detection',
    label: 'Event Detection'
  },
  {
    key: 'criticality_analysis',
    label: 'Criticality Analysis'
  },
  {
    key: 'root_cause_analysis',
    label: 'Root Cause Analysis'
  },
  {
    key: 'narrative_generation',
    label: 'Narrative Generation'
  }
];

export const DEFAULT_TIMESTAMP_COLUMN = 'time_stamp';
