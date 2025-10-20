export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const WEBHOOK_ENDPOINTS = [
  {
    key: 'anomalies',
    label: 'Anomaly Events',
    path: '/webhooks/anomalies'
  },
  {
    key: 'critical',
    label: 'Critical Anomaly Events',
    path: '/webhooks/critical'
  },
  {
    key: 'rca',
    label: 'Root Cause Analysis',
    path: '/webhooks/rca'
  }
];

export const NARRATIVE_ENDPOINT = '/api/narratives';
