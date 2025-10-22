import { useCallback, useEffect, useMemo, useState } from 'react';
import dayjs from 'dayjs';
import {
  Alert,
  Button,
  Col,
  ConfigProvider,
  FloatButton,
  Form,
  Input,
  Layout,
  Modal,
  Row,
  Space,
  Typography,
  message
} from 'antd';
import { DownloadOutlined, InfoCircleOutlined } from '@ant-design/icons';
import FileUploadPanel from './components/FileUploadPanel.jsx';
import JobFilters from './components/JobFilters.jsx';
import JobGrid from './components/JobGrid.jsx';
import WebhookStatusPills from './components/WebhookStatusPills.jsx';
import WebhookEventFeed from './components/WebhookEventFeed.jsx';
import AnomalyChart from './components/AnomalyChart.jsx';
import InsightCards from './components/InsightCards.jsx';
import NarrativePanel from './components/NarrativePanel.jsx';
import { usePredictionJob } from './hooks/usePredictionJob.js';
import {
  createPredictionJob,
  fetchJobs,
  login,
  setAuthToken,
  clearAuthToken
} from './services/api.js';
import { PIPELINE_STEPS } from './config.js';

const { Header, Content } = Layout;
const { Paragraph, Title, Text } = Typography;

const themeOverrides = {
  token: {
    colorPrimary: '#38bdf8',
    colorBgElevated: 'rgba(15, 23, 42, 0.85)',
    colorTextBase: '#e2e8f0',
    colorBgBase: 'transparent',
    borderRadius: 18,
    fontFamily: 'Inter, system-ui, sans-serif'
  },
  components: {
    Layout: {
      headerBg: 'transparent'
    },
    Card: {
      colorBgContainer: 'rgba(15, 23, 42, 0.7)',
      colorBorderSecondary: 'rgba(148, 163, 184, 0.2)'
    },
    Button: {
      colorPrimary: '#38bdf8',
      colorPrimaryHover: '#0ea5e9',
      colorPrimaryActive: '#0369a1',
      lineWidth: 0
    }
  }
};

const createInitialStatus = () =>
  PIPELINE_STEPS.reduce((acc, step) => {
    acc[step.key] = { state: 'waiting', updatedAt: null };
    return acc;
  }, {});

const buildPipelineStatus = (status) => {
  const nextStatus = createInitialStatus();
  if (!status) {
    return nextStatus;
  }

  PIPELINE_STEPS.forEach((step) => {
    if (status.steps?.[step.key]) {
      nextStatus[step.key] = {
        state: 'ready',
        updatedAt: status.updated_at,
        detail: status.steps[step.key]
      };
    } else if (status.status === 'failed') {
      nextStatus[step.key] = {
        state: 'error',
        updatedAt: status.updated_at,
        error: status.error ? new Error(status.error) : undefined
      };
    }
  });

  return nextStatus;
};

const deriveJobDetails = (status) => {
  if (!status) {
    return {};
  }

  const isComplete = status.status === 'completed';
  const resultPayload = status.result;

  const anomalyEvents =
    status.steps?.event_detection?.payload?.events ??
    (isComplete ? resultPayload?.events ?? [] : undefined);

  const criticalEvents =
    status.steps?.criticality_analysis?.payload?.events ??
    (isComplete ? [] : undefined);

  const rcaFindings =
    status.steps?.root_cause_analysis?.payload?.root_cause_analysis ??
    (isComplete ? [] : undefined);

  let narrativeEntries =
    status.steps?.narrative_generation?.payload?.narratives ??
    (isComplete ? [] : undefined);

  if (narrativeEntries !== undefined && !Array.isArray(narrativeEntries)) {
    narrativeEntries = [];
  }

  const narrative =
    narrativeEntries === undefined
      ? undefined
      : narrativeEntries.length
        ? narrativeEntries.map((entry) => entry.narrative).join('\n\n')
        : null;

  return {
    predictionResult: resultPayload ?? (isComplete ? {} : undefined),
    anomalyEvents,
    criticalEvents,
    rcaFindings,
    narrativeEntries,
    narrative
  };
};

const mergeJobDetails = (previous = {}, next = {}) => ({
  predictionResult:
    next.predictionResult !== undefined
      ? next.predictionResult
      : previous.predictionResult ?? null,
  anomalyEvents:
    next.anomalyEvents !== undefined ? next.anomalyEvents : previous.anomalyEvents ?? [],
  criticalEvents:
    next.criticalEvents !== undefined ? next.criticalEvents : previous.criticalEvents ?? [],
  rcaFindings:
    next.rcaFindings !== undefined ? next.rcaFindings : previous.rcaFindings ?? [],
  narrativeEntries:
    next.narrativeEntries !== undefined
      ? next.narrativeEntries
      : previous.narrativeEntries ?? [],
  narrative: next.narrative !== undefined ? next.narrative : previous.narrative ?? null
});

const buildCsvReport = (result, rootCause) => {
  if (!result?.events?.length) {
    return null;
  }
  const headers = ['event_id', 'start', 'end', 'duration_seconds', 'top_sensors'];
  const rcLookup = new Map();
  (rootCause || []).forEach((entry) => {
    const ranked = entry.ranked_sensors || [];
    const description = ranked
      .slice(0, 3)
      .map(([name, score]) => `${name} (${score.toFixed?.(2) ?? score})`)
      .join('; ');
    rcLookup.set(entry.event_id, description);
  });
  const rows = result.events.map((event) => {
    const topSensors = rcLookup.get(event.event_id) || '';
    return [
      event.event_id,
      event.start,
      event.end,
      event.duration_seconds,
      topSensors.includes(',') ? `"${topSensors}"` : topSensors
    ];
  });
  return [headers.join(','), ...rows.map((row) => row.join(','))].join('\n');
};

const createDefaultFilterValues = () => ({
  status: 'in_progress',
  dateRange: null,
  assetName: '',
  farmName: '',
  anomalyRange: { min: null, max: null },
  criticalRange: { min: null, max: null }
});

const createClearedFilterValues = () => ({
  status: 'all',
  dateRange: null,
  assetName: '',
  farmName: '',
  anomalyRange: { min: null, max: null },
  criticalRange: { min: null, max: null }
});

const buildFilterQuery = (values) => {
  const query = {};
  const status = values.status ?? 'in_progress';
  query.status = status || 'in_progress';

  if (values.dateRange && values.dateRange.length === 2) {
    const [start, end] = values.dateRange;
    if (start) {
      query.startDate = dayjs(start).toISOString();
    }
    if (end) {
      query.endDate = dayjs(end).toISOString();
    }
  }

  if (values.assetName) {
    query.assetName = values.assetName.trim();
  }

  if (values.farmName) {
    query.farmName = values.farmName.trim();
  }

  const anomalyMin = values.anomalyRange?.min;
  const anomalyMax = values.anomalyRange?.max;
  if (anomalyMin !== undefined && anomalyMin !== null) {
    query.minAnomalies = anomalyMin;
  }
  if (anomalyMax !== undefined && anomalyMax !== null) {
    query.maxAnomalies = anomalyMax;
  }

  const criticalMin = values.criticalRange?.min;
  const criticalMax = values.criticalRange?.max;
  if (criticalMin !== undefined && criticalMin !== null) {
    query.minCritical = criticalMin;
  }
  if (criticalMax !== undefined && criticalMax !== null) {
    query.maxCritical = criticalMax;
  }

  return query;
};

const computeCountsFromStatus = (status) => {
  if (!status) {
    return { anomaliesCount: null, criticalAnomaliesCount: null };
  }
  const anomalySource = status.steps?.event_detection?.payload?.events ?? status.result?.events;
  const anomaliesCount = Array.isArray(anomalySource) ? anomalySource.length : null;
  const criticalSource = status.steps?.criticality_analysis?.payload?.events;
  const criticalAnomaliesCount = Array.isArray(criticalSource) ? criticalSource.length : null;
  return { anomaliesCount, criticalAnomaliesCount };
};

const mapJobFromResponse = (job) => {
  const snapshot = job.snapshot || null;
  const details = deriveJobDetails(snapshot);
  const anomaliesCount =
    job.anomaliesCount ??
    (details.anomalyEvents !== undefined ? details.anomalyEvents?.length ?? null : null);
  const criticalAnomaliesCount =
    job.criticalAnomaliesCount ??
    (details.criticalEvents !== undefined ? details.criticalEvents?.length ?? null : null);

  return {
    jobId: job.jobId,
    predictionId: job.predictionId || null,
    assetName: job.assetName || '',
    farmName: job.farmName || '',
    modelName: job.modelName || '',
    fileName: job.fileName || '',
    batchName: job.batchName || '',
    status: job.status || snapshot?.status || null,
    submittedAt: job.submittedAt || snapshot?.created_at || null,
    updatedAt: job.updatedAt || snapshot?.updated_at || null,
    pipelineStatus: buildPipelineStatus(snapshot),
    snapshot,
    error: job.error || snapshot?.error || null,
    details: mergeJobDetails({}, details),
    anomaliesCount,
    criticalAnomaliesCount
  };
};

const App = () => {
  const [authForm] = Form.useForm();
  const [authState, setAuthState] = useState({
    organizationId: '',
    username: '',
    authToken: null,
    expiresAt: null
  });
  const [authModalVisible, setAuthModalVisible] = useState(true);
  const [authenticating, setAuthenticating] = useState(false);

  const [predictionId, setPredictionId] = useState(null);
  const [activeJobId, setActiveJobId] = useState(null);
  const [jobSnapshot, setJobSnapshot] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState(createInitialStatus);
  const [jobs, setJobs] = useState([]);
  const [jobsLoading, setJobsLoading] = useState(false);
  const [jobFilters, setJobFilters] = useState(buildFilterQuery(createDefaultFilterValues()));
  const [filterValues, setFilterValues] = useState(createDefaultFilterValues());

  const [predictionResult, setPredictionResult] = useState(null);
  const [anomalyEvents, setAnomalyEvents] = useState([]);
  const [criticalEvents, setCriticalEvents] = useState([]);
  const [rcaFindings, setRcaFindings] = useState([]);
  const [narrativeEntries, setNarrativeEntries] = useState([]);
  const [narrative, setNarrative] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [jobError, setJobError] = useState(null);

  const clearActiveJob = useCallback(() => {
    setActiveJobId(null);
    setJobSnapshot(null);
    setPipelineStatus(createInitialStatus());
    setPredictionResult(null);
    setAnomalyEvents([]);
    setCriticalEvents([]);
    setRcaFindings([]);
    setNarrativeEntries([]);
    setNarrative(null);
    setJobError(null);
  }, []);

  const handleSessionExpired = useCallback(() => {
    clearAuthToken();
    setAuthState({ organizationId: '', username: '', authToken: null, expiresAt: null });
    setJobs([]);
    setPredictionId(null);
    clearActiveJob();
    setAuthModalVisible(true);
    message.warning('Session expired. Please authenticate again.');
  }, [clearActiveJob]);

  const processJobStatus = useCallback(
    (status, { updateDetail = false, notify = false, jobMetadata = null } = {}) => {
      if (!status) {
        return;
      }

      const jobIdentifier = status.job_id;
      if (!jobIdentifier) {
        return;
      }

      const nextPipeline = buildPipelineStatus(status);
      const detailPatch = deriveJobDetails(status);
      const countsFromStatus = computeCountsFromStatus(status);
      let previousStatus = null;
      let matchedJob = false;

      setJobs((prevJobs) => {
        const updated = prevJobs.map((job) => {
          if (job.jobId !== jobIdentifier) {
            return job;
          }
          matchedJob = true;
          previousStatus = job.status || null;
          const mergedDetails = mergeJobDetails(job.details, detailPatch);
          return {
            ...job,
            status: status.status || job.status,
            submittedAt:
              jobMetadata?.submittedAt || job.submittedAt || status.created_at || jobMetadata?.snapshot?.created_at || job.submittedAt,
            updatedAt: status.updated_at || job.updatedAt,
            assetName: jobMetadata?.assetName ?? job.assetName,
            farmName: jobMetadata?.farmName ?? job.farmName,
            modelName: jobMetadata?.modelName ?? job.modelName,
            fileName: jobMetadata?.fileName ?? job.fileName,
            batchName: jobMetadata?.batchName ?? job.batchName,
            pipelineStatus: nextPipeline,
            snapshot: status,
            error: status.error || jobMetadata?.error || null,
            details: mergedDetails,
            anomaliesCount:
              jobMetadata?.anomaliesCount ?? countsFromStatus.anomaliesCount,
            criticalAnomaliesCount:
              jobMetadata?.criticalAnomaliesCount ?? countsFromStatus.criticalAnomaliesCount
          };
        });

        if (!matchedJob) {
          const meta = jobMetadata || {};
          updated.push({
            jobId: jobIdentifier,
            predictionId: status.prediction_id || meta.predictionId || null,
            assetName: meta.assetName || status.asset_name || '',
            farmName: meta.farmName || status.farm_name || '',
            modelName: meta.modelName || status.model_name || '',
            fileName: meta.fileName || status.file_name || '',
            batchName: meta.batchName || status.batch_name || '',
            status: status.status || meta.status || null,
            submittedAt: meta.submittedAt || status.created_at || null,
            updatedAt: status.updated_at || meta.updatedAt || null,
            pipelineStatus: nextPipeline,
            snapshot: status,
            error: status.error || meta.error || null,
            details: mergeJobDetails({}, detailPatch),
            anomaliesCount: meta.anomaliesCount ?? countsFromStatus.anomaliesCount,
            criticalAnomaliesCount:
              meta.criticalAnomaliesCount ?? countsFromStatus.criticalAnomaliesCount
          });
        }

        return updated.sort((a, b) => {
          const first = a.updatedAt ? new Date(a.updatedAt).getTime() : 0;
          const second = b.updatedAt ? new Date(b.updatedAt).getTime() : 0;
          return second - first;
        });
      });

      if (updateDetail) {
        const mergedDetails = mergeJobDetails(
          {
            predictionResult,
            anomalyEvents,
            criticalEvents,
            rcaFindings,
            narrativeEntries,
            narrative
          },
          detailPatch
        );
        setJobSnapshot(status);
        setPipelineStatus(nextPipeline);
        setPredictionResult(mergedDetails.predictionResult);
        setAnomalyEvents(mergedDetails.anomalyEvents);
        setCriticalEvents(mergedDetails.criticalEvents);
        setRcaFindings(mergedDetails.rcaFindings);
        setNarrativeEntries(mergedDetails.narrativeEntries);
        setNarrative(mergedDetails.narrative);
        setJobError(
          status.status === 'failed'
            ? new Error(status.error || 'Prediction job failed.')
            : null
        );
      }

      if (notify) {
        if (status.status === 'completed' && previousStatus !== 'completed') {
          message.success(`Prediction job ${jobIdentifier} completed.`);
        } else if (status.status === 'failed' && previousStatus !== 'failed') {
          message.error(
            status.error
              ? `Prediction job ${jobIdentifier} failed: ${status.error}`
              : `Prediction job ${jobIdentifier} failed.`
          );
        }
      }
    },
    [
      anomalyEvents,
      criticalEvents,
      narrative,
      narrativeEntries,
      predictionResult,
      rcaFindings
    ]
  );

  const loadJobs = useCallback(
    async (filters = jobFilters, { silent = false } = {}) => {
      try {
        if (!silent) {
          setJobsLoading(true);
        }
        const response = await fetchJobs(filters);
        const mapped = (response.jobs || []).map(mapJobFromResponse);
        mapped.sort((a, b) => {
          const first = a.updatedAt ? new Date(a.updatedAt).getTime() : 0;
          const second = b.updatedAt ? new Date(b.updatedAt).getTime() : 0;
          return second - first;
        });
        setJobs(mapped);
      } catch (error) {
        if (error?.response?.status === 401) {
          handleSessionExpired();
        } else {
          console.error('Failed to load jobs', error);
          if (!silent) {
            message.error('Failed to load jobs. Check the Node service logs.');
          }
        }
      } finally {
        if (!silent) {
          setJobsLoading(false);
        }
      }
    },
    [jobFilters, handleSessionExpired]
  );

  const { error: jobPollingError } = usePredictionJob(activeJobId, {
    onUpdate: (status, jobDoc) =>
      processJobStatus(status, { updateDetail: true, notify: true, jobMetadata: jobDoc }),
    interval: 4000
  });

  useEffect(() => {
    if (jobPollingError) {
      setJobError(jobPollingError);
    }
  }, [jobPollingError]);

  useEffect(() => {
    if (!authState.authToken) {
      setAuthModalVisible(true);
      setJobs([]);
      clearActiveJob();
    }
  }, [authState.authToken, clearActiveJob]);

  useEffect(() => {
    if (!authState.authToken) {
      return undefined;
    }

    const timer = window.setInterval(() => {
      loadJobs(jobFilters, { silent: true }).catch(() => {});
    }, 10000);

    return () => {
      window.clearInterval(timer);
    };
  }, [authState.authToken, jobFilters, loadJobs]);

  useEffect(() => {
    if (authState.authToken) {
      loadJobs(jobFilters).catch(() => {});
    }
  }, [authState.authToken, jobFilters, loadJobs]);

  const handleSelectJob = useCallback((jobId) => {
    if (!jobId) {
      return;
    }
    setJobError(null);
    setActiveJobId(jobId);
  }, []);

  const handleAuthSubmit = async (values) => {
    try {
      setAuthenticating(true);
      const payload = {
        organizationId: values.organizationId.trim(),
        username: values.username.trim(),
        password: values.password,
        seedToken: values.seedToken.trim()
      };
      const response = await login(payload);
      setAuthToken(response.authToken);
      setAuthState({
        organizationId: payload.organizationId,
        username: payload.username,
        authToken: response.authToken,
        expiresAt: response.expiresAt
      });
      setAuthModalVisible(false);
      message.success('Authenticated successfully.');
      const defaults = createDefaultFilterValues();
      setFilterValues(defaults);
      const initialFilters = buildFilterQuery(defaults);
      setJobFilters(initialFilters);
      await loadJobs(initialFilters);
    } catch (error) {
      console.error('Authentication failed:', error);
      message.error('Authentication failed. Verify credentials and seed token.');
    } finally {
      setAuthenticating(false);
    }
  };

  const handleUpload = async (file, metadata) => {
    if (!authState.authToken) {
      message.warning('Authenticate with the service before uploading data.');
      setAuthModalVisible(true);
      return null;
    }

    try {
      setUploading(true);
      message.loading({ content: 'Uploading dataset…', key: 'upload', duration: 0 });
      const response = await createPredictionJob(file, metadata);
      const job = response.job;
      if (job?.predictionId) {
        setPredictionId(job.predictionId);
      }
      clearActiveJob();
      await loadJobs(jobFilters, { silent: false });
      message.success({
        content: job?.jobId
          ? `Prediction job ${job.jobId} accepted. Tracking pipeline…`
          : 'Prediction job accepted. Tracking pipeline…',
        key: 'upload'
      });
      return job;
    } catch (error) {
      if (error?.response?.status === 401) {
        handleSessionExpired();
      } else {
        console.error('Prediction submission failed:', error);
        message.error({
          content: 'Failed to start the prediction job. Check logs for details.',
          key: 'upload'
        });
      }
      throw error;
    } finally {
      setUploading(false);
    }
  };

  const handleReportDownload = useCallback(() => {
    if (!predictionResult) {
      message.warning('Run a prediction before downloading a report.');
      return;
    }
    const csvContent = buildCsvReport(predictionResult, rcaFindings);
    if (!csvContent) {
      message.warning('No events available to include in the report.');
      return;
    }
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `prediction-${activeJobId || predictionId || 'result'}.csv`;
    link.click();
    window.URL.revokeObjectURL(url);
  }, [predictionResult, rcaFindings, activeJobId, predictionId]);

  const activeJob = useMemo(
    () => jobs.find((job) => job.jobId === activeJobId) || null,
    [jobs, activeJobId]
  );

  useEffect(() => {
    if (!activeJobId) {
      return;
    }
    const currentJob = jobs.find((job) => job.jobId === activeJobId);
    if (!currentJob) {
      return;
    }
    const details = mergeJobDetails({}, currentJob.details || {});
    setJobSnapshot(currentJob.snapshot || null);
    setPipelineStatus(currentJob.pipelineStatus || createInitialStatus());
    setPredictionResult(details.predictionResult);
    setAnomalyEvents(details.anomalyEvents);
    setCriticalEvents(details.criticalEvents);
    setRcaFindings(details.rcaFindings);
    setNarrativeEntries(details.narrativeEntries);
    setNarrative(details.narrative);
    if (currentJob.error) {
      setJobError(
        currentJob.error instanceof Error ? currentJob.error : new Error(currentJob.error)
      );
    }
  }, [activeJobId, jobs]);

  const handleApplyFilters = async (formValues) => {
    const nextValues = {
      status: formValues.status ?? 'in_progress',
      dateRange: formValues.dateRange ?? null,
      assetName: formValues.assetName ?? '',
      farmName: formValues.farmName ?? '',
      anomalyRange: {
        min: formValues.anomalyRange?.min ?? null,
        max: formValues.anomalyRange?.max ?? null
      },
      criticalRange: {
        min: formValues.criticalRange?.min ?? null,
        max: formValues.criticalRange?.max ?? null
      }
    };
    setFilterValues(nextValues);
    const query = buildFilterQuery(nextValues);
    setJobFilters(query);
    await loadJobs(query);
  };

  const handleResetFilters = async () => {
    const cleared = createClearedFilterValues();
    setFilterValues(cleared);
    const query = buildFilterQuery(cleared);
    setJobFilters(query);
    await loadJobs(query);
  };

  const chartData = useMemo(() => {
    if (!predictionResult?.event_sensor_data?.length) {
      return [];
    }
    const criticalIds = new Set((criticalEvents || []).map((item) => item.event_id));
    const samples = [];
    predictionResult.event_sensor_data.forEach((eventData) => {
      (eventData.points || []).forEach((point) => {
        samples.push({
          timestamp: point.timestamp,
          score: Number(point.anomaly_score ?? 0),
          criticalScore: criticalIds.has(eventData.event_id)
            ? Number(point.anomaly_score ?? 0)
            : 0,
          severity: criticalIds.has(eventData.event_id) ? 'critical' : 'anomaly',
          type: criticalIds.has(eventData.event_id) ? 'critical' : 'anomaly',
          channel:
            Object.keys(point.sensors || {})[0] || `Event ${eventData.event_id}`,
          eventId: eventData.event_id
        });
      });
    });
    return samples.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  }, [predictionResult, criticalEvents]);

  const kpis = useMemo(() => {
    const totalEvents = predictionResult?.events?.length || 0;
    const totalCritical = criticalEvents.length;
    const topSensor = rcaFindings[0]?.ranked_sensors?.[0];
    return [
      {
        title: 'Detected events',
        value: totalEvents,
        description: 'Total anomaly events detected by the trained model.',
        trend: totalEvents ? 'Processing complete' : 'No events'
      },
      {
        title: 'Critical events',
        value: totalCritical,
        description: 'Events meeting the configured criticality criteria.',
        trend: totalCritical ? 'Action required' : 'Stable'
      },
      {
        title: 'Top contributor',
        value: topSensor ? `${topSensor[0]} (${topSensor[1].toFixed?.(2) ?? topSensor[1]})` : '—',
        description: 'Highest ranked sensor from root-cause analysis.',
        trend: topSensor ? 'Investigate' : 'Awaiting results'
      }
    ];
  }, [predictionResult, criticalEvents.length, rcaFindings]);

  const hasErrors = Boolean(jobError) || jobSnapshot?.status === 'failed';
  const errorDescription = jobError?.message || jobSnapshot?.error;
  const isNarrativeReady = Boolean(narrative);

  return (
    <ConfigProvider theme={themeOverrides}>
      <Layout style={{ minHeight: '100vh', padding: '24px 48px 48px' }}>
        <Header style={{ padding: 0 }}>
          <Row justify="space-between" align="middle">
            <Col>
              <Title level={2} style={{ color: '#38bdf8', marginBottom: 0 }}>
                Energy Fault Detector Console
              </Title>
              <Paragraph className="text-subtle" style={{ marginTop: 8 }}>
                Authenticate with the Node service, launch predictions, and review the stored
                analytics from MongoDB-backed job history.
              </Paragraph>
            </Col>
            <Col>
              <Space size="middle">
                {authState.authToken ? (
                  <Space direction="vertical" size={0} style={{ textAlign: 'right' }}>
                    <Text type="secondary">Org: {authState.organizationId}</Text>
                    <Text type="secondary">User: {authState.username}</Text>
                  </Space>
                ) : (
                  <Text type="secondary">Authenticate to begin</Text>
                )}
                <Button onClick={() => setAuthModalVisible(true)}>
                  {authState.authToken ? 'Re-authenticate' : 'Authenticate'}
                </Button>
              </Space>
            </Col>
          </Row>
        </Header>
        <Content style={{ marginTop: 32 }}>
          {activeJobId ? (
            <div className="glass-panel" style={{ padding: 32 }}>
              <Row justify="space-between" align="top">
                <Col>
                  <Title level={3} style={{ marginBottom: 0, color: '#38bdf8' }}>
                    Prediction dashboard
                  </Title>
                  <Paragraph className="text-subtle" style={{ marginTop: 8 }}>
                    Review anomalies, RCA findings, and narratives for job {activeJobId}.
                  </Paragraph>
                  <Space size="large" wrap style={{ marginTop: 12 }}>
                    {[
                      {
                        label: 'Status',
                        value: activeJob?.status ? activeJob.status.replace(/_/g, ' ') : '—',
                        capitalize: true
                      },
                      { label: 'Asset', value: activeJob?.assetName || '—' },
                      { label: 'Farm', value: activeJob?.farmName || '—' },
                      { label: 'Model', value: activeJob?.modelName || '—' },
                      { label: 'File', value: activeJob?.fileName || '—' }
                    ].map((item) => (
                      <Space key={item.label} size={4}>
                        <Text type="secondary">{item.label}:</Text>
                        <Text
                          strong
                          style={item.capitalize ? { textTransform: 'capitalize' } : undefined}
                        >
                          {item.value}
                        </Text>
                      </Space>
                    ))}
                  </Space>
                </Col>
                <Col>
                  <Space>
                    <Button
                      icon={<DownloadOutlined />}
                      type="primary"
                      onClick={handleReportDownload}
                      disabled={!predictionResult}
                    >
                      Download report
                    </Button>
                    <Button onClick={clearActiveJob} ghost>
                      Back to jobs
                    </Button>
                  </Space>
                </Col>
              </Row>
              <div style={{ marginTop: 24 }}>
                <WebhookStatusPills
                  status={pipelineStatus}
                  steps={PIPELINE_STEPS}
                  hasErrors={hasErrors}
                />
              </div>
              <Row gutter={[32, 32]} style={{ marginTop: 32 }}>
                <Col xs={24} lg={14}>
                  <AnomalyChart data={chartData} />
                </Col>
                <Col xs={24} lg={10}>
                  <InsightCards items={kpis} />
                </Col>
              </Row>
              <Row gutter={[32, 32]} style={{ marginTop: 24 }}>
                <Col xs={24} md={14}>
                  <WebhookEventFeed
                    anomalies={anomalyEvents}
                    criticalEvents={criticalEvents}
                    rcaFindings={rcaFindings}
                  />
                </Col>
                <Col xs={24} md={10}>
                  <NarrativePanel
                    jobId={activeJobId}
                    narrative={narrative}
                    narratives={narrativeEntries}
                    ready={isNarrativeReady}
                  />
                </Col>
              </Row>
            </div>
          ) : (
            <div className="glass-panel" style={{ padding: 32 }}>
              <Row gutter={[32, 32]}>
                <Col xs={24} lg={8}>
                  <FileUploadPanel
                    onUpload={handleUpload}
                    loading={uploading}
                    predictionId={predictionId}
                    disabled={!authState.authToken}
                  />
                  <Paragraph className="text-subtle" style={{ marginTop: 16 }}>
                    Submit a dataset to start a new asynchronous prediction job. Completed and
                    in-progress jobs appear in the grid.
                  </Paragraph>
                </Col>
                <Col xs={24} lg={16}>
                  <Title level={3} style={{ color: '#38bdf8' }}>
                    Submitted prediction jobs
                  </Title>
                  <Paragraph className="text-subtle" style={{ marginBottom: 16 }}>
                    Jobs are served from MongoDB, keeping track of who submitted them and their
                    current status.
                  </Paragraph>
                  <JobFilters
                    values={filterValues}
                    loading={jobsLoading}
                    onApply={handleApplyFilters}
                    onReset={handleResetFilters}
                  />
                  <JobGrid jobs={jobs} loading={jobsLoading} onSelectJob={handleSelectJob} />
                </Col>
              </Row>
            </div>
          )}
          <FloatButton
            icon={<InfoCircleOutlined />}
            description="About"
            shape="square"
            tooltip="Show pipeline details"
            onClick={() => {
              Modal.info({
                title: 'Prediction pipeline',
                content: (
                  <Space direction="vertical" size="large">
                    <Paragraph>
                      The browser talks exclusively to the Node.js service. The service encrypts
                      credentials, forwards prediction jobs to the FastAPI backend, and stores
                      job metadata plus results in MongoDB.
                    </Paragraph>
                    <Paragraph>
                      Configure the service URL via <code>VITE_API_BASE_URL</code>. Jobs are
                      hydrated from MongoDB on login so operators can filter historical runs by
                      status, asset, or anomaly counts.
                    </Paragraph>
                  </Space>
                )
              });
            }}
            style={{ insetInlineEnd: 24, insetBlockEnd: 32 }}
          />
        </Content>
      </Layout>
      {hasErrors && (
        <Alert
          type="error"
          message="Prediction pipeline encountered an error."
          description={
            errorDescription ||
            'Review the logs or retry the run after validating credentials and model configuration.'
          }
          showIcon
          closable
          style={{ position: 'fixed', bottom: 24, right: 24, maxWidth: 420 }}
        />
      )}
      <Modal
        title="Authenticate with the Node service"
        open={authModalVisible}
        onCancel={() => {
          if (authState.authToken) {
            setAuthModalVisible(false);
          }
        }}
        onOk={() => authForm.submit()}
        okText={authenticating ? 'Authenticating…' : 'Authenticate'}
        confirmLoading={authenticating}
        destroyOnClose
      >
        <Form
          layout="vertical"
          form={authForm}
          onFinish={handleAuthSubmit}
          initialValues={{
            organizationId: authState.organizationId || 'sample-org',
            username: authState.username || 'analyst'
          }}
        >
          <Form.Item
            label="Organization ID"
            name="organizationId"
            rules={[{ required: true, message: 'Enter the organization identifier.' }]}
          >
            <Input placeholder="sample-org" autoComplete="organization" />
          </Form.Item>
          <Form.Item
            label="Username"
            name="username"
            rules={[{ required: true, message: 'Enter the API username.' }]}
          >
            <Input placeholder="analyst" autoComplete="username" />
          </Form.Item>
          <Form.Item
            label="Password"
            name="password"
            rules={[{ required: true, message: 'Enter the API password.' }]}
          >
            <Input.Password placeholder="••••••••" autoComplete="current-password" />
          </Form.Item>
          <Form.Item
            label="Seed token"
            name="seedToken"
            rules={[{ required: true, message: 'Enter the tenant seed token.' }]}
          >
            <Input placeholder="sample-seed-token" autoComplete="off" />
          </Form.Item>
        </Form>
      </Modal>
    </ConfigProvider>
  );
};

export default App;
