import { useCallback, useEffect, useMemo, useState } from 'react';
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
import JobGrid from './components/JobGrid.jsx';
import WebhookStatusPills from './components/WebhookStatusPills.jsx';
import WebhookEventFeed from './components/WebhookEventFeed.jsx';
import AnomalyChart from './components/AnomalyChart.jsx';
import InsightCards from './components/InsightCards.jsx';
import NarrativePanel from './components/NarrativePanel.jsx';
import { usePredictionJob } from './hooks/usePredictionJob.js';
import { authenticate, fetchJobStatus, submitAsyncPrediction, uploadDataset } from './services/api.js';
import { PIPELINE_STEPS, DEFAULT_TIMESTAMP_COLUMN } from './config.js';
import { AUTH_CONTEXT, encryptPayload, hashAuthToken } from './services/crypto.js';

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

const normaliseNumber = (value) => {
  if (value === undefined || value === null || value === '') {
    return undefined;
  }
  const parsed = Number(value);
  return Number.isNaN(parsed) ? undefined : parsed;
};

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

const App = () => {
  const [authForm] = Form.useForm();
  const [authState, setAuthState] = useState({
    organizationId: '',
    username: '',
    seedToken: '',
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
  const [predictionResult, setPredictionResult] = useState(null);
  const [anomalyEvents, setAnomalyEvents] = useState([]);
  const [criticalEvents, setCriticalEvents] = useState([]);
  const [rcaFindings, setRcaFindings] = useState([]);
  const [narrativeEntries, setNarrativeEntries] = useState([]);
  const [narrative, setNarrative] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [jobError, setJobError] = useState(null);

  const processJobStatus = useCallback(
    (status, { updateDetail = false, notify = false } = {}) => {
      if (!status) {
        return;
      }

      const jobIdentifier = status.job_id;
      if (!jobIdentifier) {
        return;
      }

      const nextPipeline = buildPipelineStatus(status);
      const detailPatch = deriveJobDetails(status);
      let previousStatus = null;
      let matchedJob = false;

      setJobs((prevJobs) => {
        const updated = prevJobs.map((job) => {
          if (job.jobId !== jobIdentifier) {
            return job;
          }
          matchedJob = true;
          previousStatus = job.status || null;
          return {
            ...job,
            status: status.status || job.status,
            updatedAt: status.updated_at || job.updatedAt,
            pipelineStatus: nextPipeline,
            snapshot: status,
            error: status.error || null,
            details: mergeJobDetails(job.details, detailPatch)
          };
        });

        if (!matchedJob) {
          updated.push({
            jobId: jobIdentifier,
            predictionId: status.prediction_id || null,
            assetName: status.asset_name || '',
            farmName: status.farm_name || '',
            fileName: status.file_name || '',
            batchName: status.batch_name || '',
            modelName: status.model_name || '',
            status: status.status || null,
            updatedAt: status.updated_at || null,
            pipelineStatus: nextPipeline,
            snapshot: status,
            error: status.error || null,
            details: mergeJobDetails({}, detailPatch)
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

  const { error: jobPollingError } = usePredictionJob(activeJobId, authState.authToken, {
    onUpdate: (status) => processJobStatus(status, { updateDetail: true, notify: true }),
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
    }
  }, [authState.authToken]);

  const jobIds = useMemo(() => jobs.map((job) => job.jobId).filter(Boolean), [jobs]);
  const jobIdsKey = useMemo(() => jobIds.slice().sort().join('|'), [jobIds]);

  useEffect(() => {
    if (!activeJobId) {
      return;
    }
    const activeJob = jobs.find((job) => job.jobId === activeJobId);
    if (!activeJob) {
      return;
    }
    const details = mergeJobDetails({}, activeJob.details || {});
    setJobSnapshot(activeJob.snapshot || null);
    setPipelineStatus(activeJob.pipelineStatus || createInitialStatus());
    setPredictionResult(details.predictionResult);
    setAnomalyEvents(details.anomalyEvents);
    setCriticalEvents(details.criticalEvents);
    setRcaFindings(details.rcaFindings);
    setNarrativeEntries(details.narrativeEntries);
    setNarrative(details.narrative);
    if (activeJob.error) {
      setJobError(
        activeJob.error instanceof Error ? activeJob.error : new Error(activeJob.error)
      );
    }
  }, [activeJobId, jobs]);

  useEffect(() => {
    if (!authState.authToken || !jobIds.length) {
      return undefined;
    }

    let cancelled = false;

    const pollAll = async () => {
      try {
        const statuses = await Promise.all(
          jobIds.map((jobId) =>
            fetchJobStatus(jobId, authState.authToken)
              .then((data) => ({ jobId, data }))
              .catch((error) => ({ jobId, error }))
          )
        );
        if (cancelled) {
          return;
        }
        statuses.forEach(({ jobId, data, error }) => {
          if (error || !data) {
            if (error) {
              console.error(`Failed to refresh job ${jobId}:`, error);
            }
            return;
          }
          processJobStatus(data, {
            updateDetail: jobId === activeJobId,
            notify: jobId !== activeJobId
          });
        });
      } catch (error) {
        console.error('Failed to refresh job list', error);
      }
    };

    pollAll();
    const timer = window.setInterval(pollAll, 8000);

    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [authState.authToken, jobIdsKey, processJobStatus, activeJobId, jobIds]);

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
      const seedToken = values.seedToken.trim();
      const organizationId = values.organizationId.trim();
      const username = values.username.trim();
      const encryptedCredentials = await encryptPayload(
        seedToken,
        {
          username,
          password: values.password
        },
        AUTH_CONTEXT
      );
      const response = await authenticate({
        organizationId,
        credentialsEncrypted: encryptedCredentials
      });
      setAuthState({
        organizationId,
        username,
        seedToken,
        authToken: response.auth_token,
        expiresAt: response.expires_at
      });
      message.success('Authenticated successfully.');
      setAuthModalVisible(false);
    } catch (error) {
      console.error('Authentication failed:', error);
      message.error('Authentication failed. Verify credentials and seed token.');
    } finally {
      setAuthenticating(false);
    }
  };

  const handleUpload = async (file, metadata) => {
    if (!authState.authToken || !authState.seedToken || !authState.organizationId) {
      message.warning('Authenticate with the prediction API before uploading data.');
      setAuthModalVisible(true);
      return null;
    }

    if (!metadata.model_name) {
      message.error('Model name is required to resolve the trained detector.');
      return null;
    }

    try {
      setUploading(true);
      message.loading({ content: 'Uploading dataset…', key: 'upload', duration: 0 });
      const uploadResponse = await uploadDataset(file, metadata);
      if (!uploadResponse?.data_path) {
        throw new Error('Upload response did not include the dataset path.');
      }
      setPredictionId(uploadResponse.prediction_id);
      clearActiveJob();
      const minEventLength = normaliseNumber(metadata.min_event_length);
      const requestPayload = {
        organization_id: authState.organizationId,
        request: {
          model_name: metadata.model_name,
          data_path: uploadResponse.data_path,
          timestamp_column: metadata.timestamp_column || DEFAULT_TIMESTAMP_COLUMN
        }
      };
      if (metadata.asset_name) {
        requestPayload.request.asset_name = metadata.asset_name;
      }
      if (metadata.farm_name) {
        requestPayload.request.farm_name = metadata.farm_name;
      }
      if (metadata.model_version) {
        requestPayload.request.model_version = metadata.model_version;
      }
      if (minEventLength !== undefined) {
        requestPayload.request.min_event_length = minEventLength;
      }
      const minEventDuration = metadata.min_event_duration;
      if (
        minEventDuration !== undefined &&
        minEventDuration !== null &&
        `${minEventDuration}`.trim() !== ''
      ) {
        requestPayload.request.min_event_duration = minEventDuration;
      }
      if (metadata.enable_narrative === false) {
        requestPayload.request.enable_narrative = false;
      }

      const authHash = await hashAuthToken(authState.authToken, authState.seedToken);
      const encryptedPayload = await encryptPayload(
        authState.seedToken,
        requestPayload,
        authHash
      );
      const predictionResponse = await submitAsyncPrediction({
        authToken: authState.authToken,
        authHash,
        payloadEncrypted: encryptedPayload
      });
      const submittedAt = new Date().toISOString();
      const jobMetadata = {
        jobId: predictionResponse.job_id,
        predictionId: uploadResponse.prediction_id,
        assetName: metadata.asset_name?.trim() || '',
        farmName: metadata.farm_name?.trim() || '',
        fileName: file.name,
        batchName: metadata.batch_name,
        modelName: metadata.model_name
      };
      setJobs((prevJobs) => {
        const filtered = prevJobs.filter((job) => job.jobId !== jobMetadata.jobId);
        return [
          {
            ...jobMetadata,
            status: 'queued',
            updatedAt: submittedAt,
            pipelineStatus: createInitialStatus(),
            snapshot: null,
            error: null,
            details: mergeJobDetails({}, {})
          },
          ...filtered
        ];
      });
      setActiveJobId(null);
      message.success({
        content: `Prediction job ${predictionResponse.job_id} accepted. Tracking pipeline…`,
        key: 'upload'
      });
      return predictionResponse;
    } catch (error) {
      console.error('Prediction submission failed:', error);
      message.error({
        content: 'Failed to start the prediction job. Check logs for details.',
        key: 'upload'
      });
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
                Authenticate, submit predictions against trained asset models, and inspect the
                asynchronous pipeline output.
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
                    Monitor pipeline progress and open a job to explore its analytics.
                  </Paragraph>
                  <JobGrid jobs={jobs} onSelectJob={handleSelectJob} />
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
                      The UI submits uploaded datasets to the asynchronous prediction API.
                      Each pipeline stage updates the job record, which the dashboard polls
                      to surface anomaly events, critical findings, and RCA metrics.
                    </Paragraph>
                    <Paragraph>
                      Configure API endpoints via <code>VITE_API_BASE_URL</code> for dataset
                      storage and <code>VITE_ASYNC_API_BASE_URL</code> for the prediction
                      service.
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
        title="Authenticate with the Prediction API"
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
