import { useCallback, useMemo, useState } from 'react';
import {
  Alert,
  Button,
  Col,
  ConfigProvider,
  FloatButton,
  Layout,
  message,
  Modal,
  Row,
  Space,
  Typography
} from 'antd';
import { DownloadOutlined, InfoCircleOutlined, SendOutlined } from '@ant-design/icons';
import FileUploadPanel from './components/FileUploadPanel.jsx';
import WebhookStatusPills from './components/WebhookStatusPills.jsx';
import WebhookEventFeed from './components/WebhookEventFeed.jsx';
import AnomalyChart from './components/AnomalyChart.jsx';
import InsightCards from './components/InsightCards.jsx';
import NarrativePanel from './components/NarrativePanel.jsx';
import { useWebhookStream } from './hooks/useWebhookStream.js';
import { downloadReport, fetchNarrative, uploadDataset } from './services/api.js';
import { WEBHOOK_ENDPOINTS } from './config.js';

const { Header, Content } = Layout;
const { Title, Paragraph } = Typography;

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

const App = () => {
  const [predictionId, setPredictionId] = useState(null);
  const [anomalyEvents, setAnomalyEvents] = useState([]);
  const [criticalEvents, setCriticalEvents] = useState([]);
  const [rcaFindings, setRcaFindings] = useState([]);
  const [narrative, setNarrative] = useState(null);
  const [narrativeLoading, setNarrativeLoading] = useState(false);
  const [uploading, setUploading] = useState(false);

  const resetDashboard = () => {
    setPredictionId(null);
    setAnomalyEvents([]);
    setCriticalEvents([]);
    setRcaFindings([]);
    setNarrative(null);
    setUploading(false);
  };

  const onWebhookEvent = useCallback((hook, payload, { allEvents }) => {
    const streamPayload = Array.isArray(payload) ? payload : [payload];
    switch (hook.key) {
      case 'anomalies':
        setAnomalyEvents((prev) => [...prev, ...streamPayload]);
        break;
      case 'critical':
        setCriticalEvents((prev) => [...prev, ...streamPayload]);
        break;
      case 'rca':
        setRcaFindings((prev) => [...prev, ...streamPayload]);
        break;
      default:
        break;
    }

    if (allEvents) {
      setNarrative(null);
    }
  }, []);

  const { status, hasErrors } = useWebhookStream(predictionId, {
    onEvent: onWebhookEvent
  });

  const isNarrativeReady = useMemo(() => {
    const readyKeys = WEBHOOK_ENDPOINTS.filter((hook) => status[hook.key]?.state === 'ready');
    return readyKeys.length === WEBHOOK_ENDPOINTS.length;
  }, [status]);

  const handleUpload = async (file, metadata) => {
    try {
      setUploading(true);
      message.loading({ content: 'Uploading dataset…', key: 'upload', duration: 0 });
      console.info('Uploading dataset', {
        filename: file?.name,
        size: file?.size,
        metadata
      });
      const response = await uploadDataset(file, metadata);
      setPredictionId(response?.prediction_id);
      setAnomalyEvents([]);
      setCriticalEvents([]);
      setRcaFindings([]);
      setNarrative(null);
      message.success({ content: 'Upload successful. Listening for webhooks…', key: 'upload' });
      console.info('Upload completed', {
        predictionId: response?.prediction_id
      });
      return response;
    } catch (error) {
      console.group('Dataset upload failed');
      console.error('Upload error:', error);
      if (error?.response) {
        console.error('Response status:', error.response.status);
        console.error('Response body:', error.response.data);
      }
      console.groupEnd();
      message.error({ content: 'Failed to upload dataset. Please try again.', key: 'upload' });
      throw error;
    } finally {
      setUploading(false);
    }
  };

  const handleNarrative = async () => {
    if (!predictionId) {
      message.warning('Upload data and wait for webhooks before requesting a narrative.');
      return;
    }
    try {
      setNarrativeLoading(true);
      const response = await fetchNarrative(predictionId);
      setNarrative(response?.summary ?? 'Narrative not available.');
      message.success('Narrative generated successfully.');
    } catch (error) {
      console.error(error);
      message.error('Failed to generate narrative.');
    } finally {
      setNarrativeLoading(false);
    }
  };

  const handleReportDownload = async () => {
    if (!predictionId) {
      message.warning('No prediction report available yet.');
      return;
    }
    try {
      const blob = await downloadReport(predictionId);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `prediction-${predictionId}.pdf`;
      link.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error(error);
      message.error('Unable to download the report.');
    }
  };

  const chartData = useMemo(() => {
    const anomalySeries = anomalyEvents.map((item) => ({
      timestamp: item.timestamp || item.event_time || item.time,
      channel: item.channel || item.metric || item.source || 'Unknown',
      score: Number(item.score ?? item.anomaly_score ?? item.value ?? 0),
      criticalScore: 0,
      severity: item.severity || item.category || 'anomaly',
      type: 'anomaly'
    }));

    const criticalSeries = criticalEvents.map((item) => ({
      timestamp: item.timestamp || item.event_time || item.time,
      channel: item.channel || item.metric || item.source || 'Unknown',
      score: 0,
      criticalScore: Number(item.score ?? item.anomaly_score ?? item.value ?? 0),
      severity: item.severity || item.criticality || 'critical',
      type: 'critical'
    }));

    return [...anomalySeries, ...criticalSeries]
      .filter((event) => event.timestamp)
      .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
  }, [anomalyEvents, criticalEvents]);

  const kpis = useMemo(
    () => [
      {
        title: 'Anomaly Events',
        value: anomalyEvents.length,
        description: 'Events flagged via anomaly web-hook.',
        trend: '+12% vs last run'
      },
      {
        title: 'Critical Events',
        value: criticalEvents.length,
        description: 'Immediate attention required events.',
        trend: '+5%'
      },
      {
        title: 'Root Causes',
        value: rcaFindings.length,
        description: 'RCA hypotheses collected from pipeline.',
        trend: 'Stable'
      }
    ],
    [anomalyEvents.length, criticalEvents.length, rcaFindings.length]
  );

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
                Upload, monitor, and narrate predictions in real-time.
              </Paragraph>
            </Col>
            <Col>
              {predictionId && (
                <Space>
                  <Button icon={<DownloadOutlined />} onClick={handleReportDownload} type="primary">
                    Download report
                  </Button>
                  <Button onClick={resetDashboard} ghost>
                    Reset
                  </Button>
                </Space>
              )}
            </Col>
          </Row>
        </Header>
        <Content style={{ marginTop: 32 }}>
          <div className="glass-panel" style={{ padding: 32 }}>
            <Row gutter={[32, 32]}>
              <Col xs={24} lg={10}>
                <FileUploadPanel
                  onUpload={handleUpload}
                  loading={uploading}
                  predictionId={predictionId}
                />
                <div style={{ marginTop: 24 }}>
                  <WebhookStatusPills status={status} endpoints={WEBHOOK_ENDPOINTS} hasErrors={hasErrors} />
                </div>
              </Col>
              <Col xs={24} lg={14}>
                <AnomalyChart data={chartData} />
              </Col>
            </Row>
            <Row gutter={[32, 32]} style={{ marginTop: 32 }}>
              <Col span={24}>
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
                  predictionId={predictionId}
                  onNarrate={handleNarrative}
                  loading={narrativeLoading}
                  ready={isNarrativeReady}
                  narrative={narrative}
                />
              </Col>
            </Row>
          </div>
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
                      The UI listens to anomaly, critical anomaly, and RCA web-hooks. Predictions
                      begin right after the dataset is uploaded. Narrative summaries stay disabled
                      until all web-hooks finish streaming events.
                    </Paragraph>
                    <Paragraph>
                      Configure API endpoints through <code>VITE_API_BASE_URL</code>. Each webhook
                      should stream Server-Sent Events (SSE) returning JSON payloads.
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
          message="One or more web-hooks failed."
          description="Please verify the webhook endpoints or retry the prediction run."
          showIcon
          closable
          style={{ position: 'fixed', bottom: 24, right: 24, maxWidth: 360 }}
        />
      )}
      <FloatButton.Group shape="circle" style={{ insetInlineEnd: 24, insetBlockEnd: 96 }}>
        <FloatButton
          icon={<SendOutlined />}
          tooltip="Generate narrative"
          onClick={handleNarrative}
          disabled={!isNarrativeReady || narrativeLoading}
        />
      </FloatButton.Group>
    </ConfigProvider>
  );
};

export default App;
