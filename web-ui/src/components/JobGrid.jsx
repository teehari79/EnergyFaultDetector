import { Button, Space, Table, Tag, Typography } from 'antd';
import dayjs from 'dayjs';

const STEP_COLUMNS = [
  { key: 'event_detection', title: 'Determining anomalies' },
  { key: 'criticality_analysis', title: 'Detecting critical anomaly' },
  { key: 'root_cause_analysis', title: 'Detecting RCA' }
];

const statusColorMap = {
  completed: 'success',
  ready: 'success',
  running: 'processing',
  pending: 'default',
  waiting: 'default',
  queued: 'processing',
  accepted: 'processing',
  'in-progress': 'processing',
  partial: 'processing',
  error: 'error',
  failed: 'error'
};

const stepStateLabel = {
  ready: 'Completed',
  completed: 'Completed',
  waiting: 'Pending',
  pending: 'Pending',
  queued: 'Queued',
  accepted: 'Queued',
  running: 'In progress',
  'in-progress': 'In progress',
  partial: 'In progress',
  error: 'Error'
};

const formatStepStatus = (job, stepKey) => {
  const entry = job.pipelineStatus?.[stepKey] || {};
  const baseState = entry.state || job.status || 'pending';
  let displayState = baseState;

  if (entry.state === 'ready') {
    displayState = 'completed';
  } else if (entry.state === 'error') {
    displayState = 'error';
  } else if (job.status === 'failed') {
    displayState = 'error';
  } else if (job.status === 'completed' && entry.state !== 'ready') {
    displayState = 'completed';
  } else if (job.status === 'running' && !['ready', 'error'].includes(entry.state)) {
    displayState = 'running';
  }

  const label = stepStateLabel[displayState] || stepStateLabel[entry.state] || 'Pending';
  const color = statusColorMap[displayState] || 'default';

  return { label, color };
};

const JobGrid = ({ jobs, loading, onSelectJob }) => {
  const columns = [
    {
      title: 'Asset name',
      dataIndex: 'assetName',
      key: 'assetName',
      render: (value) => value || '—'
    },
    {
      title: 'Farm name',
      dataIndex: 'farmName',
      key: 'farmName',
      render: (value) => value || '—'
    },
    {
      title: 'File name',
      dataIndex: 'fileName',
      key: 'fileName',
      ellipsis: true,
      render: (value) => value || '—'
    },
    {
      title: 'Job ID',
      dataIndex: 'jobId',
      key: 'jobId',
      render: (value) => <Typography.Text code>{value}</Typography.Text>
    },
    ...STEP_COLUMNS.map((step) => ({
      title: step.title,
      key: step.key,
      render: (_, record) => {
        const { label, color } = formatStepStatus(record, step.key);
        return <Tag color={color}>{label}</Tag>;
      }
    })),
    {
      title: 'Overall status',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const normalized = (status || 'pending').toLowerCase();
        const color = statusColorMap[normalized] || 'default';
        const label = stepStateLabel[normalized] || normalized;
        return <Tag color={color}>{label.charAt(0).toUpperCase() + label.slice(1)}</Tag>;
      }
    },
    {
      title: 'Submitted at',
      dataIndex: 'submittedAt',
      key: 'submittedAt',
      render: (value) => (value ? dayjs(value).format('YYYY-MM-DD HH:mm:ss') : '—')
    },
    {
      title: 'Last update',
      dataIndex: 'updatedAt',
      key: 'updatedAt',
      render: (value) => (value ? dayjs(value).format('YYYY-MM-DD HH:mm:ss') : '—')
    },
    {
      title: 'Anomalies',
      dataIndex: 'anomaliesCount',
      key: 'anomaliesCount',
      render: (value) => (value === null || value === undefined ? '—' : value)
    },
    {
      title: 'Critical anomalies',
      dataIndex: 'criticalAnomaliesCount',
      key: 'criticalAnomaliesCount',
      render: (value) => (value === null || value === undefined ? '—' : value)
    },
    {
      title: 'Actions',
      key: 'actions',
      render: (_, record) => {
        const normalizedStatus = (record.status || '').toLowerCase();
        const canInspect = ['running', 'completed', 'partial', 'in-progress'].includes(
          normalizedStatus
        );
        return (
          <Space>
            <Button type="link" onClick={() => onSelectJob(record.jobId)} disabled={!canInspect}>
              View details
            </Button>
          </Space>
        );
      }
    }
  ];

  return (
    <Table
      rowKey={(record) => record.jobId}
      columns={columns}
      dataSource={jobs}
      loading={loading}
      pagination={false}
      locale={{ emptyText: 'No prediction jobs submitted yet.' }}
      scroll={{ x: 'max-content' }}
    />
  );
};

export default JobGrid;
