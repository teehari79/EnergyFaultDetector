import { Card, Empty, Tabs, Timeline, Typography } from 'antd';
import dayjs from 'dayjs';

const { Paragraph, Text, Title } = Typography;

const renderTimelineItem = (event, index, color) => {
  const timestamp = event.timestamp || event.event_time || event.time;
  return {
    color,
    key: `${timestamp}-${index}`,
    dot: <span className="status-pill ready" style={{ padding: '2px 8px' }}>#{index + 1}</span>,
    children: (
      <div className="timeline-card">
        <Text strong>{event.metric || event.channel || event.source || 'Signal'}</Text>
        <Paragraph style={{ marginBottom: 4 }}>
          {event.message || event.description || event.summary || 'Event detected'}
        </Paragraph>
        <Text type="secondary" style={{ display: 'block' }}>
          {timestamp ? dayjs(timestamp).format('MMM D, YYYY HH:mm:ss') : 'Timestamp unavailable'}
        </Text>
        {event.severity && (
          <Text type="warning">Severity: {event.severity}</Text>
        )}
        {event.anomaly_score && (
          <Text type="secondary" style={{ marginLeft: 12 }}>
            Score: {Number(event.anomaly_score).toFixed(2)}
          </Text>
        )}
      </div>
    )
  };
};

const createTab = (key, title, events, color) => ({
  key,
  label: `${title} (${events.length})`,
  children: events.length ? (
    <Timeline items={events.map((event, index) => renderTimelineItem(event, index, color))} mode="left" />
  ) : (
    <Empty description={`No ${title.toLowerCase()} yet.`} image={Empty.PRESENTED_IMAGE_SIMPLE} />
  )
});

const WebhookEventFeed = ({ anomalies, criticalEvents, rcaFindings }) => {
  const items = [
    createTab('anomalies', 'Anomalies', anomalies, '#38bdf8'),
    createTab('critical', 'Critical', criticalEvents, '#f97316'),
    createTab('rca', 'Root Cause', rcaFindings, '#facc15')
  ];

  return (
    <Card className="glass-panel" style={{ background: 'rgba(15, 23, 42, 0.5)' }}>
      <Title level={4} className="section-title">
        Live Event Stream
      </Title>
      <Tabs items={items} defaultActiveKey="anomalies" animated={{ inkBar: true, tabPane: true }} />
    </Card>
  );
};

export default WebhookEventFeed;
