import { Card, Empty, Tabs, Timeline, Typography } from 'antd';
import dayjs from 'dayjs';

const { Paragraph, Text, Title } = Typography;

const formatRange = (start, end) => {
  if (!start && !end) {
    return 'Timestamp unavailable';
  }
  if (start && end) {
    return `${dayjs(start).format('MMM D, YYYY HH:mm:ss')} → ${dayjs(end).format(
      'HH:mm:ss'
    )}`;
  }
  const value = start || end;
  return dayjs(value).format('MMM D, YYYY HH:mm:ss');
};

const renderTimelineItem = (event, index, color) => {
  const { title, description, start, end, footer } = event;
  return {
    color,
    key: `${title}-${index}`,
    dot: <span className="status-pill ready" style={{ padding: '2px 8px' }}>#{index + 1}</span>,
    children: (
      <div className="timeline-card">
        <Text strong>{title}</Text>
        {description && <Paragraph style={{ marginBottom: 4 }}>{description}</Paragraph>}
        <Text type="secondary" style={{ display: 'block' }}>
          {formatRange(start, end)}
        </Text>
        {footer && (
          <Text type="secondary" style={{ display: 'block', marginTop: 4 }}>
            {footer}
          </Text>
        )}
      </div>
    )
  };
};

const normaliseAnomalies = (events) =>
  (events || []).map((event) => ({
    title: `Event #${event.event_id}`,
    description: `Duration ${Number(event.duration_seconds ?? 0).toFixed(1)} seconds`,
    start: event.start,
    end: event.end
  }));

const normaliseCritical = (events) =>
  (events || []).map((event) => ({
    title: `Critical event #${event.event_id}`,
    description: event.sample_count
      ? `Sample count ${event.sample_count}`
      : 'Critical threshold met',
    start: event.start,
    end: event.end,
    footer: event.reason || null
  }));

const normaliseRootCause = (findings) =>
  (findings || []).map((entry) => ({
    title: `Root cause ranking #${entry.event_id}`,
    description: (entry.ranked_sensors || [])
      .slice(0, 3)
      .map(([name, score]) => `${name} (${Number(score).toFixed?.(2) ?? score})`)
      .join(', ') || 'No dominant sensors',
    start: null,
    end: null
  }));

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
  const anomalyItems = normaliseAnomalies(anomalies);
  const criticalItems = normaliseCritical(criticalEvents);
  const rcaItems = normaliseRootCause(rcaFindings);

  const items = [
    createTab('anomalies', 'Anomalies', anomalyItems, '#38bdf8'),
    createTab('critical', 'Critical', criticalItems, '#f97316'),
    createTab('rca', 'Root Cause', rcaItems, '#facc15')
  ];

  return (
    <Card className="glass-panel" style={{ background: 'rgba(15, 23, 42, 0.5)' }}>
      <Title level={4} className="section-title">
        Pipeline Events
      </Title>
      <Tabs items={items} defaultActiveKey="anomalies" animated={{ inkBar: true, tabPane: true }} />
    </Card>
  );
};

export default WebhookEventFeed;
