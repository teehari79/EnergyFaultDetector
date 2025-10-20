import { Button, Card, Empty, Space, Typography } from 'antd';
import { BulbOutlined, ClockCircleOutlined } from '@ant-design/icons';

const { Paragraph, Text, Title } = Typography;

const NarrativePanel = ({ predictionId, onNarrate, loading, ready, narrative }) => (
  <Card className="glass-panel" style={{ height: '100%', background: 'rgba(15, 23, 42, 0.55)' }}>
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <div>
        <Title level={4} className="section-title">
          Narrative Intelligence
        </Title>
        <Paragraph className="text-subtle">
          Summarize anomalies, critical events, and RCA findings once the data stream is complete.
        </Paragraph>
      </div>
      <div>
        <Text className={`status-pill ${ready ? 'ready' : 'waiting'}`}>
          {ready ? 'Ready for narrative' : 'Awaiting webhook data'}
        </Text>
      </div>
      <Button
        type="primary"
        icon={<BulbOutlined />}
        block
        size="large"
        disabled={!ready || loading}
        onClick={onNarrate}
        loading={loading}
      >
        Generate narrative
      </Button>
      {narrative ? (
        <Card
          size="small"
          style={{
            background: 'rgba(15, 23, 42, 0.65)',
            borderRadius: 16,
            border: '1px solid rgba(148, 163, 184, 0.2)'
          }}
        >
          <Paragraph style={{ color: '#e2e8f0', whiteSpace: 'pre-line' }}>{narrative}</Paragraph>
        </Card>
      ) : (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={
            <span className="text-subtle">
              {ready
                ? 'Click generate to receive a natural language summary.'
                : 'The narrative button unlocks after all webhooks have streamed their data.'}
            </span>
          }
        />
      )}
      {predictionId && (
        <Space direction="horizontal" style={{ color: '#94a3b8' }}>
          <ClockCircleOutlined />
          <Text>Prediction ID: {predictionId}</Text>
        </Space>
      )}
    </Space>
  </Card>
);

export default NarrativePanel;
