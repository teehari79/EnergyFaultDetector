import { Card, Empty, List, Space, Typography } from 'antd';
import { ClockCircleOutlined } from '@ant-design/icons';

const { Paragraph, Text, Title } = Typography;

const NarrativePanel = ({ jobId, narrative, narratives = [], ready }) => (
  <Card className="glass-panel" style={{ height: '100%', background: 'rgba(15, 23, 42, 0.55)' }}>
    <Space direction="vertical" size="large" style={{ width: '100%' }}>
      <div>
        <Title level={4} className="section-title">
          Narrative Intelligence
        </Title>
        <Paragraph className="text-subtle">
          Contextual summaries are generated automatically once the asynchronous pipeline completes.
        </Paragraph>
      </div>
      <Text className={`status-pill ${ready ? 'ready' : 'waiting'}`}>
        {ready ? 'Narrative ready' : 'Awaiting pipeline completion'}
      </Text>
      {ready && narrative ? (
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
          description="Narrative summaries will appear once the job finishes."
        />
      )}
      {ready && narratives.length > 1 && (
        <List
          size="small"
          bordered
          dataSource={narratives}
          renderItem={(item) => (
            <List.Item style={{ background: 'rgba(15, 23, 42, 0.35)' }}>
              <Text>{item.narrative}</Text>
            </List.Item>
          )}
        />
      )}
      {jobId && (
        <Space direction="horizontal" style={{ color: '#94a3b8' }}>
          <ClockCircleOutlined />
          <Text>Job ID: {jobId}</Text>
        </Space>
      )}
    </Space>
  </Card>
);

export default NarrativePanel;
