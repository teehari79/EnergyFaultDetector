import { Badge, Space, Tooltip, Typography } from 'antd';
import dayjs from 'dayjs';

const { Text } = Typography;

const colors = {
  waiting: 'processing',
  ready: 'success',
  error: 'error'
};

const WebhookStatusPills = ({ status, endpoints, hasErrors }) => (
  <Space wrap size={[12, 12]}>
    {endpoints.map((endpoint) => {
      const entry = status[endpoint.key] || {};
      const state = entry.state || 'waiting';
      const color = colors[state] || 'default';
      const description = entry.updatedAt
        ? dayjs(entry.updatedAt).format('HH:mm:ss')
        : 'Pending';
      return (
        <Tooltip
          key={endpoint.key}
          title={
            entry.error
              ? entry.error.message || 'Unknown error'
              : `Last update: ${description}`
          }
        >
          <span className={`status-pill ${state}`}>
            <Badge status={color} />
            <Text style={{ marginLeft: 8 }} strong>
              {endpoint.label}
            </Text>
          </span>
        </Tooltip>
      );
    })}
    {hasErrors && <Text type="danger">Check your webhook configuration.</Text>}
  </Space>
);

export default WebhookStatusPills;
