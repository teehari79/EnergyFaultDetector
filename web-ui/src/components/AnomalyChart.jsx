import { Card, Empty, Typography } from 'antd';
import {
  Area,
  AreaChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from 'recharts';
import dayjs from 'dayjs';

const { Title } = Typography;

const colors = {
  anomaly: '#38bdf8',
  critical: '#f97316'
};

const formatTimestamp = (value) => (value ? dayjs(value).format('HH:mm:ss') : '');

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  const [event] = payload;
  const data = event.payload || {};
  return (
    <div
      style={{
        padding: '12px 16px',
        borderRadius: 12,
        background: 'rgba(15, 23, 42, 0.95)',
        border: '1px solid rgba(148, 163, 184, 0.3)',
        color: '#f8fafc',
        minWidth: 220
      }}
    >
      <Title level={5} style={{ color: '#38bdf8', marginBottom: 8 }}>
        {data.channel || data.metric || data.type}
      </Title>
      <p style={{ margin: 0, fontSize: 12 }}>{dayjs(label).format('MMM D, YYYY HH:mm:ss')}</p>
      <p style={{ margin: '4px 0 0', fontWeight: 600 }}>Score: {data.score?.toFixed?.(3) ?? data.score}</p>
      <p style={{ margin: 0 }}>Severity: {data.severity || data.type}</p>
    </div>
  );
};

const AnomalyChart = ({ data }) => (
  <Card className="glass-panel" style={{ height: '100%', background: 'rgba(15, 23, 42, 0.5)' }}>
    <Title level={4} className="section-title">
      Event density & severity
    </Title>
    {data.length ? (
      <ResponsiveContainer width="100%" height={320}>
        <AreaChart data={data} margin={{ top: 32, right: 16, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="colorAnomaly" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.9} />
              <stop offset="95%" stopColor="#38bdf8" stopOpacity={0.05} />
            </linearGradient>
            <linearGradient id="colorCritical" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#f97316" stopOpacity={0.9} />
              <stop offset="95%" stopColor="#f97316" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
          <XAxis dataKey="timestamp" tickFormatter={formatTimestamp} stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" domain={[0, 'auto']} />
          <Tooltip content={<CustomTooltip />} />
          <Legend />
          <Area
            type="monotone"
            dataKey="score"
            name="Anomaly score"
            stroke={colors.anomaly}
            fill="url(#colorAnomaly)"
          />
          <Area
            type="monotone"
            dataKey="criticalScore"
            name="Critical intensity"
            stroke={colors.critical}
            fill="url(#colorCritical)"
          />
        </AreaChart>
      </ResponsiveContainer>
    ) : (
      <Empty description="Upload a dataset to see anomaly trends." />
    )}
  </Card>
);

export default AnomalyChart;
