import { Card, Col, Row, Typography } from 'antd';

const { Paragraph, Text, Title } = Typography;

const InsightCards = ({ items }) => (
  <Row gutter={[24, 24]} className="dashboard-grid">
    {items.map((item) => (
      <Col key={item.title} xs={24} md={12} xl={8}>
        <Card className="glass-panel" style={{ height: '100%' }}>
          <Title level={4} style={{ color: '#e0f2fe', marginBottom: 12 }}>
            {item.title}
          </Title>
          <Text className="metric-value">{item.value}</Text>
          <Paragraph className="text-subtle" style={{ marginTop: 12 }}>
            {item.description}
          </Paragraph>
          <Text type="success">{item.trend}</Text>
        </Card>
      </Col>
    ))}
  </Row>
);

export default InsightCards;
