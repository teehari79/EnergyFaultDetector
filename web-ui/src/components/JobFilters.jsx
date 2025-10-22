import { useEffect } from 'react';
import { Button, Col, DatePicker, Form, Input, InputNumber, Row, Select, Space } from 'antd';

const { RangePicker } = DatePicker;

const statusOptions = [
  { value: 'in_progress', label: 'In progress' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
  { value: 'all', label: 'All jobs' }
];

const JobFilters = ({ values, loading, onApply, onReset }) => {
  const [form] = Form.useForm();

  useEffect(() => {
    form.setFieldsValue(values);
  }, [values, form]);

  const handleFinish = (submitted) => {
    onApply(submitted);
  };

  const handleReset = () => {
    form.resetFields();
    onReset();
  };

  return (
    <Form
      form={form}
      layout="vertical"
      onFinish={handleFinish}
      initialValues={values}
      style={{ marginBottom: 24 }}
    >
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Form.Item label="Status" name="status">
            <Select options={statusOptions} placeholder="Job status" allowClear />
          </Form.Item>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Form.Item label="Job window" name="dateRange">
            <RangePicker showTime allowClear style={{ width: '100%' }} />
          </Form.Item>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Form.Item label="Asset name" name="assetName">
            <Input placeholder="e.g. turbine-12" allowClear />
          </Form.Item>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Form.Item label="Farm name" name="farmName">
            <Input placeholder="e.g. north-farm" allowClear />
          </Form.Item>
        </Col>
      </Row>
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Row gutter={8}>
            <Col span={12}>
              <Form.Item label="Anomalies ≥" name={['anomalyRange', 'min']}>
                <InputNumber min={0} style={{ width: '100%' }} placeholder="Min" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="Anomalies ≤" name={['anomalyRange', 'max']}>
                <InputNumber min={0} style={{ width: '100%' }} placeholder="Max" />
              </Form.Item>
            </Col>
          </Row>
        </Col>
        <Col xs={24} sm={12} md={8} lg={6}>
          <Row gutter={8}>
            <Col span={12}>
              <Form.Item label="Critical ≥" name={['criticalRange', 'min']}>
                <InputNumber min={0} style={{ width: '100%' }} placeholder="Min" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="Critical ≤" name={['criticalRange', 'max']}>
                <InputNumber min={0} style={{ width: '100%' }} placeholder="Max" />
              </Form.Item>
            </Col>
          </Row>
        </Col>
      </Row>
      <Space>
        <Button type="primary" htmlType="submit" loading={loading}>
          Apply filters
        </Button>
        <Button onClick={handleReset} disabled={loading}>
          Clear filters
        </Button>
      </Space>
    </Form>
  );
};

export default JobFilters;
