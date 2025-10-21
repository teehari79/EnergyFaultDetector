import { useState } from 'react';
import { InboxOutlined } from '@ant-design/icons';
import { Button, Form, Input, InputNumber, Switch, Upload } from 'antd';

const { Dragger } = Upload;

const FileUploadPanel = ({ onUpload, loading, predictionId, disabled }) => {
  const [form] = Form.useForm();
  const [fileList, setFileList] = useState([]);

  const beforeUpload = (file) => {
    setFileList([file]);
    return false;
  };

  const handleRemove = () => {
    setFileList([]);
  };

  const handleSubmit = async (values) => {
    if (!fileList.length || disabled) {
      return;
    }
    try {
      await onUpload(fileList[0], values);
      setFileList([]);
      form.resetFields([
        'notes',
        'min_event_length',
        'min_event_duration',
        'enable_narrative'
      ]);
    } catch (error) {
      // Surface error via onUpload's messaging
    }
  };

  return (
    <div className="gradient-card">
      <h3>Upload data for prediction</h3>
      <p className="text-subtle" style={{ marginBottom: 16 }}>
        Drag and drop your telemetry CSV/Parquet or select from your device to trigger the
        prediction pipeline.
      </p>
      {disabled && (
        <p className="text-subtle" style={{ marginBottom: 12 }}>
          Authenticate with the prediction API to enable uploads.
        </p>
      )}
      <Form form={form} layout="vertical" onFinish={handleSubmit} disabled={disabled}>
        <Form.Item
          name="batch_name"
          label="Batch name"
          initialValue={`Batch-${Date.now().toString().slice(-6)}`}
          rules={[{ required: true, message: 'Please provide a batch name' }]}
        >
          <Input placeholder="Shift-42 Night Run" />
        </Form.Item>
        <Form.Item
          name="model_name"
          label="Model name"
          rules={[{ required: true, message: 'Specify the trained model name.' }]}
        >
          <Input placeholder="farm-c" />
        </Form.Item>
        <Form.Item name="asset_name" label="Asset name">
          <Input placeholder="Optional asset identifier" />
        </Form.Item>
        <Form.Item name="model_version" label="Model version">
          <Input placeholder="Latest" />
        </Form.Item>
        <Form.Item
          name="timestamp_column"
          label="Timestamp column"
          initialValue="time_stamp"
        >
          <Input placeholder="time_stamp" />
        </Form.Item>
        <Form.Item name="notes" label="Notes">
          <Input.TextArea placeholder="Describe the scenario" autoSize={{ minRows: 2, maxRows: 4 }} />
        </Form.Item>
        <Form.Item name="min_event_length" label="Minimum event length">
          <InputNumber min={0} style={{ width: '100%' }} placeholder="Use model default" />
        </Form.Item>
        <Form.Item
          name="min_event_duration"
          label="Minimum event duration"
          tooltip="Accepts pandas timedelta strings or seconds"
        >
          <Input placeholder="e.g. 5min or 120" />
        </Form.Item>
        <Form.Item
          name="enable_narrative"
          label="Generate narrative"
          initialValue
          valuePropName="checked"
        >
          <Switch />
        </Form.Item>
        <Dragger
          multiple={false}
          maxCount={1}
          beforeUpload={beforeUpload}
          onRemove={handleRemove}
          fileList={fileList}
          accept=".csv,.parquet,.json"
          disabled={loading || disabled}
          style={{ borderRadius: 16, background: 'rgba(15, 23, 42, 0.4)', border: '1px dashed #38bdf8' }}
        >
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">Click or drag data file to this area</p>
          <p className="ant-upload-hint">
            Supports CSV, JSON, and Parquet files. Max 200MB. Pipeline will start immediately after
            upload completes.
          </p>
        </Dragger>
        <Form.Item style={{ marginTop: 16 }}>
          <Button
            type="primary"
            block
            htmlType="submit"
            loading={loading}
            disabled={!fileList.length || disabled}
          >
            {predictionId ? 'Upload another batch' : 'Start prediction'}
          </Button>
        </Form.Item>
      </Form>
    </div>
  );
};

export default FileUploadPanel;
