import React, { useState } from 'react';
import { Upload, message, Typography, Spin, Card, Row, Col, Result, Button } from 'antd';
import { InboxOutlined, FileTextOutlined, CheckCircleOutlined } from '@ant-design/icons';

const { Dragger } = Upload;
const { Title, Text, Paragraph } = Typography;

const FileUpload = () => {
  const [fileList, setFileList] = useState([]);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [showThankYou, setShowThankYou] = useState(false);

  const props = {
    name: 'file',
    multiple: false,
    action: 'http://localhost:5000/analyze',
    accept: '.mp3,.wav',
    onChange(info) {
      const { status } = info.file;
      if (status === 'uploading') {
        setLoading(true);
        setAnalysisResult(null);
        setShowThankYou(false);
      }
      if (status === 'done') {
        setLoading(false);
        const result = info.file.response;
        setAnalysisResult(result);
        setShowThankYou(true);
        message.success(`${info.file.name} file analyzed successfully.`);
      } else if (status === 'error') {
        setLoading(false);
        message.error(`${info.file.name} file analysis failed.`);
      }
    },
  };

  const resetAnalysis = () => {
    setAnalysisResult(null);
    setShowThankYou(false);
  };

  return (
    <div style={{ padding: '40px', maxWidth: '1000px', margin: '0 auto' }}>
      <Row gutter={[24, 24]} justify="center">
        <Col span={24}>
          <Title level={2} style={{ textAlign: 'center', color: '#1890ff' }}>
            <FileTextOutlined /> Music File Analysis
          </Title>
        </Col>
        {!showThankYou ? (
          <>
            <Col xs={24} md={16}>
              <Card hoverable style={{ borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
                <Dragger {...props}>
                  <p className="ant-upload-drag-icon">
                    <InboxOutlined style={{ fontSize: '48px', color: '#1890ff' }} />
                  </p>
                  <p className="ant-upload-text" style={{ fontSize: '18px', color: '#333' }}>
                    Click or drag MP3 or WAV files to this area to upload
                  </p>
                  <p className="ant-upload-hint" style={{ fontSize: '14px', color: '#666' }}>
                    Support for single file upload. The file will be analyzed for genre and emotions.
                  </p>
                </Dragger>
              </Card>
            </Col>
            <Col xs={24} md={8}>
              <Card title="Analysis Result" style={{ borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
                {loading ? (
                  <div style={{ textAlign: 'center', padding: '20px' }}>
                    <Spin size="large" />
                    <Paragraph style={{ marginTop: '10px' }}>Analyzing...</Paragraph>
                  </div>
                ) : analysisResult ? (
                  <>
                    <p><Text strong>Genre:</Text> {analysisResult.genre}</p>
                    <p><Text strong>Emotions:</Text> {analysisResult.emotions.join(', ')}</p>
                  </>
                ) : (
                  <Paragraph style={{ textAlign: 'center' }}>
                    Upload a file to see the analysis result
                  </Paragraph>
                )}
              </Card>
            </Col>
          </>
        ) : (
          <Col span={24}>
            <Result
              icon={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
              title="Thank you for using our Music Analysis Service!"
              subTitle={
                <div>
                  <Paragraph>
                    We hope you found the analysis insightful. Here's a summary of the results:
                  </Paragraph>
                  <Card style={{ marginTop: '20px', borderRadius: '8px', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}>
                    <p><Text strong>Genre:</Text> {analysisResult.genre}</p>
                    <p><Text strong>Emotions:</Text> {analysisResult.emotions.join(', ')}</p>
                  </Card>
                </div>
              }
              extra={[
                <Button type="primary" key="console" onClick={resetAnalysis}>
                  Analyze Another File
                </Button>,
              ]}
            />
          </Col>
        )}
      </Row>
    </div>
  );
};

export default FileUpload;
