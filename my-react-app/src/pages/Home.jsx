import React from 'react';
import { Typography, Card } from 'antd';
import { CustomerServiceOutlined } from '@ant-design/icons';

const { Title, Paragraph } = Typography;

const Home = () => {
  return (
    <div style={{
      padding: '60px 40px',
      maxWidth: '900px',
      margin: '0 auto',
      textAlign: 'center',
      background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
      borderRadius: '15px',
      boxShadow: '0 10px 20px rgba(0,0,0,0.1)'
    }}>
      <Title level={1} style={{ 
        color: '#1890ff', 
        marginBottom: '30px',
        fontSize: '3.5em',
        textShadow: '2px 2px 4px rgba(0,0,0,0.1)'
      }}>
        <CustomerServiceOutlined style={{ fontSize: '1.2em', marginRight: '10px' }} /> 
        Unveiling the Feels
      </Title>
      <Paragraph style={{ 
        fontSize: '1.5em', 
        marginBottom: '40px',
        color: '#333',
        fontWeight: '300'
      }}>
        Genre-Specific Emotion Recognition in Music
      </Paragraph>
      
      <Card style={{ 
        backgroundColor: 'rgba(255,255,255,0.8)', 
        borderRadius: '12px', 
        marginBottom: '40px',
        boxShadow: '0 5px 15px rgba(0,0,0,0.08)'
      }}>
        <Paragraph style={{ fontSize: '1.1em', lineHeight: '1.6' }}>
          Explore the emotional landscape of music through our advanced genre-specific emotion recognition system. 
          Upload your favorite tracks and discover the intricate emotional patterns within different musical genres.
        </Paragraph>
      </Card>

      <Paragraph style={{ fontSize: '1.2em', color: '#1890ff' }}>
        To get started, navigate to the File Upload page using the menu above.
      </Paragraph>
    </div>
  );
};

export default Home;
