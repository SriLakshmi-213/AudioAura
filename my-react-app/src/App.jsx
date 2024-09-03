import React, { useState } from 'react';
import { Layout, Menu, Typography, Avatar, Breadcrumb } from 'antd';
import {
  HomeOutlined,
  UploadOutlined,
  UserOutlined,
} from '@ant-design/icons';

import Home from './pages/Home';
import FileUpload from './pages/FileUpload';

const { Header, Content, Footer } = Layout;
const { Title } = Typography;

const items = [
  { key: 'Home', icon: <HomeOutlined />, label: 'Home' },
  { key: 'FileUpload', icon: <UploadOutlined />, label: 'File Upload' },
];

const App = () => {
  const [currentPage, setCurrentPage] = useState('Home');

  const renderPage = () => {
    switch (currentPage) {
      case 'Home': return <Home />;
      case 'FileUpload': return <FileUpload />;
      default: return <Home />;
    }
  };

  return (
    <Layout className="layout">
      <Header style={{ display: 'flex', alignItems: 'center' }}>
        <div className="logo" style={{ marginRight: '24px' }}>
          <Title level={3} style={{ color: 'white', margin: 0 }}>Music Upload</Title>
        </div>
        <Menu
          theme="dark"
          mode="horizontal"
          defaultSelectedKeys={['Home']}
          items={items}
          onClick={({ key }) => setCurrentPage(key)}
          style={{ flex: 1, minWidth: 0 }}
        />
        <Avatar icon={<UserOutlined />} style={{ marginLeft: '24px' }} />
      </Header>
      <Content style={{ padding: '0 50px' }}>
        <Breadcrumb style={{ margin: '16px 0' }}>
          <Breadcrumb.Item>Home</Breadcrumb.Item>
          <Breadcrumb.Item>{currentPage}</Breadcrumb.Item>
        </Breadcrumb>
        <div className="site-layout-content" style={{ background: '#fff', padding: 24, minHeight: 280 }}>
          {renderPage()}
        </div>
      </Content>
      <Footer style={{ textAlign: 'center' }}>Music Upload App ©2023 Created by Your Name</Footer>
    </Layout>
  );
};

export default App;
