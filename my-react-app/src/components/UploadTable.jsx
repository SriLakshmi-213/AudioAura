import React from 'react';
import { Table, Upload, Button } from 'antd';
import { UploadOutlined } from '@ant-design/icons';

const UploadTable = () => {
  const columns = [
    {
      title: 'File Name',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: 'Upload',
      key: 'upload',
      render: () => (
        <Upload>
          <Button icon={<UploadOutlined />}>Click to Upload</Button>
        </Upload>
      ),
    },
  ];

  const data = [
    { key: 1, name: 'Song 1' },
    { key: 2, name: 'Song 2' },
    { key: 3, name: 'Song 3' },
  ];

  return <Table columns={columns} dataSource={data} />;
};

export default UploadTable;
