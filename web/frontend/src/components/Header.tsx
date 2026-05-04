import React from "react";
import { Layout, Select, Space, Typography } from "antd";
import type { WindowInfo } from "../types/api";
import { formatStart, formatWindow } from "../utils/format";

const { Header: AntHeader } = Layout;
const { Title } = Typography;

export interface HeaderProps {
  windows: WindowInfo[];
  selectedWindow: string | null;
  selectedStart: string | null;
  onChangeWindow: (name: string) => void;
  onChangeStart: (start: string) => void;
}

export const Header: React.FC<HeaderProps> = ({
  windows,
  selectedWindow,
  selectedStart,
  onChangeWindow,
  onChangeStart,
}) => {
  const currentWindow = windows.find((w) => w.name === selectedWindow);
  const startOptions = currentWindow?.starts ?? [];

  return (
    <AntHeader style={{ background: "#fff", padding: "0 24px", height: "auto", lineHeight: "1.5" }}>
      <div style={{ padding: "16px 0" }}>
        <Title level={3} style={{ margin: 0 }}>
          PM2.5 多模型预测对比仪表盘
        </Title>
        <Space size="large" style={{ marginTop: 12 }}>
          <Space>
            <span>时间窗口:</span>
            <Select
              data-testid="window-select"
              style={{ minWidth: 280 }}
              value={selectedWindow ?? undefined}
              onChange={onChangeWindow}
              options={windows.map((w) => ({
                value: w.name,
                label: formatWindow(w.name),
              }))}
            />
          </Space>
          <Space>
            <span>预测起点:</span>
            <Select
              data-testid="start-select"
              style={{ minWidth: 240 }}
              value={selectedStart ?? undefined}
              onChange={onChangeStart}
              disabled={startOptions.length === 0}
              placeholder={startOptions.length === 0 ? "该窗口暂无预测结果" : undefined}
              options={startOptions.map((s) => ({
                value: s,
                label: formatStart(s),
              }))}
            />
          </Space>
        </Space>
      </div>
    </AntHeader>
  );
};
