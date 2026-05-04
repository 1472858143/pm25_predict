import React, { useEffect, useState } from "react";
import { Alert, Empty, Layout, Spin } from "antd";
import { Header } from "./components/Header";
import { MetricsTable } from "./components/MetricsTable";
import { fetchWindows } from "./api/client";
import type { WindowInfo } from "./types/api";
import { buildSearchString, readSelectionFromUrl } from "./utils/url";

const { Content } = Layout;

const App: React.FC = () => {
  const [windows, setWindows] = useState<WindowInfo[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedWindow, setSelectedWindow] = useState<string | null>(null);
  const [selectedStart, setSelectedStart] = useState<string | null>(null);

  useEffect(() => {
    fetchWindows()
      .then((res) => {
        setWindows(res.windows);
        const fromUrl = readSelectionFromUrl(window.location.search);
        const firstWithStart = res.windows.find((w) => w.starts.length > 0);
        const initialWindow =
          (fromUrl.window && res.windows.find((w) => w.name === fromUrl.window)?.name) ??
          firstWithStart?.name ??
          res.windows[0]?.name ??
          null;
        const target = res.windows.find((w) => w.name === initialWindow);
        const initialStart =
          (fromUrl.start && target?.starts.includes(fromUrl.start) ? fromUrl.start : null) ??
          target?.starts[0] ??
          null;
        setSelectedWindow(initialWindow);
        setSelectedStart(initialStart);
      })
      .catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    const search = buildSearchString({ window: selectedWindow, start: selectedStart });
    const newUrl = `${window.location.pathname}${search}`;
    window.history.replaceState(null, "", newUrl);
  }, [selectedWindow, selectedStart]);

  const handleWindowChange = (name: string) => {
    setSelectedWindow(name);
    const target = windows?.find((w) => w.name === name);
    setSelectedStart(target?.starts[0] ?? null);
  };

  if (error) {
    return <Alert type="error" message={`加载窗口列表失败: ${error}`} showIcon style={{ margin: 24 }} />;
  }

  if (!windows) {
    return (
      <div style={{ padding: 48, textAlign: "center" }}>
        <Spin tip="加载中..." />
      </div>
    );
  }

  if (windows.length === 0) {
    return (
      <Empty
        description="未发现任何预测输出，请先运行 prepare_data / train_model / predict_model"
        style={{ marginTop: 96 }}
      />
    );
  }

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Header
        windows={windows}
        selectedWindow={selectedWindow}
        selectedStart={selectedStart}
        onChangeWindow={handleWindowChange}
        onChangeStart={setSelectedStart}
      />
      <Content style={{ padding: 24 }}>
        {selectedStart === null ? (
          <Empty description="该时间窗口暂无预测结果" />
        ) : (
          <>
              <MetricsTable window={selectedWindow!} start={selectedStart} />
            </>
        )}
      </Content>
    </Layout>
  );
};

export default App;
