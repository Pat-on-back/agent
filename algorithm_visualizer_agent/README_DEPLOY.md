# 算法过程可视化智能体部署说明

## 一、项目简介

本项目实现一个“算法过程可视化智能体”。用户输入一个算法学习问题，例如“用动画解释快速排序”或“可视化 Dijkstra 最短路径过程”，系统会自动完成：

1. 识别算法类型；
2. 生成算法执行轨迹；
3. 逐帧显示可视化过程；
4. 调用公开可访问的大模型 API 生成教学讲解；
5. 自动导出 PDF 求解报告。

项目满足课程要求：

- 智能体可公开访问；
- 使用公开可访问的大模型 API 或网页端，不使用内部未公开模型；
- 使用提示词工程 + 轻量工具调用；
- 能解决一个具体问题；
- 求解过程可以录制为 MP4；
- 求解过程能自动输出 PDF 文档。

## 二、推荐部署方式：Streamlit Community Cloud

### 1. 准备 GitHub 仓库

把本文件夹中的所有文件上传到一个 GitHub 仓库：

```text
algorithm_visualizer_agent/
├── app.py
├── requirements.txt
├── AGENT_PROMPT.md
├── README_DEPLOY.md
└── SUBMISSION_CHECKLIST.md
```

### 2. 部署到 Streamlit

进入 Streamlit Community Cloud，选择你的 GitHub 仓库，将主文件设置为：

```text
app.py
```

部署完成后，系统会生成公开访问链接，一般形式为：

```text
https://你的应用名.streamlit.app/
```

### 3. 配置 API Key

进入 Streamlit 应用的 Secrets 设置，加入以下内容。

如果使用 OpenAI：

```toml
OPENAI_API_KEY = "你的 API Key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "gpt-4o-mini"
```

如果使用通义千问 DashScope OpenAI 兼容模式：

```toml
OPENAI_API_KEY = "你的 DashScope API Key"
OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus"
```

如果使用豆包/火山方舟 OpenAI 兼容模式：

```toml
OPENAI_API_KEY = "你的火山方舟 API Key"
OPENAI_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
MODEL_NAME = "你的模型 ID 或推理接入点 ID"
```

## 三、演示问题建议

推荐演示快速排序，因为它有明显的 pivot、比较、交换和递归分治过程，适合录屏展示。

输入问题：

```text
请用可视化方式解释快速排序如何把数组 [8, 3, 5, 1, 9, 6, 2, 7] 排序。
```

输入数据：

```text
[8, 3, 5, 1, 9, 6, 2, 7]
```

点击：

```text
开始可视化求解并生成报告
```

录屏时展示：

1. 公开访问链接；
2. 输入问题和数据；
3. 智能体识别算法；
4. 滑动步骤条展示可视化过程；
5. 展示智能体讲解和状态变量；
6. 展示轻量工具调用记录；
7. 下载并打开 PDF 求解报告。

## 四、提交材料

最终提交：

1. 可访问链接：Streamlit 公开链接；
2. MP4 视频：录制完整求解过程；
3. PDF 文档：由智能体在网页中自动生成的报告。
