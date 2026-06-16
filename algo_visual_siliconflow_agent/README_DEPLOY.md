# 基于硅基流动 API 的通用算法过程可视化智能体部署说明

## 1. 项目简介

本项目是一个可以公开访问的 Streamlit Web 智能体。它支持多类算法过程可视化，包括：

- 01 背包：回溯法、动态规划、分支限界、遗传算法、模拟退火；
- N 皇后：回溯搜索过程；
- 最长公共子序列 LCS：动态规划状态表；
- 最短路径：Dijkstra 距离更新过程；
- 旅行商 TSP：遗传算法和模拟退火的迭代搜索过程。

本地代码生成算法轨迹，硅基流动 API 生成自然语言讲解和报告。

## 2. 部署到 Streamlit Community Cloud

1. 将整个文件夹上传到 GitHub 仓库。
2. 打开 Streamlit Community Cloud。
3. 选择 GitHub 仓库。
4. 主文件路径填写：`app.py`。
5. 在应用的 Secrets 中配置硅基流动 API：

```toml
SILICONFLOW_API_KEY = "你的硅基流动 API Key"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_MODEL = "Qwen/Qwen2.5-72B-Instruct"
```

如果你的模型名称不同，请在硅基流动控制台的模型广场中复制可用模型名称。

## 3. 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

也可以在终端中使用环境变量：

```bash
export SILICONFLOW_API_KEY="你的 Key"
export SILICONFLOW_BASE_URL="https://api.siliconflow.cn/v1"
export SILICONFLOW_MODEL="Qwen/Qwen2.5-72B-Instruct"
streamlit run app.py
```

## 4. 推荐录屏流程

1. 打开公开访问链接；
2. 展示侧边栏中的硅基流动 API 配置；
3. 选择“01 背包”，勾选五种算法；
4. 点击生成可视化；
5. 展示动态规划表、回溯搜索、分支限界剪枝、遗传算法收敛、模拟退火降温；
6. 切换到 LCS、N 皇后或最短路径，说明它不是只支持 01 背包；
7. 点击生成 PDF 报告并下载；
8. 最后展示提交材料：公开链接、MP4 视频、PDF 文档。
