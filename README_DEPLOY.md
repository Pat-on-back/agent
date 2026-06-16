# 算法过程可视化 - 硅基流动 API 大模型版

## 1. 项目简介

项目名称：算法过程可视化

本项目基于硅基流动 API 调用公开大模型，让大模型完成算法题目的自动识别、求解过程拆解、逐帧可视化设计、教学讲解和 PDF 报告生成。

本项目强调：**算法过程主要由大模型生成，不在本地写固定算法求解器。**

本地程序只做：

1. 调用硅基流动 API；
2. 渲染大模型返回的 JSON、表格、数组和 Mermaid 图；
3. 将大模型生成的求解过程、可视化帧和教学报告自动排版为详细 PDF。

注意：PDF 是智能体求解后自动生成的材料，不需要人工另写文档。

## 2. 文件结构

```text
算法过程可视化_LLM版/
├── app.py
├── requirements.txt
├── AGENT_PROMPT.md
├── README_DEPLOY.md
├── MP4_RECORDING_SCRIPT.md
├── SUBMISSION_CHECKLIST.md
└── SAMPLE_INPUTS.md
```

## 3. 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 4. Streamlit Secrets 配置

在 Streamlit Community Cloud 的 Secrets 中加入：

```toml
SILICONFLOW_API_KEY = "你的硅基流动 API Key"
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
SILICONFLOW_MODEL = "Qwen/Qwen2.5-72B-Instruct"
```

模型名以硅基流动控制台中实际可调用的模型为准。

## 5. 部署步骤

1. 将本文件夹上传到 GitHub 仓库。
2. 打开 Streamlit Community Cloud。
3. 新建应用，选择仓库和 `app.py`。
4. 配置 Secrets。
5. 点击 Deploy。
6. 获得公开访问链接。

## 6. 推荐录屏输入

```text
请可视化求解 01 背包问题。背包容量 15，物品 A 重量2 价值6，B 重量3 价值10，C 重量4 价值12，D 重量5 价值14，E 重量9 价值20，F 重量7 价值18。请用回溯法、动态规划、分支限界、遗传算法、模拟退火进行对比。
```

## 7. 项目特点

- 不需要用户选择固定问题类型；
- 用户自然语言输入算法题目；
- 大模型自动识别问题；
- 大模型生成算法过程可视化帧；
- 支持 Mermaid 决策树、DP 表格、数组状态、变量面板；
- 求解完成后自动生成详细 PDF 报告，包含题目识别、逐算法步骤、关键帧、变量变化、算法对比和完整教学报告；
- 满足“公开 API + 提示词工程 + 轻量工具调用”的要求。

## 8. 注意事项

由于算法过程由大模型生成，对于复杂大规模问题可能存在误差。建议录屏和提交时使用小规模、可人工核验的示例题，以突出教学可视化效果。
