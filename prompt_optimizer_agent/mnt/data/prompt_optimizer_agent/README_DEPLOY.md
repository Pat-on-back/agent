# 通用问题优化智能体部署与提交说明

## 1. 项目目标

本项目基于公开可访问的大模型API或网页端能力，使用提示词工程和轻量工具调用，构建一个“通用问题优化智能体”。它可以把用户的模糊问题转化为清晰、可执行、可评价的高质量提示词，并自动输出PDF求解报告。

## 2. 推荐部署路线

推荐使用 Streamlit Community Cloud 部署为公开网页。部署完成后会获得形如 `https://你的应用名.streamlit.app` 的访问链接。

## 3. 文件说明

- `app.py`：智能体网页应用主程序。
- `requirements.txt`：Python依赖。
- `AGENT_PROMPT.md`：核心系统提示词。
- `README_DEPLOY.md`：部署与提交说明。

## 4. 本地运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 5. API配置

该项目使用OpenAI兼容接口，支持：

- OpenAI：`https://api.openai.com/v1`
- 通义千问/Qwen：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- 豆包/火山方舟：`https://ark.cn-beijing.volces.com/api/v3`
- 其他OpenAI兼容接口

建议在Streamlit Cloud的Secrets中配置：

```toml
OPENAI_API_KEY = "你的API Key"
OPENAI_BASE_URL = "https://api.openai.com/v1"
MODEL_NAME = "gpt-4.1-mini"
```

如使用通义千问或豆包，只需替换Base URL和Model。

## 6. 录制MP4视频建议

录屏内容建议控制在3-5分钟：

1. 展示公开访问链接。
2. 输入一个模糊问题。
3. 点击“开始优化并生成报告”。
4. 展示任务分类、优化前后评分、优化后提示词和求解过程。
5. 点击下载PDF报告。
6. 打开PDF，证明文档由智能体自动生成。

## 7. 提交材料

最终提交：

- 可访问链接：Streamlit公开网页链接。
- MP4视频：录制智能体求解过程。
- PDF文档：应用中下载的自动生成报告。

## 8. 验收标准

- 智能体可通过公开链接访问。
- 使用公开可访问的大模型API，不使用内部未公开模型。
- 智能体能解决一个明确问题。
- 求解过程有录屏MP4。
- 智能体能自动生成详细PDF文档。
