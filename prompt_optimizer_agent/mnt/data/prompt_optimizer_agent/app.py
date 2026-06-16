import json
import re
from datetime import datetime
from io import BytesIO
from typing import Dict, Any, List

import streamlit as st
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics

APP_TITLE = "通用问题优化智能体"
APP_SUBTITLE = "基于公开可访问大模型 API 的提示词工程与轻量工具调用演示"

SYSTEM_PROMPT = """
你是“通用问题优化智能体”。你的目标不是直接替用户随便回答，而是把一个模糊、零散、缺少约束的问题优化成可执行、可评估、可复用的高质量问题/提示词，并给出求解过程。

工作流程：
1. 识别用户原始问题的任务类型、目标、隐含约束和缺失信息。
2. 用MECE方式列出问题缺陷：目标不清、背景不足、约束缺失、输出格式缺失、评价标准缺失、风险点。
3. 在不改变用户原意的前提下，重写为更好的“优化后问题/提示词”。
4. 给出解决该问题的执行步骤，要求步骤具体、可操作。
5. 给出可直接复制使用的最终提示词。
6. 给出自检清单，说明优化前后提升在哪里。

输出必须是严格JSON，不要Markdown代码块。JSON字段如下：
{
  "task_type": "任务类型",
  "user_goal": "用户目标",
  "missing_information": ["缺失信息1", "缺失信息2"],
  "problem_diagnosis": ["缺陷1", "缺陷2"],
  "optimized_question": "优化后的问题/提示词",
  "solution_process": ["步骤1", "步骤2", "步骤3"],
  "final_answer_or_plan": "针对优化后问题给出的具体答案或执行方案",
  "evaluation_checklist": ["检查项1", "检查项2"],
  "risk_control": ["风险控制1", "风险控制2"],
  "one_sentence_summary": "一句话总结"
}
""".strip()


def safe_get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def classify_task(text: str) -> str:
    rules = [
        ("论文/报告写作", ["论文", "报告", "综述", "摘要", "引言", "markdown"]),
        ("代码/程序开发", ["代码", "python", "bug", "运行", "报错", "api", "部署", "前端", "后端"]),
        ("数据分析/实验设计", ["数据", "实验", "指标", "图表", "模型", "训练", "测试"]),
        ("图像/可视化生成", ["图片", "图", "黑白图", "架构图", "示意图", "生成"]),
        ("学习答疑/概念解释", ["解释", "证明", "复杂度", "原理", "为什么"]),
    ]
    for label, keywords in rules:
        if any(k.lower() in text.lower() for k in keywords):
            return label
    return "通用问题求解"


def score_prompt(text: str) -> Dict[str, Any]:
    text_lower = text.lower()
    metrics = []
    checks = [
        ("目标清晰", any(k in text for k in ["目标", "解决", "完成", "生成", "输出", "要求"]), 20),
        ("背景充分", len(text) >= 80 or any(k in text for k in ["背景", "场景", "已有", "基于", "用于"]), 20),
        ("约束明确", any(k in text for k in ["不能", "禁止", "必须", "不要", "限制", "约束"]), 20),
        ("输出格式明确", any(k in text_lower for k in ["pdf", "mp4", "markdown", "表格", "json", "文档", "格式"]), 20),
        ("评价标准明确", any(k in text for k in ["评分", "评价", "验收", "标准", "检查", "精确", "可访问"]), 20),
    ]
    total = 0
    for name, passed, pts in checks:
        score = pts if passed else 0
        total += score
        metrics.append({"维度": name, "得分": score, "满分": pts, "判断": "满足" if passed else "需补充"})
    return {"total": total, "metrics": metrics}


def extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            return json.loads(match.group(0))
        raise ValueError("模型输出不是可解析JSON，请重试或更换模型。")


def call_llm(api_key: str, base_url: str, model: str, user_payload: str) -> Dict[str, Any]:
    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_payload},
        ],
    )
    content = completion.choices[0].message.content or "{}"
    return extract_json(content)


def demo_llm(original: str, task_type: str, before_score: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "task_type": task_type,
        "user_goal": "把原始问题改写为清晰、可执行、可评价的高质量提示词，并给出求解路径。",
        "missing_information": ["具体应用场景", "目标受众", "输出长度", "评价标准", "禁止事项或边界条件"],
        "problem_diagnosis": [
            "原始问题的目标方向存在，但可执行步骤不足。",
            "缺少输出格式和验收标准，导致模型回答容易发散。",
            "缺少上下文背景，难以保证结果贴合实际需求。",
        ],
        "optimized_question": f"请作为专业问题优化智能体，分析以下原始问题：{original}\n要求：1. 保留原意；2. 补全背景、目标、约束、输出格式和验收标准；3. 给出优化后提示词；4. 给出执行步骤；5. 输出结构化结果。",
        "solution_process": [
            "调用任务分类工具，判断问题属于哪类任务。",
            "调用提示词评分工具，从目标、背景、约束、格式、评价五个维度评分。",
            "使用大模型根据评分结果进行重写和补全。",
            "再次评分，并输出可下载PDF文档。",
        ],
        "final_answer_or_plan": "建议采用“任务背景-目标-输入-约束-输出格式-评价标准”的六段式提示词模板，将模糊问题转化为可执行任务。",
        "evaluation_checklist": ["是否保留原意", "是否明确输出格式", "是否有约束", "是否可评价", "是否可直接复制使用"],
        "risk_control": ["不编造用户未提供的关键事实", "对不确定信息标注为待补充", "避免泄露API Key或个人隐私"],
        "one_sentence_summary": "该智能体通过分类、评分、重写和文档生成，把模糊问题优化为可提交、可复用的高质量提示词。",
    }


def register_fonts() -> str:
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        return "STSong-Light"
    except Exception:
        return "Helvetica"


def para(text: Any, style: ParagraphStyle) -> Paragraph:
    s = "" if text is None else str(text)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = s.replace("\n", "<br/>")
    return Paragraph(s, style)


def make_pdf(original: str, task_type: str, before_score: Dict[str, Any], after_score: Dict[str, Any], result: Dict[str, Any], tool_log: List[str]) -> bytes:
    font_name = register_fonts()
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("CNTitle", parent=styles["Title"], fontName=font_name, fontSize=18, leading=24, alignment=1)
    h1 = ParagraphStyle("CNH1", parent=styles["Heading1"], fontName=font_name, fontSize=14, leading=20, spaceBefore=12)
    body = ParagraphStyle("CNBody", parent=styles["BodyText"], fontName=font_name, fontSize=10.5, leading=16, spaceAfter=6)
    small = ParagraphStyle("CNSmall", parent=styles["BodyText"], fontName=font_name, fontSize=9, leading=13)

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm)
    story = []
    story.append(para("通用问题优化智能体求解报告", title_style))
    story.append(para(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", small))
    story.append(Spacer(1, 0.3*cm))

    story.append(para("1. 项目说明", h1))
    story.append(para("本报告由通用问题优化智能体自动生成。智能体基于公开可访问的大模型API，通过提示词工程和轻量工具调用，对用户原始问题进行分类、评分、诊断、优化和求解。", body))

    story.append(para("2. 原始问题", h1))
    story.append(para(original, body))
    story.append(para(f"任务分类工具结果：{task_type}", body))

    story.append(para("3. 轻量工具调用记录", h1))
    for item in tool_log:
        story.append(para("- " + item, body))

    story.append(para("4. 提示词质量评分", h1))
    table_data = [["阶段", "总分", "目标", "背景", "约束", "格式", "评价"]]
    for stage, score in [("优化前", before_score), ("优化后", after_score)]:
        vals = [m["得分"] for m in score["metrics"]]
        table_data.append([stage, str(score["total"]), *[str(v) for v in vals]])
    table = Table(table_data, colWidths=[2.2*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.6*cm, 1.6*cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), font_name),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(table)

    fields = [
        ("5. 用户目标", result.get("user_goal")),
        ("6. 缺失信息", result.get("missing_information")),
        ("7. 问题诊断", result.get("problem_diagnosis")),
        ("8. 优化后问题/提示词", result.get("optimized_question")),
        ("9. 求解过程", result.get("solution_process")),
        ("10. 最终答案或执行方案", result.get("final_answer_or_plan")),
        ("11. 评价清单", result.get("evaluation_checklist")),
        ("12. 风险控制", result.get("risk_control")),
        ("13. 一句话总结", result.get("one_sentence_summary")),
    ]
    for heading, content in fields:
        story.append(para(heading, h1))
        if isinstance(content, list):
            for i, x in enumerate(content, 1):
                story.append(para(f"{i}. {x}", body))
        else:
            story.append(para(content, body))

    doc.build(story)
    return buf.getvalue()


def build_user_payload(original: str, domain: str, audience: str, output_type: str, constraints: str, before_score: Dict[str, Any], task_type: str) -> str:
    return json.dumps({
        "原始问题": original,
        "应用领域": domain,
        "目标受众": audience,
        "期望输出": output_type,
        "额外约束": constraints,
        "工具调用结果": {
            "任务分类": task_type,
            "优化前评分": before_score,
        },
        "要求": "请严格按照系统提示词输出JSON。"
    }, ensure_ascii=False)


st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

with st.sidebar:
    st.header("模型配置")
    preset = st.selectbox("API预设", ["OpenAI", "通义千问/Qwen", "豆包/火山方舟", "自定义OpenAI兼容接口"])
    preset_map = {
        "OpenAI": ("https://api.openai.com/v1", "gpt-4.1-mini"),
        "通义千问/Qwen": ("https://dashscope.aliyuncs.com/compatible-mode/v1", "qwen-plus"),
        "豆包/火山方舟": ("https://ark.cn-beijing.volces.com/api/v3", "请填写你的模型或推理接入点ID"),
        "自定义OpenAI兼容接口": (safe_get_secret("OPENAI_BASE_URL", "https://api.openai.com/v1"), safe_get_secret("MODEL_NAME", "gpt-4.1-mini")),
    }
    default_base, default_model = preset_map[preset]
    api_key = safe_get_secret("OPENAI_API_KEY", "") or st.text_input("API Key（部署时建议放入Secrets）", type="password")
    base_url = st.text_input("Base URL", value=safe_get_secret("OPENAI_BASE_URL", default_base))
    model = st.text_input("Model", value=safe_get_secret("MODEL_NAME", default_model))
    demo_mode = st.checkbox("无API时使用演示模式", value=False)

st.subheader("输入需要优化的原始问题")
original = st.text_area(
    "原始问题",
    value="帮我做一个关于机器学习课程选题的报告，要有创新点，还要能答辩。",
    height=120,
)

col1, col2, col3 = st.columns(3)
with col1:
    domain = st.text_input("应用领域", value="学习/科研/项目申报")
with col2:
    audience = st.text_input("目标受众", value="学生、教师或项目评审")
with col3:
    output_type = st.text_input("期望输出", value="优化后提示词、求解步骤、PDF报告")
constraints = st.text_area("额外约束", value="保留原意；不要编造事实；输出要可直接复制；给出评价标准。", height=80)

if st.button("开始优化并生成报告", type="primary"):
    if not original.strip():
        st.error("请先输入原始问题。")
        st.stop()

    task_type = classify_task(original)
    before_score = score_prompt(original)
    tool_log = [
        f"调用 classify_task：识别任务类型为“{task_type}”。",
        f"调用 score_prompt：优化前总分为 {before_score['total']}/100。",
    ]

    payload = build_user_payload(original, domain, audience, output_type, constraints, before_score, task_type)
    with st.spinner("智能体正在分析、优化并组织结果..."):
        try:
            if demo_mode or not api_key:
                result = demo_llm(original, task_type, before_score)
            else:
                result = call_llm(api_key, base_url, model, payload)
        except Exception as e:
            st.error(f"调用模型失败：{e}")
            st.info("可以检查API Key、Base URL和Model是否正确，或临时勾选演示模式查看流程。")
            st.stop()

    optimized_text = result.get("optimized_question", "")
    after_score = score_prompt(optimized_text)
    tool_log.append(f"调用 score_prompt：优化后总分为 {after_score['total']}/100。")
    tool_log.append("调用 make_pdf：生成智能体求解过程PDF。")

    st.success("优化完成")
    m1, m2, m3 = st.columns(3)
    m1.metric("任务类型", task_type)
    m2.metric("优化前评分", f"{before_score['total']}/100")
    m3.metric("优化后评分", f"{after_score['total']}/100")

    st.subheader("优化后问题/提示词")
    st.code(optimized_text, language="text")

    st.subheader("求解过程")
    for i, step in enumerate(result.get("solution_process", []), 1):
        st.write(f"{i}. {step}")

    st.subheader("最终答案或执行方案")
    st.write(result.get("final_answer_or_plan", ""))

    st.subheader("工具调用记录")
    for item in tool_log:
        st.write("- " + item)

    pdf_bytes = make_pdf(original, task_type, before_score, after_score, result, tool_log)
    st.download_button(
        "下载PDF求解报告",
        data=pdf_bytes,
        file_name="prompt_optimizer_agent_report.pdf",
        mime="application/pdf",
    )

    with st.expander("查看JSON结果"):
        st.json(result)
else:
    st.info("点击按钮后，智能体会完成：任务分类 -> 质量评分 -> 大模型优化 -> 再评分 -> PDF报告生成。")
