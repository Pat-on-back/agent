# -*- coding: utf-8 -*-
"""
"""

import os
import re
import json
import html
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Tuple

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

APP_TITLE = "算法过程可视化"
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "nex-agi/Nex-N2-Pro"

DETECT_SYSTEM_PROMPT = """
你是“算法过程可视化”智能体的算法识别模块。
你的任务是从用户输入中识别：题目类型、用户要求使用的唯一算法、输入数据摘要、适合的可视化方式。

关键规则：
1. 只允许返回一个 algorithm_name。
2. 如果用户明确写了某一种算法，例如“动态规划”“回溯法”“Dijkstra”“快速排序”，就只使用这个算法。
3. 如果用户写了多个算法，只选择最先被明确提出的一个算法，并在 note 中说明“本系统单次只处理一个算法”。
4. 如果用户没有写算法名，则选择最适合该题目的一个算法。
5. 只输出严格 JSON，不要代码块，不要解释。

JSON 结构：
{
  "task_title": "题目标题",
  "problem_type": "问题类型",
  "algorithm_name": "本次唯一处理的算法",
  "input_summary": "输入数据摘要",
  "visualization_style": "建议可视化方式，如状态表/搜索树/数组条形图/图遍历/路径图",
  "learning_goal": "学习目标",
  "note": "补充说明"
}
"""


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, None)
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(name, default)


def get_config(names: List[str], default: str = "") -> str:
    for name in names:
        value = get_secret(name, "")
        if value:
            return value
    return default


def create_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def is_quota_or_permission_error(err: Exception) -> bool:
    text = str(err).lower()
    keywords = ["insufficient", "balance", "quota", "403", "30001", "permission", "forbidden", "unauthorized", "invalid api key"]
    return any(k in text for k in keywords)


def extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        raise ValueError("模型没有返回内容")
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def detect_single_algorithm(client: OpenAI, model: str, user_problem: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DETECT_SYSTEM_PROMPT},
            {"role": "user", "content": user_problem},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    content = response.choices[0].message.content or ""
    return extract_json_object(content)


def build_single_algorithm_prompt(user_problem: str, plan: Dict[str, Any], max_frames: int) -> str:
    algorithm = plan.get("algorithm_name", "指定算法")
    return f"""
你是“算法过程可视化”智能体。请根据用户题目，用唯一指定算法生成求解分析和可视化内容。

极其重要：
1. 本次只处理一种算法：{algorithm}。
2. 不要生成其他算法，不要做多算法对比。
3. 算法求解分析由你生成，本地程序不会运行算法求解器。
4. 输出必须是公开可展示的求解过程，不要输出隐藏思维链。
5. 生成内容要适合网页清晰渲染和 PDF 自动报告。
6. 最多生成 {max_frames} 个关键可视化帧。
7. 每一帧都要能让初学者看懂“当前状态发生了什么变化”。
8. 如果需要表格，请用 Markdown 表格；如果需要树或流程，请用 Mermaid 代码块。
9. 对最终结果进行自检：变量是否一致，结果是否可信，是否存在近似性。

用户题目：
{user_problem}

识别结果：
- 题目类型：{plan.get('problem_type', '')}
- 唯一算法：{algorithm}
- 输入摘要：{plan.get('input_summary', '')}
- 建议可视化方式：{plan.get('visualization_style', '')}
- 学习目标：{plan.get('learning_goal', '')}

请严格按照下面 Markdown 结构输出：

# {plan.get('task_title', '算法过程可视化')}

## 1. 题目理解
说明输入、目标和约束。

## 2. 本次使用的唯一算法：{algorithm}
说明为什么本题使用该算法，以及该算法的核心思想。

## 3. 可视化求解过程

### 帧1：...
- 当前动作：...
- 关键变量：...
- 当前状态：...
- 状态变化：...
- 教学解释：...

### 帧2：...
- 当前动作：...
- 关键变量：...
- 当前状态：...
- 状态变化：...
- 教学解释：...

继续到最多 {max_frames} 帧。

## 4. 变量变化表
用 Markdown 表格总结关键变量如何变化。

## 5. 可视化图示
如果适合，请给出一个 Mermaid 图；如果不适合，就用表格或列表表达。

## 6. 最终结果
给出最终答案，并说明它是如何从过程得到的。

## 7. 学习总结
总结学习者应该掌握的算法核心思想。

## 8. 智能体自检
说明结果是否确定、是否可能有近似性、哪些地方需要人工核对。
"""


def stream_text(client: OpenAI, model: str, messages: List[Dict[str, str]], placeholder, max_tokens: int = 2200, temperature: float = 0.2) -> str:
    buffer = ""
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        delta = ""
        try:
            delta = chunk.choices[0].delta.content or ""
        except Exception:
            pass
        if not delta:
            continue
        buffer += delta
        placeholder.markdown(buffer + "\n\n▌")
    placeholder.markdown(buffer)
    return buffer.strip()


def stream_single_algorithm_analysis(client: OpenAI, model: str, user_problem: str, plan: Dict[str, Any], max_frames: int, placeholder) -> str:
    messages = [
        {"role": "system", "content": "你是算法过程可视化智能体。只处理用户指定的一种算法，用公开可展示 Markdown 输出，不输出隐藏思维链。"},
        {"role": "user", "content": build_single_algorithm_prompt(user_problem, plan, max_frames)},
    ]
    return stream_text(client, model, messages, placeholder, max_tokens=2600, temperature=0.2)


def split_visual_frames(markdown_text: str) -> List[Tuple[str, str]]:
    """从模型 Markdown 中提取 ### 帧... 小节，用于单独卡片化渲染。"""
    pattern = re.compile(r"(?m)^###\s*(帧\s*\d+[:：]?.*)$")
    matches = list(pattern.finditer(markdown_text))
    frames: List[Tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown_text)
        title = m.group(1).strip()
        body = markdown_text[start:end].strip()
        # 避免把后续一级/二级标题全部吞进去
        next_h2 = re.search(r"(?m)^##\s+", body)
        if next_h2:
            body = body[:next_h2.start()].strip()
        frames.append((title, body))
    return frames


def extract_mermaid_blocks(markdown_text: str) -> List[str]:
    pattern = re.compile(r"```mermaid\s*(.*?)```", re.DOTALL | re.IGNORECASE)
    return [m.group(1).strip() for m in pattern.finditer(markdown_text)]


def render_mermaid_block(code: str, height: int = 380) -> None:
    safe_code = html.escape(code.strip())
    mermaid_html = f"""
    <div style="background:white; padding:12px; border-radius:10px; border:1px solid #ddd; overflow:auto;">
      <pre class="mermaid">{safe_code}</pre>
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
    </script>
    """
    components.html(mermaid_html, height=height, scrolling=True)


def render_markdown_without_mermaid(markdown_text: str) -> None:
    cleaned = re.sub(r"```mermaid\s*.*?```", "", markdown_text, flags=re.DOTALL | re.IGNORECASE)
    st.markdown(cleaned)


def render_visual_result(analysis_markdown: str, plan: Dict[str, Any]) -> None:
    st.markdown("## 清晰可视化渲染")
    st.caption("下面不是重新求解，而是根据大模型已经返回的算法分析内容进行页面化、卡片化和图示化渲染。")

    frames = split_visual_frames(analysis_markdown)
    if frames:
        st.markdown("### 逐帧过程卡片")
        for idx, (title, body) in enumerate(frames, start=1):
            with st.container(border=True):
                st.markdown(f"#### {idx}. {title}")
                st.markdown(body)
    else:
        st.info("没有识别到“### 帧1”格式的小节，将直接展示完整 Markdown。")

    mermaid_blocks = extract_mermaid_blocks(analysis_markdown)
    if mermaid_blocks:
        st.markdown("### 图示渲染")
        for code in mermaid_blocks:
            render_mermaid_block(code)

    st.markdown("### 完整分析文本")
    render_markdown_without_mermaid(analysis_markdown)


def safe_para(text: Any) -> str:
    return html.escape(str(text)).replace("\n", "<br/>")


def add_markdown_to_story(story: List[Any], markdown_text: str, styles: Dict[str, Any]) -> None:
    in_code = False
    code_buffer: List[str] = []
    for raw in str(markdown_text).split("\n"):
        line = raw.rstrip()
        if line.strip().startswith("```"):
            if in_code:
                story.append(Paragraph(safe_para("\n".join(code_buffer)), styles["ChineseCode"]))
                code_buffer = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_buffer.append(line)
            continue
        if not line.strip():
            story.append(Spacer(1, 0.08 * cm))
            continue
        stripped = line.strip()
        if stripped.startswith("###"):
            story.append(Paragraph(safe_para(stripped.lstrip("# ")), styles["ChineseSubHeading"]))
        elif stripped.startswith("##"):
            story.append(Paragraph(safe_para(stripped.lstrip("# ")), styles["ChineseHeading"]))
        elif stripped.startswith("#"):
            story.append(Paragraph(safe_para(stripped.lstrip("# ")), styles["ChineseHeading"]))
        elif stripped.startswith("|"):
            story.append(Paragraph(safe_para(stripped), styles["ChineseCode"]))
        else:
            story.append(Paragraph(safe_para(stripped), styles["ChineseBody"]))
    if code_buffer:
        story.append(Paragraph(safe_para("\n".join(code_buffer)), styles["ChineseCode"]))


def make_pdf(user_problem: str, plan: Dict[str, Any], analysis_markdown: str) -> str:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    base_styles = getSampleStyleSheet()
    styles = {
        "Title": ParagraphStyle("ChineseTitle", parent=base_styles["Title"], fontName="STSong-Light", fontSize=22, leading=28, alignment=1, spaceAfter=16),
        "ChineseHeading": ParagraphStyle("ChineseHeading", parent=base_styles["Heading2"], fontName="STSong-Light", fontSize=15, leading=20, spaceBefore=10, spaceAfter=6),
        "ChineseSubHeading": ParagraphStyle("ChineseSubHeading", parent=base_styles["Heading3"], fontName="STSong-Light", fontSize=12.5, leading=17, spaceBefore=7, spaceAfter=4),
        "ChineseBody": ParagraphStyle("ChineseBody", parent=base_styles["BodyText"], fontName="STSong-Light", fontSize=10.5, leading=16, spaceAfter=4),
        "ChineseCode": ParagraphStyle("ChineseCode", parent=base_styles["BodyText"], fontName="STSong-Light", fontSize=8.2, leading=11.5, leftIndent=8, rightIndent=8, backColor=colors.whitesmoke, borderColor=colors.lightgrey, borderWidth=0.3, borderPadding=4, spaceAfter=4),
    }
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.close()
    doc = SimpleDocTemplate(tmp.name, pagesize=A4, rightMargin=1.6*cm, leftMargin=1.6*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
    story: List[Any] = []
    story.append(Paragraph("算法过程可视化", styles["Title"]))
    story.append(Paragraph("智能体自动生成的单算法求解过程 PDF", styles["ChineseHeading"]))
    story.append(Paragraph("生成时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["ChineseBody"]))
    story.append(Paragraph("说明：本报告中的算法识别、求解分析、变量变化和可视化帧均由 OpenAI 兼容接口的大模型生成；本地程序不实现算法求解器，只负责展示与 PDF 排版。", styles["ChineseBody"]))

    story.append(Paragraph("一、用户输入题目", styles["ChineseHeading"]))
    story.append(Paragraph(safe_para(user_problem), styles["ChineseBody"]))

    story.append(Paragraph("二、算法识别结果", styles["ChineseHeading"]))
    rows = [
        ["项目", "内容"],
        ["题目标题", plan.get("task_title", "")],
        ["问题类型", plan.get("problem_type", "")],
        ["本次唯一算法", plan.get("algorithm_name", "")],
        ["输入摘要", plan.get("input_summary", "")],
        ["可视化方式", plan.get("visualization_style", "")],
        ["学习目标", plan.get("learning_goal", "")],
    ]
    table = Table([[Paragraph(safe_para(c), styles["ChineseBody"]) for c in r] for r in rows], colWidths=[3.4*cm, 12.6*cm], repeatRows=1)
    table.setStyle(TableStyle([("FONTNAME", (0,0), (-1,-1), "STSong-Light"), ("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("GRID", (0,0), (-1,-1), 0.3, colors.grey), ("VALIGN", (0,0), (-1,-1), "TOP")]))
    story.append(table)

    story.append(PageBreak())
    story.append(Paragraph("三、大模型生成的算法求解分析与可视化过程", styles["ChineseHeading"]))
    add_markdown_to_story(story, analysis_markdown, styles)

    story.append(Paragraph("四、系统说明", styles["ChineseHeading"]))
    story.append(Paragraph("本系统采用单算法处理逻辑：用户输入某一种算法，智能体只围绕这一种算法进行求解分析与可视化渲染，不再展开多个算法，避免等待过久和内容发散。", styles["ChineseBody"]))
    doc.build(story)
    return tmp.name


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("输入一个算法题目和一种算法，大模型生成求解分析，网页根据分析结果进行可视化渲染并自动导出 PDF。")

    with st.sidebar:
        st.header("OpenAI 兼容接口配置")
        api_key = st.text_input("OPENAI_API_KEY", value=get_config(["OPENAI_API_KEY", "SILICONFLOW_API_KEY"], ""), type="password")
        base_url = st.text_input("OPENAI_BASE_URL", value=get_config(["OPENAI_BASE_URL", "SILICONFLOW_BASE_URL"], DEFAULT_BASE_URL))
        model = st.text_input("OPENAI_MODEL", value=get_config(["OPENAI_MODEL", "SILICONFLOW_MODEL"], DEFAULT_MODEL))
        max_frames = st.slider("最多关键帧", 2, 8, 4, help="只处理一种算法。帧数越少，生成越快；录屏建议 4 帧。")
        st.markdown("---")
        st.info("算法执行过程可视化")

    sample = "请用动态规划可视化求解 01 背包问题。背包容量 15，物品 A 重量2 价值6，B 重量3 价值10，C 重量4 价值12，D 重量5 价值14，E 重量9 价值20，F 重量7 价值18。"
    user_problem = st.text_area("请输入算法题目", value=sample, height=150)

    c1, c2 = st.columns([1, 3])
    with c1:
        run = st.button("生成算法过程可视化", type="primary", use_container_width=True)
    with c2:
        st.write("可视化算法过程")

    if run:
        if not api_key:
            st.error("请先配置 OPENAI_API_KEY。")
            return
        if not user_problem.strip():
            st.error("请输入算法题目。")
            return

        client = create_client(api_key, base_url)
        for key in ["plan", "analysis_markdown", "pdf_path", "user_problem"]:
            st.session_state.pop(key, None)

        try:
            st.markdown("## 1. 快速识别唯一算法")
            with st.spinner("正在识别题目类型和本次唯一处理的算法……"):
                plan = detect_single_algorithm(client, model, user_problem)
            st.success(f"识别完成：本次只处理「{plan.get('algorithm_name', '指定算法')}」。")
            st.json(plan)

            st.markdown("## 2. 大模型实时生成算法求解分析")
            st.caption("这里实时显示公开版求解分析，不是隐藏思维链；适合录屏展示。")
            placeholder = st.empty()
            analysis_markdown = stream_single_algorithm_analysis(client, model, user_problem, plan, max_frames, placeholder)

            st.session_state["plan"] = plan
            st.session_state["analysis_markdown"] = analysis_markdown
            st.session_state["user_problem"] = user_problem
            st.success("大模型求解分析生成完成，下面开始根据该分析进行可视化渲染。")
        except Exception as exc:
            if is_quota_or_permission_error(exc):
                st.error("接口调用失败：请检查 API Key、账户余额、模型权限或模型名是否可用。")
                st.code(str(exc), language="text")
            else:
                st.error("生成失败。建议减少关键帧数量，或更换响应更快的模型。")
                st.code(str(exc), language="text")
            return

    plan = st.session_state.get("plan")
    analysis_markdown = st.session_state.get("analysis_markdown", "")
    if not plan or not analysis_markdown:
        return

    st.markdown("---")
    st.markdown("## 结果总览")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("题目类型", plan.get("problem_type", "未识别"))
    col_b.metric("唯一算法", plan.get("algorithm_name", "未识别"))
    col_c.metric("可视化方式", plan.get("visualization_style", "自动"))
    st.write(plan.get("input_summary", ""))

    render_visual_result(analysis_markdown, plan)

    st.markdown("## 智能体自动输出 PDF")
    st.info("PDF 由大模型返回的单算法求解分析、变量变化、逐帧过程和可视化说明自动排版得到。")
    pdf_path = st.session_state.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        pdf_path = make_pdf(
            user_problem=st.session_state.get("user_problem", ""),
            plan=plan,
            analysis_markdown=analysis_markdown,
        )
        st.session_state["pdf_path"] = pdf_path
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="下载智能体自动生成的详细 PDF 报告",
            data=f.read(),
            file_name="算法过程可视化_单算法报告.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
