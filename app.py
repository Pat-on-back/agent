# -*- coding: utf-8 -*-
"""
算法过程可视化
一个面向算法学习与教学的过程可视化智能体。
用户输入一个算法题目后，系统自动分析题意、拆解求解步骤、渲染关键状态变化，并生成详细 PDF 报告。

运行：
    streamlit run app.py
"""

from __future__ import annotations

import html
import os
import re
import time
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ==========================================================
# 1. 代码配置区：在这里填写你的接口信息
# ==========================================================
# 注意：如果上传公开 GitHub，不建议把真实 Key 写进代码。
# 课程本地演示可以直接在这里填写。
CODE_OPENAI_API_KEY = ""  # 例如："sk-xxxxxxxxxxxxxxxxxxxxxxxx"
CODE_OPENAI_BASE_URL = "https://api.siliconflow.cn/v1"
CODE_OPENAI_MODEL = "nex-agi/Nex-N2-Pro"

# 默认生成设置。为了速度，建议每次只处理一个算法，帧数控制在 4-6 帧。
DEFAULT_MAX_FRAMES = 5
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MAX_TOKENS = 3200


# ==========================================================
# 2. 页面基础设置
# ==========================================================
st.set_page_config(
    page_title="算法过程可视化",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CUSTOM_CSS = """
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 750;
    margin-bottom: 0.25rem;
}
.sub-title {
    color: #555;
    font-size: 1rem;
    margin-bottom: 1.2rem;
}
.step-box {
    border: 1px solid #e8e8e8;
    border-radius: 14px;
    padding: 14px 16px;
    background: #ffffff;
    box-shadow: 0 1px 5px rgba(0,0,0,0.04);
    margin-bottom: 12px;
}
.step-title {
    font-weight: 700;
    font-size: 1.05rem;
    margin-bottom: 6px;
}
.muted {
    color: #666;
    font-size: 0.92rem;
}
.status-line {
    padding: 6px 0;
    border-bottom: 1px dashed #eeeeee;
}
.small-tag {
    display: inline-block;
    border: 1px solid #e5e5e5;
    border-radius: 999px;
    padding: 2px 10px;
    margin-right: 6px;
    font-size: 0.85rem;
    color: #444;
    background: #fafafa;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==========================================================
# 3. 工具函数：配置、提示词、接口调用
# ==========================================================
def get_config_value(name: str, code_value: str, default: str = "") -> str:
    """优先读取代码配置，其次读取 st.secrets，再其次读取环境变量。"""
    if code_value:
        return code_value
    try:
        if name in st.secrets and st.secrets[name]:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


def get_client() -> Tuple[Optional[OpenAI], str, str, str]:
    api_key = get_config_value("OPENAI_API_KEY", CODE_OPENAI_API_KEY)
    base_url = get_config_value("OPENAI_BASE_URL", CODE_OPENAI_BASE_URL, "https://api.siliconflow.cn/v1")
    model = get_config_value("OPENAI_MODEL", CODE_OPENAI_MODEL, "nex-agi/Nex-N2-Pro")
    if not api_key:
        return None, api_key, base_url, model
    return OpenAI(api_key=api_key, base_url=base_url), api_key, base_url, model


def build_system_prompt(max_frames: int) -> str:
    """系统提示词：对外包装为算法过程可视化智能体，不在页面中暴露底层实现。"""
    return f"""
你是“算法过程可视化”智能体，面向算法学习与教学场景。
你的任务是：根据用户输入的算法题目，自动分析题意、识别唯一目标算法、拆解求解过程，并生成清晰易懂的过程可视化说明和详细报告。

重要要求：
1. 只处理一个算法。
   - 如果用户明确指定了算法，例如“动态规划”“回溯法”“Dijkstra”“快速排序”，就只处理这个算法。
   - 如果用户没有明确指定算法，就选择最适合该题的一个算法，并说明选择理由。
   - 如果用户同时列出多个算法，也只选择最先出现或最核心的一个算法进行可视化，不要展开多个算法。
2. 不要只给最终答案，必须展示求解过程。
3. 可视化内容要围绕“状态变化、关键变量、当前决策、下一步依据”来写。
4. 每个可视化帧都要短而清楚，最多输出 {max_frames} 帧。
5. 输出必须是 Markdown，不要输出 JSON。
6. 报告中不要出现“调用接口”“模型返回”“JSON”“提示词”等实现细节。
7. 如果题目数据不足，先给出合理假设，再继续演示。
8. 需要说明最终答案的确定性；如果是启发式算法，要说明结果可能是近似解。

请严格按照以下 Markdown 结构输出：

# 算法过程可视化报告

## 题目分析
- 题目类型：
- 输入信息：
- 求解目标：
- 需要观察的核心过程：

## 算法识别
- 当前算法：
- 选择理由：
- 核心思想：
- 时间复杂度：
- 空间复杂度：

## 过程可视化

### 第1帧：帧标题
**当前步骤：**
**关键变量：**
**状态变化：**
**可视化说明：**
**伪代码焦点：**

### 第2帧：帧标题
**当前步骤：**
**关键变量：**
**状态变化：**
**可视化说明：**
**伪代码焦点：**

根据题目继续输出后续帧，但不要超过 {max_frames} 帧。

如有助于理解，可以在过程可视化之后输出一个 Mermaid 图，格式如下：
```mermaid
graph TD
    A[开始] --> B[关键步骤]
```

## 结果总结
- 最终结果：
- 为什么得到该结果：
- 学习者应该掌握的核心思想：

## 详细报告
请用完整段落总结本题的算法过程、关键状态变化、可视化理解方式和学习价值。
""".strip()


def build_user_prompt(question: str, max_frames: int) -> str:
    return f"""
请对下面的算法题进行过程可视化分析。只处理一个算法，并把结果写成清晰的 Markdown 报告。

用户题目：
{question}

生成要求：
- 只识别并处理一个算法。
- 重点展示该算法的求解过程，而不是多算法对比。
- 最多生成 {max_frames} 个过程可视化帧。
- 每一帧都要包含当前步骤、关键变量、状态变化、可视化说明和伪代码焦点。
- 最后生成可以直接导出 PDF 的详细报告。
""".strip()


def stream_agent_report(question: str, max_frames: int, temperature: float, max_tokens: int):
    client, api_key, base_url, model = get_client()
    if client is None:
        raise RuntimeError("未配置 API Key。请在 app.py 顶部 CODE_OPENAI_API_KEY 中填写可用 Key。")

    messages = [
        {"role": "system", "content": build_system_prompt(max_frames)},
        {"role": "user", "content": build_user_prompt(question, max_frames)},
    ]
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content or ""
        except Exception:
            delta = ""
        if delta:
            yield delta


# ==========================================================
# 4. 文本解析与可视化渲染
# ==========================================================
def extract_section(markdown_text: str, section_title: str) -> str:
    pattern = rf"(?ms)^##\s*{re.escape(section_title)}\s*\n(.*?)(?=^##\s+|\Z)"
    m = re.search(pattern, markdown_text)
    return m.group(1).strip() if m else ""


def split_frames(markdown_text: str) -> List[Tuple[str, str]]:
    section = extract_section(markdown_text, "过程可视化") or markdown_text
    pattern = re.compile(r"(?m)^###\s*第\s*(\d+)\s*帧\s*[:：]?\s*(.*)$")
    matches = list(pattern.finditer(section))
    frames: List[Tuple[str, str]] = []
    if not matches:
        return frames
    for idx, m in enumerate(matches):
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(section)
        title = f"第{m.group(1)}帧：{m.group(2).strip() or '过程状态'}"
        body = section[start:end].strip()
        # 去掉 Mermaid 代码块，图单独渲染
        body = re.sub(r"```mermaid[\s\S]*?```", "", body).strip()
        frames.append((title, body))
    return frames


def extract_mermaid_blocks(markdown_text: str) -> List[str]:
    return [m.strip() for m in re.findall(r"```mermaid\s*([\s\S]*?)```", markdown_text)]


def render_mermaid(code: str, height: int = 360):
    safe_code = html.escape(code)
    components.html(
        f"""
        <div class="mermaid">
        {safe_code}
        </div>
        <script type="module">
          import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
          mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
        </script>
        """,
        height=height,
        scrolling=True,
    )


def render_frames(markdown_text: str):
    frames = split_frames(markdown_text)
    if not frames:
        st.info("暂未解析到标准帧结构，已在下方展示完整报告。")
        return

    st.markdown("## 过程可视化")
    st.caption("以下内容根据智能体的求解分析自动渲染，突出算法每一步的状态变化。")

    for idx, (title, body) in enumerate(frames, start=1):
        st.markdown(
            f"""
            <div class="step-box">
              <div class="step-title">{html.escape(title)}</div>
              <div class="muted">关键状态帧 {idx}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("查看本帧详细说明", expanded=True):
            st.markdown(body)


def render_summary_cards(markdown_text: str):
    analysis = extract_section(markdown_text, "题目分析")
    algo = extract_section(markdown_text, "算法识别")
    result = extract_section(markdown_text, "结果总结")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("## 题目分析")
        if analysis:
            st.markdown(analysis)
        else:
            st.info("正在整理题目分析。")
    with col2:
        st.markdown("## 算法识别")
        if algo:
            st.markdown(algo)
        else:
            st.info("正在整理算法识别结果。")

    if result:
        st.markdown("## 结果总结")
        st.markdown(result)


# ==========================================================
# 5. PDF 生成
# ==========================================================
def setup_pdf_styles() -> Dict[str, ParagraphStyle]:
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        base_font = "STSong-Light"
    except Exception:
        base_font = "Helvetica"

    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ChineseTitle",
            parent=styles["Title"],
            fontName=base_font,
            fontSize=20,
            leading=26,
            alignment=TA_CENTER,
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "ChineseH1",
            parent=styles["Heading1"],
            fontName=base_font,
            fontSize=16,
            leading=22,
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "ChineseH2",
            parent=styles["Heading2"],
            fontName=base_font,
            fontSize=13,
            leading=18,
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "ChineseBody",
            parent=styles["BodyText"],
            fontName=base_font,
            fontSize=10.5,
            leading=16,
            alignment=TA_LEFT,
            spaceAfter=5,
        ),
        "code": ParagraphStyle(
            "ChineseCode",
            parent=styles["Code"],
            fontName=base_font,
            fontSize=8.5,
            leading=12,
            leftIndent=8,
            rightIndent=8,
            backColor=colors.whitesmoke,
            borderColor=colors.lightgrey,
            borderWidth=0.5,
            borderPadding=6,
            spaceBefore=6,
            spaceAfter=6,
        ),
    }


def is_markdown_table(lines: List[str], idx: int) -> bool:
    return (
        idx + 1 < len(lines)
        and lines[idx].strip().startswith("|")
        and "|" in lines[idx].strip()[1:]
        and re.match(r"^\s*\|?\s*[-:]+", lines[idx + 1].strip()) is not None
    )


def parse_table(lines: List[str], start: int) -> Tuple[List[List[str]], int]:
    table_lines = []
    i = start
    while i < len(lines) and lines[i].strip().startswith("|"):
        # 跳过分隔行
        if not re.match(r"^\s*\|?\s*[-:| ]+\s*$", lines[i].strip()):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            table_lines.append(cells)
        i += 1
    return table_lines, i


def markdown_to_pdf_bytes(markdown_text: str, source_question: str = "") -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        title="算法过程可视化报告",
    )
    styles = setup_pdf_styles()
    story = []

    story.append(Paragraph("算法过程可视化报告", styles["title"]))
    if source_question:
        story.append(Paragraph("用户输入题目", styles["h1"]))
        story.append(Paragraph(html.escape(source_question), styles["body"]))
        story.append(Spacer(1, 8))

    lines = markdown_text.splitlines()
    i = 0
    in_code = False
    code_buffer: List[str] = []

    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            if not in_code:
                in_code = True
                code_buffer = []
            else:
                in_code = False
                story.append(Preformatted("\n".join(code_buffer), styles["code"]))
                code_buffer = []
            i += 1
            continue

        if in_code:
            code_buffer.append(line)
            i += 1
            continue

        if not stripped:
            story.append(Spacer(1, 4))
            i += 1
            continue

        if is_markdown_table(lines, i):
            table_data, next_i = parse_table(lines, i)
            if table_data:
                t = Table(table_data, repeatRows=1)
                t.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                            ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
                            ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                            ("FONTSIZE", (0, 0), (-1, -1), 8),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("LEFTPADDING", (0, 0), (-1, -1), 4),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ]
                    )
                )
                story.append(t)
                story.append(Spacer(1, 8))
            i = next_i
            continue

        # Markdown 标题转换
        if stripped.startswith("# "):
            title = stripped[2:].strip()
            # 避免重复封面标题
            if title != "算法过程可视化报告":
                story.append(Paragraph(html.escape(title), styles["h1"]))
        elif stripped.startswith("## "):
            story.append(Paragraph(html.escape(stripped[3:].strip()), styles["h1"]))
        elif stripped.startswith("### "):
            story.append(Paragraph(html.escape(stripped[4:].strip()), styles["h2"]))
        else:
            # 简单处理粗体标记
            text = html.escape(stripped)
            text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
            story.append(Paragraph(text, styles["body"]))
        i += 1

    doc.build(story)
    return buffer.getvalue()


# ==========================================================
# 6. 主页面
# ==========================================================
def main():
    st.markdown('<div class="main-title">算法过程可视化</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">输入算法题目，智能体将自动分析求解过程，生成清晰易懂的过程可视化与详细报告。</div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 运行设置")
        max_frames = st.slider("过程可视化帧数", min_value=3, max_value=8, value=DEFAULT_MAX_FRAMES, step=1)
        max_tokens = st.slider("报告详细程度", min_value=1600, max_value=6000, value=DEFAULT_MAX_TOKENS, step=400)
        temperature = st.slider("表达稳定性", min_value=0.0, max_value=0.8, value=DEFAULT_TEMPERATURE, step=0.1)
        st.markdown("---")
        st.caption("建议一次只输入一个算法题目，例如“请用动态规划可视化求解 01 背包问题”。")

    sample = (
        "请用动态规划可视化求解 01 背包问题。背包容量 15，"
        "物品 A 重量2 价值6，B 重量3 价值10，C 重量4 价值12，"
        "D 重量5 价值14，E 重量9 价值20，F 重量7 价值18。"
    )

    question = st.text_area(
        "请输入算法题目",
        value=sample,
        height=150,
        placeholder="例如：请用 Dijkstra 算法可视化求解从 A 到其他节点的最短路径……",
    )

    run_btn = st.button("生成算法过程可视化", type="primary", use_container_width=True)

    if "last_report" not in st.session_state:
        st.session_state.last_report = ""
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

    if run_btn:
        if not question.strip():
            st.warning("请先输入算法题目。")
            st.stop()

        st.session_state.last_report = ""
        st.session_state.last_question = question.strip()

        st.markdown("## 智能体实时求解过程")
        st.caption("展示智能体对题目的分析、拆解与可视化生成过程。")

        status_steps = [
            "正在解析题目描述……",
            "正在识别目标算法……",
            "正在提取关键输入信息……",
            "正在分析核心状态变量……",
            "正在拆解求解步骤……",
            "正在生成过程可视化……",
            "正在整理详细报告……",
        ]
        status_box = st.empty()
        progress = st.progress(0)
        for idx, step in enumerate(status_steps[:3], start=1):
            status_box.markdown("\n".join([f"<div class='status-line'>✅ {s}</div>" for s in status_steps[: idx - 1]] + [f"<div class='status-line'>⏳ {step}</div>"]), unsafe_allow_html=True)
            progress.progress(int(idx / len(status_steps) * 100))
            time.sleep(0.15)

        stream_box = st.empty()
        full_text = ""
        try:
            for chunk in stream_agent_report(question.strip(), max_frames=max_frames, temperature=temperature, max_tokens=max_tokens):
                full_text += chunk
                stream_box.markdown(full_text + "\n\n▌")
                # 依据文本长度推进进度，不依赖内部实现细节
                current = min(95, 35 + len(full_text) // 80)
                progress.progress(current)

            progress.progress(100)
            status_box.markdown(
                "\n".join([f"<div class='status-line'>✅ {s}</div>" for s in status_steps]),
                unsafe_allow_html=True,
            )
            stream_box.markdown(full_text)
            st.session_state.last_report = full_text
            st.success("算法过程可视化生成完成。")
        except Exception as e:
            message = str(e)
            st.error("生成失败。请检查代码配置区中的 API Key、接口地址和模型名称。")
            if "insufficient" in message.lower() or "balance" in message.lower() or "30001" in message:
                st.warning("当前账号可能没有可用额度。请更换可用 Key 或处理账户额度后重新运行。")
            with st.expander("查看错误详情"):
                st.code(message)
            st.stop()

    report = st.session_state.last_report
    if report:
        st.markdown("---")
        render_summary_cards(report)
        st.markdown("---")
        render_frames(report)

        mermaid_blocks = extract_mermaid_blocks(report)
        if mermaid_blocks:
            st.markdown("## 结构图示")
            for block in mermaid_blocks:
                render_mermaid(block)

        st.markdown("---")
        st.markdown("## 详细报告")
        with st.expander("查看完整报告内容", expanded=True):
            st.markdown(report)

        pdf_bytes = markdown_to_pdf_bytes(report, source_question=st.session_state.last_question)
        st.download_button(
            label="下载详细 PDF 报告",
            data=pdf_bytes,
            file_name="算法过程可视化报告.pdf",
            mime="application/pdf",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
