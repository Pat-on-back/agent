# -*- coding: utf-8 -*-
"""
算法过程可视化

说明：
1. 页面面向用户包装为“算法过程可视化智能体”，不在界面中展示底层模型或接口细节。
2. 一次只处理用户输入中的一个算法：用户指定哪个算法，就可视化哪个算法；未指定时自动选择最合适的一个。
3. 智能体返回算法分析与可视化帧，前端负责清晰渲染和 PDF 导出。
"""

import os
import re
import json
import html
import time
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional

import streamlit as st
import pandas as pd
import altair as alt
from openai import OpenAI

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Preformatted,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# ============================================================
# 1. 代码配置区：在这里填写你的 API 信息
# ============================================================
# 课程本地演示可以直接填在这里。
# 如果上传公开 GitHub，建议不要写真实 Key，改用环境变量或 Streamlit Secrets。
CODE_OPENAI_API_KEY = "sk-xpdideuepwpxalyhosclmczxzfbydlqelgarxlthigpimjfd"  # 例如：sk-xxxxxxxxxxxxxxxxxxxxxxxx
CODE_OPENAI_BASE_URL = "https://api.siliconflow.cn/v1"
CODE_OPENAI_MODEL = "nex-agi/Nex-N2-Pro"

# 可视化控制
MAX_VISUAL_FRAMES = 6
MAX_OUTPUT_TOKENS = 4096
TEMPERATURE = 0.2

# ============================================================
# 2. 页面基础设置
# ============================================================
st.set_page_config(
    page_title="算法过程可视化",
    page_icon="🧩",
    layout="wide",
)

CUSTOM_CSS = """
<style>
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
.main-title {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.25rem;
}
.sub-title {
    font-size: 1.02rem;
    color: #555;
    margin-bottom: 1.5rem;
}
.status-box {
    border: 1px solid #e6e6e6;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    background: #fafafa;
    margin-bottom: 0.6rem;
}
.frame-card {
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 1rem 1.1rem;
    margin: 0.8rem 0 1.1rem 0;
    background: #ffffff;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.frame-title {
    font-weight: 750;
    font-size: 1.12rem;
    margin-bottom: 0.25rem;
}
.frame-step {
    color: #4b5563;
    font-size: 0.95rem;
    margin-bottom: 0.7rem;
}
.metric-chip {
    display: inline-block;
    border: 1px solid #e5e7eb;
    border-radius: 999px;
    padding: 0.22rem 0.6rem;
    margin: 0.15rem 0.15rem 0.15rem 0;
    background: #f9fafb;
    font-size: 0.9rem;
}
.small-muted { color: #6b7280; font-size: 0.9rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<div class="main-title">算法过程可视化</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">输入算法题目，智能体将自动分析求解过程，生成清晰易懂的过程可视化与详细报告。</div>',
    unsafe_allow_html=True,
)

# ============================================================
# 3. 工具函数：配置、解析、错误处理
# ============================================================

def get_config_value(code_value: str, env_key: str, default: str = "") -> str:
    """优先读取代码配置，其次环境变量，最后 Streamlit Secrets。"""
    if code_value and code_value.strip():
        return code_value.strip()
    if os.getenv(env_key):
        return os.getenv(env_key, "").strip()
    try:
        if env_key in st.secrets:
            return str(st.secrets[env_key]).strip()
    except Exception:
        pass
    return default


def get_client() -> Optional[OpenAI]:
    api_key = get_config_value(CODE_OPENAI_API_KEY, "OPENAI_API_KEY")
    base_url = get_config_value(CODE_OPENAI_BASE_URL, "OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=base_url)


def get_model_name() -> str:
    return get_config_value(CODE_OPENAI_MODEL, "OPENAI_MODEL", "nex-agi/Nex-N2-Pro")


def user_friendly_error(e: Exception) -> str:
    raw = str(e)
    if "balance is insufficient" in raw or "30001" in raw:
        return "智能体服务当前额度不足，请更换可用配置后重试。"
    if "401" in raw or "Unauthorized" in raw or "invalid api key" in raw.lower():
        return "智能体服务认证失败，请检查代码中的 Key 是否正确。"
    if "403" in raw:
        return "智能体服务暂不可用，可能是额度、权限或访问策略限制。"
    if "timeout" in raw.lower():
        return "生成时间过长，请减少题目规模或稍后重试。"
    return "生成过程中出现异常，请检查配置或稍后重试。"


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """从返回文本中尽量提取 JSON。"""
    if not text:
        return None
    text = text.strip()

    # 1. 直接解析
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2. 解析 ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass

    # 3. 从第一个 { 到最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            # 简单修复中文引号和尾随逗号
            candidate2 = candidate.replace("，", ",")
            candidate2 = re.sub(r",\s*([}\]])", r"\1", candidate2)
            try:
                return json.loads(candidate2)
            except Exception:
                return None
    return None


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, ensure_ascii=False, indent=2)
        except Exception:
            return str(x)
    return str(x)


def normalize_result(data: Optional[Dict[str, Any]], raw_text: str, user_question: str) -> Dict[str, Any]:
    """保证页面一定有可展示内容。"""
    if not data:
        # 如果解析失败，至少把原始分析拆成文本帧，保证页面和 PDF 不为空。
        return {
            "title": "算法过程可视化报告",
            "problem_type": "待确认的算法问题",
            "algorithm": "智能体识别结果",
            "input_summary": user_question,
            "core_idea": "系统已生成分析文本，但结构化可视化数据解析不完整。下面以过程卡片方式展示。",
            "complexity": "请参考分析内容。",
            "visualization_frames": split_text_to_frames(raw_text),
            "final_answer": "请参考下方过程分析。",
            "report_markdown": raw_text or "本次未获得有效分析内容。",
        }

    frames = data.get("visualization_frames") or data.get("frames") or []
    if not isinstance(frames, list):
        frames = []
    if not frames:
        frames = split_text_to_frames(data.get("report_markdown", raw_text))

    return {
        "title": data.get("title") or "算法过程可视化报告",
        "problem_type": data.get("problem_type") or data.get("detected_problem_type") or "算法问题",
        "algorithm": data.get("algorithm") or data.get("algorithm_name") or "当前算法",
        "input_summary": data.get("input_summary") or user_question,
        "core_idea": data.get("core_idea") or data.get("idea") or "见过程说明。",
        "complexity": data.get("complexity") or "见过程说明。",
        "visualization_frames": frames[:MAX_VISUAL_FRAMES],
        "final_answer": data.get("final_answer") or data.get("result") or "见结果总结。",
        "report_markdown": data.get("report_markdown") or build_markdown_from_data(data, user_question),
    }


def split_text_to_frames(text: str) -> List[Dict[str, Any]]:
    """把普通文本切成若干过程卡片。"""
    text = (text or "").strip()
    if not text:
        return [
            {
                "frame_id": 1,
                "title": "未获得有效过程",
                "step": "请重新生成",
                "variables": {},
                "state_change": "暂无",
                "explanation": "当前没有可展示内容。",
                "visual_type": "text",
                "visual_data": {"content": "请检查配置或重新输入题目。"},
            }
        ]

    parts = re.split(r"\n(?=#+\s|第\s*\d+\s*[步帧]|步骤\s*\d+|Frame\s*\d+)", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) < 2:
        # 按段落切
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        parts = paras[:MAX_VISUAL_FRAMES]
    frames = []
    for i, p in enumerate(parts[:MAX_VISUAL_FRAMES], start=1):
        title = p.splitlines()[0].strip("# ：:")[:60] if p.splitlines() else f"过程帧 {i}"
        frames.append(
            {
                "frame_id": i,
                "title": title or f"过程帧 {i}",
                "step": f"第 {i} 个关键步骤",
                "variables": {},
                "state_change": "见解释说明",
                "explanation": p,
                "visual_type": "text",
                "visual_data": {"content": p},
            }
        )
    return frames


def build_markdown_from_data(data: Dict[str, Any], user_question: str) -> str:
    frames = data.get("visualization_frames") or data.get("frames") or []
    lines = [
        "# 算法过程可视化报告",
        "",
        "## 1. 用户输入题目",
        user_question,
        "",
        "## 2. 题目分析",
        safe_str(data.get("input_summary") or data.get("problem_type") or ""),
        "",
        "## 3. 算法思想",
        safe_str(data.get("core_idea") or data.get("idea") or ""),
        "",
        "## 4. 过程可视化",
    ]
    for idx, frame in enumerate(frames, 1):
        lines += [
            f"### 4.{idx} {frame.get('title', f'过程帧 {idx}')}",
            f"- 当前步骤：{safe_str(frame.get('step') or frame.get('step_description'))}",
            f"- 关键变量：{safe_str(frame.get('variables'))}",
            f"- 状态变化：{safe_str(frame.get('state_change') or frame.get('state'))}",
            "",
            safe_str(frame.get("explanation")),
            "",
        ]
    lines += [
        "## 5. 结果总结",
        safe_str(data.get("final_answer") or data.get("result") or ""),
    ]
    return "\n".join(lines)

# ============================================================
# 4. 智能体提示词
# ============================================================
SYSTEM_PROMPT = """
你是“算法过程可视化”智能体。你的输出将被网页直接渲染为算法过程可视化，并被整理成 PDF 报告。

任务要求：
1. 一次只处理一个算法。
2. 如果用户明确指定算法，只处理用户指定的算法。
3. 如果用户没有指定算法，自动选择最适合题目的一个算法。
4. 不要只给最终答案，必须把求解过程拆成 4 到 6 个可视化帧。
5. 每一帧都要包含关键变量、状态变化、解释说明和可视化数据。
6. 可视化数据必须尽量具体，不要只写“见说明”。
7. 报告要详细，能直接写入 PDF。
8. 输出必须是 JSON，不要输出 Markdown 代码围栏，不要输出额外解释。

JSON 格式如下：
{
  "title": "报告标题",
  "problem_type": "问题类型",
  "algorithm": "当前算法名称，只能有一个",
  "input_summary": "对输入数据的整理",
  "core_idea": "算法核心思想",
  "complexity": "时间复杂度与空间复杂度",
  "visualization_frames": [
    {
      "frame_id": 1,
      "title": "帧标题",
      "step": "当前步骤",
      "variables": {"变量名": "变量值"},
      "state_change": "本帧状态变化",
      "explanation": "通俗解释，说明为什么这样做",
      "visual_type": "array/table/tree/graph/list/flow/text 中的一种",
      "visual_data": {
        "content": "可视化说明",
        "array": [8,3,5,1],
        "highlight_indices": [0,2],
        "table": [[0,0,0],[0,6,6]],
        "columns": ["容量0", "容量1", "容量2"],
        "rows": ["物品0", "物品1"],
        "mermaid": "graph TD; A-->B;",
        "dot": "digraph G { A -> B }",
        "items": ["队列: A,B", "已访问: A"]
      },
      "pseudocode_focus": "当前对应的伪代码片段"
    }
  ],
  "final_answer": "最终结果",
  "report_markdown": "# 算法过程可视化报告\n\n详细报告正文，包含题目、算法思想、每一帧过程、最终结果、学习总结。"
}

重要：
- 如果是动态规划，visual_type 优先使用 table，并给出 table、rows、columns。
- 如果是排序，visual_type 优先使用 array，并给出 array 和 highlight_indices。
- 如果是回溯、递归、分支限界，visual_type 优先使用 tree 或 flow，并给出 mermaid 或 dot。
- 如果是图算法，visual_type 优先使用 graph，并给出 dot 或 mermaid。
- 如果是栈、队列、种群、候选集合，visual_type 使用 list。
- 每一帧的 visual_data 必须存在。
"""


def build_user_prompt(user_question: str) -> str:
    return f"""
请根据下面的算法题目生成“单算法”的过程可视化内容。

用户题目：
{user_question}

生成要求：
1. 只处理题目中指定的一个算法；不要额外展开其他算法。
2. 如果题目没有指定算法，你只选择一个最适合的算法。
3. 输出 4 到 6 个可视化帧。
4. 每个可视化帧必须有 visual_type 和 visual_data，方便页面渲染。
5. report_markdown 要详细，能够作为 PDF 报告正文。
6. 只返回 JSON。
"""

# ============================================================
# 5. PDF 生成
# ============================================================

def register_pdf_font() -> str:
    """使用 ReportLab 内置 CID 中文字体，避免打包字体文件。"""
    font_name = "STSong-Light"
    try:
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    except Exception:
        pass
    return font_name


def markdown_to_plain_blocks(md: str) -> List[Any]:
    """把 Markdown 简化为 ReportLab 元素。"""
    font = register_pdf_font()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CNTitle", parent=styles["Title"], fontName=font, fontSize=20, leading=26, spaceAfter=16))
    styles.add(ParagraphStyle(name="CNHeading1", parent=styles["Heading1"], fontName=font, fontSize=16, leading=22, spaceBefore=12, spaceAfter=8))
    styles.add(ParagraphStyle(name="CNHeading2", parent=styles["Heading2"], fontName=font, fontSize=13, leading=18, spaceBefore=10, spaceAfter=6))
    styles.add(ParagraphStyle(name="CNBody", parent=styles["BodyText"], fontName=font, fontSize=10.5, leading=16, spaceAfter=6))
    styles.add(ParagraphStyle(name="CNCode", parent=styles["Code"], fontName=font, fontSize=8.5, leading=12))

    elements: List[Any] = []
    in_code = False
    code_lines: List[str] = []

    def flush_code():
        nonlocal code_lines
        if code_lines:
            elements.append(Preformatted("\n".join(code_lines), styles["CNCode"]))
            elements.append(Spacer(1, 0.15 * cm))
            code_lines = []

    for raw_line in md.splitlines():
        line = raw_line.rstrip()
        if line.strip().startswith("```"):
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue

        if not line.strip():
            elements.append(Spacer(1, 0.08 * cm))
            continue

        escaped = html.escape(line.strip())
        if line.startswith("# "):
            elements.append(Paragraph(html.escape(line[2:].strip()), styles["CNTitle"]))
        elif line.startswith("## "):
            elements.append(Paragraph(html.escape(line[3:].strip()), styles["CNHeading1"]))
        elif line.startswith("### "):
            elements.append(Paragraph(html.escape(line[4:].strip()), styles["CNHeading2"]))
        elif line.startswith("- "):
            elements.append(Paragraph("• " + html.escape(line[2:].strip()), styles["CNBody"]))
        else:
            elements.append(Paragraph(escaped, styles["CNBody"]))
    flush_code()
    return elements


def make_pdf(result: Dict[str, Any], user_question: str) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
    )
    font = register_pdf_font()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CNTitle2", parent=styles["Title"], fontName=font, fontSize=22, leading=28, alignment=1, spaceAfter=18))
    styles.add(ParagraphStyle(name="CNBody2", parent=styles["BodyText"], fontName=font, fontSize=10.5, leading=16, spaceAfter=8))
    styles.add(ParagraphStyle(name="CNH1B", parent=styles["Heading1"], fontName=font, fontSize=16, leading=22, spaceBefore=12, spaceAfter=8))
    styles.add(ParagraphStyle(name="CNH2B", parent=styles["Heading2"], fontName=font, fontSize=13, leading=18, spaceBefore=10, spaceAfter=6))

    story: List[Any] = []
    story.append(Paragraph("算法过程可视化报告", styles["CNTitle2"]))
    story.append(Paragraph("本报告由算法过程可视化智能体根据用户输入题目自动整理生成。", styles["CNBody2"]))
    story.append(Spacer(1, 0.3 * cm))

    summary_data = [
        ["项目", "内容"],
        ["题目类型", safe_str(result.get("problem_type"))],
        ["当前算法", safe_str(result.get("algorithm"))],
        ["核心思想", safe_str(result.get("core_idea"))[:180]],
        ["复杂度", safe_str(result.get("complexity"))],
    ]
    table = Table(summary_data, colWidths=[3.0 * cm, 12.0 * cm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("LEADING", (0, 0), (-1, -1), 13),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.4 * cm))

    story.append(Paragraph("一、用户输入题目", styles["CNH1B"]))
    story.append(Paragraph(html.escape(user_question), styles["CNBody2"]))

    story.append(Paragraph("二、题目分析与算法识别", styles["CNH1B"]))
    story.append(Paragraph("题目分析：" + html.escape(safe_str(result.get("input_summary"))), styles["CNBody2"]))
    story.append(Paragraph("当前算法：" + html.escape(safe_str(result.get("algorithm"))), styles["CNBody2"]))
    story.append(Paragraph("核心思想：" + html.escape(safe_str(result.get("core_idea"))), styles["CNBody2"]))
    story.append(Paragraph("复杂度：" + html.escape(safe_str(result.get("complexity"))), styles["CNBody2"]))

    story.append(Paragraph("三、过程可视化逐帧说明", styles["CNH1B"]))
    frames = result.get("visualization_frames") or []
    for idx, frame in enumerate(frames, 1):
        story.append(Paragraph(f"第 {idx} 帧：{html.escape(safe_str(frame.get('title')))}", styles["CNH2B"]))
        story.append(Paragraph("当前步骤：" + html.escape(safe_str(frame.get("step"))), styles["CNBody2"]))
        story.append(Paragraph("关键变量：" + html.escape(safe_str(frame.get("variables"))), styles["CNBody2"]))
        story.append(Paragraph("状态变化：" + html.escape(safe_str(frame.get("state_change"))), styles["CNBody2"]))
        story.append(Paragraph("解释说明：" + html.escape(safe_str(frame.get("explanation"))), styles["CNBody2"]))
        if frame.get("pseudocode_focus"):
            story.append(Paragraph("伪代码焦点：" + html.escape(safe_str(frame.get("pseudocode_focus"))), styles["CNBody2"]))
        story.append(Spacer(1, 0.15 * cm))

    story.append(Paragraph("四、结果总结", styles["CNH1B"]))
    story.append(Paragraph(html.escape(safe_str(result.get("final_answer"))), styles["CNBody2"]))

    story.append(PageBreak())
    story.extend(markdown_to_plain_blocks(result.get("report_markdown") or ""))

    doc.build(story)
    return buffer.getvalue()

# ============================================================
# 6. 可视化渲染函数
# ============================================================

def render_mermaid(code: str, height: int = 320):
    if not code:
        return
    escaped_code = html.escape(code)
    components_html = f"""
    <div class="mermaid">{escaped_code}</div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    </script>
    """
    st.components.v1.html(components_html, height=height, scrolling=True)


def render_array_visual(visual_data: Dict[str, Any]):
    arr = visual_data.get("array") or visual_data.get("values") or []
    if not isinstance(arr, list) or not arr:
        st.info(safe_str(visual_data.get("content") or "数组状态未提供。"))
        return
    highlight = set(visual_data.get("highlight_indices") or visual_data.get("highlight") or [])
    df = pd.DataFrame(
        {
            "位置": [str(i) for i in range(len(arr))],
            "数值": arr,
            "状态": ["当前关注" if i in highlight else "普通" for i in range(len(arr))],
        }
    )
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("位置:N", title="位置"),
            y=alt.Y("数值:Q", title="数值"),
            tooltip=["位置", "数值", "状态"],
        )
        .properties(height=230)
    )
    st.altair_chart(chart, use_container_width=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_table_visual(visual_data: Dict[str, Any]):
    table = visual_data.get("table") or visual_data.get("matrix") or []
    if not isinstance(table, list) or not table:
        st.info(safe_str(visual_data.get("content") or "状态表未提供。"))
        return
    columns = visual_data.get("columns")
    rows = visual_data.get("rows")
    try:
        df = pd.DataFrame(table)
        if columns and len(columns) == df.shape[1]:
            df.columns = columns
        if rows and len(rows) == df.shape[0]:
            df.insert(0, "状态", rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        st.write(table)


def render_list_visual(visual_data: Dict[str, Any]):
    items = visual_data.get("items") or visual_data.get("list") or []
    if isinstance(items, dict):
        items = [f"{k}: {v}" for k, v in items.items()]
    if not items:
        st.info(safe_str(visual_data.get("content") or "列表状态未提供。"))
        return
    cols = st.columns(min(4, max(1, len(items))))
    for idx, item in enumerate(items):
        with cols[idx % len(cols)]:
            st.markdown(f"<div class='status-box'>{html.escape(safe_str(item))}</div>", unsafe_allow_html=True)


def render_graph_visual(visual_data: Dict[str, Any]):
    dot = visual_data.get("dot")
    mermaid = visual_data.get("mermaid")
    if dot:
        try:
            st.graphviz_chart(dot, use_container_width=True)
            return
        except Exception:
            st.code(dot)
            return
    if mermaid:
        render_mermaid(mermaid)
        return
    st.info(safe_str(visual_data.get("content") or "图结构未提供。"))


def render_text_visual(visual_data: Dict[str, Any]):
    content = visual_data.get("content") if isinstance(visual_data, dict) else visual_data
    st.info(safe_str(content or "本帧以文字方式展示。"))


def render_visual(frame: Dict[str, Any]):
    visual_type = str(frame.get("visual_type") or frame.get("type") or "text").lower()
    visual_data = frame.get("visual_data") or {}
    if not isinstance(visual_data, dict):
        visual_data = {"content": visual_data}

    if visual_type in ["array", "bar", "bars", "sort"]:
        render_array_visual(visual_data)
    elif visual_type in ["table", "dp_table", "matrix", "distance_table"]:
        render_table_visual(visual_data)
    elif visual_type in ["tree", "graph", "flow", "mermaid", "search_tree"]:
        render_graph_visual(visual_data)
    elif visual_type in ["list", "queue", "stack", "population", "set"]:
        render_list_visual(visual_data)
    else:
        # 兜底：如果有 table/array/graph 字段，自动判断
        if visual_data.get("table") or visual_data.get("matrix"):
            render_table_visual(visual_data)
        elif visual_data.get("array") or visual_data.get("values"):
            render_array_visual(visual_data)
        elif visual_data.get("dot") or visual_data.get("mermaid"):
            render_graph_visual(visual_data)
        elif visual_data.get("items"):
            render_list_visual(visual_data)
        else:
            render_text_visual(visual_data)


def render_frame(frame: Dict[str, Any], index: int):
    title = frame.get("title") or f"过程帧 {index}"
    step = frame.get("step") or frame.get("step_description") or "当前步骤"
    explanation = frame.get("explanation") or ""
    variables = frame.get("variables") or {}
    state_change = frame.get("state_change") or frame.get("state") or ""
    pseudocode = frame.get("pseudocode_focus") or ""

    st.markdown("<div class='frame-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='frame-title'>第 {index} 帧：{html.escape(safe_str(title))}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='frame-step'>{html.escape(safe_str(step))}</div>", unsafe_allow_html=True)

    if variables:
        st.markdown("**关键变量**")
        if isinstance(variables, dict):
            chips = "".join(
                [f"<span class='metric-chip'>{html.escape(str(k))}: {html.escape(safe_str(v))}</span>" for k, v in variables.items()]
            )
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.write(variables)

    if state_change:
        st.markdown("**状态变化**")
        st.write(state_change)

    st.markdown("**可视化内容**")
    render_visual(frame)

    if explanation:
        st.markdown("**解释说明**")
        st.write(explanation)

    if pseudocode:
        with st.expander("查看当前伪代码焦点"):
            st.code(pseudocode, language="text")

    st.markdown("</div>", unsafe_allow_html=True)


def render_result(result: Dict[str, Any], user_question: str):
    st.markdown("## 题目分析")
    c1, c2, c3 = st.columns(3)
    c1.metric("题目类型", safe_str(result.get("problem_type"))[:30])
    c2.metric("当前算法", safe_str(result.get("algorithm"))[:30])
    c3.metric("可视化帧数", len(result.get("visualization_frames") or []))

    st.markdown("**输入整理**")
    st.write(result.get("input_summary") or user_question)

    st.markdown("## 算法识别")
    st.markdown("**核心思想**")
    st.write(result.get("core_idea"))
    st.markdown("**复杂度**")
    st.write(result.get("complexity"))

    st.markdown("## 过程可视化")
    frames = result.get("visualization_frames") or []
    if not frames:
        st.warning("未获得可视化帧，请重新生成。")
    else:
        for idx, frame in enumerate(frames, 1):
            render_frame(frame, idx)

    st.markdown("## 结果总结")
    st.success(result.get("final_answer") or "已完成算法过程可视化。")

    st.markdown("## 详细报告")
    with st.expander("查看报告正文", expanded=False):
        st.markdown(result.get("report_markdown") or "暂无报告正文。")

    pdf_bytes = make_pdf(result, user_question)
    st.download_button(
        label="下载详细 PDF 报告",
        data=pdf_bytes,
        file_name="算法过程可视化报告.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

# ============================================================
# 7. 调用智能体生成内容
# ============================================================

def call_agent(user_question: str) -> Dict[str, Any]:
    client = get_client()
    if client is None:
        raise RuntimeError("未配置智能体服务 Key，请在 app.py 顶部 CODE_OPENAI_API_KEY 中填写。")

    model = get_model_name()
    prompt = build_user_prompt(user_question)
    status_ph = st.empty()
    progress = st.progress(0)

    steps = [
        "正在解析题目描述……",
        "正在识别当前算法……",
        "正在提取关键输入信息……",
        "正在拆解核心步骤……",
        "正在生成可视化帧……",
        "正在整理详细报告……",
    ]

    raw = ""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_OUTPUT_TOKENS,
            stream=True,
        )

        step_idx = 0
        last_update = time.time()
        status_ph.markdown(f"<div class='status-box'>🧩 {steps[0]}</div>", unsafe_allow_html=True)
        progress.progress(5)

        for chunk in stream:
            delta = getattr(chunk.choices[0].delta, "content", None) or ""
            raw += delta
            now = time.time()
            # 按时间和长度推进公开状态，不展示底层返回文本。
            if now - last_update > 0.7 and step_idx < len(steps) - 1:
                step_idx += 1
                pct = int((step_idx + 1) / len(steps) * 85)
                status_ph.markdown(f"<div class='status-box'>🧩 {steps[step_idx]}</div>", unsafe_allow_html=True)
                progress.progress(pct)
                last_update = now

        progress.progress(100)
        status_ph.markdown("<div class='status-box'>✅ 算法过程可视化生成完成。</div>", unsafe_allow_html=True)

    except Exception as e:
        raise RuntimeError(user_friendly_error(e)) from e

    data = extract_json(raw)
    result = normalize_result(data, raw, user_question)

    # 把原始内容放进会话状态，便于调试；默认不展示。
    st.session_state["raw_agent_text"] = raw
    return result

# ============================================================
# 8. 主界面
# ============================================================

with st.sidebar:
    st.markdown("### 使用说明")
    st.write("1. 输入一个算法题目。")
    st.write("2. 题目中指定哪个算法，就只处理该算法。")
    st.write("3. 生成后会展示可视化帧，并可下载 PDF。")
    st.divider()
    st.markdown("### 示例题目")
    st.code("请用动态规划可视化求解 01 背包问题。背包容量 15，物品 A 重量2 价值6，B 重量3 价值10，C 重量4 价值12，D 重量5 价值14，E 重量9 价值20，F 重量7 价值18。", language="text")

question = st.text_area(
    "请输入算法题目",
    value="请用动态规划可视化求解 01 背包问题。背包容量 15，物品 A 重量2 价值6，B 重量3 价值10，C 重量4 价值12，D 重量5 价值14，E 重量9 价值20，F 重量7 价值18。",
    height=160,
    placeholder="例如：请用快速排序可视化数组 [8, 3, 5, 1, 9, 6, 2, 7] 的排序过程。",
)

col_a, col_b = st.columns([1, 3])
with col_a:
    run_btn = st.button("生成算法过程可视化", type="primary", use_container_width=True)
with col_b:
    st.markdown("<span class='small-muted'>建议一次只输入一个算法，例如“请用动态规划……”“请用回溯法……”“请用 Dijkstra……”</span>", unsafe_allow_html=True)

if run_btn:
    if not question.strip():
        st.warning("请先输入算法题目。")
    else:
        st.markdown("## 智能体实时求解过程")
        st.caption("展示智能体对题目的分析、拆解与可视化生成过程。")
        try:
            result = call_agent(question.strip())
            st.session_state["last_result"] = result
            st.session_state["last_question"] = question.strip()
            st.divider()
            render_result(result, question.strip())
        except Exception as e:
            st.error(str(e))
            with st.expander("查看错误详情"):
                st.code(traceback.format_exc(), language="text")

# 如果刷新页面后会话里已有结果，保留展示
if not run_btn and "last_result" in st.session_state:
    st.divider()
    render_result(st.session_state["last_result"], st.session_state.get("last_question", question))

with st.expander("调试信息：查看原始分析文本", expanded=False):
    st.caption("仅用于排查渲染问题，录屏和正式展示时可以关闭。")
    st.text_area("原始分析文本", st.session_state.get("raw_agent_text", ""), height=240)
