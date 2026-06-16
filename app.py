# -*- coding: utf-8 -*-
"""
算法过程可视化 - LLM版

设计原则：
1. 不在本地实现回溯、动态规划、分支限界、遗传算法、模拟退火等求解器。
2. 算法识别、求解步骤、可视化帧、教学讲解、报告内容全部由 OpenAI 兼容接口的大模型生成。
3. 本地程序只做三件事：调用大模型 API、渲染大模型返回的可视化 JSON、导出 PDF。
"""

import os
import re
import json
import html
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
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
# 使用通用 OpenAI 兼容接口配置。硅基流动只需替换 endpoint 即可。
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "nex-agi/Nex-N2-Pro"

DEFAULT_SYSTEM_PROMPT = r"""
你是“算法过程可视化”智能体。你的核心任务不是只给答案，而是把算法求解过程转换为可逐帧展示的可视化数据。

重要约束：
- 你必须根据用户输入自动识别问题类型和适合的算法，不能要求用户从固定列表中选择。
- 算法步骤、状态变化、可视化帧、最终报告文字都由你生成。
- 不要假设本地程序会替你运行算法；你必须自己推导并给出过程。
- 对小规模实例要尽量精确计算；不确定时要写明“近似演示”或“启发式结果”。
- 回溯法、动态规划、分支限界通常应给出精确最优解；遗传算法、模拟退火属于启发式方法，可以展示迭代过程，并说明不保证每次最优。
- 输出必须是严格 JSON，不要输出 Markdown 代码块，不要在 JSON 前后添加解释。

你需要输出如下 JSON 结构：
{
  "task_title": "题目标题",
  "detected_problem_type": "自动识别的问题类型",
  "input_summary": "对用户输入的结构化概括",
  "overall_goal": "本次可视化学习目标",
  "algorithms": [
    {
      "name": "算法名称",
      "role": "该算法为什么适合该问题",
      "guarantee": "是否保证最优以及原因",
      "core_idea": "核心思想",
      "complexity": "时间和空间复杂度，允许用常见表达式",
      "final_answer": "该算法得到的结果",
      "frames": [
        {
          "step": 1,
          "title": "当前帧标题",
          "explanation": "这一帧解释什么算法动作",
          "state": "关键状态，例如当前物品、容量、队列、温度、种群等",
          "visual_type": "mermaid|table|array|text",
          "mermaid": "若 visual_type 为 mermaid，则给 Mermaid 图代码；可为空字符串",
          "table": [["列1", "列2"], ["值1", "值2"]],
          "array": ["若 visual_type 为 array，则给数组或染色体展示"],
          "variables": {"变量名": "变量值"},
          "pseudocode_focus": "本帧对应的伪代码关键句"
        }
      ],
      "summary": "这个算法的过程总结"
    }
  ],
  "comparison": [
    {"algorithm": "算法名", "view": "主要看什么图", "strength": "优点", "limitation": "局限"}
  ],
  "teaching_report_markdown": "完整教学报告，必须足够详细，包含题目解析、输入数据、算法选择理由、逐算法求解过程、关键帧说明、结果对比、复杂度分析、学习总结和可核验结论"
}

可视化帧生成规则：
1. 每个算法建议 4-8 帧，保证录屏时清晰。
2. 如果是回溯/分支限界，优先使用 Mermaid flowchart 展示决策树、剪枝、队列、上界。
3. 如果是动态规划，优先使用 table 展示状态转移表，并在 explanation 中说明 dp 状态含义。
4. 如果是遗传算法，优先使用 table/array 展示染色体、适应度、选择、交叉、变异、最优个体变化。
5. 如果是模拟退火，优先使用 table/array 展示当前解、邻域解、温度、接受概率、历史最优。
6. Mermaid 代码要简洁，尽量使用 flowchart TD 或 graph TD，不要使用过复杂语法。
7. table 的第一行必须是表头。
8. teaching_report_markdown 必须是一份可以直接写入 PDF 的详细文档，不少于 1200 字；如果题目很小，也要解释每种算法的核心思想、每一步如何变化、最终结果如何验证。
9. 所有内容使用中文。
"""


def load_system_prompt() -> str:
    local_file = os.path.join(os.path.dirname(__file__), "AGENT_PROMPT.md")
    if os.path.exists(local_file):
        try:
            with open(local_file, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return DEFAULT_SYSTEM_PROMPT
    return DEFAULT_SYSTEM_PROMPT


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, None)
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(name, default)


def get_config(names: List[str], default: str = "") -> str:
    """按顺序读取配置，优先使用通用 OPENAI_*，兼容旧版 SILICONFLOW_*。"""
    for name in names:
        value = get_secret(name, "")
        if value:
            return value
    return default


def is_quota_or_permission_error(err: Exception) -> bool:
    text = str(err).lower()
    keywords = [
        "insufficient", "balance", "quota", "403", "30001",
        "permission", "forbidden", "unauthorized", "invalid api key"
    ]
    return any(k in text for k in keywords)


def create_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def extract_json_object(text: str) -> Dict[str, Any]:
    """从模型输出中提取 JSON。若模型误加代码块，也尽量恢复。"""
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


def ask_llm_for_visualization(client: OpenAI, model: str, user_problem: str, max_frames: int) -> Dict[str, Any]:
    system_prompt = load_system_prompt()
    user_prompt = f"""
请将下面的算法题目转化为可视化求解过程。要求：
1. 自动识别问题类型，不要依赖用户从菜单选择。
2. 由你生成算法过程、可视化帧和教学报告。
3. 若用户要求多种算法，请分别生成；若用户没有指定算法，请选择最适合教学展示的 1-3 种算法。
4. 每种算法最多生成 {max_frames} 帧。
5. teaching_report_markdown 必须是详细 PDF 报告正文，包含完整求解过程，不少于 1200 字。
6. 严格按照系统提示词给出的 JSON 结构输出。

用户题目：
{user_problem}
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or ""
    try:
        return extract_json_object(content)
    except Exception as exc:
        raise ValueError(content) from exc


def repair_json_with_llm(client: OpenAI, model: str, bad_text: str) -> Dict[str, Any]:
    prompt = """
下面文本本应是 JSON，但格式可能有错误。请只返回修复后的严格 JSON，不要解释，不要代码块。
""" + bad_text
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content or ""
    try:
        return extract_json_object(content)
    except Exception as exc:
        raise ValueError(content) from exc


def render_mermaid(code: str, height: int = 430) -> None:
    if not code.strip():
        st.info("该帧没有 Mermaid 图代码。")
        return
    safe_code = html.escape(code)
    mermaid_html = f"""
    <div style="background:white; padding: 12px; border-radius: 10px; border: 1px solid #ddd; overflow:auto;">
      <pre class="mermaid">{safe_code}</pre>
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{ startOnLoad: true, theme: 'default', securityLevel: 'loose' }});
    </script>
    """
    components.html(mermaid_html, height=height, scrolling=True)


def render_table(table_data: Any) -> None:
    if not isinstance(table_data, list) or not table_data:
        st.info("该帧没有表格数据。")
        return
    try:
        header = table_data[0]
        rows = table_data[1:]
        df = pd.DataFrame(rows, columns=header)
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        st.write(table_data)


def render_array(array_data: Any) -> None:
    if not array_data:
        st.info("该帧没有数组/染色体数据。")
        return
    if not isinstance(array_data, list):
        array_data = [str(array_data)]
    cells = "".join(
        f"<span style='display:inline-block;margin:4px;padding:10px 14px;border:1px solid #999;border-radius:8px;background:#fafafa;font-family:monospace'>{html.escape(str(x))}</span>"
        for x in array_data
    )
    st.markdown(cells, unsafe_allow_html=True)


def render_frame(frame: Dict[str, Any]) -> None:
    st.subheader(f"第 {frame.get('step', '')} 帧：{frame.get('title', '')}")
    st.write(frame.get("explanation", ""))

    visual_type = str(frame.get("visual_type", "text")).lower()
    if visual_type == "mermaid":
        render_mermaid(str(frame.get("mermaid", "")))
    elif visual_type == "table":
        render_table(frame.get("table", []))
    elif visual_type == "array":
        render_array(frame.get("array", []))
    else:
        st.info(frame.get("state", "本帧以文字解释为主。"))

    variables = frame.get("variables", {})
    if isinstance(variables, dict) and variables:
        st.markdown("**关键变量**")
        var_df = pd.DataFrame([{"变量": k, "值": v} for k, v in variables.items()])
        st.dataframe(var_df, use_container_width=True, hide_index=True)

    focus = frame.get("pseudocode_focus", "")
    if focus:
        st.code(focus, language="text")


def safe_para(text: Any) -> str:
    return html.escape(str(text)).replace("\n", "<br/>")


def _pdf_table_from_rows(rows: List[List[Any]], col_widths: List[float], font_size: float = 8.5) -> Table:
    wrapped_rows = []
    body_style = ParagraphStyle(
        name="TableChineseBody",
        fontName="STSong-Light",
        fontSize=font_size,
        leading=font_size + 3,
    )
    for row in rows:
        wrapped_rows.append([Paragraph(safe_para(cell), body_style) for cell in row])
    table = Table(wrapped_rows, colWidths=col_widths, repeatRows=1)
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 3),
    ]))
    return table


def make_pdf(data: Dict[str, Any]) -> str:
    """生成详细 PDF。

    注意：算法求解内容来自大模型返回的 JSON；本函数只负责将智能体输出排版成 PDF，
    不参与算法求解、不实现本地算法求解器。
    """
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ChineseTitle",
        parent=styles["Title"],
        fontName="STSong-Light",
        fontSize=22,
        leading=28,
        alignment=1,
        spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="ChineseHeading",
        parent=styles["Heading2"],
        fontName="STSong-Light",
        fontSize=15,
        leading=20,
        spaceBefore=12,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        name="ChineseSubHeading",
        parent=styles["Heading3"],
        fontName="STSong-Light",
        fontSize=12.5,
        leading=17,
        spaceBefore=8,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="ChineseBody",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=10.5,
        leading=16,
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="ChineseCode",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=8.5,
        leading=12,
        leftIndent=10,
        rightIndent=10,
        backColor=colors.whitesmoke,
        borderColor=colors.lightgrey,
        borderWidth=0.3,
        borderPadding=5,
        spaceAfter=6,
    ))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.close()
    doc = SimpleDocTemplate(
        tmp.name,
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    story: List[Any] = []
    story.append(Paragraph("算法过程可视化", styles["ChineseTitle"]))
    story.append(Paragraph("智能体自动生成的详细求解过程 PDF", styles["ChineseHeading"]))
    story.append(Paragraph("生成时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["ChineseBody"]))
    story.append(Paragraph("说明：本 PDF 的题目识别、算法步骤、可视化帧、变量变化和教学报告来自 OpenAI 兼容接口的大模型输出；程序仅负责把大模型输出排版为 PDF。", styles["ChineseBody"]))
    story.append(Spacer(1, 0.2 * cm))

    story.append(Paragraph("一、题目与识别结果", styles["ChineseHeading"]))
    overview_rows = [
        ["项目", "内容"],
        ["题目标题", data.get("task_title", "算法题目")],
        ["自动识别的问题类型", data.get("detected_problem_type", "未识别")],
        ["输入概括", data.get("input_summary", "")],
        ["学习目标", data.get("overall_goal", "")],
        ["算法数量", str(len(data.get("algorithms", []) or []))],
    ]
    story.append(_pdf_table_from_rows(overview_rows, [4.0 * cm, 12.0 * cm], 9))
    story.append(Spacer(1, 0.2 * cm))

    algorithms = data.get("algorithms", [])
    if isinstance(algorithms, list) and algorithms:
        story.append(Paragraph("二、逐算法详细求解过程", styles["ChineseHeading"]))
        for alg_idx, alg in enumerate(algorithms, start=1):
            story.append(Paragraph(f"{alg_idx}. {safe_para(alg.get('name', '算法'))}", styles["ChineseSubHeading"]))
            alg_rows = [["项目", "内容"]]
            for key, label in [
                ("role", "适用原因"),
                ("core_idea", "核心思想"),
                ("guarantee", "结果保证"),
                ("complexity", "复杂度"),
                ("final_answer", "最终结果"),
                ("summary", "过程总结"),
            ]:
                if alg.get(key):
                    alg_rows.append([label, alg.get(key)])
            story.append(_pdf_table_from_rows(alg_rows, [3.3 * cm, 12.7 * cm], 8.8))
            story.append(Spacer(1, 0.15 * cm))

            frames = alg.get("frames", [])
            if isinstance(frames, list) and frames:
                story.append(Paragraph("关键可视化帧与变量变化", styles["ChineseSubHeading"]))
                for fr in frames:
                    story.append(Paragraph(f"帧 {safe_para(fr.get('step', ''))}：{safe_para(fr.get('title', ''))}", styles["ChineseBody"]))
                    detail_rows = [["字段", "内容"]]
                    for key, label in [
                        ("explanation", "动作解释"),
                        ("state", "关键状态"),
                        ("pseudocode_focus", "伪代码焦点"),
                    ]:
                        if fr.get(key):
                            detail_rows.append([label, fr.get(key)])
                    variables = fr.get("variables", {})
                    if isinstance(variables, dict) and variables:
                        var_text = "；".join([f"{k}={v}" for k, v in variables.items()])
                        detail_rows.append(["关键变量", var_text])
                    visual_type = str(fr.get("visual_type", "text"))
                    detail_rows.append(["可视化类型", visual_type])
                    story.append(_pdf_table_from_rows(detail_rows, [3.0 * cm, 13.0 * cm], 8.2))

                    # 将模型返回的可视化数据也写入 PDF，便于审核者看到“智能体求解过程”。
                    if visual_type == "table" and fr.get("table"):
                        raw_table = fr.get("table", [])
                        try:
                            if isinstance(raw_table, list) and raw_table:
                                max_cols = max(len(r) if isinstance(r, list) else 1 for r in raw_table)
                                normalized = []
                                for r in raw_table[:12]:
                                    if not isinstance(r, list):
                                        r = [r]
                                    normalized.append([str(x) for x in r] + [""] * (max_cols - len(r)))
                                col_w = [16.0 * cm / max_cols] * max_cols
                                story.append(_pdf_table_from_rows(normalized, col_w, 7.2))
                        except Exception:
                            story.append(Paragraph(safe_para(raw_table), styles["ChineseCode"]))
                    elif visual_type == "array" and fr.get("array"):
                        story.append(Paragraph("数组/染色体展示：" + safe_para(fr.get("array")), styles["ChineseCode"]))
                    elif visual_type == "mermaid" and fr.get("mermaid"):
                        story.append(Paragraph("Mermaid 可视化代码：", styles["ChineseBody"]))
                        story.append(Paragraph(safe_para(fr.get("mermaid")), styles["ChineseCode"]))
                    story.append(Spacer(1, 0.12 * cm))

    comparison = data.get("comparison", [])
    if isinstance(comparison, list) and comparison:
        story.append(Paragraph("三、算法对比", styles["ChineseHeading"]))
        rows = [["算法", "可视化重点", "优点", "局限"]]
        for item in comparison:
            rows.append([
                item.get("algorithm", ""),
                item.get("view", ""),
                item.get("strength", ""),
                item.get("limitation", ""),
            ])
        story.append(_pdf_table_from_rows(rows, [2.5 * cm, 4.2 * cm, 4.7 * cm, 4.6 * cm], 7.8))

    report = data.get("teaching_report_markdown", "")
    if report:
        story.append(PageBreak())
        story.append(Paragraph("四、大模型生成的完整教学报告", styles["ChineseHeading"]))
        for para in str(report).split("\n"):
            line = para.strip()
            if not line:
                continue
            # 将 Markdown 标题轻量转换成 PDF 标题风格。
            if line.startswith("###"):
                story.append(Paragraph(safe_para(line.lstrip("# ")), styles["ChineseSubHeading"]))
            elif line.startswith("##"):
                story.append(Paragraph(safe_para(line.lstrip("# ")), styles["ChineseHeading"]))
            elif line.startswith("#"):
                story.append(Paragraph(safe_para(line.lstrip("# ")), styles["ChineseHeading"]))
            else:
                story.append(Paragraph(safe_para(line), styles["ChineseBody"]))

    doc.build(story)
    return tmp.name

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("基于 OpenAI 兼容接口的大模型算法过程可视化：输入题目，大模型生成算法识别、求解过程、可视化帧和 PDF 报告。")

    with st.sidebar:
        st.header("OpenAI 兼容接口配置")
        api_key = st.text_input("OPENAI_API_KEY", value=get_config(["OPENAI_API_KEY", "SILICONFLOW_API_KEY"], ""), type="password")
        base_url = st.text_input("OPENAI_BASE_URL", value=get_config(["OPENAI_BASE_URL", "SILICONFLOW_BASE_URL"], DEFAULT_BASE_URL))
        model = st.text_input("OPENAI_MODEL", value=get_config(["OPENAI_MODEL", "SILICONFLOW_MODEL"], DEFAULT_MODEL))
        max_frames = st.slider("每种算法最多帧数", min_value=3, max_value=10, value=5)
        st.markdown("---")
        st.info("默认使用硅基流动 OpenAI 兼容接口：base_url=https://api.siliconflow.cn/v1，模型 nex-agi/Nex-N2-Pro。")
        st.info("本项目不在本地实现算法求解器；本地只负责调用大模型、渲染可视化和导出 PDF。")

    sample = "请可视化求解 01 背包问题。背包容量 15，物品 A 重量2 价值6，B 重量3 价值10，C 重量4 价值12，D 重量5 价值14，E 重量9 价值20，F 重量7 价值18。请用回溯法、动态规划、分支限界、遗传算法、模拟退火进行对比。"
    user_problem = st.text_area("请输入算法题目", value=sample, height=160)

    col1, col2 = st.columns([1, 3])
    with col1:
        run = st.button("生成算法过程可视化", type="primary", use_container_width=True)
    with col2:
        st.write("支持自然语言输入，例如：01 背包、N 皇后、最短路径、排序、LCS、TSP、图搜索等。")

    if run:
        if not api_key:
            st.error("请先配置 OPENAI_API_KEY。")
            return
        if not user_problem.strip():
            st.error("请输入算法题目。")
            return

        client = create_client(api_key, base_url)
        with st.spinner("正在调用大模型生成算法过程可视化……"):
            try:
                data = ask_llm_for_visualization(client, model, user_problem, max_frames)
            except Exception as e:
                if is_quota_or_permission_error(e):
                    st.error("接口调用失败：请检查 API Key、模型权限、账户额度，以及模型名是否可用。")
                    st.code(str(e), language="text")
                    return
                st.warning("首次解析 JSON 失败，尝试让大模型修复输出格式。")
                try:
                    data = repair_json_with_llm(client, model, str(e))
                except Exception as e2:
                    if is_quota_or_permission_error(e2):
                        st.error("接口调用失败：请检查 API Key、模型权限、账户额度，以及模型名是否可用。")
                        st.code(str(e2), language="text")
                        return
                    st.error(f"生成失败：{e2}")
                    return

        st.session_state["visualization_data"] = data
        if "pdf_path" in st.session_state:
            del st.session_state["pdf_path"]
        st.success("大模型已生成算法过程可视化，并将自动整理为详细 PDF 报告。")

    data = st.session_state.get("visualization_data")
    if not data:
        return

    st.markdown("## 自动识别结果")
    c1, c2, c3 = st.columns(3)
    c1.metric("题目", data.get("task_title", "算法题目"))
    c2.metric("问题类型", data.get("detected_problem_type", "未识别"))
    c3.metric("算法数量", len(data.get("algorithms", []) or []))
    st.write(data.get("input_summary", ""))

    algorithms = data.get("algorithms", []) or []
    if algorithms:
        st.markdown("## 逐算法可视化")
        tabs = st.tabs([alg.get("name", f"算法{i+1}") for i, alg in enumerate(algorithms)])
        for tab, alg in zip(tabs, algorithms):
            with tab:
                st.markdown(f"### {alg.get('name', '算法')}")
                st.write("**核心思想：**", alg.get("core_idea", ""))
                st.write("**结果保证：**", alg.get("guarantee", ""))
                st.write("**复杂度：**", alg.get("complexity", ""))
                st.write("**最终结果：**", alg.get("final_answer", ""))
                frames = alg.get("frames", []) or []
                if frames:
                    idx = st.slider(
                        f"选择 {alg.get('name', '算法')} 的可视化帧",
                        min_value=1,
                        max_value=len(frames),
                        value=1,
                        key=f"slider_{alg.get('name', '')}_{id(alg)}",
                    )
                    render_frame(frames[idx - 1])
                st.markdown("#### 算法总结")
                st.write(alg.get("summary", ""))

    comparison = data.get("comparison", []) or []
    if comparison:
        st.markdown("## 算法对比")
        st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

    report = data.get("teaching_report_markdown", "")
    if report:
        st.markdown("## 大模型生成的教学报告")
        st.markdown(report)

    st.markdown("## 智能体自动输出 PDF")
    st.info("PDF 已根据大模型生成的算法识别、求解步骤、可视化帧、关键变量和教学报告自动生成。用户只需要点击下载即可提交。")
    pdf_path = st.session_state.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        pdf_path = make_pdf(data)
        st.session_state["pdf_path"] = pdf_path
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="下载智能体自动生成的详细 PDF 报告",
            data=f.read(),
            file_name="算法过程可视化_智能体求解报告.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with st.expander("查看大模型原始 JSON 输出"):
        st.json(data)


if __name__ == "__main__":
    main()
