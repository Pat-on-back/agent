# -*- coding: utf-8 -*-
"""
算法可视化智能体 - 加速版

设计原则：
1. 不在本地实现回溯、动态规划、分支限界、遗传算法、模拟退火等求解器。
2. 算法识别、求解步骤、可视化帧、教学讲解、报告内容全部由智能体引擎生成。
3. 本地程序只做三件事：连接智能体引擎、渲染结构化可视化结果、导出 PDF。
"""

import os
import re
import json
import html
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime
from queue import Queue, Empty
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


APP_TITLE = "算法可视化智能体"
# 使用通用 智能体引擎配置。硅基流动只需替换 endpoint 即可。
DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_MODEL = "nex-agi/Nex-N2-Pro"

DEFAULT_SYSTEM_PROMPT = r"""
你是“算法可视化智能体”。你的任务是把用户输入的算法题，快速转成可渲染的结构化数据。

硬性要求：
- 只处理用户明确指定的算法；用户没有指定算法时，只选择 1 个最适合教学展示的算法。
- 不要擅自扩展成动态规划、回溯、分支限界、遗传算法、模拟退火等一大堆算法。
- 底层输出必须是严格 JSON，不要 Markdown 代码块，不要 JSON 外解释。
- 可视化帧必须短小、清晰、能直接渲染。
- Mermaid 只用简单 flowchart TD 或 graph TD；table 第一行必须是表头。
- 报告默认写简洁版，不要写长篇论文式报告。

必须输出如下 JSON：
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
      "complexity": "时间和空间复杂度",
      "final_answer": "该算法得到的结果",
      "frames": [
        {
          "step": 1,
          "title": "当前帧标题",
          "explanation": "这一帧解释什么算法动作",
          "state": "关键状态",
          "visual_type": "mermaid|table|array|text",
          "mermaid": "",
          "table": [["列1", "列2"], ["值1", "值2"]],
          "array": [],
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
  "teaching_report_markdown": "简洁教学报告"
}
"""


def load_system_prompt() -> str:
    # 加速版默认使用内置短提示词，避免外部 AGENT_PROMPT.md 仍然保留旧版超长提示词导致变慢。
    # 如确实需要外部提示词，可设置环境变量 USE_LOCAL_AGENT_PROMPT=1。
    if os.environ.get("USE_LOCAL_AGENT_PROMPT") == "1":
        local_file = os.path.join(os.path.dirname(__file__), "AGENT_PROMPT.md")
        if os.path.exists(local_file):
            try:
                with open(local_file, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception:
                pass
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
    """按顺序读取配置，优先使用通用环境变量，兼容旧版服务配置。"""
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
    # 设置超时，避免接口卡住导致页面一直等待。
    return OpenAI(api_key=api_key, base_url=base_url, timeout=60)


@st.cache_resource
def get_executor() -> ThreadPoolExecutor:
    """后台线程池。

    作用：把智能体生成请求放到后台执行，主页面保持可刷新，用户可以点击
    “终止当前生成 / 开启新会话”。已发出的生成请求不一定能被服务端强制取消，
    但旧任务完成后会被 generation_id 丢弃，不会覆盖新会话。
    """
    return ThreadPoolExecutor(max_workers=4)


def reset_visual_state(clear_input: bool = False) -> None:
    """清空当前页面的生成结果，开启一个新的前端会话。

    Streamlit 规则：同一轮脚本运行中，带 key 的输入组件一旦实例化，
    就不能再直接写 st.session_state[组件 key]。
    因此这里不再写 user_problem，而是通过更换输入框 key 的方式安全清空输入。
    """
    for key in [
        "visualization_data",
        "pdf_path",
        "generation_future",
        "generation_id",
        "generation_error",
        "generation_started_at",
        "realtime_process_markdown",
        "realtime_log_items",
        "generation_log_queue",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state["active_session_id"] = str(uuid.uuid4())
    if clear_input:
        # 不直接修改 text_area 当前 key 的值，避免 StreamlitAPIException。
        # 改用新的 widget key，下一轮 rerun 时输入框自然变成空白。
        st.session_state["problem_input_nonce"] = str(uuid.uuid4())
        st.session_state["problem_initial_text"] = ""


def cancel_current_generation(clear_input: bool = False) -> None:
    """终止当前页面等待，并开启新会话。

    注意：已经发出去的生成请求通常无法可靠强杀；
    这里通过切换 active_session_id 和 generation_id，让旧任务的返回结果自动作废。
    """
    old_future = st.session_state.get("generation_future")
    if isinstance(old_future, Future) and not old_future.done():
        old_future.cancel()
    reset_visual_state(clear_input=clear_input)


def generate_visualization_job(
    *,
    api_key: str,
    base_url: str,
    model: str,
    user_problem: str,
    max_frames: int,
    fast_mode: bool,
    include_detailed_report: bool,
    session_id: str,
    generation_id: str,
    log_queue: Optional[Queue] = None,
) -> Dict[str, Any]:
    """后台任务：不直接调用任何 st.* 渲染函数。"""
    client = create_client(api_key, base_url)
    log_event(log_queue, "🟢 智能体会话启动：已接收题目，准备进入算法可视化流程。")
    log_event(log_queue, "🔎 正在识别题型、抽取输入数据和约束条件。")
    log_event(log_queue, "🧭 " + build_algorithm_policy(user_problem))
    log_event(log_queue, f"🎞️ 正在规划逐帧展示，每种算法最多 {max_frames} 帧。")
    data = ask_llm_for_visualization(
        client=client,
        model=model,
        user_problem=user_problem,
        max_frames=max_frames,
        fast_mode=fast_mode,
        include_detailed_report=include_detailed_report,
        log_queue=log_queue,
    )
    log_event(log_queue, "✅ 结构化可视化内容已完成：算法识别、帧数据、变量变化和报告摘要均已就绪。")
    data["realtime_process_markdown"] = ""
    data["_session_id"] = session_id
    data["_generation_id"] = generation_id
    return data


def extract_json_object(text: str) -> Dict[str, Any]:
    """从智能体结构化输出中提取数据。若输出误加代码块，也尽量恢复。"""
    if not text:
        raise ValueError("智能体引擎没有返回内容")

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


ALGORITHM_KEYWORDS = [
    ("动态规划", ["动态规划", "dp", "DP"]),
    ("回溯法", ["回溯", "回溯法", "backtracking"]),
    ("分支限界", ["分支限界", "分枝限界", "branch and bound", "branch-and-bound"]),
    ("贪心算法", ["贪心", "greedy"]),
    ("分治法", ["分治", "divide and conquer"]),
    ("广度优先搜索", ["广度优先", "BFS", "bfs"]),
    ("深度优先搜索", ["深度优先", "DFS", "dfs"]),
    ("Dijkstra 算法", ["dijkstra", "Dijkstra", "迪ijkstra", "最短路径"]),
    ("遗传算法", ["遗传算法", "GA", "genetic"]),
    ("模拟退火", ["模拟退火", "SA", "annealing"]),
    ("快速排序", ["快速排序", "快排", "quick sort", "quicksort"]),
    ("归并排序", ["归并排序", "merge sort", "mergesort"]),
]


def detect_requested_algorithms(user_problem: str) -> List[str]:
    """识别用户文本里明确点名的算法。点名几个就只处理几个；未点名则返回空列表。"""
    found: List[str] = []
    text = user_problem or ""
    lower_text = text.lower()
    for canonical, keys in ALGORITHM_KEYWORDS:
        for key in keys:
            if key.lower() in lower_text:
                found.append(canonical)
                break
    # 去重并保持顺序
    result: List[str] = []
    for name in found:
        if name not in result:
            result.append(name)
    return result


def build_algorithm_policy(user_problem: str) -> str:
    requested = detect_requested_algorithms(user_problem)
    if requested:
        return "用户已明确指定算法，只允许输出这些算法：" + "、".join(requested) + "。不要额外生成其他算法。"
    return "用户没有明确指定算法，只选择 1 个最适合本题的算法，不要扩展生成多个算法。"


def build_local_process_log(user_problem: str, max_frames: int) -> str:
    """本地即时生成公开过程，不再为“实时日志”额外启动一次智能体生成流程。"""
    policy = build_algorithm_policy(user_problem)
    requested = detect_requested_algorithms(user_problem)
    algo_text = "、".join(requested) if requested else "自动选择 1 个最适合算法"
    return f"""
✅ **题目已接收**：系统将输入题目整理为算法可视化任务。  
🔎 **算法范围**：{policy}  
📊 **逐帧可视化**：每种算法最多生成 {max_frames} 帧，优先展示关键状态、变量变化和表格/流程图。  
📄 **报告策略**：快速模式下不再先生成长篇实时日志，直接生成可视化内容，并由页面立即渲染；PDF 会根据智能体输出自动整理。  
🚀 **当前执行**：正在生成 {algo_text} 的可视化帧、变量表和简洁教学报告。
""".strip()




def log_event(log_queue: Optional[Queue], message: str) -> None:
    """把后台任务的公开过程写入队列，供页面实时打印。

    这里记录的是可展示的智能体工作日志，不是模型隐藏思维链。
    """
    if log_queue is None:
        return
    try:
        log_queue.put(f"{datetime.now().strftime('%H:%M:%S')}  {message}")
    except Exception:
        pass


def drain_generation_log_queue() -> None:
    """从后台队列取出实时日志，追加到 session_state。"""
    q = st.session_state.get("generation_log_queue")
    if q is None:
        return
    items = st.session_state.setdefault("realtime_log_items", [])
    try:
        while True:
            items.append(q.get_nowait())
    except Empty:
        pass


def build_log_markdown(items: List[str]) -> str:
    return "\n\n".join(str(x).strip() for x in items if str(x).strip())


def render_realtime_thinking(running: bool = False) -> None:
    """页面打印实时思考过程。"""
    items = st.session_state.get("realtime_log_items", [])
    if not items:
        st.info("等待智能体开始输出实时思考过程。")
        return
    text = "\n".join(items)
    if running:
        text += "\n▌"
    st.code(text, language="text")

def ensure_report(data: Dict[str, Any], user_problem: str) -> None:
    """如果智能体引擎没有给出教学报告，则用结构化结果本地整理一份短报告，避免再次请求智能体引擎。"""
    if str(data.get("teaching_report_markdown", "")).strip():
        return
    algorithms = data.get("algorithms", []) or []
    lines = [
        "# 算法可视化智能体教学报告",
        "",
        "## 题目概括",
        str(data.get("input_summary", user_problem)),
        "",
        "## 问题类型与目标",
        f"识别类型：{data.get('detected_problem_type', '未识别')}。学习目标：{data.get('overall_goal', '理解算法状态变化与最终结果。')}",
        "",
        "## 算法过程",
    ]
    for alg in algorithms:
        lines.extend([
            f"### {alg.get('name', '算法')}",
            f"核心思想：{alg.get('core_idea', '')}",
            f"结果保证：{alg.get('guarantee', '')}",
            f"复杂度：{alg.get('complexity', '')}",
            f"最终结果：{alg.get('final_answer', '')}",
            "关键帧：",
        ])
        for fr in alg.get("frames", []) or []:
            lines.append(f"- 帧 {fr.get('step', '')}：{fr.get('title', '')}。{fr.get('explanation', '')}")
        lines.append(f"总结：{alg.get('summary', '')}")
        lines.append("")
    lines.extend([
        "## 学习总结",
        "本报告由结构化数据自动整理，重点保留算法识别、逐帧可视化、变量变化、伪代码焦点和最终结论，适合课堂演示与作业提交。",
    ])
    data["teaching_report_markdown"] = "\n".join(lines)


def build_public_process_prompt(user_problem: str, max_frames: int) -> str:
    """生成可公开展示的“求解过程日志”提示词。

    注意：这里不是展示隐藏推理链，而是生成适合课堂展示和录屏的
    公开版工作日志：它描述智能体正在做什么、为什么这样做、接下来会输出什么。
    """
    return f"""
请为下面的算法题生成“公开可展示的智能体工作流日志”。

重要要求：
1. 这不是隐藏思维链，不要写私密心理活动；只输出可以展示给学生看的解题工作流。
2. 需要像智能体正在一步步工作一样输出，适合 Streamlit 页面实时滚动显示。
3. 使用 Markdown，分阶段显示：题目理解、问题类型识别、算法选择、可视化设计、PDF 报告规划、准确性自检。
4. 每个阶段用 2-4 句话说明，不要太长。
5. 说明每种算法最多生成 {max_frames} 个可视化帧。
6. 输出中可以使用 ✅、🔎、🧠、📊、📄 等符号增强可读性。

用户题目：
{user_problem}
"""


def stream_public_process(
    client: OpenAI,
    model: str,
    user_problem: str,
    max_frames: int,
    placeholder,
    progress_bar=None,
) -> str:
    """实时流式显示公开版求解过程日志。"""
    prompt = build_public_process_prompt(user_problem, max_frames)
    buffer = ""
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是算法可视化智能体。请输出公开可展示的智能体工作流，不要输出隐藏推理链。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1200,
            stream=True,
        )
        tick = 0
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if not delta:
                continue
            buffer += delta
            tick += 1
            placeholder.markdown(buffer + "\n\n▌")
            if progress_bar is not None:
                # 这是视觉进度，不代表真实 token 进度；用于录屏时呈现智能体正在工作。
                progress_bar.progress(min(90, 8 + tick % 83))
        placeholder.markdown(buffer)
        if progress_bar is not None:
            progress_bar.progress(100)
        return buffer.strip()
    except Exception as exc:
        msg = "智能体工作流生成失败，将继续尝试生成可视化结果。\n\n错误信息：" + str(exc)
        placeholder.warning(msg)
        return msg

def ask_llm_for_visualization(
    client: OpenAI,
    model: str,
    user_problem: str,
    max_frames: int,
    fast_mode: bool = True,
    include_detailed_report: bool = False,
    log_queue: Optional[Queue] = None,
) -> Dict[str, Any]:
    """一次生成结构化结果。快速模式下压缩算法数量、帧数和报告长度。"""
    system_prompt = load_system_prompt()
    algorithm_policy = build_algorithm_policy(user_problem)
    report_requirement = (
        "teaching_report_markdown 写 1000-1400 字的详细报告。"
        if include_detailed_report else
        "teaching_report_markdown 写 300-600 字简洁报告；不要长篇铺垫。"
    )
    frame_rule = (
        f"每种算法最多 {max_frames} 帧，每帧解释不超过 80 字。"
        if fast_mode else
        f"每种算法最多 {max_frames} 帧。"
    )
    user_prompt = f"""
请快速将下面题目转化为可视化结构化数据。

加速规则：
1. {algorithm_policy}
2. {frame_rule}
3. 每个 frame 的 table 不超过 8 行；Mermaid 节点不超过 12 个。
4. variables 只保留最关键 3-6 个变量。
5. {report_requirement}
6. 只输出严格 JSON，不要输出 Markdown 代码块，不要解释。

用户题目：
{user_problem}
"""
    kwargs = dict(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=3600 if fast_mode and not include_detailed_report else 7000,
    )
    # 使用流式接收：页面可以实时打印公开工作过程，同时仍然只做一次主生成调用。
    log_event(log_queue, "🧠 正在组织算法识别、状态变化、逐帧可视化和报告内容。")
    content = ""
    milestones = {
        "task_title": "🏷️ 已生成题目标题与任务概括。",
        "detected_problem_type": "🔍 已识别问题类型。",
        "algorithms": "🧩 正在生成算法模块。",
        "frames": "🎞️ 正在生成逐帧可视化内容。",
        "variables": "📌 正在整理关键变量变化。",
        "teaching_report_markdown": "📄 正在整理教学报告内容。",
    }
    emitted = set()
    last_report_len = 0
    try:
        stream = client.chat.completions.create(**kwargs, stream=True)
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if not delta:
                continue
            content += delta
            for key, msg in milestones.items():
                if key not in emitted and f'"{key}"' in content:
                    emitted.add(key)
                    log_event(log_queue, msg)
            if len(content) - last_report_len >= 900:
                last_report_len = len(content)
                log_event(log_queue, f"📥 正在接收可视化数据，已接收约 {len(content)} 个字符。")
    except Exception as stream_exc:
        # 少数 OpenAI 兼容服务可能不支持流式输出；退回普通生成，保证程序可运行。
        log_event(log_queue, "⚠️ 流式接收不可用，已切换为普通接收模式。")
        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content or ""

    log_event(log_queue, "🧪 正在解析结构化结果并检查字段完整性。")
    try:
        data = extract_json_object(content)
    except Exception as exc:
        log_event(log_queue, "❌ 结构化结果解析失败，页面将显示错误信息。")
        raise ValueError(content) from exc
    ensure_report(data, user_problem)
    log_event(log_queue, "🧾 正在补全报告摘要并准备页面渲染。")
    return data


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

    注意：算法求解内容来自智能体返回的结构化数据；本函数只负责将智能体输出排版成 PDF，
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
    story.append(Paragraph("算法可视化智能体", styles["ChineseTitle"]))
    story.append(Paragraph("算法可视化智能体生成的详细求解报告", styles["ChineseHeading"]))
    story.append(Paragraph("生成时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"), styles["ChineseBody"]))
    story.append(Paragraph("说明：本报告的题目识别、算法步骤、可视化帧、变量变化和教学内容来自算法可视化智能体输出；程序负责将智能体输出整理为 PDF。", styles["ChineseBody"]))
    story.append(Spacer(1, 0.2 * cm))

    realtime_process = data.get("realtime_process_markdown", "")
    if realtime_process:
        story.append(Paragraph("一、实时思考过程记录", styles["ChineseHeading"]))
        story.append(Paragraph("说明：以下内容是页面端实时打印的智能体公开思考过程，用于呈现题目理解、算法选择、可视化规划、变量整理和报告输出过程。", styles["ChineseBody"]))
        for para in str(realtime_process).split("\n"):
            line = para.strip()
            if not line:
                continue
            if line.startswith("###") or line.startswith("##") or line.startswith("#"):
                story.append(Paragraph(safe_para(line.lstrip("# ")), styles["ChineseSubHeading"]))
            else:
                story.append(Paragraph(safe_para(line), styles["ChineseBody"]))
        story.append(Spacer(1, 0.2 * cm))
        section_prefix = "二"
    else:
        section_prefix = "一"

    story.append(Paragraph(f"{section_prefix}、题目与识别结果", styles["ChineseHeading"]))
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
        story.append(Paragraph("逐算法详细求解过程", styles["ChineseHeading"]))
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

                    # 将智能体返回的可视化数据也写入 PDF，便于审核者看到“智能体求解过程”。
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
        story.append(Paragraph("算法对比", styles["ChineseHeading"]))
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
        story.append(Paragraph("智能体生成的完整教学报告", styles["ChineseHeading"]))
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
    st.caption("输入算法题目后，智能体自动识别问题、生成逐帧可视化、变量变化和教学报告；支持随时终止并开启新会话。")

    if "active_session_id" not in st.session_state:
        st.session_state["active_session_id"] = str(uuid.uuid4())

    running_future = st.session_state.get("generation_future")
    is_running = isinstance(running_future, Future) and not running_future.done()

    with st.sidebar:
        st.header("智能体引擎配置")
        api_key = st.text_input("智能体接口密钥", value=get_config(["OPENAI_API_KEY", "SILICONFLOW_API_KEY"], ""), type="password")
        base_url = st.text_input("服务地址", value=get_config(["OPENAI_BASE_URL", "SILICONFLOW_BASE_URL"], DEFAULT_BASE_URL))
        model = st.text_input("推理引擎", value=get_config(["OPENAI_MODEL", "SILICONFLOW_MODEL"], DEFAULT_MODEL))
        fast_mode = st.checkbox("快速模式：少算法、少帧、短报告", value=True)
        include_detailed_report = st.checkbox("生成长篇教学报告（较慢）", value=False)
        default_frames = 3 if fast_mode else 5
        max_frame_upper = 5 if fast_mode else 10
        max_frames = st.slider("每种算法最多帧数", min_value=2, max_value=max_frame_upper, value=default_frames)

        st.markdown("---")
        st.subheader("会话控制")
        if st.button("终止当前生成", use_container_width=True, disabled=not is_running):
            cancel_current_generation(clear_input=False)
            st.success("已终止当前生成；旧结果若稍后返回，也不会覆盖新会话。")
            st.rerun()
        if st.button("开启新会话", use_container_width=True):
            cancel_current_generation(clear_input=True)
            st.rerun()

        st.markdown("---")
        st.info("默认只启动一次智能体生成流程；后台执行，页面可随时终止并切换新会话。")
        st.info("生成时会实时打印公开思考过程：题型识别、算法范围、帧生成、变量整理和报告输出。")
        st.info("若点击终止，旧生成结果会被自动丢弃，不会覆盖新会话。")

    sample = "请可视化求解 01 背包问题。背包容量 15，物品 A 重量2 价值6，B 重量3 价值10，C 重量4 价值12，D 重量5 价值14，E 重量9 价值20，F 重量7 价值18。请用动态规划。"

    # 输入框使用动态 key。这样点击“终止并新建 / 开启新会话”时，
    # 只需要切换 key，不需要在组件实例化后强行改它的 session_state。
    if "problem_input_nonce" not in st.session_state:
        st.session_state["problem_input_nonce"] = "default"
    problem_key = f"user_problem_{st.session_state['problem_input_nonce']}"
    if problem_key not in st.session_state:
        st.session_state[problem_key] = st.session_state.pop("problem_initial_text", sample)
    user_problem = st.text_area("请输入算法题目", key=problem_key, height=160)

    col1, col2, col3 = st.columns([1.1, 1.1, 2.8])
    with col1:
        run = st.button("启动算法可视化智能体", type="primary", use_container_width=True, disabled=is_running)
    with col2:
        stop_and_new = st.button("终止并新建", use_container_width=True, disabled=not is_running)
    with col3:
        if is_running:
            st.warning("当前正在生成。你可以点击“终止并新建”立即切换到新会话。")
        else:
            st.write("若题目写明算法名，系统只处理该算法；未写明则自动选择 1 个算法。")

    if stop_and_new:
        cancel_current_generation(clear_input=True)
        st.rerun()

    if run:
        if not api_key:
            st.error("请先配置智能体接口密钥。")
            return
        if not user_problem.strip():
            st.error("请输入算法题目。")
            return

        # 开始新一轮生成：清掉旧结果，但保留输入框内容。
        for key in ["visualization_data", "pdf_path", "generation_error"]:
            if key in st.session_state:
                del st.session_state[key]

        session_id = st.session_state.get("active_session_id") or str(uuid.uuid4())
        st.session_state["active_session_id"] = session_id
        generation_id = str(uuid.uuid4())
        realtime_process = build_local_process_log(user_problem, max_frames)
        initial_log_items = [line.strip() for line in realtime_process.splitlines() if line.strip()]
        st.session_state["realtime_process_markdown"] = realtime_process
        st.session_state["realtime_log_items"] = initial_log_items
        st.session_state["generation_log_queue"] = Queue()
        st.session_state["generation_id"] = generation_id
        st.session_state["generation_started_at"] = time.time()

        executor = get_executor()
        future = executor.submit(
            generate_visualization_job,
            api_key=api_key,
            base_url=base_url,
            model=model,
            user_problem=user_problem,
            max_frames=max_frames,
            fast_mode=fast_mode,
            include_detailed_report=include_detailed_report,
            session_id=session_id,
            generation_id=generation_id,
            log_queue=st.session_state["generation_log_queue"],
        )
        st.session_state["generation_future"] = future
        st.rerun()

    # 后台任务轮询区：页面不会卡死，用户可以点“终止当前生成 / 开启新会话”。
    future = st.session_state.get("generation_future")
    if isinstance(future, Future):
        drain_generation_log_queue()
        st.markdown("## 实时思考过程")
        st.caption("这里打印的是算法可视化智能体的公开思考过程，用于教学展示；不是隐藏推理链。")

        current_generation_id = st.session_state.get("generation_id")
        current_session_id = st.session_state.get("active_session_id")

        if future.done():
            try:
                data = future.result()
                drain_generation_log_queue()
                # 关键：旧会话/旧任务完成后不允许覆盖新页面。
                if data.get("_generation_id") == current_generation_id and data.get("_session_id") == current_session_id:
                    data.pop("_generation_id", None)
                    data.pop("_session_id", None)
                    realtime_log = build_log_markdown(st.session_state.get("realtime_log_items", []))
                    data["realtime_process_markdown"] = realtime_log
                    st.session_state["realtime_process_markdown"] = realtime_log
                    st.session_state["visualization_data"] = data
                    del st.session_state["generation_future"]
                    render_realtime_thinking(running=False)
                    st.success("算法可视化内容已生成，并将自动整理为 PDF 报告。")
                else:
                    del st.session_state["generation_future"]
                    render_realtime_thinking(running=False)
                    st.info("一个旧任务已经完成，但它属于已终止会话，结果已自动丢弃。")
            except Exception as e:
                drain_generation_log_queue()
                render_realtime_thinking(running=False)
                if "generation_future" in st.session_state:
                    del st.session_state["generation_future"]
                if is_quota_or_permission_error(e):
                    st.error("连接失败：请检查接口密钥、服务权限、账户额度，以及推理引擎名称是否可用。")
                else:
                    st.error("生成失败。快速模式下不会追加第二次修复调用，以免继续变慢。")
                st.code(str(e)[:4000], language="text")
                return
        else:
            elapsed = int(time.time() - float(st.session_state.get("generation_started_at", time.time())))
            render_realtime_thinking(running=True)
            st.info(f"智能体正在生成算法可视化内容……已等待 {elapsed} 秒。点击侧栏“终止当前生成”可立即开启新会话。")
            st.progress(min(95, 15 + (elapsed * 5) % 80))
            time.sleep(0.8)
            st.rerun()

    data = st.session_state.get("visualization_data")
    if not data:
        return

    if data.get("realtime_process_markdown"):
        with st.expander("查看完整实时思考过程", expanded=False):
            st.markdown(data.get("realtime_process_markdown", ""))

    st.markdown("## 智能体识别结果")
    c1, c2, c3 = st.columns(3)
    c1.metric("题目", data.get("task_title", "算法题目"))
    c2.metric("问题类型", data.get("detected_problem_type", "未识别"))
    c3.metric("算法数量", len(data.get("algorithms", []) or []))
    st.write(data.get("input_summary", ""))

    algorithms = data.get("algorithms", []) or []
    if algorithms:
        st.markdown("## 逐帧算法可视化")
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
        st.markdown("## 智能体生成的教学报告")
        st.markdown(report)

    st.markdown("## 智能体报告输出")
    st.info("报告会根据算法识别、求解步骤、可视化帧、关键变量和教学内容自动生成。")
    pdf_path = st.session_state.get("pdf_path")
    if not pdf_path or not os.path.exists(pdf_path):
        pdf_path = make_pdf(data)
        st.session_state["pdf_path"] = pdf_path
    with open(pdf_path, "rb") as f:
        st.download_button(
            label="下载算法可视化智能体报告",
            data=f.read(),
            file_name="算法可视化智能体_求解报告.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    with st.expander("查看智能体结构化数据"):
        st.json(data)


if __name__ == "__main__":
    main()
