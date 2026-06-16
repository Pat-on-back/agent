# -*- coding: utf-8 -*-
"""
算法过程可视化 - 即时可视化版
运行：streamlit run app.py

说明：
1. 在下方代码配置区填写 API Key。
2. 页面不展示“模型 / JSON / 接口”等实现细节，只展示算法过程可视化智能体的工作过程。
3. 智能体一次只处理用户输入中指定的一个算法；若未指定，则自动选择一种最合适算法。
4. 每完成一个算法步骤，就立即在页面渲染一帧可视化内容。
5. PDF 不再等待额外长文本生成，而是直接由已展示的过程帧自动整理得到。
"""

import os
import re
import html
import time
from io import BytesIO
from typing import Dict, List, Tuple

import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# =========================================================
# 代码配置区：把你的 Key 填在这里
# =========================================================
CODE_OPENAI_API_KEY = ""  # 例如："sk-xpdideuepwpxalyhosclmczxzfbydlqelgarxlthigpimjfd"
CODE_OPENAI_BASE_URL = "https://api.siliconflow.cn/v1"
CODE_OPENAI_MODEL = "nex-agi/Nex-N2-Pro"

# 每次只生成一个算法的可视化过程，帧数不宜太多，保证录屏稳定
MAX_FRAMES = 7
MAX_TOKENS = 2600
TEMPERATURE = 0.2


# =========================================================
# 页面基础设置
# =========================================================
st.set_page_config(
    page_title="算法过程可视化",
    page_icon="🧩",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container { padding-top: 2rem; max-width: 1180px; }
.main-title { font-size: 2.25rem; font-weight: 800; margin-bottom: .2rem; }
.sub-title { color: #555; font-size: 1rem; margin-bottom: 1.2rem; }
.agent-card {
    border: 1px solid #e6e8ec;
    border-radius: 14px;
    padding: 16px 18px;
    margin: 10px 0 14px 0;
    background: #ffffff;
    box-shadow: 0 2px 8px rgba(0,0,0,0.035);
}
.frame-card {
    border: 1px solid #dfe4ea;
    border-left: 6px solid #4f6bed;
    border-radius: 14px;
    padding: 16px 18px;
    margin: 16px 0;
    background: #fbfcff;
}
.frame-title { font-size: 1.18rem; font-weight: 700; margin-bottom: 8px; }
.tag {
    display:inline-block;
    padding: 3px 9px;
    margin: 2px 4px 2px 0;
    border-radius: 999px;
    background: #eef2ff;
    color: #2f3a8f;
    font-size: .82rem;
}
.array-wrap { display:flex; align-items:flex-end; gap:8px; min-height: 150px; padding: 12px 4px; }
.array-bar-box { display:flex; flex-direction:column; align-items:center; gap:4px; }
.array-bar {
    width:34px;
    background: linear-gradient(180deg, #6f8cff, #3b5bdb);
    border-radius: 8px 8px 3px 3px;
    min-height: 12px;
}
.array-label { font-size:.82rem; color:#333; }
.status-line { color:#444; padding: 4px 0; }
.small-muted { color:#666; font-size:.9rem; }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# 工具函数
# =========================================================
def read_secret(name: str, default: str = "") -> str:
    """兼容代码配置、环境变量和 Streamlit Secrets。"""
    if name == "OPENAI_API_KEY" and CODE_OPENAI_API_KEY.strip():
        return CODE_OPENAI_API_KEY.strip()
    if name == "OPENAI_BASE_URL" and CODE_OPENAI_BASE_URL.strip():
        return CODE_OPENAI_BASE_URL.strip()
    if name == "OPENAI_MODEL" and CODE_OPENAI_MODEL.strip():
        return CODE_OPENAI_MODEL.strip()

    env_val = os.getenv(name, "").strip()
    if env_val:
        return env_val
    try:
        sec_val = str(st.secrets.get(name, "")).strip()
        if sec_val:
            return sec_val
    except Exception:
        pass
    return default


def get_client() -> Tuple[OpenAI, str]:
    api_key = read_secret("OPENAI_API_KEY")
    base_url = read_secret("OPENAI_BASE_URL", "https://api.siliconflow.cn/v1")
    model = read_secret("OPENAI_MODEL", "nex-agi/Nex-N2-Pro")
    if not api_key:
        st.error("未配置访问密钥。请在 app.py 顶部 CODE_OPENAI_API_KEY 中填写。")
        st.stop()
    return OpenAI(api_key=api_key, base_url=base_url), model


def safe_text(v) -> str:
    return "" if v is None else str(v).strip()


def extract_between(text: str, start_pat: str, end_pat: str = None) -> str:
    if end_pat:
        pattern = start_pat + r"(.*?)" + end_pat
        m = re.search(pattern, text, flags=re.S | re.I)
        return m.group(1).strip() if m else ""
    m = re.search(start_pat + r"(.*)", text, flags=re.S | re.I)
    return m.group(1).strip() if m else ""


def parse_key(frame_text: str, key: str, next_keys: List[str]) -> str:
    """解析形如 KEY: xxx 的字段，兼容多行。"""
    key_pat = rf"{re.escape(key)}\s*:\s*"
    next_pat = "|".join([rf"\n{re.escape(k)}\s*:" for k in next_keys])
    if next_pat:
        m = re.search(key_pat + rf"(.*?)(?={next_pat}|\Z)", frame_text, flags=re.S | re.I)
    else:
        m = re.search(key_pat + r"(.*)", frame_text, flags=re.S | re.I)
    return m.group(1).strip() if m else ""


def parse_frame(frame_text: str) -> Dict[str, str]:
    keys = [
        "TITLE", "STEP", "PUBLIC_PROCESS", "VARIABLES", "VISUAL_TYPE",
        "VISUAL_DATA", "EXPLANATION", "PSEUDOCODE"
    ]
    visual_block = ""
    m = re.search(r"VISUAL_DATA\s*:\s*```(?:\w+)?\s*\n(.*?)\n```", frame_text, flags=re.S | re.I)
    if m:
        visual_block = m.group(1).strip()
    else:
        visual_block = parse_key(frame_text, "VISUAL_DATA", ["EXPLANATION", "PSEUDOCODE"])

    data = {
        "title": parse_key(frame_text, "TITLE", keys[1:]) or "算法步骤",
        "step": parse_key(frame_text, "STEP", keys[2:]),
        "public_process": parse_key(frame_text, "PUBLIC_PROCESS", keys[3:]),
        "variables": parse_key(frame_text, "VARIABLES", keys[4:]),
        "visual_type": parse_key(frame_text, "VISUAL_TYPE", keys[5:]).lower() or "none",
        "visual_data": visual_block,
        "explanation": parse_key(frame_text, "EXPLANATION", ["PSEUDOCODE"]),
        "pseudocode": parse_key(frame_text, "PSEUDOCODE", []),
        "raw": frame_text.strip(),
    }
    return data


def collect_completed_frames(buffer: str) -> Tuple[List[Dict[str, str]], str]:
    """从流式文本中抽取已经结束的帧，返回 frames 和剩余未完成文本。"""
    frames = []
    while "<<<FRAME_START>>>" in buffer and "<<<FRAME_END>>>" in buffer:
        start = buffer.find("<<<FRAME_START>>>")
        end = buffer.find("<<<FRAME_END>>>", start)
        if end == -1:
            break
        body = buffer[start + len("<<<FRAME_START>>>"):end]
        frames.append(parse_frame(body))
        buffer = buffer[end + len("<<<FRAME_END>>>"):]
    return frames, buffer


def extract_report(text: str) -> str:
    if "<<<REPORT_START>>>" in text and "<<<REPORT_END>>>" in text:
        return extract_between(text, r"<<<REPORT_START>>>", r"<<<REPORT_END>>>")
    return ""


def render_mermaid(code: str, height: int = 360):
    code = code.strip()
    if not code:
        return
    escaped = html.escape(code)
    components.html(
        f"""
        <div class="mermaid">{escaped}</div>
        <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
        <script>
            mermaid.initialize({{startOnLoad: true, theme: 'default', securityLevel: 'loose'}});
        </script>
        """,
        height=height,
        scrolling=True,
    )


def parse_numbers(text: str) -> List[float]:
    nums = []
    for x in re.findall(r"-?\d+(?:\.\d+)?", text):
        try:
            nums.append(float(x))
        except Exception:
            pass
    return nums


def render_array_visual(data: str):
    nums = parse_numbers(data)
    if not nums:
        st.code(data)
        return
    max_abs = max(abs(x) for x in nums) or 1
    bars = []
    for idx, val in enumerate(nums):
        h = max(12, int(abs(val) / max_abs * 130))
        label = int(val) if float(val).is_integer() else val
        bars.append(
            f"""
            <div class="array-bar-box">
                <div class="array-bar" style="height:{h}px"></div>
                <div class="array-label">{html.escape(str(label))}</div>
                <div class="array-label">i={idx}</div>
            </div>
            """
        )
    st.markdown(f"<div class='array-wrap'>{''.join(bars)}</div>", unsafe_allow_html=True)


def looks_like_mermaid(data: str) -> bool:
    s = data.strip().lower()
    return s.startswith("graph ") or s.startswith("flowchart ") or s.startswith("sequencediagram") or s.startswith("statediagram")


def render_visual(frame: Dict[str, str]):
    vtype = safe_text(frame.get("visual_type", "none")).lower()
    vdata = safe_text(frame.get("visual_data", ""))
    if not vdata:
        st.info("本步骤以文字解释为主。")
        return

    if vtype in ["array", "bar", "数组"]:
        render_array_visual(vdata)
    elif vtype in ["tree", "graph", "flow", "mermaid", "流程图", "树图"] or looks_like_mermaid(vdata):
        render_mermaid(vdata)
        with st.expander("查看图结构源码"):
            st.code(vdata)
    elif vtype in ["table", "dp_table", "matrix", "表格", "状态表"]:
        st.markdown(vdata)
    elif vtype in ["list", "state", "text", "列表"]:
        st.markdown(vdata)
    else:
        # 自动判断：表格 / 图 / 数组 / 普通文本
        if "|" in vdata and "---" in vdata:
            st.markdown(vdata)
        elif looks_like_mermaid(vdata):
            render_mermaid(vdata)
        elif len(parse_numbers(vdata)) >= 3 and "[" in vdata:
            render_array_visual(vdata)
        else:
            st.markdown(vdata)


def render_frame(frame: Dict[str, str], index: int):
    title = html.escape(frame.get("title", f"第 {index} 步"))
    st.markdown(
        f"""
        <div class="frame-card">
            <div class="frame-title">第 {index} 步：{title}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns([1.1, 1.2])
    with cols[0]:
        if frame.get("step"):
            st.markdown("**当前操作**")
            st.write(frame["step"])
        if frame.get("public_process"):
            st.markdown("**过程说明**")
            st.write(frame["public_process"])
        if frame.get("variables"):
            st.markdown("**关键变量**")
            # 变量通常是列表，直接渲染即可
            st.markdown(frame["variables"])
        if frame.get("pseudocode"):
            st.markdown("**伪代码焦点**")
            st.code(frame["pseudocode"])
    with cols[1]:
        st.markdown("**可视化内容**")
        render_visual(frame)
    if frame.get("explanation"):
        st.markdown("**学习提示**")
        st.info(frame["explanation"])


def wrap_cjk(text: str, width: int = 42) -> List[str]:
    """简单中文换行。"""
    text = str(text).replace("\r", "")
    lines = []
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            lines.append("")
            continue
        while len(para) > width:
            lines.append(para[:width])
            para = para[width:]
        lines.append(para)
    return lines


def build_pdf(user_question: str, frames: List[Dict[str, str]], report_text: str = "") -> bytes:
    """根据已展示帧生成 PDF。无需再次请求长报告。"""
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    w, h = A4
    left = 52
    y = h - 54

    def new_page():
        nonlocal y
        c.showPage()
        c.setFont("STSong-Light", 11)
        y = h - 54

    def draw_line(line: str, size: int = 11, leading: int = 17):
        nonlocal y
        if y < 54:
            new_page()
        c.setFont("STSong-Light", size)
        c.drawString(left, y, line)
        y -= leading

    def draw_paragraph(text: str, size: int = 11, width_chars: int = 42):
        for line in wrap_cjk(text, width_chars):
            draw_line(line, size=size)

    c.setFont("STSong-Light", 18)
    c.drawString(left, y, "算法过程可视化求解报告")
    y -= 34

    draw_line("一、用户输入题目", size=14, leading=22)
    draw_paragraph(user_question, width_chars=40)
    y -= 8

    draw_line("二、智能体求解过程与可视化步骤", size=14, leading=22)
    if not frames:
        draw_paragraph("未解析到结构化可视化帧。请重新生成或简化题目。")
    for i, f in enumerate(frames, 1):
        draw_line(f"第 {i} 步：{f.get('title','算法步骤')}", size=13, leading=22)
        if f.get("step"):
            draw_paragraph("当前操作：" + f["step"])
        if f.get("public_process"):
            draw_paragraph("过程说明：" + f["public_process"])
        if f.get("variables"):
            draw_paragraph("关键变量：" + f["variables"].replace("\n", "；"))
        if f.get("visual_data"):
            draw_paragraph("可视化数据：" + f["visual_data"].replace("\n", " | "), width_chars=40)
        if f.get("explanation"):
            draw_paragraph("学习提示：" + f["explanation"])
        y -= 6

    if report_text.strip():
        draw_line("三、结果总结", size=14, leading=22)
        draw_paragraph(report_text.strip(), width_chars=40)

    draw_line("四、说明", size=14, leading=22)
    draw_paragraph("本报告由算法过程可视化智能体根据页面中已经展示的求解过程自动整理生成，内容包括题目分析、步骤拆解、关键变量、可视化数据和学习提示。")

    c.save()
    return buffer.getvalue()


def make_prompt(user_question: str) -> str:
    return f"""
你是“算法过程可视化”智能体。你的目标是把用户输入的算法题目转化为清晰、可读、可视化的求解过程。

重要要求：
1. 一次只处理一个算法。用户指定了哪个算法，就只处理那个算法；不要扩展到其他算法。
2. 如果用户没有指定算法，只选择最适合该题的一种算法。
3. 不要只说“生成完成”，必须输出每一步的具体内容。
4. 每完成一步，就输出一个完整的可视化帧。
5. 不要输出内部隐藏思维链；只输出可公开展示的题目分析、求解过程和可视化设计说明。
6. 不要提到“模型、接口、JSON、调用”等实现细节。
7. 每一帧必须包含可视化数据。动态规划给表格；排序给数组；回溯/图算法给 Mermaid 图；搜索算法给状态列表或树图。
8. 最多输出 {MAX_FRAMES} 帧，保证速度。每帧控制在简洁范围内。
9. 最后输出一个简短结果总结，不要写长篇报告。

用户题目：
{user_question}

请严格按照下面协议输出。注意：每个可视化帧一完成就立刻以 <<<FRAME_END>>> 结束。

<<<FRAME_START>>>
TITLE: 这一帧标题
STEP: 当前正在执行的算法步骤
PUBLIC_PROCESS: 用公开、简洁的话说明为什么这样做
VARIABLES:
- 变量1 = 值，含义
- 变量2 = 值，含义
VISUAL_TYPE: table / array / tree / graph / flow / list / none 中选择一种
VISUAL_DATA:
```text
这里放可视化数据。
如果是动态规划，用 Markdown 表格。
如果是排序，用数组，如 [8, 3, 5, 1]。
如果是回溯、树、图或流程，用 Mermaid，如 graph TD。
如果是搜索状态，用列表。
```
EXPLANATION: 这一帧帮助学习者理解的关键点
PSEUDOCODE: 与当前步骤对应的一小段伪代码
<<<FRAME_END>>>

最后输出：
<<<REPORT_START>>>
用 200 到 400 字总结算法思想、关键步骤、最终结果和学习价值。
<<<REPORT_END>>>
""".strip()


def stream_agent(user_question: str):
    client, model = get_client()

    st.markdown("## 智能体实时求解过程")
    st.caption("展示智能体对题目的分析、步骤拆解和可视化生成过程。每完成一步会立即显示对应内容。")

    status_box = st.empty()
    progress = st.progress(0)
    live_box = st.empty()

    st.markdown("## 过程可视化")
    frame_container = st.container()

    status_messages = [
        "正在解析题目描述……",
        "正在识别目标算法……",
        "正在提取关键输入信息……",
        "正在拆解核心步骤……",
        "正在生成第 1 个可视化步骤……",
    ]

    buffer = ""
    full_text = ""
    frames: List[Dict[str, str]] = []
    report_text = ""
    last_status_index = 0

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": make_prompt(user_question)},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        for chunk in stream:
            delta = ""
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if not delta:
                continue

            full_text += delta
            buffer += delta

            # 实时状态，不等待最终长报告
            new_frames, buffer = collect_completed_frames(buffer)
            if new_frames:
                for f in new_frames:
                    frames.append(f)
                    with frame_container:
                        render_frame(f, len(frames))
                    last_status_index = min(4 + len(frames), 8)
                    status_box.success(f"已生成第 {len(frames)} 步可视化内容：{f.get('title', '算法步骤')}")
                    progress.progress(min(95, 20 + len(frames) * 12))

            # 显示当前正在生成的片段摘要，而不是只显示“生成完成”
            preview = buffer[-900:].replace("<<<FRAME_START>>>", "").replace("<<<FRAME_END>>>", "")
            if preview.strip():
                live_box.markdown("**正在生成当前步骤：**\n\n" + preview + "\n\n▌")
            elif len(frames) == 0:
                status_box.info(status_messages[min(last_status_index, len(status_messages)-1)])
                progress.progress(12)

        # 流结束后处理剩余内容
        more_frames, buffer = collect_completed_frames(buffer)
        if more_frames:
            for f in more_frames:
                frames.append(f)
                with frame_container:
                    render_frame(f, len(frames))

        report_text = extract_report(full_text)
        progress.progress(100)
        status_box.success("算法过程可视化生成完成。")
        live_box.empty()

    except Exception as e:
        st.error("生成过程遇到问题，请检查访问密钥、服务额度或题目长度。")
        st.code(str(e))
        return [], "", full_text

    # 如果协议没有被遵守，至少把实时文本展示出来，避免“只有完成文字”
    if not frames:
        st.warning("未解析到标准可视化帧，下面展示智能体返回的原始过程内容。建议重新点击生成，或将题目写得更明确。")
        st.markdown(full_text)

    return frames, report_text, full_text


# =========================================================
# 页面主体
# =========================================================
st.markdown('<div class="main-title">算法过程可视化</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">输入算法题目，智能体将自动拆解求解步骤、渲染关键状态变化，并输出可下载报告。</div>',
    unsafe_allow_html=True,
)

with st.container():
    st.markdown("### 请输入算法题目")
    default_question = "请用动态规划可视化求解 01 背包问题。背包容量 15，物品 A 重量2 价值6，B 重量3 价值10，C 重量4 价值12，D 重量5 价值14，E 重量9 价值20，F 重量7 价值18。"
    user_question = st.text_area(
        "题目内容",
        value=default_question,
        height=140,
        label_visibility="collapsed",
    )

col_a, col_b = st.columns([1, 3])
with col_a:
    run_btn = st.button("生成算法过程可视化", type="primary", use_container_width=True)
with col_b:
    st.markdown("<span class='small-muted'>建议一次只输入一个算法，例如：动态规划、回溯法、Dijkstra、快速排序、二分查找。</span>", unsafe_allow_html=True)

if run_btn:
    if not user_question.strip():
        st.warning("请先输入算法题目。")
        st.stop()

    st.markdown("## 题目分析")
    st.markdown(f"<div class='agent-card'>{html.escape(user_question)}</div>", unsafe_allow_html=True)

    frames, report_text, raw_text = stream_agent(user_question)

    st.markdown("## 结果总结")
    if report_text.strip():
        st.markdown(report_text)
    elif frames:
        st.write("已完成算法过程拆解与可视化渲染。可下载下方报告查看完整过程。")
    else:
        st.write("本次未生成结构化可视化内容，请调整题目后重试。")

    st.markdown("## 详细报告")
    pdf_bytes = build_pdf(user_question, frames, report_text or raw_text[:800])
    st.download_button(
        label="下载详细 PDF 报告",
        data=pdf_bytes,
        file_name="算法过程可视化报告.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

    with st.expander("查看可复制的过程文本"):
        if report_text:
            st.markdown(report_text)
        if raw_text:
            st.code(raw_text[:8000])

else:
    st.markdown("## 使用说明")
    st.markdown(
        """
1. 在上方输入一个算法题目，建议明确写出要使用的算法。  
2. 点击“生成算法过程可视化”。  
3. 页面会在每一步完成后立即显示对应的可视化内容。  
4. 最后点击“下载详细 PDF 报告”。  

示例：  
- 请用快速排序可视化数组 `[8, 3, 5, 1, 9, 6]` 的排序过程。  
- 请用回溯法可视化求解 4 皇后问题。  
- 请用 Dijkstra 算法可视化求解从 A 到各节点的最短路径。  
- 请用动态规划可视化求解 01 背包问题。  
"""
    )
