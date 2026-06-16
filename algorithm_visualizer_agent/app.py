import ast
import io
import json
import math
import os
import re
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import networkx as nx
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

try:
    from openai import OpenAI
except Exception:  # deployment can still show deterministic demo without API
    OpenAI = None


# -----------------------------
# 1. Basic configuration
# -----------------------------
st.set_page_config(
    page_title="算法过程可视化智能体",
    page_icon="🧠",
    layout="wide",
)

ALGO_LABELS = {
    "auto": "自动识别",
    "quick_sort": "快速排序 Quick Sort",
    "bfs": "广度优先搜索 BFS",
    "dfs": "深度优先搜索 DFS",
    "dijkstra": "Dijkstra 最短路径",
    "binary_search": "二分查找 Binary Search",
}

DEFAULT_PROBLEM = "请用可视化方式解释快速排序如何把数组 [8, 3, 5, 1, 9, 6, 2, 7] 排序。"
DEFAULT_ARRAY = "[8, 3, 5, 1, 9, 6, 2, 7]"
DEFAULT_GRAPH = "A-B:2, A-C:5, B-C:1, B-D:4, C-D:2, C-E:3, D-E:1"

SYSTEM_PROMPT = """
你是一个算法过程可视化智能体。你的目标不是只给答案，而是把算法运行过程拆解为可观察的步骤，帮助学习者快速理解核心思想。
你需要遵守：
1. 先识别问题对应的算法；
2. 用简短语言说明算法核心思想；
3. 围绕每一步的状态变化解释为什么这样做；
4. 输出适合教学的讲解，而不是堆砌定义；
5. 不调用任何内部未公开模型，只基于公开可访问的大模型 API 或网页端能力。
"""


@dataclass
class Step:
    title: str
    description: str
    state: Dict[str, Any]


# -----------------------------
# 2. Utility functions
# -----------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        return value if value is not None else default
    except Exception:
        return os.environ.get(name, default)


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}
        return {}


def parse_int_list(text: str) -> List[int]:
    """Parse an integer array from free text."""
    try:
        obj = ast.literal_eval(text.strip())
        if isinstance(obj, list):
            values = [int(x) for x in obj]
            if values:
                return values[:30]
    except Exception:
        pass
    nums = re.findall(r"-?\d+", text)
    values = [int(x) for x in nums]
    return values[:30] if values else [8, 3, 5, 1, 9, 6, 2, 7]


def parse_graph(text: str) -> Tuple[nx.Graph, str, str]:
    """Parse graph edges like A-B:2, A C 5. Return graph, start, target."""
    G = nx.Graph()
    edge_chunks = re.split(r"[,，;；\n]+", text)
    for chunk in edge_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r"([A-Za-z0-9]+)\s*[-—>]\s*([A-Za-z0-9]+)(?:\s*[:：=]\s*(-?\d+))?", chunk)
        if not m:
            m = re.match(r"([A-Za-z0-9]+)\s+([A-Za-z0-9]+)(?:\s+(-?\d+))?", chunk)
        if m:
            u, v = m.group(1), m.group(2)
            w = int(m.group(3)) if m.group(3) else 1
            G.add_edge(u, v, weight=w)
    if G.number_of_edges() == 0:
        G.add_weighted_edges_from([
            ("A", "B", 2), ("A", "C", 5), ("B", "C", 1),
            ("B", "D", 4), ("C", "D", 2), ("C", "E", 3), ("D", "E", 1)
        ])
    nodes = sorted(G.nodes())
    start = nodes[0]
    target = nodes[-1]
    return G, start, target


def classify_algorithm(problem: str, selected: str, api_conf: Dict[str, str]) -> str:
    if selected != "auto":
        return selected

    lower = problem.lower()
    rules = [
        ("quick_sort", ["快速排序", "quicksort", "quick sort", "partition", "划分"]),
        ("bfs", ["bfs", "广度优先", "层序", "最短步数"]),
        ("dfs", ["dfs", "深度优先", "回溯", "递归遍历"]),
        ("dijkstra", ["dijkstra", "最短路径", "加权图", "单源最短"]),
        ("binary_search", ["二分", "binary search", "折半查找"]),
    ]
    for algo, keys in rules:
        if any(k in lower or k in problem for k in keys):
            return algo

    # Optional LLM classification if configured
    llm_answer = call_llm(
        api_conf,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "从 quick_sort, bfs, dfs, dijkstra, binary_search 中选择最匹配的算法，只输出 JSON，如 {\"algorithm\": \"bfs\"}。问题：" + problem},
        ],
        temperature=0,
    )
    if llm_answer:
        data = safe_json_loads(llm_answer)
        if data.get("algorithm") in ALGO_LABELS:
            return data["algorithm"]
    return "quick_sort"


def call_llm(api_conf: Dict[str, str], messages: List[Dict[str, str]], temperature: float = 0.2) -> str:
    api_key = api_conf.get("api_key", "").strip()
    base_url = api_conf.get("base_url", "").strip()
    model = api_conf.get("model", "").strip()
    if not api_key or not base_url or not model or OpenAI is None:
        return ""
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""
    except Exception as e:
        return f"LLM 调用失败，已切换为本地规则演示模式。错误信息：{e}"


# -----------------------------
# 3. Trace generators as lightweight tools
# -----------------------------
def quicksort_trace(values: List[int]) -> List[Step]:
    arr = values[:]
    steps: List[Step] = []

    def add(title: str, desc: str, **state):
        steps.append(Step(title, desc, {"array": arr[:], **state}))

    add("初始化", f"待排序数组为 {arr}。快速排序会选择枢轴 pivot，把小于等于 pivot 的元素放左边，大的放右边。")

    def partition(lo: int, hi: int) -> int:
        pivot = arr[hi]
        i = lo - 1
        add("开始划分", f"处理区间 [{lo}, {hi}]，选择末尾元素 {pivot} 作为 pivot。", lo=lo, hi=hi, pivot=hi, i=i, j=lo)
        for j in range(lo, hi):
            add("比较元素", f"比较 arr[{j}]={arr[j]} 与 pivot={pivot}。", lo=lo, hi=hi, pivot=hi, i=i, j=j)
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
                add("交换到左区间", f"因为 {arr[i]} <= {pivot}，把它交换到小元素区间末尾。", lo=lo, hi=hi, pivot=hi, i=i, j=j, swap=(i, j))
        arr[i + 1], arr[hi] = arr[hi], arr[i + 1]
        add("pivot 归位", f"将 pivot={pivot} 放到最终位置 {i + 1}，左侧不大于它，右侧大于它。", lo=lo, hi=hi, pivot=i + 1, i=i + 1, j=hi, swap=(i + 1, hi))
        return i + 1

    def qs(lo: int, hi: int):
        if lo < hi:
            p = partition(lo, hi)
            add("递归分治", f"pivot 已固定在位置 {p}，接下来分别处理左区间 [{lo}, {p-1}] 和右区间 [{p+1}, {hi}]。", lo=lo, hi=hi, pivot=p)
            qs(lo, p - 1)
            qs(p + 1, hi)
        elif lo == hi:
            add("单元素区间", f"位置 {lo} 只有一个元素，天然有序。", lo=lo, hi=hi, pivot=lo)

    qs(0, len(arr) - 1)
    add("排序完成", f"最终数组为 {arr}。", sorted=True)
    return steps[:160]


def binary_search_trace(values: List[int], target: int) -> List[Step]:
    arr = sorted(values)
    left, right = 0, len(arr) - 1
    steps = [Step("初始化", f"先保证数组有序：{arr}。目标值 target={target}。", {"array": arr[:], "left": left, "right": right, "target": target})]
    while left <= right:
        mid = (left + right) // 2
        steps.append(Step("检查中点", f"当前区间 [{left}, {right}]，中点 mid={mid}，arr[mid]={arr[mid]}。", {"array": arr[:], "left": left, "right": right, "mid": mid, "target": target}))
        if arr[mid] == target:
            steps.append(Step("查找成功", f"arr[{mid}] 正好等于 {target}，查找结束。", {"array": arr[:], "left": left, "right": right, "mid": mid, "target": target, "found": mid}))
            return steps
        if arr[mid] < target:
            left = mid + 1
            steps.append(Step("舍弃左半区", f"因为 {arr[mid]} < {target}，目标只能在右半区。", {"array": arr[:], "left": left, "right": right, "target": target}))
        else:
            right = mid - 1
            steps.append(Step("舍弃右半区", f"因为 {arr[mid]} > {target}，目标只能在左半区。", {"array": arr[:], "left": left, "right": right, "target": target}))
    steps.append(Step("查找失败", f"区间为空，数组中不存在 {target}。", {"array": arr[:], "left": left, "right": right, "target": target}))
    return steps


def bfs_trace(G: nx.Graph, start: str) -> List[Step]:
    visited = set([start])
    queue = [start]
    order = []
    steps = [Step("初始化", f"从节点 {start} 出发，把它加入队列。BFS 按层推进。", {"current": start, "visited": sorted(visited), "frontier": queue[:], "order": order[:]})]
    while queue:
        u = queue.pop(0)
        order.append(u)
        steps.append(Step("出队访问", f"取出队首节点 {u}，访问它的所有未访问邻居。", {"current": u, "visited": sorted(visited), "frontier": queue[:], "order": order[:]}))
        for v in sorted(G.neighbors(u)):
            if v not in visited:
                visited.add(v)
                queue.append(v)
                steps.append(Step("发现新节点", f"从 {u} 发现未访问节点 {v}，将 {v} 加入队尾。", {"current": u, "new": v, "visited": sorted(visited), "frontier": queue[:], "order": order[:]}))
    steps.append(Step("遍历完成", f"BFS 访问顺序为：{' -> '.join(order)}。", {"visited": sorted(visited), "frontier": [], "order": order[:]}))
    return steps


def dfs_trace(G: nx.Graph, start: str) -> List[Step]:
    visited = set()
    stack = [start]
    order = []
    steps = [Step("初始化", f"从节点 {start} 出发，把它压入栈。DFS 会尽量沿一条路走到底。", {"current": start, "visited": [], "frontier": stack[:], "order": []})]
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        order.append(u)
        steps.append(Step("弹栈访问", f"弹出节点 {u} 并访问。", {"current": u, "visited": sorted(visited), "frontier": stack[:], "order": order[:]}))
        for v in sorted(G.neighbors(u), reverse=True):
            if v not in visited:
                stack.append(v)
                steps.append(Step("压入邻居", f"节点 {v} 尚未访问，将它压入栈，后续继续深入。", {"current": u, "new": v, "visited": sorted(visited), "frontier": stack[:], "order": order[:]}))
    steps.append(Step("遍历完成", f"DFS 访问顺序为：{' -> '.join(order)}。", {"visited": sorted(visited), "frontier": [], "order": order[:]}))
    return steps


def dijkstra_trace(G: nx.Graph, start: str, target: str) -> List[Step]:
    dist = {n: math.inf for n in G.nodes()}
    prev = {n: None for n in G.nodes()}
    dist[start] = 0
    unvisited = set(G.nodes())
    steps = [Step("初始化", f"从源点 {start} 出发，令 dist[{start}]=0，其余节点距离为无穷大。", {"current": start, "visited": [], "frontier": sorted(unvisited), "dist": dist.copy(), "prev": prev.copy()})]

    while unvisited:
        u = min(unvisited, key=lambda n: dist[n])
        if math.isinf(dist[u]):
            break
        unvisited.remove(u)
        steps.append(Step("选择最近未确定节点", f"在未确定节点中选择距离最小的 {u}，当前 dist={dist[u]}。", {"current": u, "visited": sorted(set(G.nodes()) - unvisited), "frontier": sorted(unvisited), "dist": dist.copy(), "prev": prev.copy()}))
        if u == target:
            steps.append(Step("到达目标", f"目标节点 {target} 的最短距离已确定，为 {dist[target]}。", {"current": u, "visited": sorted(set(G.nodes()) - unvisited), "frontier": sorted(unvisited), "dist": dist.copy(), "prev": prev.copy(), "target": target}))
            break
        for v in sorted(G.neighbors(u)):
            if v not in unvisited:
                continue
            w = G[u][v].get("weight", 1)
            cand = dist[u] + w
            steps.append(Step("松弛边", f"检查边 {u}-{v}，候选距离 dist[{u}] + w = {dist[u]} + {w} = {cand}。", {"current": u, "new": v, "visited": sorted(set(G.nodes()) - unvisited), "frontier": sorted(unvisited), "dist": dist.copy(), "prev": prev.copy(), "edge": (u, v)}))
            if cand < dist[v]:
                dist[v] = cand
                prev[v] = u
                steps.append(Step("更新距离", f"候选距离更短，更新 dist[{v}]={cand}，前驱节点为 {u}。", {"current": u, "new": v, "visited": sorted(set(G.nodes()) - unvisited), "frontier": sorted(unvisited), "dist": dist.copy(), "prev": prev.copy(), "edge": (u, v)}))
    path = []
    cur = target
    while cur is not None and cur in prev:
        path.append(cur)
        cur = prev[cur]
    path = list(reversed(path)) if path and path[-1] == target else []
    steps.append(Step("算法结束", f"从 {start} 到 {target} 的最短路径为：{' -> '.join(path) if path else '不可达'}，距离为 {dist.get(target)}。", {"current": target, "visited": sorted(set(G.nodes()) - unvisited), "frontier": [], "dist": dist.copy(), "prev": prev.copy(), "path": path, "target": target}))
    return steps


def generate_trace(algorithm: str, data_text: str) -> Tuple[List[Step], Dict[str, Any]]:
    if algorithm in ["quick_sort", "binary_search"]:
        arr = parse_int_list(data_text)
        if algorithm == "quick_sort":
            return quicksort_trace(arr), {"type": "array", "array": arr}
        target_match = re.search(r"target\s*[:=：]?\s*(-?\d+)", data_text, re.I)
        target = int(target_match.group(1)) if target_match else sorted(arr)[len(arr) // 2]
        return binary_search_trace(arr, target), {"type": "array", "array": arr, "target": target}
    G, start, target = parse_graph(data_text)
    if algorithm == "bfs":
        return bfs_trace(G, start), {"type": "graph", "graph": G, "start": start}
    if algorithm == "dfs":
        return dfs_trace(G, start), {"type": "graph", "graph": G, "start": start}
    return dijkstra_trace(G, start, target), {"type": "graph", "graph": G, "start": start, "target": target}


# -----------------------------
# 4. Visualization renderers
# -----------------------------
def render_array_svg(step: Step, width: int = 760, height: int = 260) -> str:
    arr = step.state.get("array", [])
    n = max(len(arr), 1)
    max_val = max([abs(x) for x in arr] + [1])
    pad = 35
    gap = 8
    bar_w = (width - 2 * pad - gap * (n - 1)) / n
    baseline = height - 55
    svg = [f'<svg width="100%" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    svg.append(f'<text x="{width/2}" y="24" text-anchor="middle" font-size="18" font-weight="700" fill="#111827">{step.title}</text>')
    svg.append(f'<text x="{width/2}" y="48" text-anchor="middle" font-size="13" fill="#374151">{step.description[:95]}</text>')
    pivot = step.state.get("pivot")
    i_idx = step.state.get("i")
    j_idx = step.state.get("j")
    left = step.state.get("left")
    right = step.state.get("right")
    mid = step.state.get("mid")
    found = step.state.get("found")
    swap = step.state.get("swap")

    for idx, val in enumerate(arr):
        x = pad + idx * (bar_w + gap)
        h = max(14, abs(val) / max_val * 135)
        y = baseline - h
        fill = "#9ca3af"
        if left is not None and right is not None and left <= idx <= right:
            fill = "#60a5fa"
        if idx == pivot:
            fill = "#f97316"
        if idx == mid:
            fill = "#22c55e"
        if idx == found:
            fill = "#16a34a"
        if idx == i_idx:
            fill = "#a78bfa"
        if idx == j_idx:
            fill = "#ef4444"
        if isinstance(swap, tuple) and idx in swap:
            fill = "#facc15"
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="6" fill="{fill}" stroke="#111827" stroke-width="1"/>')
        svg.append(f'<text x="{x + bar_w/2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-size="14" fill="#111827">{val}</text>')
        svg.append(f'<text x="{x + bar_w/2:.1f}" y="{baseline + 20}" text-anchor="middle" font-size="12" fill="#4b5563">{idx}</text>')
    svg.append(f'<line x1="{pad}" y1="{baseline}" x2="{width-pad}" y2="{baseline}" stroke="#d1d5db"/>')
    legend = "橙色=pivot，紫色=i，红色=j，黄色=交换，中点/命中=绿色"
    svg.append(f'<text x="{width/2}" y="{height-16}" text-anchor="middle" font-size="12" fill="#6b7280">{legend}</text>')
    svg.append('</svg>')
    return "".join(svg)


def clean_text_for_svg(x: Any) -> str:
    return str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_graph_svg(G: nx.Graph, step: Step, width: int = 760, height: int = 420) -> str:
    pos = nx.spring_layout(G, seed=7, weight="weight")
    # normalize positions
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def map_pos(p):
        x = 80 + (p[0] - min_x) / (max_x - min_x + 1e-9) * (width - 160)
        y = 90 + (p[1] - min_y) / (max_y - min_y + 1e-9) * (height - 170)
        return x, y

    visited = set(step.state.get("visited", []))
    frontier = set(step.state.get("frontier", []))
    current = step.state.get("current")
    new = step.state.get("new")
    path = set(step.state.get("path", []))
    active_edge = step.state.get("edge")
    dist = step.state.get("dist", {})

    svg = [f'<svg width="100%" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">']
    svg.append('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>')
    svg.append(f'<text x="{width/2}" y="28" text-anchor="middle" font-size="18" font-weight="700" fill="#111827">{clean_text_for_svg(step.title)}</text>')
    svg.append(f'<text x="{width/2}" y="54" text-anchor="middle" font-size="13" fill="#374151">{clean_text_for_svg(step.description[:95])}</text>')

    for u, v, data in G.edges(data=True):
        x1, y1 = map_pos(pos[u])
        x2, y2 = map_pos(pos[v])
        stroke = "#9ca3af"
        sw = 2
        if active_edge and set(active_edge) == set((u, v)):
            stroke = "#f97316"
            sw = 5
        if u in path and v in path:
            stroke = "#16a34a"
            sw = 4
        svg.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{sw}"/>')
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        w = data.get("weight", 1)
        if w != 1:
            svg.append(f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="12" fill="#ffffff" stroke="#d1d5db"/>')
            svg.append(f'<text x="{mx:.1f}" y="{my+4:.1f}" text-anchor="middle" font-size="11" fill="#111827">{w}</text>')

    for n in sorted(G.nodes()):
        x, y = map_pos(pos[n])
        fill = "#ffffff"
        stroke = "#111827"
        if n in frontier:
            fill = "#bfdbfe"
        if n in visited:
            fill = "#bbf7d0"
        if n == current:
            fill = "#f97316"
        if n == new:
            fill = "#fde68a"
        if n in path:
            fill = "#86efac"
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="24" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        svg.append(f'<text x="{x:.1f}" y="{y+5:.1f}" text-anchor="middle" font-size="16" font-weight="700" fill="#111827">{clean_text_for_svg(n)}</text>')
        if dist and n in dist:
            d = dist[n]
            d_text = "∞" if math.isinf(d) else str(d)
            svg.append(f'<text x="{x:.1f}" y="{y+42:.1f}" text-anchor="middle" font-size="11" fill="#4b5563">d={d_text}</text>')
    svg.append(f'<text x="{width/2}" y="{height-18}" text-anchor="middle" font-size="12" fill="#6b7280">绿色=已确定/已访问，蓝色=队列/候选，橙色=当前节点，黄色=新发现节点</text>')
    svg.append('</svg>')
    return "".join(svg)


# -----------------------------
# 5. PDF report generator
# -----------------------------
def build_pdf_report(problem: str, algorithm: str, steps: List[Step], llm_explain: str, tool_log: List[Dict[str, str]]) -> bytes:
    buffer = io.BytesIO()
    try:
        pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
        cn_font = "STSong-Light"
    except Exception:
        cn_font = "Helvetica"

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CNTitle", parent=styles["Title"], fontName=cn_font, fontSize=20, leading=28, alignment=TA_CENTER
    )
    h_style = ParagraphStyle(
        "CNHeading", parent=styles["Heading2"], fontName=cn_font, fontSize=14, leading=20, spaceBefore=14, spaceAfter=8
    )
    body_style = ParagraphStyle(
        "CNBody", parent=styles["BodyText"], fontName=cn_font, fontSize=10.5, leading=16, alignment=TA_LEFT
    )
    small_style = ParagraphStyle(
        "CNSmall", parent=styles["BodyText"], fontName=cn_font, fontSize=9, leading=13
    )

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=1.6 * cm,
        leftMargin=1.6 * cm,
        topMargin=1.4 * cm,
        bottomMargin=1.4 * cm,
    )
    story = []
    story.append(Paragraph("算法过程可视化智能体 - 求解报告", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("1. 输入问题", h_style))
    story.append(Paragraph(problem, body_style))
    story.append(Paragraph("2. 识别结果", h_style))
    story.append(Paragraph(f"智能体识别出的算法为：{ALGO_LABELS.get(algorithm, algorithm)}。", body_style))
    story.append(Paragraph("3. 核心思想", h_style))
    if llm_explain:
        for para in llm_explain.split("\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))
    else:
        story.append(Paragraph("智能体根据算法轨迹，把抽象步骤转换为可观察的状态变化：当前元素/节点、候选集合、已访问集合、距离表或数组区间。", body_style))

    story.append(Paragraph("4. 可视化步骤摘要", h_style))
    table_rows = [["步骤", "动作", "说明"]]
    for idx, step in enumerate(steps[:18], 1):
        table_rows.append([str(idx), step.title, step.description[:70]])
    table = Table(table_rows, colWidths=[1.1 * cm, 3.0 * cm, 11.0 * cm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#111827")),
        ("FONTNAME", (0, 0), (-1, -1), cn_font),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d1d5db")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)

    story.append(Paragraph("5. 工具调用记录", h_style))
    for row in tool_log:
        story.append(Paragraph(f"- {row['tool']}：{row['result']}", small_style))

    story.append(Paragraph("6. 学习者可获得的理解", h_style))
    story.append(Paragraph("通过逐帧观察，学习者能够看到算法中的关键变量如何变化。例如排序中的 pivot 与左右区间、图遍历中的队列/栈和访问集合、最短路径中的距离松弛过程。相比只阅读伪代码，可视化过程能降低抽象理解难度。", body_style))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# -----------------------------
# 6. Main UI
# -----------------------------
st.title("算法过程可视化智能体")
st.caption("基于公开可访问大模型 API + 规则化轻量工具调用，把算法运行过程转换为可视化步骤和 PDF 报告。")

with st.sidebar:
    st.header("模型配置")
    provider = st.selectbox("API 提供方", ["OpenAI", "通义千问 DashScope", "豆包/火山方舟", "自定义 OpenAI 兼容接口"])
    default_base = {
        "OpenAI": "https://api.openai.com/v1",
        "通义千问 DashScope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "豆包/火山方舟": "https://ark.cn-beijing.volces.com/api/v3",
        "自定义 OpenAI 兼容接口": get_secret("OPENAI_BASE_URL", ""),
    }[provider]
    default_model = {
        "OpenAI": "gpt-4o-mini",
        "通义千问 DashScope": "qwen-plus",
        "豆包/火山方舟": "填写你的模型或推理接入点 ID",
        "自定义 OpenAI 兼容接口": get_secret("MODEL_NAME", ""),
    }[provider]
    base_url = st.text_input("Base URL", value=get_secret("OPENAI_BASE_URL", default_base))
    model = st.text_input("Model", value=get_secret("MODEL_NAME", default_model))
    api_key = st.text_input("API Key", value=get_secret("OPENAI_API_KEY", ""), type="password")
    st.info("没有 API Key 也可以运行本地规则演示；配置 API 后会自动生成更自然的教学讲解。")

api_conf = {"api_key": api_key, "base_url": base_url, "model": model}

left, right = st.columns([1.1, 0.9])
with left:
    problem = st.text_area("输入你想学习的算法问题", value=DEFAULT_PROBLEM, height=110)
    selected_label = st.selectbox("算法选择", list(ALGO_LABELS.keys()), format_func=lambda x: ALGO_LABELS[x])
with right:
    st.markdown("**输入数据**")
    st.caption("排序/二分请输入数组；图算法请输入边列表，如 A-B:2, B-C:1。")
    default_data = DEFAULT_ARRAY if selected_label in ["auto", "quick_sort", "binary_search"] else DEFAULT_GRAPH
    data_text = st.text_area("数据", value=default_data, height=110)

run = st.button("开始可视化求解并生成报告", type="primary", use_container_width=True)

if run:
    algorithm = classify_algorithm(problem, selected_label, api_conf)
    steps, data_meta = generate_trace(algorithm, data_text)
    tool_log = [
        {"tool": "classify_algorithm", "result": f"识别为 {ALGO_LABELS.get(algorithm, algorithm)}"},
        {"tool": "generate_trace", "result": f"生成 {len(steps)} 个可视化步骤"},
        {"tool": "render_visualization", "result": "根据数组或图结构生成 SVG 可视化帧"},
    ]

    explain_prompt = f"""
请为学习者解释这个算法可视化过程。
问题：{problem}
识别算法：{ALGO_LABELS.get(algorithm, algorithm)}
前 8 个步骤：
{chr(10).join([f'{i+1}. {s.title}: {s.description}' for i, s in enumerate(steps[:8])])}
要求：用中文，分为“核心思想”“看动画时关注什么”“复杂度直觉”三段，每段 2-3 句。
"""
    llm_explain = call_llm(
        api_conf,
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": explain_prompt}],
        temperature=0.3,
    )
    if not llm_explain or llm_explain.startswith("LLM 调用失败"):
        fallback = {
            "quick_sort": "核心思想：快速排序通过 pivot 把数组划分为左右两个更小问题，再递归处理。看动画时关注 pivot 归位的瞬间，因为它一旦归位就不再移动。复杂度直觉：平均情况下每次划分比较均衡，时间复杂度约为 O(n log n)；最坏情况下划分极不均衡，会退化到 O(n^2)。",
            "binary_search": "核心思想：二分查找每次比较中点，并直接舍弃不可能包含答案的一半区间。看动画时关注 left、right、mid 三个指针的变化。复杂度直觉：每一步搜索范围减半，所以时间复杂度为 O(log n)。",
            "bfs": "核心思想：BFS 使用队列按层扩展，先访问距离起点更近的节点。看动画时关注队列如何先进先出，以及节点如何一层层变为已访问。复杂度直觉：每个节点和每条边通常只处理一次，复杂度为 O(V+E)。",
            "dfs": "核心思想：DFS 使用栈或递归，沿着一条路径尽量深入，走不通再回退。看动画时关注栈的压入和弹出。复杂度直觉：每个节点和每条边通常只处理一次，复杂度为 O(V+E)。",
            "dijkstra": "核心思想：Dijkstra 每次确定当前距离最小的未确定节点，并用它更新邻居距离。看动画时关注 dist 表和松弛边的过程。复杂度直觉：朴素实现约为 O(V^2)，使用优先队列可优化到 O((V+E)logV)。",
        }
        llm_explain = fallback.get(algorithm, "智能体已生成算法步骤，可通过逐帧观察理解状态变化。")
        if llm_explain.startswith("LLM 调用失败"):
            tool_log.append({"tool": "llm_explainer", "result": llm_explain})
        else:
            tool_log.append({"tool": "llm_explainer", "result": "未配置或调用失败，使用本地教学模板"})
    else:
        tool_log.append({"tool": "llm_explainer", "result": "调用公开 API 生成教学讲解"})

    st.success(f"已生成：{ALGO_LABELS.get(algorithm, algorithm)} 的 {len(steps)} 个可视化步骤。")

    c1, c2 = st.columns([1.25, 0.75])
    with c1:
        idx = st.slider("选择步骤", min_value=0, max_value=len(steps) - 1, value=0, format="第 %d 步")
        current_step = steps[idx]
        if data_meta["type"] == "array":
            st.markdown(render_array_svg(current_step), unsafe_allow_html=True)
        else:
            st.markdown(render_graph_svg(data_meta["graph"], current_step), unsafe_allow_html=True)
        st.markdown(f"**当前步骤：** {current_step.title}")
        st.write(current_step.description)

    with c2:
        st.subheader("智能体讲解")
        st.write(llm_explain)
        st.subheader("状态变量")
        st.json(current_step.state, expanded=False)

    with st.expander("查看完整步骤表"):
        for i, s in enumerate(steps, 1):
            st.markdown(f"**{i}. {s.title}**  ")
            st.write(s.description)

    with st.expander("轻量工具调用记录"):
        st.table(tool_log)

    pdf_bytes = build_pdf_report(problem, algorithm, steps, llm_explain, tool_log + [{"tool": "make_pdf", "result": "自动生成 PDF 求解报告"}])
    st.download_button(
        "下载 PDF 求解报告",
        data=pdf_bytes,
        file_name="算法过程可视化智能体_求解报告.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

else:
    st.info("建议演示：选择“自动识别”，输入快速排序、BFS 或 Dijkstra 的问题，然后点击按钮录制完整过程。")
