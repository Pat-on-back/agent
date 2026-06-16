import io
import json
import math
import os
import random
import heapq
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import pandas as pd
import streamlit as st
from openai import OpenAI
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

# -----------------------------
# Page and API utilities
# -----------------------------

st.set_page_config(
    page_title="硅基流动 API - 通用算法过程可视化智能体",
    page_icon="🧠",
    layout="wide",
)

DEFAULT_ITEMS = [
    {"name": "A", "weight": 2, "value": 6},
    {"name": "B", "weight": 3, "value": 10},
    {"name": "C", "weight": 4, "value": 12},
    {"name": "D", "weight": 5, "value": 14},
    {"name": "E", "weight": 9, "value": 20},
    {"name": "F", "weight": 7, "value": 18},
]

SYSTEM_PROMPT = """
你是一个算法过程可视化教学智能体。请基于本地工具生成的真实轨迹进行讲解。
要求：
1. 先说明问题建模；
2. 再解释每种算法的核心思想；
3. 指出可视化中应该重点观察的变量；
4. 对启发式算法只说“近似搜索”或“本次运行得到”，不要承诺全局最优；
5. 输出中文，面向课程项目答辩。
"""


def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, os.getenv(name, default))
    except Exception:
        return os.getenv(name, default)


def call_siliconflow(api_key: str, base_url: str, model: str, user_prompt: str) -> str:
    if not api_key:
        return "未配置硅基流动 API Key。当前页面仍可使用本地算法工具生成可视化；配置 Key 后可自动生成更完整的智能体讲解。"
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception as exc:
        return f"硅基流动 API 调用失败：{exc}\n\n本地算法可视化结果仍然有效，请检查 API Key、Base URL 或模型名称。"


# -----------------------------
# 01 knapsack algorithms
# -----------------------------

def item_names(items: List[Dict[str, Any]]) -> List[str]:
    return [str(x["name"]) for x in items]


def selected_items_from_bits(items: List[Dict[str, Any]], bits: List[int]) -> List[str]:
    return [items[i]["name"] for i, b in enumerate(bits) if b]


def knapsack_value_weight(items: List[Dict[str, Any]], bits: List[int]) -> Tuple[int, int]:
    w = sum(items[i]["weight"] for i, b in enumerate(bits) if b)
    v = sum(items[i]["value"] for i, b in enumerate(bits) if b)
    return v, w


def knapsack_dp(items: List[Dict[str, Any]], capacity: int) -> Dict[str, Any]:
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    frames = []
    for i in range(1, n + 1):
        wt = items[i - 1]["weight"]
        val = items[i - 1]["value"]
        for c in range(capacity + 1):
            without = dp[i - 1][c]
            with_item = -1
            decision = "不取"
            if c >= wt:
                with_item = dp[i - 1][c - wt] + val
                if with_item > without:
                    decision = "取"
            dp[i][c] = max(without, with_item)
            if c == capacity or c in {0, wt, capacity // 2}:
                frames.append({
                    "step": len(frames) + 1,
                    "algorithm": "动态规划",
                    "i": i,
                    "capacity": c,
                    "item": items[i - 1]["name"],
                    "decision": decision,
                    "without": without,
                    "with_item": with_item if with_item >= 0 else None,
                    "best_value": dp[i][c],
                    "table": [row[:] for row in dp],
                    "explain": f"计算 dp[{i}][{c}]：比较不取物品 {items[i - 1]['name']} 的价值 {without} 与取它后的价值 {with_item if with_item >= 0 else '不可取'}，选择{decision}。",
                })
    bits = [0] * n
    c = capacity
    for i in range(n, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            bits[i - 1] = 1
            c -= items[i - 1]["weight"]
    return {"best_value": dp[n][capacity], "bits": bits, "frames": frames, "dp": dp}


def knapsack_backtracking(items: List[Dict[str, Any]], capacity: int) -> Dict[str, Any]:
    n = len(items)
    best_value = 0
    best_bits = [0] * n
    frames = []

    def dfs(i: int, current_w: int, current_v: int, bits: List[int], depth: int):
        nonlocal best_value, best_bits
        frames.append({
            "step": len(frames) + 1,
            "algorithm": "回溯法",
            "level": i,
            "depth": depth,
            "weight": current_w,
            "value": current_v,
            "bits": bits[:],
            "status": "访问节点",
            "best_value": best_value,
            "explain": f"到达第 {i} 层，当前重量 {current_w}，当前价值 {current_v}。",
        })
        if current_w > capacity:
            frames[-1]["status"] = "超重剪枝"
            frames[-1]["explain"] = f"当前重量 {current_w} 超过容量 {capacity}，该分支被剪枝。"
            return
        if i == n:
            if current_v > best_value:
                best_value = current_v
                best_bits = bits[:]
                frames.append({
                    "step": len(frames) + 1,
                    "algorithm": "回溯法",
                    "level": i,
                    "depth": depth,
                    "weight": current_w,
                    "value": current_v,
                    "bits": bits[:],
                    "status": "更新最优",
                    "best_value": best_value,
                    "explain": f"到达叶子节点，发现更优解，最优价值更新为 {best_value}。",
                })
            return
        bits[i] = 1
        dfs(i + 1, current_w + items[i]["weight"], current_v + items[i]["value"], bits, depth + 1)
        bits[i] = 0
        dfs(i + 1, current_w, current_v, bits, depth + 1)

    dfs(0, 0, 0, [0] * n, 0)
    return {"best_value": best_value, "bits": best_bits, "frames": frames[:120]}


def fractional_bound(items: List[Dict[str, Any]], capacity: int, level: int, weight: int, value: int, order: List[int]) -> float:
    if weight >= capacity:
        return float(value) if weight == capacity else 0.0
    bound = float(value)
    total_w = weight
    for idx in order:
        if idx < level:
            continue
        wt = items[idx]["weight"]
        val = items[idx]["value"]
        if total_w + wt <= capacity:
            total_w += wt
            bound += val
        else:
            remain = capacity - total_w
            bound += val * remain / wt
            break
    return bound


def knapsack_branch_bound(items: List[Dict[str, Any]], capacity: int) -> Dict[str, Any]:
    n = len(items)
    order = list(range(n))
    # The visualized decision still follows original item index. The bound uses value density order for optimism.
    density_order = sorted(range(n), key=lambda i: items[i]["value"] / items[i]["weight"], reverse=True)
    best_value = 0
    best_bits = [0] * n
    frames = []
    heap = []
    start_bound = fractional_bound(items, capacity, 0, 0, 0, density_order)
    heapq.heappush(heap, (-start_bound, 0, 0, 0, [0] * n))

    while heap and len(frames) < 120:
        neg_bound, level, weight, value, bits = heapq.heappop(heap)
        bound = -neg_bound
        if bound <= best_value:
            frames.append({
                "step": len(frames) + 1,
                "algorithm": "分支限界",
                "level": level,
                "weight": weight,
                "value": value,
                "bound": round(bound, 2),
                "best_value": best_value,
                "queue_size": len(heap),
                "bits": bits[:],
                "status": "上界剪枝",
                "explain": f"节点上界 {bound:.2f} 不超过当前最优 {best_value}，不再扩展。",
            })
            continue
        frames.append({
            "step": len(frames) + 1,
            "algorithm": "分支限界",
            "level": level,
            "weight": weight,
            "value": value,
            "bound": round(bound, 2),
            "best_value": best_value,
            "queue_size": len(heap),
            "bits": bits[:],
            "status": "扩展节点",
            "explain": f"从优先队列取出上界最高的节点，上界为 {bound:.2f}，尝试扩展第 {level + 1} 个物品。",
        })
        if level == n:
            continue
        idx = level
        for take in [1, 0]:
            new_bits = bits[:]
            new_bits[idx] = take
            new_w = weight + take * items[idx]["weight"]
            new_v = value + take * items[idx]["value"]
            if new_w <= capacity and new_v > best_value:
                best_value = new_v
                best_bits = new_bits[:]
            new_bound = fractional_bound(items, capacity, level + 1, new_w, new_v, density_order)
            if new_w <= capacity and new_bound > best_value:
                heapq.heappush(heap, (-new_bound, level + 1, new_w, new_v, new_bits))
    return {"best_value": best_value, "bits": best_bits, "frames": frames}


def repair_bits(items: List[Dict[str, Any]], capacity: int, bits: List[int]) -> List[int]:
    bits = bits[:]
    while knapsack_value_weight(items, bits)[1] > capacity:
        selected = [i for i, b in enumerate(bits) if b]
        if not selected:
            break
        worst = min(selected, key=lambda i: items[i]["value"] / items[i]["weight"])
        bits[worst] = 0
    return bits


def knapsack_ga(items: List[Dict[str, Any]], capacity: int, pop_size: int = 18, generations: int = 25, seed: int = 7) -> Dict[str, Any]:
    rnd = random.Random(seed)
    n = len(items)

    def fitness(bits):
        bits = repair_bits(items, capacity, bits)
        v, w = knapsack_value_weight(items, bits)
        return v, w, bits

    population = [[rnd.randint(0, 1) for _ in range(n)] for _ in range(pop_size)]
    best_bits = [0] * n
    best_value = 0
    frames = []
    for gen in range(generations + 1):
        scored = []
        for chrom in population:
            v, w, fixed = fitness(chrom)
            scored.append((v, w, fixed))
            if v > best_value:
                best_value = v
                best_bits = fixed[:]
        scored.sort(reverse=True, key=lambda x: x[0])
        avg = sum(x[0] for x in scored) / len(scored)
        frames.append({
            "step": gen,
            "algorithm": "遗传算法",
            "generation": gen,
            "best_value": best_value,
            "generation_best": scored[0][0],
            "avg_value": round(avg, 2),
            "best_bits": best_bits[:],
            "status": "种群评估",
            "explain": f"第 {gen} 代：评估种群适应度，本代最优 {scored[0][0]}，历史最优 {best_value}。",
        })
        if gen == generations:
            break
        parents = [x[2] for x in scored[: max(4, pop_size // 3)]]
        new_pop = parents[:2]
        while len(new_pop) < pop_size:
            p1, p2 = rnd.sample(parents, 2)
            cut = rnd.randint(1, n - 1)
            child = p1[:cut] + p2[cut:]
            if rnd.random() < 0.25:
                m = rnd.randrange(n)
                child[m] = 1 - child[m]
            new_pop.append(child)
        population = new_pop
    return {"best_value": best_value, "bits": best_bits, "frames": frames}


def knapsack_sa(items: List[Dict[str, Any]], capacity: int, iterations: int = 45, seed: int = 3) -> Dict[str, Any]:
    rnd = random.Random(seed)
    n = len(items)
    current = repair_bits(items, capacity, [rnd.randint(0, 1) for _ in range(n)])
    current_v, current_w = knapsack_value_weight(items, current)
    best_bits = current[:]
    best_value = current_v
    temp = 10.0
    frames = []
    for it in range(iterations + 1):
        candidate = current[:]
        pos = rnd.randrange(n)
        candidate[pos] = 1 - candidate[pos]
        candidate = repair_bits(items, capacity, candidate)
        cand_v, cand_w = knapsack_value_weight(items, candidate)
        delta = cand_v - current_v
        accept_prob = 1.0 if delta >= 0 else math.exp(delta / max(temp, 1e-9))
        accepted = rnd.random() < accept_prob
        if accepted:
            current, current_v, current_w = candidate, cand_v, cand_w
        if current_v > best_value:
            best_value = current_v
            best_bits = current[:]
        frames.append({
            "step": it,
            "algorithm": "模拟退火",
            "temperature": round(temp, 4),
            "current_value": current_v,
            "candidate_value": cand_v,
            "delta": delta,
            "accept_prob": round(accept_prob, 4),
            "accepted": accepted,
            "best_value": best_value,
            "bits": current[:],
            "status": "接受" if accepted else "拒绝",
            "explain": f"第 {it} 次扰动：候选价值 {cand_v}，当前价值 {current_v}，接受概率 {accept_prob:.3f}，结果为{'接受' if accepted else '拒绝'}。",
        })
        temp *= 0.90
    return {"best_value": best_value, "bits": best_bits, "frames": frames}


# -----------------------------
# Other problem visualizers
# -----------------------------

def lcs_dp(a: str, b: str) -> Dict[str, Any]:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    frames = []
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                reason = "字符相同，来自左上角 + 1"
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                reason = "字符不同，取上方和左方最大值"
            frames.append({
                "step": len(frames) + 1,
                "algorithm": "LCS 动态规划",
                "i": i,
                "j": j,
                "char_a": a[i - 1],
                "char_b": b[j - 1],
                "value": dp[i][j],
                "table": [row[:] for row in dp],
                "explain": f"计算 dp[{i}][{j}]：{a[i - 1]} 与 {b[j - 1]}，{reason}，得到 {dp[i][j]}。",
            })
    # reconstruct
    i, j = m, n
    seq = []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            seq.append(a[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return {"lcs": "".join(reversed(seq)), "length": dp[m][n], "frames": frames, "dp": dp}


def nqueens_backtracking(n: int) -> Dict[str, Any]:
    cols = set()
    diag1 = set()
    diag2 = set()
    board = [-1] * n
    frames = []
    solutions = []

    def dfs(row: int):
        if len(frames) > 160:
            return
        if row == n:
            solutions.append(board[:])
            frames.append({
                "step": len(frames) + 1,
                "algorithm": "N 皇后回溯",
                "row": row,
                "board": board[:],
                "status": "找到解",
                "explain": f"所有 {n} 行都放置完成，找到一个合法方案。",
            })
            return
        for col in range(n):
            conflict = col in cols or (row - col) in diag1 or (row + col) in diag2
            frames.append({
                "step": len(frames) + 1,
                "algorithm": "N 皇后回溯",
                "row": row,
                "col": col,
                "board": board[:],
                "status": "冲突剪枝" if conflict else "尝试放置",
                "explain": f"尝试在第 {row + 1} 行第 {col + 1} 列放皇后：{'与已有皇后冲突，剪枝' if conflict else '暂时合法，继续下一行'}。",
            })
            if conflict:
                continue
            board[row] = col
            cols.add(col); diag1.add(row - col); diag2.add(row + col)
            dfs(row + 1)
            cols.remove(col); diag1.remove(row - col); diag2.remove(row + col)
            board[row] = -1
            frames.append({
                "step": len(frames) + 1,
                "algorithm": "N 皇后回溯",
                "row": row,
                "col": col,
                "board": board[:],
                "status": "回退",
                "explain": f"从第 {row + 1} 行第 {col + 1} 列回退，尝试其他列。",
            })
    dfs(0)
    return {"solutions": solutions, "frames": frames}


def dijkstra_visual(nodes: List[str], edges: List[Tuple[str, str, int]], start: str) -> Dict[str, Any]:
    graph = {u: [] for u in nodes}
    for u, v, w in edges:
        graph.setdefault(u, []).append((v, w))
        graph.setdefault(v, []).append((u, w))
    dist = {u: float("inf") for u in graph}
    prev = {u: None for u in graph}
    dist[start] = 0
    pq = [(0, start)]
    visited = set()
    frames = []
    while pq and len(frames) < 120:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        frames.append({
            "step": len(frames) + 1,
            "algorithm": "Dijkstra",
            "current": u,
            "visited": sorted(list(visited)),
            "dist": {k: ("∞" if v == float("inf") else v) for k, v in dist.items()},
            "status": "确定最短点",
            "explain": f"选择未确定节点中距离最小的 {u}，其当前最短距离为 {d}。",
        })
        for v, w in graph[u]:
            if v in visited:
                continue
            if dist[u] + w < dist[v]:
                old = dist[v]
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
                frames.append({
                    "step": len(frames) + 1,
                    "algorithm": "Dijkstra",
                    "current": u,
                    "relaxed": v,
                    "edge_weight": w,
                    "visited": sorted(list(visited)),
                    "dist": {k: ("∞" if val == float("inf") else val) for k, val in dist.items()},
                    "status": "松弛边",
                    "explain": f"用边 {u}-{v} 松弛：原距离 {old if old != float('inf') else '∞'}，新距离 {dist[v]}。",
                })
    return {"dist": dist, "prev": prev, "frames": frames}


def tsp_distance(route: List[int], cities: List[Tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(route)):
        x1, y1 = cities[route[i]]
        x2, y2 = cities[route[(i + 1) % len(route)]]
        total += math.hypot(x1 - x2, y1 - y2)
    return total


def tsp_sa(cities: List[Tuple[float, float]], iterations: int = 60, seed: int = 11) -> Dict[str, Any]:
    rnd = random.Random(seed)
    n = len(cities)
    current = list(range(n))
    rnd.shuffle(current)
    current_d = tsp_distance(current, cities)
    best = current[:]
    best_d = current_d
    temp = 20.0
    frames = []
    for it in range(iterations + 1):
        a, b = sorted(rnd.sample(range(n), 2))
        cand = current[:]
        cand[a:b+1] = reversed(cand[a:b+1])
        cand_d = tsp_distance(cand, cities)
        delta = cand_d - current_d
        accept_prob = 1.0 if delta < 0 else math.exp(-delta / max(temp, 1e-9))
        accepted = rnd.random() < accept_prob
        if accepted:
            current, current_d = cand, cand_d
        if current_d < best_d:
            best, best_d = current[:], current_d
        frames.append({
            "step": it,
            "algorithm": "TSP 模拟退火",
            "temperature": round(temp, 3),
            "current_distance": round(current_d, 3),
            "candidate_distance": round(cand_d, 3),
            "best_distance": round(best_d, 3),
            "accepted": accepted,
            "route": current[:],
            "explain": f"第 {it} 次交换路径片段，候选距离 {cand_d:.2f}，接受概率 {accept_prob:.3f}，{'接受' if accepted else '拒绝'}。",
        })
        temp *= 0.92
    return {"best_route": best, "best_distance": best_d, "frames": frames}


# -----------------------------
# Rendering helpers
# -----------------------------

def bits_table(items, bits):
    data = []
    for i, item in enumerate(items):
        data.append({
            "物品": item["name"],
            "重量": item["weight"],
            "价值": item["value"],
            "是否选择": "1 取" if bits[i] else "0 不取",
        })
    return pd.DataFrame(data)


def show_frame_table(frame: Dict[str, Any]):
    st.markdown(f"**第 {frame.get('step', 0)} 步：{frame.get('status', frame.get('algorithm', ''))}**")
    st.write(frame.get("explain", ""))
    cols = st.columns(4)
    for idx, key in enumerate(["best_value", "weight", "value", "bound", "temperature", "current_value", "avg_value", "queue_size"]):
        if key in frame:
            cols[idx % 4].metric(key, frame[key])


def dataframe_from_matrix(matrix: List[List[int]], row_prefix="i", col_prefix="c") -> pd.DataFrame:
    return pd.DataFrame(matrix, index=[f"{row_prefix}{i}" for i in range(len(matrix))], columns=[f"{col_prefix}{j}" for j in range(len(matrix[0]))])


def render_knapsack_result(name: str, result: Dict[str, Any], items: List[Dict[str, Any]], capacity: int):
    st.subheader(name)
    cols = st.columns(3)
    cols[0].metric("最优/本次最好价值", result.get("best_value", "-"))
    if "bits" in result:
        v, w = knapsack_value_weight(items, result["bits"])
        cols[1].metric("总重量", w)
        cols[2].metric("选择物品", ", ".join(selected_items_from_bits(items, result["bits"])) or "空")
        st.dataframe(bits_table(items, result["bits"]), use_container_width=True)
    frames = result.get("frames", [])
    if frames:
        idx = st.slider(f"选择{name}步骤", 0, len(frames) - 1, min(len(frames) - 1, 0), key=f"slider_{name}")
        frame = frames[idx]
        show_frame_table(frame)
        if "table" in frame:
            st.dataframe(dataframe_from_matrix(frame["table"]), use_container_width=True)
        if "bits" in frame:
            st.dataframe(bits_table(items, frame["bits"]), use_container_width=True)
        if name in ["遗传算法", "模拟退火"]:
            chart_data = pd.DataFrame(frames)
            y_cols = [x for x in ["best_value", "generation_best", "avg_value", "current_value", "candidate_value"] if x in chart_data.columns]
            if y_cols:
                st.line_chart(chart_data[y_cols])


def render_lcs(result: Dict[str, Any]):
    st.metric("LCS 长度", result["length"])
    st.success(f"最长公共子序列：{result['lcs']}")
    frames = result["frames"]
    idx = st.slider("选择 LCS 动态规划步骤", 0, len(frames) - 1, 0)
    frame = frames[idx]
    show_frame_table(frame)
    st.dataframe(dataframe_from_matrix(frame["table"], "i", "j"), use_container_width=True)


def render_nqueens(result: Dict[str, Any], n: int):
    st.metric("已找到方案数量", len(result["solutions"]))
    frames = result["frames"]
    idx = st.slider("选择 N 皇后回溯步骤", 0, len(frames) - 1, 0)
    frame = frames[idx]
    show_frame_table(frame)
    board = frame.get("board", [-1] * n)
    grid = []
    for r in range(n):
        row = []
        for c in range(n):
            row.append("♛" if board[r] == c else "·")
        grid.append(row)
    st.dataframe(pd.DataFrame(grid), use_container_width=True)


def render_dijkstra(result: Dict[str, Any]):
    dist = {k: ("∞" if v == float("inf") else v) for k, v in result["dist"].items()}
    st.write("最终最短距离：", dist)
    frames = result["frames"]
    idx = st.slider("选择 Dijkstra 步骤", 0, len(frames) - 1, 0)
    frame = frames[idx]
    show_frame_table(frame)
    st.dataframe(pd.DataFrame([frame["dist"]]), use_container_width=True)
    st.write("已确定节点：", ", ".join(frame.get("visited", [])))


def render_tsp(result: Dict[str, Any], cities: List[Tuple[float, float]]):
    st.metric("本次最好路径距离", f"{result['best_distance']:.3f}")
    st.write("本次最好路径：", " → ".join(map(str, result["best_route"] + [result["best_route"][0]])))
    frames = result["frames"]
    idx = st.slider("选择 TSP 模拟退火步骤", 0, len(frames) - 1, 0)
    frame = frames[idx]
    show_frame_table(frame)
    chart = pd.DataFrame(frames)[["current_distance", "candidate_distance", "best_distance"]]
    st.line_chart(chart)
    route = frame["route"]
    route_points = []
    for order, city_id in enumerate(route + [route[0]]):
        x, y = cities[city_id]
        route_points.append({"顺序": order, "城市": city_id, "x": x, "y": y})
    st.dataframe(pd.DataFrame(route_points), use_container_width=True)


# -----------------------------
# PDF report generation
# -----------------------------

def build_pdf_report(title: str, problem: str, summary: str, results_summary: List[Dict[str, Any]], llm_text: str) -> bytes:
    buffer = io.BytesIO()
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=1.6 * cm, leftMargin=1.6 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CJKTitle", parent=styles["Title"], fontName="STSong-Light", fontSize=18, leading=24))
    styles.add(ParagraphStyle(name="CJKHeading", parent=styles["Heading2"], fontName="STSong-Light", fontSize=13, leading=18, spaceBefore=8, spaceAfter=6))
    styles.add(ParagraphStyle(name="CJKBody", parent=styles["BodyText"], fontName="STSong-Light", fontSize=10.5, leading=16))
    story = []
    story.append(Paragraph(title, styles["CJKTitle"]))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph("一、项目定位", styles["CJKHeading"]))
    story.append(Paragraph("本智能体基于硅基流动公开 API 与本地轻量算法工具构建。它将算法运行过程拆成可视化帧，帮助学习者理解状态转移、搜索、剪枝和随机优化等核心思想。", styles["CJKBody"]))
    story.append(Paragraph("二、当前演示问题", styles["CJKHeading"]))
    story.append(Paragraph(problem.replace("\n", "<br/>"), styles["CJKBody"]))
    story.append(Paragraph("三、本地工具求解摘要", styles["CJKHeading"]))
    table_data = [["算法", "结果", "关键观察点"]]
    for row in results_summary:
        table_data.append([row.get("算法", ""), str(row.get("结果", "")), row.get("关键观察点", "")])
    tbl = Table(table_data, colWidths=[3.0 * cm, 4.0 * cm, 9.0 * cm])
    tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "STSong-Light"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(tbl)
    story.append(Paragraph("四、硅基流动 API 生成的教学讲解", styles["CJKHeading"]))
    for para in llm_text.split("\n"):
        if para.strip():
            story.append(Paragraph(para.strip(), styles["CJKBody"]))
            story.append(Spacer(1, 0.08 * cm))
    story.append(Paragraph("五、提交说明", styles["CJKHeading"]))
    story.append(Paragraph("最终提交包括：1）公开访问链接；2）MP4 录屏，展示输入问题、算法可视化和 PDF 下载过程；3）本 PDF 文档。", styles["CJKBody"]))
    doc.build(story)
    return buffer.getvalue()


# -----------------------------
# Main UI
# -----------------------------

st.title("🧠 基于硅基流动 API 的通用算法过程可视化智能体")
st.caption("本地工具生成算法轨迹；硅基流动 API 生成教学讲解与报告。适合课程项目：公开链接 + MP4 录屏 + PDF 文档。")

with st.sidebar:
    st.header("硅基流动 API 设置")
    api_key_default = get_secret("SILICONFLOW_API_KEY", "")
    base_url_default = get_secret("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    model_default = get_secret("SILICONFLOW_MODEL", "Qwen/Qwen2.5-72B-Instruct")
    api_key = st.text_input("SILICONFLOW_API_KEY", value=api_key_default, type="password")
    base_url = st.text_input("Base URL", value=base_url_default)
    model = st.text_input("模型名称", value=model_default)
    st.info("提示：未配置 Key 时仍能演示本地算法可视化；配置后可生成更完整的智能体讲解。")

problem_type = st.selectbox(
    "选择问题类型",
    ["01 背包 - 五类算法对比", "最长公共子序列 LCS - 动态规划", "N 皇后 - 回溯法", "最短路径 - Dijkstra", "旅行商 TSP - 模拟退火"],
)

results_summary = []
llm_text = ""
problem_text = ""

if problem_type == "01 背包 - 五类算法对比":
    st.header("01 背包：回溯、动态规划、分支限界、遗传算法、模拟退火")
    capacity = st.number_input("背包容量", min_value=1, value=15, step=1)
    items_text = st.text_area("物品 JSON", value=json.dumps(DEFAULT_ITEMS, ensure_ascii=False, indent=2), height=180)
    algorithms = st.multiselect(
        "选择要可视化的算法",
        ["动态规划", "回溯法", "分支限界", "遗传算法", "模拟退火"],
        default=["动态规划", "回溯法", "分支限界", "遗传算法", "模拟退火"],
    )
    if st.button("生成 01 背包多算法可视化", type="primary"):
        try:
            items = json.loads(items_text)
            st.session_state["items"] = items
            st.session_state["capacity"] = int(capacity)
            st.session_state["knapsack_results"] = {}
            if "动态规划" in algorithms:
                st.session_state["knapsack_results"]["动态规划"] = knapsack_dp(items, int(capacity))
            if "回溯法" in algorithms:
                st.session_state["knapsack_results"]["回溯法"] = knapsack_backtracking(items, int(capacity))
            if "分支限界" in algorithms:
                st.session_state["knapsack_results"]["分支限界"] = knapsack_branch_bound(items, int(capacity))
            if "遗传算法" in algorithms:
                st.session_state["knapsack_results"]["遗传算法"] = knapsack_ga(items, int(capacity))
            if "模拟退火" in algorithms:
                st.session_state["knapsack_results"]["模拟退火"] = knapsack_sa(items, int(capacity))
            st.session_state["active_problem"] = "knapsack"
        except Exception as exc:
            st.error(f"输入解析失败：{exc}")

    if st.session_state.get("active_problem") == "knapsack":
        items = st.session_state["items"]
        capacity = st.session_state["capacity"]
        results = st.session_state["knapsack_results"]
        st.write("物品表：")
        st.dataframe(pd.DataFrame(items), use_container_width=True)
        tabs = st.tabs(list(results.keys()))
        for tab, name in zip(tabs, results.keys()):
            with tab:
                render_knapsack_result(name, results[name], items, capacity)
        for name, res in results.items():
            bits = res.get("bits", [])
            chosen = ", ".join(selected_items_from_bits(items, bits)) if bits else "-"
            results_summary.append({"算法": name, "结果": f"价值 {res.get('best_value', '-')}; 选择 {chosen}", "关键观察点": "观察状态转移、搜索路径、剪枝或迭代收敛过程。"})
        problem_text = f"01 背包问题，容量 {capacity}，物品为 {items}。"

elif problem_type == "最长公共子序列 LCS - 动态规划":
    st.header("LCS：动态规划状态表可视化")
    a = st.text_input("字符串 A", value="ABCBDAB")
    b = st.text_input("字符串 B", value="BDCABA")
    if st.button("生成 LCS 可视化", type="primary"):
        st.session_state["lcs_result"] = lcs_dp(a, b)
        st.session_state["active_problem"] = "lcs"
        st.session_state["lcs_a"] = a
        st.session_state["lcs_b"] = b
    if st.session_state.get("active_problem") == "lcs":
        render_lcs(st.session_state["lcs_result"])
        res = st.session_state["lcs_result"]
        results_summary.append({"算法": "动态规划", "结果": f"LCS={res['lcs']}，长度={res['length']}", "关键观察点": "观察 dp[i][j] 如何由左、上、左上转移而来。"})
        problem_text = f"最长公共子序列问题：A={st.session_state['lcs_a']}，B={st.session_state['lcs_b']}。"

elif problem_type == "N 皇后 - 回溯法":
    st.header("N 皇后：回溯搜索树可视化")
    n = st.slider("N", min_value=4, max_value=8, value=4)
    if st.button("生成 N 皇后回溯可视化", type="primary"):
        st.session_state["nq_result"] = nqueens_backtracking(n)
        st.session_state["nq_n"] = n
        st.session_state["active_problem"] = "nq"
    if st.session_state.get("active_problem") == "nq":
        render_nqueens(st.session_state["nq_result"], st.session_state["nq_n"])
        results_summary.append({"算法": "回溯法", "结果": f"找到 {len(st.session_state['nq_result']['solutions'])} 个方案", "关键观察点": "观察列冲突、主对角线冲突、副对角线冲突如何导致剪枝。"})
        problem_text = f"N 皇后问题，N={st.session_state['nq_n']}。"

elif problem_type == "最短路径 - Dijkstra":
    st.header("Dijkstra：最短路径距离更新可视化")
    nodes_text = st.text_input("节点列表", value="A,B,C,D,E,F")
    edges_text = st.text_area("边列表 JSON: [起点,终点,权重]", value=json.dumps([
        ["A", "B", 4], ["A", "C", 2], ["B", "C", 1], ["B", "D", 5],
        ["C", "D", 8], ["C", "E", 10], ["D", "E", 2], ["D", "F", 6], ["E", "F", 3]
    ], ensure_ascii=False, indent=2), height=180)
    start = st.text_input("起点", value="A")
    if st.button("生成 Dijkstra 可视化", type="primary"):
        nodes = [x.strip() for x in nodes_text.split(",") if x.strip()]
        edges = [(u, v, int(w)) for u, v, w in json.loads(edges_text)]
        st.session_state["dij_result"] = dijkstra_visual(nodes, edges, start)
        st.session_state["active_problem"] = "dij"
        st.session_state["dij_nodes"] = nodes
        st.session_state["dij_edges"] = edges
        st.session_state["dij_start"] = start
    if st.session_state.get("active_problem") == "dij":
        render_dijkstra(st.session_state["dij_result"])
        dist = {k: ("∞" if v == float("inf") else v) for k, v in st.session_state["dij_result"]["dist"].items()}
        results_summary.append({"算法": "Dijkstra", "结果": str(dist), "关键观察点": "观察每次选择的当前最短节点，以及边松弛后距离表如何更新。"})
        problem_text = f"最短路径问题，起点 {st.session_state['dij_start']}，边为 {st.session_state['dij_edges']}。"

else:
    st.header("TSP：模拟退火路径优化可视化")
    cities_text = st.text_area("城市坐标 JSON", value=json.dumps([[0, 0], [1, 5], [2, 3], [5, 4], [6, 1], [3, 0]], ensure_ascii=False, indent=2), height=160)
    if st.button("生成 TSP 模拟退火可视化", type="primary"):
        cities = [tuple(map(float, x)) for x in json.loads(cities_text)]
        st.session_state["tsp_result"] = tsp_sa(cities)
        st.session_state["tsp_cities"] = cities
        st.session_state["active_problem"] = "tsp"
    if st.session_state.get("active_problem") == "tsp":
        render_tsp(st.session_state["tsp_result"], st.session_state["tsp_cities"])
        res = st.session_state["tsp_result"]
        results_summary.append({"算法": "模拟退火", "结果": f"本次最好距离 {res['best_distance']:.3f}", "关键观察点": "观察温度下降后，接受较差解的概率逐步减小，搜索由探索转向收敛。"})
        problem_text = f"旅行商 TSP 问题，城市坐标为 {st.session_state['tsp_cities']}。"

if results_summary:
    st.divider()
    st.header("硅基流动 API 讲解与 PDF 报告")
    summary_text = json.dumps(results_summary, ensure_ascii=False, indent=2)
    prompt = f"请为以下算法可视化结果生成课程项目答辩式讲解。\n问题：{problem_text}\n结果摘要：{summary_text}"
    if st.button("调用硅基流动 API 生成讲解"):
        st.session_state["llm_text"] = call_siliconflow(api_key, base_url, model, prompt)
    llm_text = st.session_state.get("llm_text", "点击上方按钮可调用硅基流动 API 生成讲解。")
    st.markdown(llm_text)
    pdf_bytes = build_pdf_report(
        "基于硅基流动 API 的通用算法过程可视化智能体报告",
        problem_text,
        summary_text,
        results_summary,
        llm_text,
    )
    st.download_button(
        "下载 PDF 报告",
        data=pdf_bytes,
        file_name="算法过程可视化智能体报告.pdf",
        mime="application/pdf",
    )
