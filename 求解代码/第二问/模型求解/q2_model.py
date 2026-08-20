#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2024 高教社杯全国大学生数学建模竞赛 B 题——问题二
信息保持型闭环 Bellman–Markov 决策模型

严格依据《问题二建模手册》实现：
1. 表1六种情形参数结构化录入与口径校验；
2. 解析概率传播 + 几何分布闭式模型（无拆解基准）；
3. 信息保持型闭环 Bellman–Markov 主模型；
4. 2^6=64 个原始策略全枚举，按四类生产决策 (x1,x2,y,z) 比较全局最优；
5. 解析退化检验、参考结果复现、谱半径/吸收性检验、极端情形检验；
6. Monte Carlo 离散事件仿真独立验证；
7. 基于题目真实参数区间的单参数临界值灵敏度分析；
8. p2-L 二维策略相图数据导出（只导数据，不绘图）。

依赖：Python >= 3.10, numpy, pandas
运行（PyCharm 直接运行本文件即可）：
    python q2_model.py
可选：
    python q2_model.py --output-dir D:/your/output --mc-orders 100000 --sens-points 101 --phase-points 31
    python q2_model.py --skip-mc
    python q2_model.py --quick

默认输出目录：脚本所在目录下 ../结果输出/
（即 求解代码/第二问/结果输出/），供 MATLAB 绘图复用。

说明：
- Python 仅负责计算与 CSV 导出；后续 MATLAB 读取 CSV 绘图。
- 所有成本单位均为 元/件；利润口径为“最终成功交付 1 件合格产品”的单位订单期望利润。
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

# 在 Windows GBK 默认编码的控制台中，print 中文会乱码；
# 强制以 UTF-8 输出，PyCharm Terminal / 现代 Windows Terminal 均能正确显示。
# 如果运行环境不支持 reconfigure（极旧 Python），则忽略异常。
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    if _stream is not None and hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:  # pragma: no cover
            pass

import numpy as np
import pandas as pd


# ============================================================
# 0. 全局设置
# ============================================================

EPS = 1e-12
RHO_TOL = 1e-10
STATE_K = "K"       # 已检测并确认合格
STATE_UG = "U_G"    # 企业未知，后台潜在真实状态=合格
STATE_UB = "U_B"    # 企业未知，后台潜在真实状态=不合格
PAIR_STATES: List[Tuple[str, str]] = list(
    itertools.product([STATE_K, STATE_UG, STATE_UB], repeat=2)
)
STATE_N = "N"       # 重新从供应商准备一套零件

COST_CATEGORIES = [
    "purchase",
    "component_inspection",
    "assembly",
    "final_inspection",
    "replacement",
    "disassembly",
]

COUNT_CATEGORIES = [
    "purchase_1",
    "purchase_2",
    "inspect_1",
    "inspect_2",
    "assemblies",
    "final_inspections",
    "replacements",
    "disassemblies",
]


# ============================================================
# 1. 参数与策略
# ============================================================

@dataclass(frozen=True)
class ScenarioParams:
    """问题二表1的一行情形参数。"""

    scenario_id: Union[int, str]
    p1: float
    a1: float
    b1: float
    p2: float
    a2: float
    b2: float
    pf: float
    assembly_cost: float
    bf: float
    sale_price: float
    replacement_loss: float
    disassembly_cost: float

    def validate(self) -> None:
        for name in ("p1", "p2", "pf"):
            v = float(getattr(self, name))
            if not (0.0 <= v < 1.0):
                raise ValueError(f"{name} 必须满足 0 <= {name} < 1，当前={v}")
        for name in (
            "a1", "b1", "a2", "b2", "assembly_cost", "bf",
            "sale_price", "replacement_loss", "disassembly_cost"
        ):
            v = float(getattr(self, name))
            if v < 0:
                raise ValueError(f"成本/售价字段 {name} 不能为负，当前={v}")


@dataclass(frozen=True)
class Policy:
    """
    完整策略 pi=(x1,x2,y,z,r1,r2)

    x1,x2 : 首次使用零配件1/2前是否检测
    y     : 成品是否检测
    z     : 不合格成品是否拆解
    r1,r2 : 拆解后，对“质量未知”的回收零件1/2是否检测
    """

    x1: int
    x2: int
    y: int
    z: int
    r1: int
    r2: int

    def validate(self) -> None:
        vals = (self.x1, self.x2, self.y, self.z, self.r1, self.r2)
        if any(v not in (0, 1) for v in vals):
            raise ValueError(f"策略变量必须均为0/1，当前={vals}")

    def canonical(self) -> "Policy":
        """z=0 时 r1,r2 无实际意义，统一规范化为0。"""
        if self.z == 0:
            return Policy(self.x1, self.x2, self.y, 0, 0, 0)
        return self

    @property
    def base4(self) -> Tuple[int, int, int, int]:
        return self.x1, self.x2, self.y, self.z

    @property
    def code4(self) -> str:
        return f"{self.x1}{self.x2}{self.y}{self.z}"

    @property
    def code6(self) -> str:
        return f"{self.x1}{self.x2}{self.y}{self.z}{self.r1}{self.r2}"


@dataclass
class PolicyEvaluation:
    scenario_id: Union[int, str]
    policy: Policy
    feasible: bool
    expected_cost: float
    expected_profit: float
    spectral_radius: float
    condition_number: float
    bellman_residual_inf: float
    absorption_probability: float
    reachable_states: Tuple[Union[str, Tuple[str, str]], ...]
    cost_breakdown: Dict[str, float]
    event_counts: Dict[str, float]
    reason: str = ""


SCENARIOS: List[ScenarioParams] = [
    ScenarioParams(1, 0.10, 4, 2, 0.10, 18, 3, 0.10, 6, 3, 56, 6, 5),
    ScenarioParams(2, 0.20, 4, 2, 0.20, 18, 3, 0.20, 6, 3, 56, 6, 5),
    ScenarioParams(3, 0.10, 4, 2, 0.10, 18, 3, 0.10, 6, 3, 56, 30, 5),
    ScenarioParams(4, 0.20, 4, 1, 0.20, 18, 1, 0.20, 6, 2, 56, 30, 5),
    ScenarioParams(5, 0.10, 4, 8, 0.20, 18, 1, 0.10, 6, 2, 56, 10, 5),
    ScenarioParams(6, 0.05, 4, 2, 0.05, 18, 3, 0.05, 6, 3, 56, 10, 40),
]

# 手册中的外部交叉检查结果，仅用于“验证”，绝不参与优化。
REFERENCE_BEST = {
    1: ("1001", 18.8411),
    2: ("1101", 12.0000),
    3: ("1011", 16.4744),
    4: ("1111", 14.7500),
    5: ("0101", 14.9633),
    6: ("0000", 21.6787),
}


# ============================================================
# 2. 数据处理与基础概率
# ============================================================

def params_to_dataframe(params_list: Sequence[ScenarioParams]) -> pd.DataFrame:
    rows = []
    for p in params_list:
        p.validate()
        rows.append({
            "scenario_id": p.scenario_id,
            "p1": p.p1, "a1": p.a1, "b1": p.b1,
            "p2": p.p2, "a2": p.a2, "b2": p.b2,
            "pf": p.pf,
            "assembly_cost": p.assembly_cost,
            "bf": p.bf,
            "sale_price": p.sale_price,
            "replacement_loss": p.replacement_loss,
            "disassembly_cost": p.disassembly_cost,
        })
    return pd.DataFrame(rows)


def total_defect_probability_no_component_inspection(p: ScenarioParams) -> float:
    """两个零件均不检测时，一次装配的总次品概率。"""
    return 1.0 - (1.0 - p.p1) * (1.0 - p.p2) * (1.0 - p.pf)


def initial_latent_branches(defect_rate: float, inspect: int) -> List[Tuple[float, str]]:
    """
    首次准备某类零件后的潜在状态分支。
    若检测：不断采购+检测直到合格，确定进入 K；
    若不检测：一次采购，后台潜在真实状态为 U_G/U_B。
    """
    if inspect == 1:
        return [(1.0, STATE_K)]
    return [(1.0 - defect_rate, STATE_UG), (defect_rate, STATE_UB)]


def is_actually_good(state: str) -> bool:
    return state in (STATE_K, STATE_UG)


# ============================================================
# 3. 模型A：无拆解解析基准（几何分布）
# ============================================================

def prepared_component_expected_cost(a: float, b: float, p: float, x: int) -> float:
    """手册式(7)：首次准备一个可投入装配的零件的期望成本。"""
    return (1 - x) * a + x * (a + b) / (1.0 - p)


def residual_defect_rate(p: float, x: int) -> float:
    """手册式(8)：检测后进入装配的残余次品率。"""
    return (1 - x) * p


def no_disassembly_closed_form(p: ScenarioParams, x1: int, x2: int, y: int) -> Dict[str, float]:
    """
    手册式(9)-(12)：z=0 时，失败后整套报废并从头重来。
    返回成功概率、单位成功订单期望成本与利润。
    """
    q1 = residual_defect_rate(p.p1, x1)
    q2 = residual_defect_rate(p.p2, x2)
    g = (1.0 - q1) * (1.0 - q2) * (1.0 - p.pf)
    if g <= 0:
        return {"g": g, "cost": math.inf, "profit": -math.inf}

    c_try = (
        prepared_component_expected_cost(p.a1, p.b1, p.p1, x1)
        + prepared_component_expected_cost(p.a2, p.b2, p.p2, x2)
        + p.assembly_cost
        + y * p.bf
    )
    cost = c_try / g + (1 - y) * p.replacement_loss * (1.0 - g) / g
    profit = p.sale_price - cost
    return {"g": g, "cost": cost, "profit": profit}


# ============================================================
# 4. 模型B：信息保持型闭环 Bellman–Markov 主模型
# ============================================================

def recovery_process_component(
    state: str,
    inspect_recovered: int,
    p_defect: float,
    purchase_cost: float,
    inspect_cost: float,
) -> Tuple[str, Dict[str, float], Dict[str, float]]:
    """
    对拆解后的某个零件执行 r_i。

    K：已知合格，不重复检测；
    U_G + r=1：付一次检测费后转 K；
    U_B + r=1：先检出坏件，再不断新购+检测直到得到合格件，转 K；
    U_* + r=0：保持未知潜在状态。
    """
    cost = {k: 0.0 for k in COST_CATEGORIES}
    count = {k: 0.0 for k in COUNT_CATEGORIES}

    if state == STATE_K or inspect_recovered == 0:
        return state, cost, count

    if state == STATE_UG:
        cost["component_inspection"] += inspect_cost
        count_key = "inspect_1"  # 这里只先占位，调用端会映射到零件2
        count[count_key] += 1.0
        return STATE_K, cost, count

    if state == STATE_UB:
        # 先检测当前回收坏件 1 次；随后新购+全检至合格。
        expected_new_trials = 1.0 / (1.0 - p_defect)
        cost["purchase"] += purchase_cost * expected_new_trials
        cost["component_inspection"] += inspect_cost * (1.0 + expected_new_trials)
        count["purchase_1"] += expected_new_trials
        count["inspect_1"] += 1.0 + expected_new_trials
        return STATE_K, cost, count

    raise ValueError(f"未知零件状态: {state}")


def remap_component_count_keys(counts: Dict[str, float], component: int) -> Dict[str, float]:
    """recovery_process_component 内部统一用 *_1，占位后映射到零件1或2。"""
    out = {k: 0.0 for k in COUNT_CATEGORIES}
    if component == 1:
        out.update(counts)
    elif component == 2:
        for k, v in counts.items():
            if k == "purchase_1":
                out["purchase_2"] += v
            elif k == "inspect_1":
                out["inspect_2"] += v
            else:
                out[k] += v
    else:
        raise ValueError("component 只能为1或2")
    return out


def add_dict(dst: Dict[str, float], src: Dict[str, float], weight: float = 1.0) -> None:
    for k, v in src.items():
        dst[k] += weight * v


def product_failure_probability(states: Tuple[str, str], p: ScenarioParams) -> float:
    """
    若任一零件实际不合格，成品必不合格；
    若两零件实际合格，则以 pf 概率因装配工艺形成次品。
    """
    both_good = is_actually_good(states[0]) and is_actually_good(states[1])
    return p.pf if both_good else 1.0


def reachable_indices(Q: np.ndarray, start: int = 0, tol: float = 1e-15) -> List[int]:
    """从 N 出发，仅保留当前策略真正可达的暂态状态。"""
    reached = {start}
    stack = [start]
    while stack:
        i = stack.pop()
        js = np.flatnonzero(Q[i] > tol)
        for j in js:
            jj = int(j)
            if jj not in reached:
                reached.add(jj)
                stack.append(jj)
    return sorted(reached)


def _initial_preparation_reward(p: ScenarioParams, policy: Policy):
    """首次从供应商准备一套零件的期望成本/事件次数（装配前）。"""
    cost = {k: 0.0 for k in COST_CATEGORIES}
    count = {k: 0.0 for k in COUNT_CATEGORIES}

    for comp, defect, a, b, x in [
        (1, p.p1, p.a1, p.b1, policy.x1),
        (2, p.p2, p.a2, p.b2, policy.x2),
    ]:
        if x == 1:
            trials = 1.0 / (1.0 - defect)
            cost["purchase"] += a * trials
            cost["component_inspection"] += b * trials
            count[f"purchase_{comp}"] += trials
            count[f"inspect_{comp}"] += trials
        else:
            cost["purchase"] += a
            count[f"purchase_{comp}"] += 1.0
    return cost, count


def build_markov_model(
    p: ScenarioParams,
    policy: Policy,
) -> Tuple[
    List[Union[str, Tuple[str, str]]],
    np.ndarray,
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    """
    构造完整 10 状态的暂态转移矩阵 Q，以及每类即时成本/事件次数奖励向量。

    状态：N + 9 个 (s1,s2)，s_i∈{K,U_G,U_B}。
    注意：真正求解时只取从 N 可达的子矩阵，避免不可达自循环状态错误污染谱半径。
    """
    p.validate()
    policy = policy.canonical()
    policy.validate()

    states: List[Union[str, Tuple[str, str]]] = [STATE_N] + PAIR_STATES
    index = {s: i for i, s in enumerate(states)}
    n = len(states)

    Q = np.zeros((n, n), dtype=float)
    cost_reward = {k: np.zeros(n, dtype=float) for k in COST_CATEGORIES}
    count_reward = {k: np.zeros(n, dtype=float) for k in COUNT_CATEGORIES}

    # ---------- 状态 N：首次采购/检测，然后立即完成一轮装配 ----------
    prep_cost, prep_count = _initial_preparation_reward(p, policy)
    for k, v in prep_cost.items():
        cost_reward[k][0] += v
    for k, v in prep_count.items():
        count_reward[k][0] += v

    cost_reward["assembly"][0] += p.assembly_cost
    count_reward["assemblies"][0] += 1.0
    if policy.y == 1:
        cost_reward["final_inspection"][0] += p.bf
        count_reward["final_inspections"][0] += 1.0

    for prob1, s1 in initial_latent_branches(p.p1, policy.x1):
        for prob2, s2 in initial_latent_branches(p.p2, policy.x2):
            branch_prob = prob1 * prob2
            state_pair = (s1, s2)
            fail_prob = product_failure_probability(state_pair, p)

            if policy.y == 0:
                cost_reward["replacement"][0] += branch_prob * fail_prob * p.replacement_loss
                count_reward["replacements"][0] += branch_prob * fail_prob

            if policy.z == 1:
                cost_reward["disassembly"][0] += branch_prob * fail_prob * p.disassembly_cost
                count_reward["disassemblies"][0] += branch_prob * fail_prob
                Q[0, index[state_pair]] += branch_prob * fail_prob
            else:
                # 坏品报废，重新采购整套零件
                Q[0, index[STATE_N]] += branch_prob * fail_prob
            # 成功概率质量 1-fail_prob 直接进入吸收态 T，不写入 Q

    # ---------- 回收状态 (s1,s2)：先按 r1/r2 处理，再装配 ----------
    for state_pair in PAIR_STATES:
        row = index[state_pair]

        new1, c1, n1 = recovery_process_component(
            state_pair[0], policy.r1, p.p1, p.a1, p.b1
        )
        new2, c2, n2_raw = recovery_process_component(
            state_pair[1], policy.r2, p.p2, p.a2, p.b2
        )
        n2 = remap_component_count_keys(n2_raw, component=2)

        # recovery_process_component 对零件1返回的计数已经可直接使用
        for k, v in c1.items():
            cost_reward[k][row] += v
        for k, v in c2.items():
            cost_reward[k][row] += v
        for k, v in n1.items():
            count_reward[k][row] += v
        for k, v in n2.items():
            count_reward[k][row] += v

        processed_pair = (new1, new2)

        cost_reward["assembly"][row] += p.assembly_cost
        count_reward["assemblies"][row] += 1.0
        if policy.y == 1:
            cost_reward["final_inspection"][row] += p.bf
            count_reward["final_inspections"][row] += 1.0

        fail_prob = product_failure_probability(processed_pair, p)

        if policy.y == 0:
            cost_reward["replacement"][row] += fail_prob * p.replacement_loss
            count_reward["replacements"][row] += fail_prob

        if policy.z == 1:
            cost_reward["disassembly"][row] += fail_prob * p.disassembly_cost
            count_reward["disassemblies"][row] += fail_prob
            Q[row, index[processed_pair]] += fail_prob
        else:
            Q[row, index[STATE_N]] += fail_prob

    return states, Q, cost_reward, count_reward


def evaluate_policy(p: ScenarioParams, policy: Policy) -> PolicyEvaluation:
    """对固定策略解 Bellman 线性方程并返回完整指标。"""
    policy = policy.canonical()
    states, Q, cost_reward, count_reward = build_markov_model(p, policy)

    # 基本概率合法性检查
    row_sums = Q.sum(axis=1)
    if np.any(row_sums > 1.0 + 1e-10) or np.any(Q < -1e-14):
        return PolicyEvaluation(
            p.scenario_id, policy, False, math.inf, -math.inf,
            math.inf, math.inf, math.inf, 0.0, tuple(), {}, {},
            reason="Q矩阵存在非法概率（行和>1或负概率）"
        )

    reachable = reachable_indices(Q, start=0)
    Qr = Q[np.ix_(reachable, reachable)]

    if Qr.size == 0:
        rho = 0.0
    else:
        eigvals = np.linalg.eigvals(Qr)
        rho = float(np.max(np.abs(eigvals))) if len(eigvals) else 0.0

    reachable_state_names = tuple(states[i] for i in reachable)

    if rho >= 1.0 - RHO_TOL:
        return PolicyEvaluation(
            p.scenario_id, policy, False, math.inf, -math.inf,
            rho, math.inf, math.inf, 0.0, reachable_state_names, {}, {},
            reason="暂态矩阵谱半径>=1，存在永久循环/期望成本发散"
        )

    A = np.eye(len(reachable)) - Qr
    cond = float(np.linalg.cond(A))

    total_immediate = np.zeros(len(states), dtype=float)
    for rv in cost_reward.values():
        total_immediate += rv
    c = total_immediate[reachable]

    try:
        V = np.linalg.solve(A, c)
    except np.linalg.LinAlgError as exc:
        return PolicyEvaluation(
            p.scenario_id, policy, False, math.inf, -math.inf,
            rho, cond, math.inf, 0.0, reachable_state_names, {}, {},
            reason=f"Bellman线性方程求解失败: {exc}"
        )

    start_pos = reachable.index(0)
    expected_cost = float(V[start_pos])
    expected_profit = float(p.sale_price - expected_cost)
    residual = float(np.max(np.abs(A @ V - c)))

    # 吸收概率 h = a + Q h，其中 a=1-行和(Q)
    absorb_now = 1.0 - Qr.sum(axis=1)
    h = np.linalg.solve(A, absorb_now)
    absorption_probability = float(h[start_pos])

    breakdown: Dict[str, float] = {}
    for name, rv in cost_reward.items():
        vv = np.linalg.solve(A, rv[reachable])
        breakdown[name] = float(vv[start_pos])

    counts: Dict[str, float] = {}
    for name, rv in count_reward.items():
        vv = np.linalg.solve(A, rv[reachable])
        counts[name] = float(vv[start_pos])

    # 成本分解之和必须回到总成本
    if abs(sum(breakdown.values()) - expected_cost) > 1e-8:
        raise RuntimeError("内部错误：成本分解之和与Bellman总成本不一致")

    return PolicyEvaluation(
        scenario_id=p.scenario_id,
        policy=policy,
        feasible=True,
        expected_cost=expected_cost,
        expected_profit=expected_profit,
        spectral_radius=rho,
        condition_number=cond,
        bellman_residual_inf=residual,
        absorption_probability=absorption_probability,
        reachable_states=reachable_state_names,
        cost_breakdown=breakdown,
        event_counts=counts,
        reason="",
    )


# ============================================================
# 5. 64策略枚举与全局最优
# ============================================================

def generate_raw_64_policies() -> List[Policy]:
    return [Policy(*bits) for bits in itertools.product([0, 1], repeat=6)]


def unique_effective_policies() -> List[Policy]:
    """64个原始策略在 z=0 规范化后得到40个有效六位策略。"""
    seen = {}
    for pol in generate_raw_64_policies():
        can = pol.canonical()
        seen[can.code6] = can
    return [seen[k] for k in sorted(seen)]


def raw64_policy_mapping() -> pd.DataFrame:
    """导出64个原始策略及z=0规范化后的有效策略映射。"""
    rows = []
    for i, pol in enumerate(generate_raw_64_policies(), start=1):
        can = pol.canonical()
        rows.append({
            "raw_index": i,
            "raw_policy6": pol.code6,
            "effective_policy6": can.code6,
            "policy4": can.code4,
            "canonicalized": pol != can,
        })
    return pd.DataFrame(rows)


def evaluate_all_effective_policies(p: ScenarioParams) -> pd.DataFrame:
    rows = []
    for pol in unique_effective_policies():
        ev = evaluate_policy(p, pol)
        row = {
            "scenario_id": p.scenario_id,
            "policy4": pol.code4,
            "policy6": pol.code6,
            "x1": pol.x1, "x2": pol.x2, "y": pol.y, "z": pol.z,
            "r1": pol.r1, "r2": pol.r2,
            "feasible": ev.feasible,
            "expected_cost": ev.expected_cost,
            "expected_profit": ev.expected_profit,
            "spectral_radius": ev.spectral_radius,
            "condition_number": ev.condition_number,
            "bellman_residual_inf": ev.bellman_residual_inf,
            "absorption_probability": ev.absorption_probability,
            "reachable_state_count": len(ev.reachable_states),
            "reason": ev.reason,
        }
        for k, v in ev.cost_breakdown.items():
            row[f"cost_{k}"] = v
        for k, v in ev.event_counts.items():
            row[f"count_{k}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def rank_base4_policies(p: ScenarioParams) -> pd.DataFrame:
    """
    先对每个四位生产策略 (x1,x2,y,z) 选择最优 r1,r2，
    再在16个不同四位策略之间排序。
    """
    df = evaluate_all_effective_policies(p)
    feasible = df[df["feasible"]].copy()
    if feasible.empty:
        raise RuntimeError(f"情形{p.scenario_id}没有可行策略")

    # 每个四位策略中，选利润最高的内部回收规则
    idx = feasible.groupby("policy4", sort=True)["expected_profit"].idxmax()
    ranked = feasible.loc[idx].copy()
    ranked = ranked.sort_values(
        ["expected_profit", "policy4"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def solve_scenario(p: ScenarioParams) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    ranked = rank_base4_policies(p)
    best = ranked.iloc[0]
    second = ranked.iloc[1]
    return ranked, best, second


def policy4_to_text(code: str) -> str:
    x1, x2, y, z = map(int, code)
    return (
        f"零件1{'检' if x1 else '不检'}；"
        f"零件2{'检' if x2 else '不检'}；"
        f"成品{'检' if y else '不检'}；"
        f"坏品{'拆解' if z else '报废'}"
    )


# ============================================================
# 6. 模型检验
# ============================================================

def validate_analytic_degeneracy(params_list: Sequence[ScenarioParams]) -> pd.DataFrame:
    """z=0 时 Bellman 必须退化为几何分布闭式解。"""
    rows = []
    for p in params_list:
        for x1, x2, y in itertools.product([0, 1], repeat=3):
            pol = Policy(x1, x2, y, 0, 0, 0)
            ev = evaluate_policy(p, pol)
            cf = no_disassembly_closed_form(p, x1, x2, y)
            abs_err = abs(ev.expected_cost - cf["cost"])
            rel_err = abs_err / max(1.0, abs(cf["cost"]))
            rows.append({
                "scenario_id": p.scenario_id,
                "policy4": pol.code4,
                "bellman_cost": ev.expected_cost,
                "closed_form_cost": cf["cost"],
                "abs_error": abs_err,
                "relative_error": rel_err,
                "pass_1e-10": rel_err < 1e-10,
            })
    return pd.DataFrame(rows)


def validate_reference_results(params_list: Sequence[ScenarioParams]) -> pd.DataFrame:
    """复现建模手册/优秀解析论文的六情形最优利润。"""
    rows = []
    for p in params_list:
        ranked, best, second = solve_scenario(p)
        ref_code, ref_profit = REFERENCE_BEST[int(p.scenario_id)]
        rows.append({
            "scenario_id": p.scenario_id,
            "model_policy4": best["policy4"],
            "reference_policy4": ref_code,
            "model_profit": best["expected_profit"],
            "reference_profit": ref_profit,
            "profit_abs_error": abs(float(best["expected_profit"]) - ref_profit),
            "policy_match": best["policy4"] == ref_code,
            "profit_match_5e-4": abs(float(best["expected_profit"]) - ref_profit) < 5e-4,
            "second_policy4": second["policy4"],
            "profit_gap": float(best["expected_profit"] - second["expected_profit"]),
        })
    return pd.DataFrame(rows)


def validate_extreme_cases() -> pd.DataFrame:
    """不依赖参考答案的结构性/极端情形检验。"""
    base = SCENARIOS[0]
    rows = []

    # 1) 零缺陷 + 检测有成本：不应检测任何环节
    p_zero = replace(base, scenario_id="ext_zero_defect", p1=0.0, p2=0.0, pf=0.0)
    _, best, _ = solve_scenario(p_zero)
    passed = int(best.x1) == 0 and int(best.x2) == 0 and int(best.y) == 0
    rows.append({
        "test": "零缺陷且检测费>0",
        "expected_behavior": "x1=x2=y=0",
        "observed_policy4": best.policy4,
        "pass": passed,
        "detail": policy4_to_text(best.policy4),
    })

    # 2) 调换损失极高：成品检测应开启
    p_high_L = replace(base, scenario_id="ext_high_L", replacement_loss=1000.0)
    _, best, _ = solve_scenario(p_high_L)
    rows.append({
        "test": "调换损失极高",
        "expected_behavior": "y=1",
        "observed_policy4": best.policy4,
        "pass": int(best.y) == 1,
        "detail": policy4_to_text(best.policy4),
    })

    # 3) 拆解费用极高：拆解应关闭
    p_high_d = replace(base, scenario_id="ext_high_d", disassembly_cost=1000.0)
    _, best, _ = solve_scenario(p_high_d)
    rows.append({
        "test": "拆解费用极高",
        "expected_behavior": "z=0",
        "observed_policy4": best.policy4,
        "pass": int(best.z) == 0,
        "detail": policy4_to_text(best.policy4),
    })

    # 4) 高风险 + 检测完全免费：不应减少检测动作
    p_free = replace(
        SCENARIOS[1], scenario_id="ext_free_inspection", b1=0.0, b2=0.0, bf=0.0
    )
    _, best, _ = solve_scenario(p_free)
    rows.append({
        "test": "高风险且检测免费",
        "expected_behavior": "x1=x2=y=1",
        "observed_policy4": best.policy4,
        "pass": int(best.x1) == 1 and int(best.x2) == 1 and int(best.y) == 1,
        "detail": policy4_to_text(best.policy4),
    })

    # 5) 两个零件都已检测合格，一轮成功率应为1-pf
    p = SCENARIOS[0]
    theoretical = 1.0 - p.pf
    state_pair = (STATE_K, STATE_K)
    actual = 1.0 - product_failure_probability(state_pair, p)
    rows.append({
        "test": "两零件均已知合格时的一轮成功率",
        "expected_behavior": f"1-pf={theoretical:.12f}",
        "observed_policy4": "-",
        "pass": abs(actual - theoretical) < 1e-14,
        "detail": f"model_success={actual:.12f}",
    })

    # 6) 谱半径陷阱：坏的未知回收件永不检测，会永久循环
    trap_policy = Policy(0, 1, 1, 1, 0, 0)
    ev = evaluate_policy(base, trap_policy)
    rows.append({
        "test": "未知坏回收件永不检测的永久循环",
        "expected_behavior": "策略不可行，rho>=1",
        "observed_policy4": trap_policy.code4,
        "pass": (not ev.feasible) and ev.spectral_radius >= 1.0 - RHO_TOL,
        "detail": f"feasible={ev.feasible}, rho={ev.spectral_radius:.12f}",
    })

    return pd.DataFrame(rows)


def validate_matrix_numerics(params_list: Sequence[ScenarioParams]) -> pd.DataFrame:
    """对所有可行策略检查 Bellman 残差、吸收概率、成本分解闭合。"""
    rows = []
    for p in params_list:
        for pol in unique_effective_policies():
            ev = evaluate_policy(p, pol)
            if not ev.feasible:
                continue
            cost_sum = sum(ev.cost_breakdown.values())
            rows.append({
                "scenario_id": p.scenario_id,
                "policy6": pol.code6,
                "spectral_radius": ev.spectral_radius,
                "condition_number": ev.condition_number,
                "bellman_residual_inf": ev.bellman_residual_inf,
                "absorption_probability": ev.absorption_probability,
                "cost_breakdown_error": abs(cost_sum - ev.expected_cost),
                "pass": (
                    ev.spectral_radius < 1.0 - RHO_TOL
                    and ev.bellman_residual_inf < 1e-9
                    and abs(ev.absorption_probability - 1.0) < 1e-9
                    and abs(cost_sum - ev.expected_cost) < 1e-9
                ),
            })
    return pd.DataFrame(rows)


# ============================================================
# 7. Monte Carlo 独立离散事件仿真
# ============================================================

class RunningStats:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.n < 2:
            return float("nan")
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance) if self.n >= 2 else float("nan")

    @property
    def se(self) -> float:
        return self.std / math.sqrt(self.n) if self.n >= 2 else float("nan")


def _purchase_until_good(
    rng: np.random.Generator,
    defect_rate: float,
    purchase_cost: float,
    inspect_cost: float,
    component: int,
    costs: Dict[str, float],
    counts: Dict[str, float],
) -> Tuple[str, bool]:
    """不断新购+检测至合格，返回 (K, True)。"""
    while True:
        costs["purchase"] += purchase_cost
        costs["component_inspection"] += inspect_cost
        counts[f"purchase_{component}"] += 1.0
        counts[f"inspect_{component}"] += 1.0
        good = rng.random() >= defect_rate
        if good:
            return STATE_K, True


def _prepare_initial_component(
    rng: np.random.Generator,
    defect_rate: float,
    purchase_cost: float,
    inspect_cost: float,
    inspect: int,
    component: int,
    costs: Dict[str, float],
    counts: Dict[str, float],
) -> Tuple[str, bool]:
    if inspect == 1:
        return _purchase_until_good(
            rng, defect_rate, purchase_cost, inspect_cost,
            component, costs, counts
        )
    costs["purchase"] += purchase_cost
    counts[f"purchase_{component}"] += 1.0
    good = rng.random() >= defect_rate
    return (STATE_UG if good else STATE_UB), good


def _inspect_recovered_component_mc(
    rng: np.random.Generator,
    state: str,
    actual_good: bool,
    inspect_recovered: int,
    defect_rate: float,
    purchase_cost: float,
    inspect_cost: float,
    component: int,
    costs: Dict[str, float],
    counts: Dict[str, float],
) -> Tuple[str, bool]:
    if state == STATE_K or inspect_recovered == 0:
        return state, actual_good

    # 检测当前回收件一次
    costs["component_inspection"] += inspect_cost
    counts[f"inspect_{component}"] += 1.0
    if actual_good:
        return STATE_K, True

    # 当前回收件为坏件，被检出丢弃；新购+全检至合格
    return _purchase_until_good(
        rng, defect_rate, purchase_cost, inspect_cost,
        component, costs, counts
    )


def simulate_one_order(
    p: ScenarioParams,
    policy: Policy,
    rng: np.random.Generator,
    max_cycles: int = 1_000_000,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """事件级仿真一笔订单，直到客户最终得到合格产品。"""
    policy = policy.canonical()
    costs = {k: 0.0 for k in COST_CATEGORIES}
    counts = {k: 0.0 for k in COUNT_CATEGORIES}

    need_new_set = True
    s1 = s2 = STATE_K
    good1 = good2 = True

    for _ in range(max_cycles):
        if need_new_set:
            s1, good1 = _prepare_initial_component(
                rng, p.p1, p.a1, p.b1, policy.x1, 1, costs, counts
            )
            s2, good2 = _prepare_initial_component(
                rng, p.p2, p.a2, p.b2, policy.x2, 2, costs, counts
            )
            need_new_set = False
        else:
            s1, good1 = _inspect_recovered_component_mc(
                rng, s1, good1, policy.r1,
                p.p1, p.a1, p.b1, 1, costs, counts
            )
            s2, good2 = _inspect_recovered_component_mc(
                rng, s2, good2, policy.r2,
                p.p2, p.a2, p.b2, 2, costs, counts
            )

        # 装配
        costs["assembly"] += p.assembly_cost
        counts["assemblies"] += 1.0

        process_good = rng.random() >= p.pf if (good1 and good2) else False
        product_good = good1 and good2 and process_good

        if policy.y == 1:
            costs["final_inspection"] += p.bf
            counts["final_inspections"] += 1.0

        if product_good:
            total_cost = sum(costs.values())
            return total_cost, costs, counts

        # 产品坏：若不检成品，则已经流入市场并发生换货损失
        if policy.y == 0:
            costs["replacement"] += p.replacement_loss
            counts["replacements"] += 1.0

        if policy.z == 1:
            costs["disassembly"] += p.disassembly_cost
            counts["disassemblies"] += 1.0
            # 拆解无损，保留零件实际质量及其已知/未知信息状态
            need_new_set = False
        else:
            # 整件报废，重新从供应商准备一套
            need_new_set = True

    raise RuntimeError(
        "Monte Carlo 单笔订单超过 max_cycles。若当前策略理论上可行，请检查实现；"
        "否则可能是永久循环策略。"
    )


def monte_carlo_validate_scenario(
    p: ScenarioParams,
    policy: Policy,
    theory: PolicyEvaluation,
    n_orders: int,
    seed: int,
) -> Tuple[dict, List[dict], List[dict]]:
    rng = np.random.default_rng(seed)
    total_stats = RunningStats()
    cost_sums = {k: 0.0 for k in COST_CATEGORIES}
    count_sums = {k: 0.0 for k in COUNT_CATEGORIES}

    for _ in range(n_orders):
        total_cost, costs, counts = simulate_one_order(p, policy, rng)
        total_stats.update(total_cost)
        for k in COST_CATEGORIES:
            cost_sums[k] += costs[k]
        for k in COUNT_CATEGORIES:
            count_sums[k] += counts[k]

    mean_cost = total_stats.mean
    mean_profit = p.sale_price - mean_cost
    se_profit = total_stats.se
    ci_low = mean_profit - 1.96 * se_profit
    ci_high = mean_profit + 1.96 * se_profit
    z_score = (
        (mean_profit - theory.expected_profit) / se_profit
        if se_profit > 0 else 0.0
    )

    summary = {
        "scenario_id": p.scenario_id,
        "policy4": policy.code4,
        "policy6": policy.code6,
        "n_orders": n_orders,
        "seed": seed,
        "theory_cost": theory.expected_cost,
        "mc_mean_cost": mean_cost,
        "theory_profit": theory.expected_profit,
        "mc_mean_profit": mean_profit,
        "mc_profit_se": se_profit,
        "mc_profit_ci95_low": ci_low,
        "mc_profit_ci95_high": ci_high,
        "theory_in_mc_95ci": ci_low <= theory.expected_profit <= ci_high,
        "z_score": z_score,
        "abs_profit_error": abs(mean_profit - theory.expected_profit),
    }

    cost_rows = []
    for k in COST_CATEGORIES:
        mc_mean = cost_sums[k] / n_orders
        theo = theory.cost_breakdown[k]
        cost_rows.append({
            "scenario_id": p.scenario_id,
            "policy4": policy.code4,
            "category": k,
            "theory": theo,
            "mc_mean": mc_mean,
            "abs_error": abs(mc_mean - theo),
            "relative_error": abs(mc_mean - theo) / max(1.0, abs(theo)),
        })

    count_rows = []
    for k in COUNT_CATEGORIES:
        mc_mean = count_sums[k] / n_orders
        theo = theory.event_counts[k]
        count_rows.append({
            "scenario_id": p.scenario_id,
            "policy4": policy.code4,
            "event": k,
            "theory": theo,
            "mc_mean": mc_mean,
            "abs_error": abs(mc_mean - theo),
            "relative_error": abs(mc_mean - theo) / max(1.0, abs(theo)),
        })

    return summary, cost_rows, count_rows


def monte_carlo_batches_scenario4(
    p: ScenarioParams,
    policy: Policy,
    theory: PolicyEvaluation,
    n_batches: int,
    n_per_batch: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """以预先固定的 seed=1,...,n_batches 复核情形4的单次随机偏差。"""
    batch_rows = []
    for batch in range(1, n_batches + 1):
        summary, cost_rows, count_rows = monte_carlo_validate_scenario(
            p, policy, theory, n_orders=n_per_batch, seed=batch
        )
        costs = {row["category"]: row["mc_mean"] for row in cost_rows}
        counts = {row["event"]: row["mc_mean"] for row in count_rows}
        batch_rows.append({
            "batch": batch,
            "seed": batch,
            "n_sim": n_per_batch,
            "theoretical_profit": theory.expected_profit,
            "mc_profit": summary["mc_mean_profit"],
            "se": summary["mc_profit_se"],
            "ci_low": summary["mc_profit_ci95_low"],
            "ci_high": summary["mc_profit_ci95_high"],
            "z_score": summary["z_score"],
            "purchase_cost": costs["purchase"],
            "component_inspection_cost": costs["component_inspection"],
            "assembly_cost": costs["assembly"],
            "final_inspection_cost": costs["final_inspection"],
            "replacement_cost": costs["replacement"],
            "disassembly_cost": costs["disassembly"],
            "mean_purchase_1": counts["purchase_1"],
            "mean_purchase_2": counts["purchase_2"],
            "mean_assembly_count": counts["assemblies"],
        })

    batches = pd.DataFrame(batch_rows)
    batch_sd = float(batches["mc_profit"].std(ddof=1)) if n_batches > 1 else 0.0
    mean_mc_profit = float(batches["mc_profit"].mean())
    mean_bias = mean_mc_profit - theory.expected_profit
    summary = pd.DataFrame([{
        "n_batches": n_batches,
        "n_per_batch": n_per_batch,
        "total_simulations": n_batches * n_per_batch,
        "theoretical_profit": theory.expected_profit,
        "mean_mc_profit": mean_mc_profit,
        "batch_sd": batch_sd,
        "batch_se": batch_sd / math.sqrt(n_batches),
        "mean_bias": mean_bias,
        "relative_bias": mean_bias / theory.expected_profit,
        "mean_abs_z": float(batches["z_score"].abs().mean()),
        "max_abs_z": float(batches["z_score"].abs().max()),
        "coverage_rate": float(
            ((batches["ci_low"] <= theory.expected_profit)
             & (theory.expected_profit <= batches["ci_high"])).mean()
        ),
    }])
    return batches, summary


# ============================================================
# 8. 灵敏度分析
# ============================================================

SENSITIVITY_RANGES = {
    "p1": (0.05, 0.20),
    "p2": (0.05, 0.20),
    "pf": (0.05, 0.20),
    "b1": (1.0, 8.0),
    "b2": (1.0, 8.0),
    "bf": (2.0, 3.0),
    "replacement_loss": (6.0, 30.0),
    "disassembly_cost": (5.0, 40.0),
}

PARAM_LABELS = {
    "p1": "p1",
    "p2": "p2",
    "pf": "pf",
    "b1": "b1",
    "b2": "b2",
    "bf": "bf",
    "replacement_loss": "L",
    "disassembly_cost": "d",
}


def set_param(p: ScenarioParams, name: str, value: float) -> ScenarioParams:
    return replace(p, **{name: float(value)})


def best_profit_for_base4(
    p: ScenarioParams,
    base4: Tuple[int, int, int, int],
) -> Optional[Tuple[float, PolicyEvaluation]]:
    x1, x2, y, z = base4
    rr = [(0, 0)] if z == 0 else list(itertools.product([0, 1], repeat=2))
    candidates = []
    for r1, r2 in rr:
        ev = evaluate_policy(p, Policy(x1, x2, y, z, r1, r2))
        if ev.feasible:
            candidates.append((ev.expected_profit, ev))
    if not candidates:
        return None
    return max(candidates, key=lambda t: t[0])


def best_base4_at(p: ScenarioParams) -> Tuple[Tuple[int, int, int, int], PolicyEvaluation, float]:
    ranked, best, second = solve_scenario(p)
    best_tuple = tuple(map(int, [best.x1, best.x2, best.y, best.z]))
    best_ev = evaluate_policy(
        p, Policy(int(best.x1), int(best.x2), int(best.y), int(best.z), int(best.r1), int(best.r2))
    )
    gap = float(best.expected_profit - second.expected_profit)
    return best_tuple, best_ev, gap


def scan_sensitivity_1d(
    base: ScenarioParams,
    parameter: str,
    low: float,
    high: float,
    points: int,
) -> pd.DataFrame:
    rows = []
    for value in np.linspace(low, high, points):
        p = set_param(base, parameter, float(value))
        ranked, best, second = solve_scenario(p)
        rows.append({
            "scenario_id": base.scenario_id,
            "parameter": parameter,
            "parameter_label": PARAM_LABELS[parameter],
            "value": float(value),
            "best_policy4": best.policy4,
            "best_policy6": best.policy6,
            "best_profit": float(best.expected_profit),
            "second_policy4": second.policy4,
            "second_profit": float(second.expected_profit),
            "profit_gap": float(best.expected_profit - second.expected_profit),
        })
    return pd.DataFrame(rows)


def _policy_code_to_tuple(code: str) -> Tuple[int, int, int, int]:
    return tuple(int(ch) for ch in code)  # type: ignore[return-value]


def refine_switch_threshold(
    base: ScenarioParams,
    parameter: str,
    left: float,
    right: float,
    policy_left_code: str,
    policy_right_code: str,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """
    在相邻扫描点之间，以两个四位候选策略的“最优内部r规则后利润差=0”做二分求根。
    """
    A = _policy_code_to_tuple(policy_left_code)
    B = _policy_code_to_tuple(policy_right_code)

    def f(x: float) -> float:
        p = set_param(base, parameter, x)
        a = best_profit_for_base4(p, A)
        b = best_profit_for_base4(p, B)
        if a is None or b is None:
            return float("nan")
        return a[0] - b[0]

    fl = f(left)
    fr = f(right)
    if not np.isfinite(fl) or not np.isfinite(fr) or fl * fr > 0:
        return (left + right) / 2.0

    a, b = left, right
    fa, fb = fl, fr
    for _ in range(max_iter):
        m = 0.5 * (a + b)
        fm = f(m)
        if not np.isfinite(fm):
            break
        if abs(fm) < 1e-12 or (b - a) < tol:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5 * (a + b)


def detect_sensitivity_thresholds(
    base: ScenarioParams,
    scan_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for parameter, g in scan_df.groupby("parameter", sort=False):
        g = g.sort_values("value").reset_index(drop=True)
        for i in range(1, len(g)):
            prev_code = str(g.loc[i - 1, "best_policy4"])
            curr_code = str(g.loc[i, "best_policy4"])
            if prev_code == curr_code:
                continue
            left = float(g.loc[i - 1, "value"])
            right = float(g.loc[i, "value"])
            threshold = refine_switch_threshold(
                base, parameter, left, right, prev_code, curr_code
            )
            p_star = set_param(base, parameter, threshold)
            pa = best_profit_for_base4(p_star, _policy_code_to_tuple(prev_code))
            pb = best_profit_for_base4(p_star, _policy_code_to_tuple(curr_code))
            profit_a = pa[0] if pa else float("nan")
            profit_b = pb[0] if pb else float("nan")
            rows.append({
                "scenario_id": base.scenario_id,
                "parameter": parameter,
                "parameter_label": PARAM_LABELS[parameter],
                "threshold": threshold,
                "policy_before": prev_code,
                "policy_after": curr_code,
                "profit_before_at_threshold": profit_a,
                "profit_after_at_threshold": profit_b,
                "profit_difference_at_threshold": profit_a - profit_b,
            })
    return pd.DataFrame(rows)


def phase_map_p2_L(
    base: ScenarioParams,
    p2_low: float,
    p2_high: float,
    L_low: float,
    L_high: float,
    points: int,
) -> pd.DataFrame:
    """问题二正式灵敏度图的数据底稿：p2-L 二维最优策略相图。"""
    rows = []
    p2_values = np.linspace(p2_low, p2_high, points)
    L_values = np.linspace(L_low, L_high, points)
    for p2v in p2_values:
        for Lv in L_values:
            p = replace(base, p2=float(p2v), replacement_loss=float(Lv))
            ranked, best, second = solve_scenario(p)
            rows.append({
                "scenario_id": base.scenario_id,
                "p2": float(p2v),
                "replacement_loss": float(Lv),
                "best_policy4": best.policy4,
                "best_policy6": best.policy6,
                "best_profit": float(best.expected_profit),
                "profit_gap": float(best.expected_profit - second.expected_profit),
            })
    return pd.DataFrame(rows)


# ============================================================
# 9. 导出与主流程
# ============================================================

def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def print_scenario_summary(summary_df: pd.DataFrame) -> None:
    print("\n" + "=" * 100)
    print("问题二：六种情形最优决策（单位订单口径）")
    print("=" * 100)
    cols = [
        "scenario_id", "best_policy4", "best_policy6", "expected_profit",
        "expected_cost", "second_policy4", "profit_gap"
    ]
    print(summary_df[cols].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\n四位编码顺序：x1 x2 y z；1=执行该动作，0=不执行。")


def run_all(args: argparse.Namespace) -> None:
    # 默认输出到脚本所在目录的 ../结果输出/，
    # 即 求解代码/第二问/结果输出/；PyCharm 直接运行即可生效。
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------- 1. 原始参数 ----------
    params_df = params_to_dataframe(SCENARIOS)
    save_csv(params_df, output_dir / "params_table1.csv")
    save_csv(raw64_policy_mapping(), output_dir / "raw_64_policy_mapping.csv")

    # ---------- 2. 全策略求解 ----------
    all_effective_rows = []
    all_rank_rows = []
    summary_rows = []
    cost_rows = []
    count_rows = []
    optimal_evaluations: Dict[int, Tuple[ScenarioParams, Policy, PolicyEvaluation]] = {}

    for p in SCENARIOS:
        all_eval = evaluate_all_effective_policies(p)
        all_effective_rows.append(all_eval)

        ranked, best, second = solve_scenario(p)
        all_rank_rows.append(ranked)

        best_policy = Policy(
            int(best.x1), int(best.x2), int(best.y), int(best.z), int(best.r1), int(best.r2)
        )
        best_ev = evaluate_policy(p, best_policy)
        optimal_evaluations[int(p.scenario_id)] = (p, best_policy, best_ev)

        summary_rows.append({
            "scenario_id": p.scenario_id,
            "best_policy4": best.policy4,
            "best_policy6": best.policy6,
            "decision_text": policy4_to_text(best.policy4),
            "x1": int(best.x1), "x2": int(best.x2),
            "y": int(best.y), "z": int(best.z),
            "r1": int(best.r1), "r2": int(best.r2),
            "expected_profit": best_ev.expected_profit,
            "expected_cost": best_ev.expected_cost,
            "second_policy4": second.policy4,
            "second_profit": float(second.expected_profit),
            "profit_gap": float(best.expected_profit - second.expected_profit),
            "spectral_radius": best_ev.spectral_radius,
            "bellman_residual_inf": best_ev.bellman_residual_inf,
            "absorption_probability": best_ev.absorption_probability,
        })

        crow = {"scenario_id": p.scenario_id, "policy4": best.policy4}
        crow.update(best_ev.cost_breakdown)
        crow["total_expected_cost"] = best_ev.expected_cost
        cost_rows.append(crow)

        nrow = {"scenario_id": p.scenario_id, "policy4": best.policy4}
        nrow.update(best_ev.event_counts)
        count_rows.append(nrow)

    all_effective_df = pd.concat(all_effective_rows, ignore_index=True)
    all_rank_df = pd.concat(all_rank_rows, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)
    cost_df = pd.DataFrame(cost_rows)
    count_df = pd.DataFrame(count_rows)

    save_csv(all_effective_df, output_dir / "all_effective_40_policy_evaluations.csv")
    save_csv(all_rank_df, output_dir / "policy_rank_16_base_policies.csv")
    save_csv(summary_df, output_dir / "optimal_policy_summary.csv")
    save_csv(cost_df, output_dir / "optimal_cost_breakdown.csv")
    save_csv(count_df, output_dir / "optimal_event_counts.csv")

    print_scenario_summary(summary_df)

    # ---------- 3. 解析退化检验 ----------
    analytic_df = validate_analytic_degeneracy(SCENARIOS)
    save_csv(analytic_df, output_dir / "validation_analytic_degeneracy.csv")
    max_rel_err = float(analytic_df["relative_error"].max())

    # ---------- 4. 参考结果复现 ----------
    ref_df = validate_reference_results(SCENARIOS)
    save_csv(ref_df, output_dir / "validation_reference_reproduction.csv")

    # ---------- 5. 数值/吸收性检验 ----------
    matrix_df = validate_matrix_numerics(SCENARIOS)
    save_csv(matrix_df, output_dir / "validation_matrix_numerics.csv")

    # ---------- 6. 极端情形检验 ----------
    extreme_df = validate_extreme_cases()
    save_csv(extreme_df, output_dir / "validation_extreme_cases.csv")

    print("\n" + "-" * 100)
    print("模型检验摘要")
    print("-" * 100)
    print(f"解析退化最大相对误差: {max_rel_err:.3e}  （目标 < 1e-10）")
    print(f"参考结果策略全部复现: {bool(ref_df['policy_match'].all())}")
    print(f"参考利润全部在5e-4内复现: {bool(ref_df['profit_match_5e-4'].all())}")
    print(f"所有可行策略Bellman/吸收性数值检查通过: {bool(matrix_df['pass'].all())}")
    print(f"所有极端情形结构检查通过: {bool(extreme_df['pass'].all())}")

    # 这些是确定性检验，应当全部通过；否则立即报错，不输出“看似正常”的结果。
    if max_rel_err >= 1e-10:
        raise AssertionError("解析退化检验失败：Bellman模型未正确退化到几何分布闭式模型")
    if not bool(ref_df["policy_match"].all() and ref_df["profit_match_5e-4"].all()):
        raise AssertionError("参考结果复现检验失败")
    if not bool(matrix_df["pass"].all()):
        raise AssertionError("Bellman数值/吸收性检验失败")
    if not bool(extreme_df["pass"].all()):
        raise AssertionError("极端情形结构检验失败")

    # ---------- 7. Monte Carlo ----------
    if not args.skip_mc:
        mc_summary_rows = []
        mc_cost_rows = []
        mc_count_rows = []
        for scenario_id, (p, pol, theory) in optimal_evaluations.items():
            summary, crows, nrows = monte_carlo_validate_scenario(
                p, pol, theory,
                n_orders=args.mc_orders,
                seed=args.seed + scenario_id * 1009,
            )
            mc_summary_rows.append(summary)
            mc_cost_rows.extend(crows)
            mc_count_rows.extend(nrows)

        mc_summary_df = pd.DataFrame(mc_summary_rows)
        mc_cost_df = pd.DataFrame(mc_cost_rows)
        mc_count_df = pd.DataFrame(mc_count_rows)
        save_csv(mc_summary_df, output_dir / "validation_mc_summary.csv")
        save_csv(mc_cost_df, output_dir / "validation_mc_cost_breakdown.csv")
        save_csv(mc_count_df, output_dir / "validation_mc_event_counts.csv")

        # 情形4原单次区间曾偶然漏覆盖理论值；用运行前固定的多批次方案复核，
        # 不更换种子追求“通过”，也不改变原六情形单次验证结果。
        p4, pol4, theory4 = optimal_evaluations[4]
        mc_batches4_df, mc_batches4_summary_df = monte_carlo_batches_scenario4(
            p4, pol4, theory4,
            n_batches=args.mc_batches_scenario4,
            n_per_batch=args.mc_batch_orders,
        )
        save_csv(
            mc_batches4_df,
            output_dir / "validation_mc_batches_scenario4.csv",
        )
        save_csv(
            mc_batches4_summary_df,
            output_dir / "validation_mc_batches_summary_scenario4.csv",
        )

        print("\nMonte Carlo 独立验证：")
        print(mc_summary_df[[
            "scenario_id", "policy4", "n_orders", "theory_profit", "mc_mean_profit",
            "mc_profit_ci95_low", "mc_profit_ci95_high", "theory_in_mc_95ci", "z_score"
        ]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
        print("\n情形4多批次 Monte Carlo 复核：")
        print(mc_batches4_summary_df.to_string(
            index=False, float_format=lambda x: f"{x:.6f}"
        ))

    # ---------- 8. 灵敏度：情形1，题目实际区间 ----------
    base = SCENARIOS[0]
    sens_frames = []
    for parameter, (low, high) in SENSITIVITY_RANGES.items():
        df = scan_sensitivity_1d(base, parameter, low, high, args.sens_points)
        sens_frames.append(df)
    sens_df = pd.concat(sens_frames, ignore_index=True)
    threshold_df = detect_sensitivity_thresholds(base, sens_df)
    save_csv(sens_df, output_dir / "sensitivity_1d_scenario1.csv")
    save_csv(threshold_df, output_dir / "sensitivity_thresholds_scenario1.csv")

    print("\n" + "-" * 100)
    print("情形1：单参数决策切换临界值（由全策略重求最优 + 二分求根自动得到）")
    print("-" * 100)
    if threshold_df.empty:
        print("在指定题目区间内未检测到策略切换。")
    else:
        print(threshold_df[[
            "parameter_label", "threshold", "policy_before", "policy_after",
            "profit_difference_at_threshold"
        ]].to_string(index=False, float_format=lambda x: f"{x:.8f}"))

    # ---------- 9. p2-L 二维相图数据 ----------
    phase_df = phase_map_p2_L(
        base,
        p2_low=SENSITIVITY_RANGES["p2"][0],
        p2_high=SENSITIVITY_RANGES["p2"][1],
        L_low=SENSITIVITY_RANGES["replacement_loss"][0],
        L_high=SENSITIVITY_RANGES["replacement_loss"][1],
        points=args.phase_points,
    )
    save_csv(phase_df, output_dir / "sensitivity_phase_p2_L_scenario1.csv")

    # ---------- 10. 运行总结 ----------
    with open(output_dir / "run_summary.txt", "w", encoding="utf-8") as f:
        f.write("2024国赛B题问题二——Python求解运行总结\n")
        f.write("=" * 72 + "\n")
        f.write("模型：信息保持型闭环 Bellman–Markov + 64原始策略全枚举\n")
        f.write("核算口径：最终成功交付1件合格成品\n\n")
        f.write(summary_df.to_string(index=False) + "\n\n")
        f.write(f"解析退化最大相对误差 = {max_rel_err:.3e}\n")
        f.write(f"参考策略复现全部通过 = {bool(ref_df['policy_match'].all())}\n")
        f.write(f"参考利润复现全部通过 = {bool(ref_df['profit_match_5e-4'].all())}\n")
        f.write(f"数值/吸收性检查全部通过 = {bool(matrix_df['pass'].all())}\n")
        f.write(f"极端情形检查全部通过 = {bool(extreme_df['pass'].all())}\n")
        if not args.skip_mc:
            f.write("Monte Carlo结果见 validation_mc_summary.csv\n")
            f.write(
                "情形4多批次复核见 validation_mc_batches_scenario4.csv 与 "
                "validation_mc_batches_summary_scenario4.csv\n"
            )
        f.write("\n灵敏度临界值：\n")
        if threshold_df.empty:
            f.write("指定区间内无策略切换。\n")
        else:
            f.write(threshold_df.to_string(index=False) + "\n")

    print(f"\n全部计算完成，结果已输出到：{output_dir.resolve()}")
    print("未绘制任何图；后续 MATLAB 可直接读取 sensitivity / cost / policy CSV。")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2024国赛B题问题二：闭环Bellman–Markov生产决策模型"
    )
    parser.add_argument(
        "--output-dir", type=str, default="../结果输出",
        help=(
            "结果输出目录，默认相对脚本所在目录的 ../结果输出。"
            "PyCharm 中直接运行即可把全部 CSV 与运行总结写入"
            "求解代码/第二问/结果输出/，供 MATLAB 绘图复用。"
        ),
    )
    parser.add_argument(
        "--mc-orders", type=int, default=100_000,
        help="每个最优策略的Monte Carlo订单数，手册建议至少1e5，默认100000"
    )
    parser.add_argument(
        "--mc-batches-scenario4", type=int, default=10,
        help="情形4独立Monte Carlo复核批次数，默认10（固定seed=1,...,10）"
    )
    parser.add_argument(
        "--mc-batch-orders", type=int, default=100_000,
        help="情形4每个复核批次的订单数，默认100000"
    )
    parser.add_argument(
        "--sens-points", type=int, default=101,
        help="每个单参数灵敏度扫描网格点数，默认101；临界值随后用二分法精确细化"
    )
    parser.add_argument(
        "--phase-points", type=int, default=31,
        help="p2-L二维相图每个轴网格点数，默认31；可按论文制图需要提高"
    )
    parser.add_argument(
        "--seed", type=int, default=20240820,
        help="Monte Carlo随机种子，默认20240820"
    )
    parser.add_argument(
        "--skip-mc", action="store_true",
        help="跳过Monte Carlo，仅做精确求解、确定性检验和灵敏度"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="快速测试模式：MC=10000，1D=101点，2D=31点"
    )
    args = parser.parse_args(argv)

    if args.quick:
        args.mc_orders = 10_000
        args.mc_batch_orders = 10_000
        args.sens_points = 51
        args.phase_points = 21

    if args.mc_orders <= 0:
        parser.error("--mc-orders 必须为正整数")
    if args.mc_batches_scenario4 <= 0:
        parser.error("--mc-batches-scenario4 必须为正整数")
    if args.mc_batch_orders <= 0:
        parser.error("--mc-batch-orders 必须为正整数")
    if args.sens_points < 21:
        parser.error("--sens-points 建议至少21")
    if args.phase_points < 11:
        parser.error("--phase-points 建议至少11")
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    run_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
