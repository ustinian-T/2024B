# -*- coding: utf-8 -*-
"""
2024 高教社杯数学建模国赛 B 题 —— 问题三
信息保持型 BOM 树状态压缩动态规划 + 65536 静态策略精确枚举 + 模型检验 + 灵敏度分析

对应建模手册：
《问题三建模手册——基于信息保持型 BOM 树状态压缩动态规划与精确枚举交叉验证》

运行（PyCharm 直接运行本文件即可）：
    python Q3_model_full.py

默认结果输出目录：脚本所在目录的 ../结果输出/
（即 求解代码/第三问/结果输出/），供 MATLAB 绘图复用。

可选参数：
    python Q3_model_full.py --outdir ../结果输出 --mc 200000 --generic-mc 10000
    python Q3_model_full.py --skip-generic-mc
    python Q3_model_full.py --skip-sensitivity

输出：
    01_all_strategies.csv
    02_top10_strategies.csv
    03_optimal_policy.csv
    04_cost_breakdown.csv
    05_validation_summary.csv
    06_mc_vectorized_summary.csv
    07_mc_event_summary.csv (若未跳过)
    08_mc_convergence.csv
    09_sensitivity_thresholds.csv
    10_sensitivity_component_fee.csv
    11_sensitivity_final_phase.csv
    12_sensitivity_local_curves.csv
    13_sensitivity_global_spotcheck.csv
    14_extreme_case_checks.csv
    run_summary.txt

注意：
1) 不使用遗传算法、蚁群算法、模拟退火等元启发式算法；本例 16 个 0-1 位只有 2^16=65536 个静态策略，
   精确枚举更透明，且能证明静态策略空间的全局最优。
2) 半成品/成品的 10% 是“所有输入均合格时，装配工序自身产生次品”的条件次品率，不能直接当总次品率。
3) 拆解无损，已检测合格对象保持 Known-good；父节点失败后，对未知子节点使用 Bayes 条件概率更新。
4) 主核算口径为“最终成功交付 1 件合格成品的期望成本/利润”，售价只记一次。
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple, Sequence, Optional

import numpy as np
import pandas as pd


# ============================================================
# 0. 常量与数据结构
# ============================================================

EPS = 1e-12
POLICY_NAMES = [
    *(f"x_C{i}" for i in range(1, 9)),
    "y_H1", "y_H2", "y_H3",
    "z_H1", "z_H2", "z_H3",
    "y_F", "z_F",
]

COST_CATEGORIES = [
    "purchase",               # 零配件采购
    "component_inspection",   # 零配件检测
    "assembly",               # 半成品+成品装配
    "semi_inspection",        # 半成品检测
    "final_inspection",       # 成品检测
    "replacement",            # 市场调换损失
    "disassembly",            # 半成品+成品拆解
]

COST_CN = {
    "purchase": "采购成本",
    "component_inspection": "零配件检测成本",
    "assembly": "装配成本",
    "semi_inspection": "半成品检测成本",
    "final_inspection": "成品检测成本",
    "replacement": "调换损失",
    "disassembly": "拆解成本",
}


@dataclass
class Q3Parameters:
    """题目表2及BOM结构参数。所有金额单位：元/件。"""

    p_comp: List[float]
    purchase_comp: List[float]
    inspect_comp: List[float]

    children: Dict[str, List[int]]
    p_semi: Dict[str, float]
    asm_semi: Dict[str, float]
    inspect_semi: Dict[str, float]
    disasm_semi: Dict[str, float]

    p_final: float
    asm_final: float
    inspect_final: float
    disasm_final: float
    sale_price: float
    replacement_loss: float

    def clone(self) -> "Q3Parameters":
        return copy.deepcopy(self)


@dataclass
class NodeDesc:
    """
    子树状态压缩摘要。

    C      : 按当前初始策略获得一个节点输出的期望成本（fresh_cost）
    q      : 该输出实际为合格品的概率（good_prob）
    known  : 企业是否已经知道该输出必然合格（Known-good）
    K      : 从头获得一个“已知合格”节点的期望成本（certified_cost）
    D      : 一个节点已被检测确认不合格后，将其恢复为已知合格的期望成本（repair_cost）
    b      : 该节点单次检测费，用于父节点拆解后认证未知回收件
    """

    C: float
    q: float
    known: bool
    K: float
    D: float
    b: float


@dataclass
class VecNodeDesc:
    """与 NodeDesc 相同，但 C/K/D 用成本分解向量表示。仅对少量策略使用。"""

    C: np.ndarray
    q: float
    known: bool
    K: np.ndarray
    D: np.ndarray
    b: float
    inspect_category: str


@dataclass
class LocalCandidate:
    name: str
    comp_indices: Tuple[int, ...]
    x_bits: Tuple[int, ...]
    y: int
    z: int
    desc: NodeDesc


@dataclass
class SimItem:
    """Monte Carlo 离散事件模拟中的实物状态。good 是潜在真实质量，known 是企业信息状态。"""

    node: str
    good: bool
    known: bool
    children: Optional[List["SimItem"]] = None


# ============================================================
# 1. 题目表2参数与数据校验
# ============================================================

def default_parameters() -> Q3Parameters:
    """严格按题目表2录入参数，不引入任何额外经验参数。"""
    return Q3Parameters(
        p_comp=[0.10] * 8,
        purchase_comp=[2, 8, 12, 2, 8, 12, 8, 12],
        inspect_comp=[1, 1, 2, 1, 1, 2, 1, 2],
        children={
            "H1": [0, 1, 2],
            "H2": [3, 4, 5],
            "H3": [6, 7],
        },
        p_semi={"H1": 0.10, "H2": 0.10, "H3": 0.10},
        asm_semi={"H1": 8.0, "H2": 8.0, "H3": 8.0},
        inspect_semi={"H1": 4.0, "H2": 4.0, "H3": 4.0},
        disasm_semi={"H1": 6.0, "H2": 6.0, "H3": 6.0},
        p_final=0.10,
        asm_final=8.0,
        inspect_final=6.0,
        disasm_final=10.0,
        sale_price=200.0,
        replacement_loss=40.0,
    )


def validate_parameters(par: Q3Parameters) -> None:
    """数据预处理/一致性检查：概率、成本、BOM覆盖性。"""
    assert len(par.p_comp) == len(par.purchase_comp) == len(par.inspect_comp) == 8

    all_probs = list(par.p_comp) + list(par.p_semi.values()) + [par.p_final]
    if not all(0.0 <= p < 1.0 for p in all_probs):
        raise ValueError("所有次品率必须满足 0 <= p < 1。")

    all_costs = (
        list(par.purchase_comp)
        + list(par.inspect_comp)
        + list(par.asm_semi.values())
        + list(par.inspect_semi.values())
        + list(par.disasm_semi.values())
        + [
            par.asm_final,
            par.inspect_final,
            par.disasm_final,
            par.sale_price,
            par.replacement_loss,
        ]
    )
    if not all(c >= 0 for c in all_costs):
        raise ValueError("所有成本/售价参数必须非负。")

    # 正向BOM必须覆盖8个零件，且每个零件在本题树结构中只属于一个半成品。
    flattened = [i for name in ("H1", "H2", "H3") for i in par.children[name]]
    if sorted(flattened) != list(range(8)):
        raise ValueError("BOM边表错误：H1/H2/H3 的子零件必须恰好覆盖 C1~C8。")

    if set(par.children) != {"H1", "H2", "H3"}:
        raise ValueError("本题具体实例必须包含 H1/H2/H3 三个半成品节点。")


# ============================================================
# 2. 标量精确评价器：公式(7)~(26)
# ============================================================

def leaf_desc(par: Q3Parameters, i: int, x: int) -> NodeDesc:
    """手册式(7)~(9)。"""
    p = par.p_comp[i]
    a = par.purchase_comp[i]
    b = par.inspect_comp[i]
    K = (a + b) / (1.0 - p)
    D = K
    if x == 1:
        return NodeDesc(C=K, q=1.0, known=True, K=K, D=D, b=b)
    return NodeDesc(C=a, q=1.0 - p, known=False, K=K, D=D, b=b)


def recovery_cost(desc: NodeDesc, q_cond: float) -> float:
    """手册式(14)~(15)：未知回收件认证成本 R_u(q)。"""
    if desc.known:
        return 0.0
    q_cond = min(1.0, max(0.0, q_cond))
    return desc.b + (1.0 - q_cond) * desc.D


def internal_desc(
    ch_descs: Sequence[NodeDesc],
    p_proc: float,
    asm_cost: float,
    inspect_cost: float,
    disasm_cost: float,
    y: int,
    z: int,
) -> NodeDesc:
    """半成品/被检测成品的状态压缩递推，对应手册式(10)~(21)。"""
    B = asm_cost + sum(c.C for c in ch_descs)
    g = (1.0 - p_proc) * math.prod(c.q for c in ch_descs)

    if not (0.0 < g <= 1.0 + EPS):
        raise ArithmeticError(f"节点一次成功概率 g={g} 非法。")
    g = min(1.0, g)

    if z == 0:
        # 失败后整节点报废，从头生产并检测，直到获得已知合格输出。
        K = (B + inspect_cost) / g
        D = K
    else:
        # 父节点失败后，若拆解，则对未知子件做Bayes条件更新并认证。
        fail_prob = 1.0 - g
        cert = 0.0
        if fail_prob > EPS:
            for c in ch_descs:
                if c.known:
                    continue
                # 手册式(12)(13): P(u坏|v坏)=(1-q_u)/(1-g_v)
                bad_post = (1.0 - c.q) / fail_prob
                bad_post = min(1.0, max(0.0, bad_post))
                q_post = 1.0 - bad_post
                cert += recovery_cost(c, q_post)

        # 当全部输入已知合格，只剩工序自身失败；对工序失败不断拆解重装。
        K_proc = (asm_cost + inspect_cost + p_proc * disasm_cost) / (1.0 - p_proc)
        D = disasm_cost + cert + K_proc
        K = B + inspect_cost + fail_prob * D

    if y == 1:
        return NodeDesc(C=K, q=1.0, known=True, K=K, D=D, b=inspect_cost)
    return NodeDesc(C=B, q=g, known=False, K=K, D=D, b=inspect_cost)


def final_cost(
    par: Q3Parameters,
    h_descs: Sequence[NodeDesc],
    y_f: int,
    z_f: int,
) -> float:
    """根节点市场退换闭环，对应手册式(22)~(24)。"""
    B = par.asm_final + sum(c.C for c in h_descs)
    g = (1.0 - par.p_final) * math.prod(c.q for c in h_descs)
    if not (0.0 < g <= 1.0 + EPS):
        raise ArithmeticError(f"成品一次成功概率 g_F={g} 非法。")
    g = min(1.0, g)

    if y_f == 1:
        # 成品出厂检测：坏品在厂内截获，无市场调换损失。
        desc = internal_desc(
            h_descs,
            par.p_final,
            par.asm_final,
            par.inspect_final,
            par.disasm_final,
            y=1,
            z=z_f,
        )
        return desc.C

    # 不检测成品：失败会流入市场并产生调换损失。
    if z_f == 0:
        # 每次失败整套报废，从头开始；几何重复闭式。
        return (B + (1.0 - g) * par.replacement_loss) / g

    # 不检测 + 用户退回后拆解。
    fail_prob = 1.0 - g
    cert = 0.0
    if fail_prob > EPS:
        for c in h_descs:
            if c.known:
                continue
            bad_post = (1.0 - c.q) / fail_prob
            bad_post = min(1.0, max(0.0, bad_post))
            q_post = 1.0 - bad_post
            cert += recovery_cost(c, q_post)

    # 三个半成品全部认证合格后，只剩最终装配工序自身随机失败。
    M_f = (
        par.asm_final
        + par.p_final * (par.replacement_loss + par.disasm_final)
    ) / (1.0 - par.p_final)

    return B + fail_prob * (
        par.replacement_loss + par.disasm_final + cert + M_f
    )


def evaluate_policy(par: Q3Parameters, bits: Sequence[int]) -> Tuple[float, float]:
    """返回 (单位订单期望利润, 单位订单期望成本)。"""
    if len(bits) != 16 or any(b not in (0, 1) for b in bits):
        raise ValueError("策略必须是长度16的0-1序列。")

    xs = bits[:8]
    ys = bits[8:11]
    zs = bits[11:14]
    y_f, z_f = bits[14:16]

    leaves = [leaf_desc(par, i, xs[i]) for i in range(8)]
    h_descs: List[NodeDesc] = []
    for j, name in enumerate(("H1", "H2", "H3")):
        ch = [leaves[i] for i in par.children[name]]
        h_descs.append(
            internal_desc(
                ch,
                par.p_semi[name],
                par.asm_semi[name],
                par.inspect_semi[name],
                par.disasm_semi[name],
                ys[j],
                zs[j],
            )
        )

    V = final_cost(par, h_descs, y_f, z_f)
    profit = par.sale_price - V
    return profit, V


# ============================================================
# 3. 成本分解评价器：与标量评价器同一递推，但追踪成本来源
# ============================================================

def zvec() -> np.ndarray:
    return np.zeros(len(COST_CATEGORIES), dtype=float)


def unit_vec(category: str, amount: float) -> np.ndarray:
    v = zvec()
    v[COST_CATEGORIES.index(category)] = amount
    return v


def leaf_vec_desc(par: Q3Parameters, i: int, x: int) -> VecNodeDesc:
    p, a, b = par.p_comp[i], par.purchase_comp[i], par.inspect_comp[i]
    K = (
        unit_vec("purchase", a)
        + unit_vec("component_inspection", b)
    ) / (1.0 - p)
    if x == 1:
        return VecNodeDesc(K.copy(), 1.0, True, K.copy(), K.copy(), b, "component_inspection")
    C = unit_vec("purchase", a)
    return VecNodeDesc(C, 1.0 - p, False, K.copy(), K.copy(), b, "component_inspection")


def recovery_vec(desc: VecNodeDesc, q_cond: float) -> np.ndarray:
    if desc.known:
        return zvec()
    q_cond = min(1.0, max(0.0, q_cond))
    return unit_vec(desc.inspect_category, desc.b) + (1.0 - q_cond) * desc.D


def internal_vec_desc(
    ch_descs: Sequence[VecNodeDesc],
    p_proc: float,
    asm_cost: float,
    inspect_cost: float,
    disasm_cost: float,
    y: int,
    z: int,
    inspect_category: str,
) -> VecNodeDesc:
    B = unit_vec("assembly", asm_cost)
    for c in ch_descs:
        B = B + c.C
    g = (1.0 - p_proc) * math.prod(c.q for c in ch_descs)
    fail_prob = 1.0 - g

    if z == 0:
        K = (B + unit_vec(inspect_category, inspect_cost)) / g
        D = K.copy()
    else:
        cert = zvec()
        if fail_prob > EPS:
            for c in ch_descs:
                if c.known:
                    continue
                bad_post = min(1.0, max(0.0, (1.0 - c.q) / fail_prob))
                cert += recovery_vec(c, 1.0 - bad_post)

        K_proc = (
            unit_vec("assembly", asm_cost)
            + unit_vec(inspect_category, inspect_cost)
            + p_proc * unit_vec("disassembly", disasm_cost)
        ) / (1.0 - p_proc)
        D = unit_vec("disassembly", disasm_cost) + cert + K_proc
        K = B + unit_vec(inspect_category, inspect_cost) + fail_prob * D

    if y == 1:
        return VecNodeDesc(K.copy(), 1.0, True, K.copy(), D.copy(), inspect_cost, inspect_category)
    return VecNodeDesc(B.copy(), g, False, K.copy(), D.copy(), inspect_cost, inspect_category)


def final_cost_vec(par: Q3Parameters, h_descs: Sequence[VecNodeDesc], y_f: int, z_f: int) -> np.ndarray:
    B = unit_vec("assembly", par.asm_final)
    for c in h_descs:
        B += c.C
    g = (1.0 - par.p_final) * math.prod(c.q for c in h_descs)
    fail_prob = 1.0 - g

    if y_f == 1:
        desc = internal_vec_desc(
            h_descs,
            par.p_final,
            par.asm_final,
            par.inspect_final,
            par.disasm_final,
            1,
            z_f,
            "final_inspection",
        )
        return desc.C

    if z_f == 0:
        return B / g + unit_vec("replacement", par.replacement_loss) * (fail_prob / g)

    cert = zvec()
    if fail_prob > EPS:
        for c in h_descs:
            if c.known:
                continue
            bad_post = min(1.0, max(0.0, (1.0 - c.q) / fail_prob))
            cert += recovery_vec(c, 1.0 - bad_post)

    M = (
        unit_vec("assembly", par.asm_final)
        + par.p_final
        * (
            unit_vec("replacement", par.replacement_loss)
            + unit_vec("disassembly", par.disasm_final)
        )
    ) / (1.0 - par.p_final)

    return B + fail_prob * (
        unit_vec("replacement", par.replacement_loss)
        + unit_vec("disassembly", par.disasm_final)
        + cert
        + M
    )


def evaluate_policy_breakdown(par: Q3Parameters, bits: Sequence[int]) -> Tuple[float, np.ndarray]:
    xs = bits[:8]
    ys = bits[8:11]
    zs = bits[11:14]
    y_f, z_f = bits[14:16]

    leaves = [leaf_vec_desc(par, i, xs[i]) for i in range(8)]
    hs: List[VecNodeDesc] = []
    for j, name in enumerate(("H1", "H2", "H3")):
        ch = [leaves[i] for i in par.children[name]]
        hs.append(
            internal_vec_desc(
                ch,
                par.p_semi[name],
                par.asm_semi[name],
                par.inspect_semi[name],
                par.disasm_semi[name],
                ys[j],
                zs[j],
                "semi_inspection",
            )
        )
    vec = final_cost_vec(par, hs, y_f, z_f)
    return float(vec.sum()), vec


# ============================================================
# 4. 两条求解路径：树形DP候选表 + 16位平坦精确枚举
# ============================================================

def bits_to_string(bits: Sequence[int]) -> str:
    return "".join(str(int(b)) for b in bits)


def bits_to_dict(bits: Sequence[int]) -> Dict[str, int]:
    return dict(zip(POLICY_NAMES, map(int, bits)))


def build_local_candidates(par: Q3Parameters, name: str) -> List[LocalCandidate]:
    inds = tuple(par.children[name])
    out: List[LocalCandidate] = []
    for x_bits in product((0, 1), repeat=len(inds)):
        leaves = [leaf_desc(par, idx, x) for idx, x in zip(inds, x_bits)]
        for y, z in product((0, 1), repeat=2):
            desc = internal_desc(
                leaves,
                par.p_semi[name],
                par.asm_semi[name],
                par.inspect_semi[name],
                par.disasm_semi[name],
                y,
                z,
            )
            out.append(LocalCandidate(name, inds, tuple(x_bits), y, z, desc))
    return out


def tree_dp_search(par: Q3Parameters, top_k: int = 10, return_all: bool = False):
    """
    按手册“子树候选表”组织搜索。
    H1=32, H2=32, H3=16，根节点4种决策，总计65536个静态方案。
    """
    cand = {name: build_local_candidates(par, name) for name in ("H1", "H2", "H3")}
    assert len(cand["H1"]) == 32
    assert len(cand["H2"]) == 32
    assert len(cand["H3"]) == 16

    rows = [] if return_all else None
    best_rows: List[Dict] = []

    # top_k很小，直接维护列表即可。
    for h1, h2, h3 in product(cand["H1"], cand["H2"], cand["H3"]):
        hs = [h1.desc, h2.desc, h3.desc]
        for y_f, z_f in product((0, 1), repeat=2):
            cost = final_cost(par, hs, y_f, z_f)
            profit = par.sale_price - cost

            xs = [0] * 8
            for c in (h1, h2, h3):
                for idx, x in zip(c.comp_indices, c.x_bits):
                    xs[idx] = x
            bits = tuple(xs + [h1.y, h2.y, h3.y] + [h1.z, h2.z, h3.z] + [y_f, z_f])
            rec = {
                "policy_bits": bits_to_string(bits),
                "profit": float(profit),
                "cost": float(cost),
                "bits": bits,
            }
            if rows is not None:
                rows.append(rec)

            best_rows.append(rec)
            if len(best_rows) > top_k * 4:
                best_rows.sort(key=lambda r: (-r["profit"], r["policy_bits"]))
                del best_rows[top_k:]

    best_rows.sort(key=lambda r: (-r["profit"], r["policy_bits"]))
    best_rows = best_rows[:top_k]
    if return_all:
        df = pd.DataFrame(rows).sort_values(["profit", "policy_bits"], ascending=[False, True]).reset_index(drop=True)
        return best_rows, df
    return best_rows


def flat_enumeration(par: Q3Parameters) -> pd.DataFrame:
    """独立16位平坦枚举：2^16=65536，静态策略空间全局最优。"""
    rows = []
    for bits in product((0, 1), repeat=16):
        profit, cost = evaluate_policy(par, bits)
        rec = {
            "policy_bits": bits_to_string(bits),
            "profit": float(profit),
            "cost": float(cost),
        }
        rec.update(bits_to_dict(bits))
        rows.append(rec)
    return pd.DataFrame(rows).sort_values(["profit", "policy_bits"], ascending=[False, True]).reset_index(drop=True)


def best_only_tree(par: Q3Parameters) -> Dict:
    return tree_dp_search(par, top_k=1, return_all=False)[0]


# ============================================================
# 5. 手算闭式核验：手册式(27)~(31)
# ============================================================

def manual_closed_form_check(par: Q3Parameters) -> Dict[str, float]:
    """只用于题目基准参数、手册给出的最优结构的独立手算核验。"""
    comp_layer = sum(
        (a + b) / (1.0 - p)
        for a, b, p in zip(par.purchase_comp, par.inspect_comp, par.p_comp)
    )

    semi_each = [
        (
            par.asm_semi[h]
            + par.inspect_semi[h]
            + par.p_semi[h] * par.disasm_semi[h]
        ) / (1.0 - par.p_semi[h])
        for h in ("H1", "H2", "H3")
    ]
    semi_layer = sum(semi_each)

    final_layer = (
        par.asm_final
        + par.p_final * (par.replacement_loss + par.disasm_final)
    ) / (1.0 - par.p_final)

    total = comp_layer + semi_layer + final_layer
    profit = par.sale_price - total
    return {
        "component_layer_cost": comp_layer,
        "semi_layer_cost": semi_layer,
        "final_layer_cost": final_layer,
        "total_cost": total,
        "profit": profit,
    }


# ============================================================
# 6. Monte Carlo 检验
# ============================================================

def vectorized_mc_for_manual_optimum(
    par: Q3Parameters,
    n_orders: int = 200_000,
    seed: int = 20260820,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    对手册最优结构进行200,000笔独立订单向量化仿真。
    这是对随机过程的独立验证，不参与优化。
    """
    rng = np.random.default_rng(seed)
    costs = np.zeros(n_orders, dtype=float)

    # 8个零配件全部检测：每类直到抽到合格件，几何分布参数1-p_i。
    comp_attempts = np.zeros((n_orders, 8), dtype=np.int32)
    for i, (a, b, p) in enumerate(zip(par.purchase_comp, par.inspect_comp, par.p_comp)):
        n = rng.geometric(1.0 - p, size=n_orders)
        comp_attempts[:, i] = n
        costs += n * (a + b)

    # 三个半成品：输入已知合格，反复装配+检测；失败则拆解后重装。
    semi_attempts = np.zeros((n_orders, 3), dtype=np.int32)
    for j, h in enumerate(("H1", "H2", "H3")):
        n = rng.geometric(1.0 - par.p_semi[h], size=n_orders)
        semi_attempts[:, j] = n
        costs += n * (par.asm_semi[h] + par.inspect_semi[h])
        costs += (n - 1) * par.disasm_semi[h]

    # 成品不检测；失败流入市场，发生换货+拆解，再用已知合格半成品重新装配。
    final_attempts = rng.geometric(1.0 - par.p_final, size=n_orders)
    costs += final_attempts * par.asm_final
    costs += (final_attempts - 1) * (par.replacement_loss + par.disasm_final)

    profits = par.sale_price - costs
    mean = float(profits.mean())
    sd = float(profits.std(ddof=1))
    se = sd / math.sqrt(n_orders)
    ci_low = mean - 1.96 * se
    ci_high = mean + 1.96 * se

    summary = {
        "n_orders": n_orders,
        "seed": seed,
        "mean_profit": mean,
        "sample_sd": sd,
        "standard_error": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "mean_component_attempts_each": float(comp_attempts.mean()),
        "mean_semi_attempts_total": float(semi_attempts.sum(axis=1).mean()),
        "mean_semi_failures_total": float((semi_attempts - 1).sum(axis=1).mean()),
        "mean_final_attempts": float(final_attempts.mean()),
        "mean_replacements": float((final_attempts - 1).mean()),
        "mean_final_disassemblies": float((final_attempts - 1).mean()),
    }

    checkpoints = np.unique(np.linspace(max(100, n_orders // 200), n_orders, 100, dtype=int))
    running = np.array([profits[:k].mean() for k in checkpoints])
    conv = pd.DataFrame({"n_orders": checkpoints, "running_mean_profit": running})
    return summary, conv


class EventSimulator:
    """
    逐事件Monte Carlo模拟器。
    它不调用解析期望公式，而是按“采购/检测/装配/检测/拆解/换货”的物理事件逐项发生，
    用于验证状态转移语义。默认只跑1万笔，避免拖慢主程序。
    """

    def __init__(self, par: Q3Parameters, bits: Sequence[int], rng: np.random.Generator):
        self.par = par
        self.bits = tuple(map(int, bits))
        self.rng = rng
        self.xs = self.bits[:8]
        self.ys = self.bits[8:11]
        self.zs = self.bits[11:14]
        self.yf, self.zf = self.bits[14:16]
        self.h_index = {"H1": 0, "H2": 1, "H3": 2}
        self.cost = 0.0
        self.event_counts = {
            "purchase": 0,
            "component_inspection": 0,
            "semi_assembly": 0,
            "semi_inspection": 0,
            "semi_disassembly": 0,
            "final_assembly": 0,
            "final_inspection": 0,
            "replacement": 0,
            "final_disassembly": 0,
        }

    def add(self, amount: float, event: str, count: int = 1) -> None:
        self.cost += amount
        self.event_counts[event] += count

    def buy_leaf_forward(self, i: int) -> SimItem:
        p, a, b = self.par.p_comp[i], self.par.purchase_comp[i], self.par.inspect_comp[i]
        if self.xs[i] == 0:
            self.add(a, "purchase")
            good = bool(self.rng.random() >= p)
            return SimItem(f"C{i+1}", good, False, None)

        while True:
            self.add(a, "purchase")
            self.add(b, "component_inspection")
            good = bool(self.rng.random() >= p)
            if good:
                return SimItem(f"C{i+1}", True, True, None)

    def certify_leaf_existing(self, i: int, item: SimItem) -> SimItem:
        # 先检测当前未知回收件。
        b = self.par.inspect_comp[i]
        self.add(b, "component_inspection")
        if item.good:
            item.known = True
            return item

        # 当前件已确认坏，丢弃并反复购买+检测直到获得已知合格件。
        p, a = self.par.p_comp[i], self.par.purchase_comp[i]
        while True:
            self.add(a, "purchase")
            self.add(b, "component_inspection")
            good = bool(self.rng.random() >= p)
            if good:
                return SimItem(f"C{i+1}", True, True, None)

    def produce_h_forward(self, h: str) -> SimItem:
        j = self.h_index[h]
        children = [self.buy_leaf_forward(i) for i in self.par.children[h]]
        self.add(self.par.asm_semi[h], "semi_assembly")
        good = all(c.good for c in children) and bool(self.rng.random() >= self.par.p_semi[h])
        item = SimItem(h, good, False, children)

        if self.ys[j] == 0:
            return item

        self.add(self.par.inspect_semi[h], "semi_inspection")
        if item.good:
            item.known = True
            return item
        return self.recover_failed_h(h, item)

    def certify_h_existing(self, h: str, item: SimItem) -> SimItem:
        # 父层拆解后，对未知半成品先做一次检测。
        self.add(self.par.inspect_semi[h], "semi_inspection")
        if item.good:
            item.known = True
            return item
        return self.recover_failed_h(h, item)

    def recover_failed_h(self, h: str, item: SimItem) -> SimItem:
        j = self.h_index[h]
        z = self.zs[j]

        if z == 0:
            # 整个坏半成品报废；从头构造一个新半成品并检测，直到合格。
            while True:
                children = [self.buy_leaf_forward(i) for i in self.par.children[h]]
                self.add(self.par.asm_semi[h], "semi_assembly")
                good = all(c.good for c in children) and bool(self.rng.random() >= self.par.p_semi[h])
                self.add(self.par.inspect_semi[h], "semi_inspection")
                if good:
                    return SimItem(h, True, True, children)

        # z=1：拆解当前坏半成品，保留已知合格子件；未知子件逐个检测认证。
        self.add(self.par.disasm_semi[h], "semi_disassembly")
        assert item.children is not None
        new_children = []
        for idx, child in zip(self.par.children[h], item.children):
            if child.known:
                new_children.append(child)
            else:
                new_children.append(self.certify_leaf_existing(idx, child))

        # 现在所有零件已知合格，只剩工序自身失效；失败则拆解重装。
        while True:
            self.add(self.par.asm_semi[h], "semi_assembly")
            self.add(self.par.inspect_semi[h], "semi_inspection")
            good = bool(self.rng.random() >= self.par.p_semi[h])
            if good:
                return SimItem(h, True, True, new_children)
            self.add(self.par.disasm_semi[h], "semi_disassembly")

    def run_order(self) -> Tuple[float, Dict[str, int]]:
        self.cost = 0.0
        for k in self.event_counts:
            self.event_counts[k] = 0

        while True:
            hs = [self.produce_h_forward(h) for h in ("H1", "H2", "H3")]
            self.add(self.par.asm_final, "final_assembly")
            good = all(h.good for h in hs) and bool(self.rng.random() >= self.par.p_final)

            if self.yf == 1:
                self.add(self.par.inspect_final, "final_inspection")
                if good:
                    return self.par.sale_price - self.cost, dict(self.event_counts)

                if self.zf == 0:
                    # 成品坏品直接报废，重新开始整笔生产。
                    continue

                # 成品检测失败后拆解：无市场换货损失。
                self.add(self.par.disasm_final, "final_disassembly")
                cert_hs = []
                for h_name, h_item in zip(("H1", "H2", "H3"), hs):
                    if h_item.known:
                        cert_hs.append(h_item)
                    else:
                        cert_hs.append(self.certify_h_existing(h_name, h_item))

                # 三个半成品已知合格，只剩成品工序自身失败；每轮都检测。
                while True:
                    self.add(self.par.asm_final, "final_assembly")
                    self.add(self.par.inspect_final, "final_inspection")
                    if self.rng.random() >= self.par.p_final:
                        return self.par.sale_price - self.cost, dict(self.event_counts)
                    self.add(self.par.disasm_final, "final_disassembly")

            # yF=0：产品直接进入市场。
            if good:
                return self.par.sale_price - self.cost, dict(self.event_counts)

            self.add(self.par.replacement_loss, "replacement")
            if self.zf == 0:
                # 用户退回后不拆解，整套报废，从头重做订单。
                continue

            # 用户退回坏品后拆解。
            self.add(self.par.disasm_final, "final_disassembly")
            cert_hs = []
            for h_name, h_item in zip(("H1", "H2", "H3"), hs):
                if h_item.known:
                    cert_hs.append(h_item)
                else:
                    cert_hs.append(self.certify_h_existing(h_name, h_item))

            # 三个半成品已知合格后，无出厂检测，每次成品工序失败都流入市场并换货+拆解。
            while True:
                self.add(self.par.asm_final, "final_assembly")
                if self.rng.random() >= self.par.p_final:
                    return self.par.sale_price - self.cost, dict(self.event_counts)
                self.add(self.par.replacement_loss, "replacement")
                self.add(self.par.disasm_final, "final_disassembly")


def generic_event_mc(
    par: Q3Parameters,
    bits: Sequence[int],
    n_orders: int = 10_000,
    seed: int = 20260821,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    profits = np.empty(n_orders, dtype=float)
    event_sum: Dict[str, float] = {}

    for r in range(n_orders):
        sim = EventSimulator(par, bits, rng)
        profit, counts = sim.run_order()
        profits[r] = profit
        for k, v in counts.items():
            event_sum[k] = event_sum.get(k, 0.0) + v

    mean = float(profits.mean())
    sd = float(profits.std(ddof=1))
    se = sd / math.sqrt(n_orders)
    out = {
        "n_orders": n_orders,
        "seed": seed,
        "mean_profit": mean,
        "sample_sd": sd,
        "standard_error": se,
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }
    for k, v in event_sum.items():
        out[f"mean_{k}"] = v / n_orders
    return out


# ============================================================
# 7. 模型检验
# ============================================================

def run_validations(
    par: Q3Parameters,
    flat_df: pd.DataFrame,
    tree_top: List[Dict],
    mc_summary: Dict[str, float],
    generic_mc_summary: Optional[Dict[str, float]],
) -> pd.DataFrame:
    rows = []

    best = flat_df.iloc[0]
    best_bits = tuple(int(c) for c in best["policy_bits"])
    best_profit = float(best["profit"])
    best_cost = float(best["cost"])

    # 1) 手册固定基准结果
    expected_bits = "1111111111111101"
    expected_cost = 139.77777777777777
    expected_profit = 60.22222222222222
    rows.append({
        "test": "manual_target_policy",
        "passed": bool(best["policy_bits"] == expected_bits),
        "metric": best["policy_bits"],
        "target": expected_bits,
    })
    rows.append({
        "test": "manual_target_cost",
        "passed": abs(best_cost - expected_cost) < 1e-10,
        "metric": best_cost,
        "target": expected_cost,
    })
    rows.append({
        "test": "manual_target_profit",
        "passed": abs(best_profit - expected_profit) < 1e-10,
        "metric": best_profit,
        "target": expected_profit,
    })

    # 2) 手算闭式核验
    hand = manual_closed_form_check(par)
    rows.append({
        "test": "hand_closed_form_vs_exact",
        "passed": abs(hand["total_cost"] - best_cost) < 1e-10,
        "metric": abs(hand["total_cost"] - best_cost),
        "target": "<1e-10",
    })

    # 3) DP候选表与平坦全枚举一致
    tree_best = tree_top[0]
    rows.append({
        "test": "tree_dp_vs_flat_policy",
        "passed": tree_best["policy_bits"] == best["policy_bits"],
        "metric": tree_best["policy_bits"],
        "target": best["policy_bits"],
    })
    rows.append({
        "test": "tree_dp_vs_flat_profit",
        "passed": abs(tree_best["profit"] - best_profit) < 1e-10,
        "metric": abs(tree_best["profit"] - best_profit),
        "target": "<1e-10",
    })

    # 4) 成本分解向量与标量评价器一致
    vec_cost, _ = evaluate_policy_breakdown(par, best_bits)
    rows.append({
        "test": "cost_breakdown_sum_vs_scalar",
        "passed": abs(vec_cost - best_cost) < 1e-10,
        "metric": abs(vec_cost - best_cost),
        "target": "<1e-10",
    })

    # 5) Monte Carlo 95% CI 包含理论期望
    rows.append({
        "test": "vectorized_mc_ci_contains_theory",
        "passed": mc_summary["ci95_low"] <= best_profit <= mc_summary["ci95_high"],
        "metric": f"[{mc_summary['ci95_low']:.6f}, {mc_summary['ci95_high']:.6f}]",
        "target": f"contains {best_profit:.6f}",
    })

    if generic_mc_summary is not None:
        rows.append({
            "test": "generic_event_mc_ci_contains_theory",
            "passed": generic_mc_summary["ci95_low"] <= best_profit <= generic_mc_summary["ci95_high"],
            "metric": f"[{generic_mc_summary['ci95_low']:.6f}, {generic_mc_summary['ci95_high']:.6f}]",
            "target": f"contains {best_profit:.6f}",
        })

    # 6) 理论事件次数与200k仿真均值
    expected = {
        "mean_component_attempts_each": 1.0 / 0.9,
        "mean_semi_attempts_total": 3.0 / 0.9,
        "mean_semi_failures_total": 3.0 * 0.1 / 0.9,
        "mean_final_attempts": 1.0 / 0.9,
        "mean_replacements": 0.1 / 0.9,
        "mean_final_disassemblies": 0.1 / 0.9,
    }
    for key, target in expected.items():
        # 以绝对误差0.01作为200k样本的宽松事件级验收阈值；不作为模型参数，只是程序测试容差。
        err = abs(mc_summary[key] - target)
        rows.append({
            "test": f"mc_event_count_{key}",
            "passed": err < 0.01,
            "metric": err,
            "target": "abs error < 0.01",
        })

    return pd.DataFrame(rows)


# ============================================================
# 8. 灵敏度分析：经济阈值 + 数值交叉核验
# ============================================================

def analytic_thresholds(par: Q3Parameters) -> pd.DataFrame:
    rows = []

    # 成品检测阈值：b_F < p_F L
    L_star = par.inspect_final / par.p_final
    pF_star = par.inspect_final / par.replacement_loss
    rows += [
        {
            "parameter": "replacement_loss_L",
            "actual": par.replacement_loss,
            "threshold": L_star,
            "rule": "检测成品 iff L > b_F/p_F",
            "baseline_side": "不检测" if par.replacement_loss < L_star else "检测",
        },
        {
            "parameter": "final_process_defect_pF",
            "actual": par.p_final,
            "threshold": pF_star,
            "rule": "检测成品 iff p_F > b_F/L",
            "baseline_side": "不检测" if par.p_final < pF_star else "检测",
        },
    ]

    # 零配件检测费阈值（按所属半成品参数计算）
    idx_to_h = {}
    for h, inds in par.children.items():
        for i in inds:
            idx_to_h[i] = h
    for i in range(8):
        h = idx_to_h[i]
        p_i = par.p_comp[i]
        b_star = p_i * (
            par.asm_semi[h] + par.inspect_semi[h] + par.disasm_semi[h]
        ) / ((1.0 - p_i) * (1.0 - par.p_semi[h]))
        rows.append({
            "parameter": f"inspect_cost_C{i+1}",
            "actual": par.inspect_comp[i],
            "threshold": b_star,
            "rule": "检测零件 iff b_i < b_i*（局部最优条件）",
            "baseline_side": "检测" if par.inspect_comp[i] < b_star else "不检测",
        })

    # 半成品检测费阈值
    for h in ("H1", "H2", "H3"):
        p_h = par.p_semi[h]
        b_star = p_h * (
            par.replacement_loss + par.asm_final + par.disasm_final
        ) / ((1.0 - p_h) * (1.0 - par.p_final))
        rows.append({
            "parameter": f"inspect_cost_{h}",
            "actual": par.inspect_semi[h],
            "threshold": b_star,
            "rule": "检测半成品 iff b_H < b_H*（基准其余节点已知合格）",
            "baseline_side": "检测" if par.inspect_semi[h] < b_star else "不检测",
        })

    # 半成品拆解阈值：子零件已知合格价值之和
    K_leaf = [
        (a + b) / (1.0 - p)
        for a, b, p in zip(par.purchase_comp, par.inspect_comp, par.p_comp)
    ]
    for h in ("H1", "H2", "H3"):
        d_star = sum(K_leaf[i] for i in par.children[h])
        rows.append({
            "parameter": f"disassembly_cost_{h}",
            "actual": par.disasm_semi[h],
            "threshold": d_star,
            "rule": "拆解半成品 iff d_H < 子零件已知合格重置价值",
            "baseline_side": "拆解" if par.disasm_semi[h] < d_star else "报废",
        })

    # 成品拆解阈值：三个已知合格半成品的替代价值之和
    leaf_known = [leaf_desc(par, i, 1) for i in range(8)]
    K_h = []
    for h in ("H1", "H2", "H3"):
        ch = [leaf_known[i] for i in par.children[h]]
        d = internal_desc(
            ch,
            par.p_semi[h],
            par.asm_semi[h],
            par.inspect_semi[h],
            par.disasm_semi[h],
            y=1,
            z=1,
        )
        K_h.append(d.K)
    dF_star = sum(K_h)
    rows.append({
        "parameter": "disassembly_cost_F",
        "actual": par.disasm_final,
        "threshold": dF_star,
        "rule": "拆解成品 iff d_F < 三个已知合格半成品重置价值",
        "baseline_side": "拆解" if par.disasm_final < dF_star else "报废",
    })

    return pd.DataFrame(rows)


def component_fee_sensitivity(par: Q3Parameters, base_bits: Sequence[int]) -> pd.DataFrame:
    """对8类零配件检测费做局部切换曲线：只切换对应x_i，其余决策保持基准最优。"""
    rows = []
    grid = np.unique(np.r_[np.linspace(0.0, 4.0, 81), 2.2222222222222223])
    for i in range(8):
        for b in grid:
            p2 = par.clone()
            p2.inspect_comp[i] = float(b)
            bits1 = list(base_bits)
            bits0 = list(base_bits)
            bits1[i] = 1
            bits0[i] = 0
            prof1, _ = evaluate_policy(p2, bits1)
            prof0, _ = evaluate_policy(p2, bits0)
            rows.append({
                "component": f"C{i+1}",
                "inspect_cost": float(b),
                "profit_if_inspect": prof1,
                "profit_if_not_inspect": prof0,
                "profit_diff_inspect_minus_no": prof1 - prof0,
                "preferred": "inspect" if prof1 > prof0 + 1e-12 else ("no_inspect" if prof0 > prof1 + 1e-12 else "tie"),
            })
    return pd.DataFrame(rows)


def final_phase_sensitivity(par: Q3Parameters, base_bits: Sequence[int]) -> pd.DataFrame:
    """(p_F,L)二维相图数据：比较只切换成品检测位，其余保持基准最优。"""
    rows = []
    p_grid = np.unique(np.r_[np.linspace(0.04, 0.25, 43), 0.10, 0.15])
    L_grid = np.unique(np.r_[np.linspace(0.0, 100.0, 41), 40.0, 60.0])
    for p_f in p_grid:
        for L in L_grid:
            p2 = par.clone()
            p2.p_final = float(p_f)
            p2.replacement_loss = float(L)
            b0 = list(base_bits); b0[14] = 0
            b1 = list(base_bits); b1[14] = 1
            prof0, _ = evaluate_policy(p2, b0)
            prof1, _ = evaluate_policy(p2, b1)
            rows.append({
                "p_final": float(p_f),
                "replacement_loss": float(L),
                "profit_no_final_inspection": prof0,
                "profit_final_inspection": prof1,
                "profit_diff_inspect_minus_no": prof1 - prof0,
                "preferred": "inspect" if prof1 > prof0 + 1e-12 else ("no_inspect" if prof0 > prof1 + 1e-12 else "tie"),
                "analytic_rule": "inspect" if par.inspect_final < p_f * L else ("tie" if abs(par.inspect_final - p_f * L) < 1e-12 else "no_inspect"),
            })
    return pd.DataFrame(rows)


def local_sensitivity_curves(par: Q3Parameters, base_bits: Sequence[int]) -> pd.DataFrame:
    """
    经济意义导向的一维扫描，不采用机械统一±5%。
    关注：H1检测费、H1拆解费、F拆解费、L、pF。
    """
    scans = {
        "inspect_H1": np.unique(np.r_[np.linspace(0, 10, 101), 7.160493827160494]),
        "disasm_H1": np.unique(np.r_[np.linspace(0, 40, 161), 28.88888888888889]),
        "disasm_F": np.unique(np.r_[np.linspace(0, 160, 161), 125.33333333333333]),
        "replacement_L": np.unique(np.r_[np.linspace(0, 100, 101), 60.0]),
        "p_final": np.unique(np.r_[np.linspace(0.03, 0.25, 111), 0.15]),
    }
    rows = []
    for name, grid in scans.items():
        for val in grid:
            p2 = par.clone()
            bits_a = list(base_bits)
            bits_b = list(base_bits)
            if name == "inspect_H1":
                p2.inspect_semi["H1"] = float(val)
                bits_a[8], bits_b[8] = 1, 0
            elif name == "disasm_H1":
                p2.disasm_semi["H1"] = float(val)
                bits_a[11], bits_b[11] = 1, 0
            elif name == "disasm_F":
                p2.disasm_final = float(val)
                bits_a[15], bits_b[15] = 1, 0
            elif name == "replacement_L":
                p2.replacement_loss = float(val)
                bits_a[14], bits_b[14] = 1, 0
            elif name == "p_final":
                p2.p_final = float(val)
                bits_a[14], bits_b[14] = 1, 0
            else:
                raise RuntimeError(name)

            pa, _ = evaluate_policy(p2, bits_a)
            pb, _ = evaluate_policy(p2, bits_b)
            rows.append({
                "parameter": name,
                "value": float(val),
                "profit_action_1": pa,
                "profit_action_0": pb,
                "difference_1_minus_0": pa - pb,
            })
    return pd.DataFrame(rows)


def global_sensitivity_spotcheck(par: Q3Parameters) -> pd.DataFrame:
    """
    在解析阈值附近及远离阈值处，重新对65536个静态策略做全局搜索。
    这些点不是人为±5%，而是由手册经济阈值与题目基准值共同确定。
    """
    cases = [
        ("L", 40.0), ("L", 59.0), ("L", 61.0), ("L", 80.0),
        ("pF", 0.10), ("pF", 0.14), ("pF", 0.16), ("pF", 0.20),
        ("bC3", 2.0), ("bC3", 2.20), ("bC3", 2.25), ("bC3", 3.0),
        ("bH1", 4.0), ("bH1", 7.0), ("bH1", 7.3), ("bH1", 9.0),
        ("dH1", 6.0), ("dH1", 28.0), ("dH1", 30.0), ("dH1", 35.0),
        ("dF", 10.0), ("dF", 120.0), ("dF", 130.0), ("dF", 150.0),
    ]

    rows = []
    for param_name, val in cases:
        p2 = par.clone()
        if param_name == "L":
            p2.replacement_loss = val
        elif param_name == "pF":
            p2.p_final = val
        elif param_name == "bC3":
            p2.inspect_comp[2] = val
        elif param_name == "bH1":
            p2.inspect_semi["H1"] = val
        elif param_name == "dH1":
            p2.disasm_semi["H1"] = val
        elif param_name == "dF":
            p2.disasm_final = val
        else:
            raise RuntimeError(param_name)

        best = best_only_tree(p2)
        rows.append({
            "parameter": param_name,
            "value": val,
            "best_policy_bits": best["policy_bits"],
            "best_profit": best["profit"],
            "best_cost": best["cost"],
            "x_C3": int(best["policy_bits"][2]),
            "y_H1": int(best["policy_bits"][8]),
            "z_H1": int(best["policy_bits"][11]),
            "y_F": int(best["policy_bits"][14]),
            "z_F": int(best["policy_bits"][15]),
        })
    return pd.DataFrame(rows)


# ============================================================
# 9. 极端情形检验
# ============================================================

def extreme_case_checks(par: Q3Parameters) -> pd.DataFrame:
    rows = []

    # Case A: 所有次品率为0，且检测费>0 -> 检测没有收益；应存在全不检的最优策略。
    p0 = par.clone()
    p0.p_comp = [0.0] * 8
    p0.p_semi = {k: 0.0 for k in p0.p_semi}
    p0.p_final = 0.0
    best0 = best_only_tree(p0)
    bits0 = best0["policy_bits"]
    pass0 = bits0[:8] == "00000000" and bits0[8:11] == "000" and bits0[14] == "0"
    rows.append({
        "case": "all_defect_rates_zero",
        "passed": pass0,
        "best_policy_bits": bits0,
        "best_profit": best0["profit"],
        "expected_behavior": "零件、半成品、成品均无需检测；拆解位因无失败可出现等价",
    })

    # Case B: 调换损失极大 -> 成品检测应开启。
    pL = par.clone()
    pL.replacement_loss = 1000.0
    bestL = best_only_tree(pL)
    passL = bestL["policy_bits"][14] == "1"
    rows.append({
        "case": "very_large_replacement_loss",
        "passed": passL,
        "best_policy_bits": bestL["policy_bits"],
        "best_profit": bestL["profit"],
        "expected_behavior": "成品检测 y_F=1",
    })

    # Case C: 所有拆解费极大 -> 拆解应失去经济性。
    p_dis = par.clone()
    p_dis.disasm_semi = {k: 1e6 for k in p_dis.disasm_semi}
    p_dis.disasm_final = 1e6
    bestd = best_only_tree(p_dis)
    bitsd = bestd["policy_bits"]
    passd = bitsd[11:14] == "000" and bitsd[15] == "0"
    rows.append({
        "case": "very_large_disassembly_cost",
        "passed": passd,
        "best_policy_bits": bitsd,
        "best_profit": bestd["profit"],
        "expected_behavior": "H1/H2/H3/F拆解位均为0",
    })

    # Case D: 零件检测费极大 -> 不应检测零件。
    pb = par.clone()
    pb.inspect_comp = [1e6] * 8
    bestb = best_only_tree(pb)
    bitsb = bestb["policy_bits"]
    passb = bitsb[:8] == "00000000"
    rows.append({
        "case": "very_large_component_inspection_cost",
        "passed": passb,
        "best_policy_bits": bitsb,
        "best_profit": bestb["profit"],
        "expected_behavior": "8个零件检测位均为0",
    })

    return pd.DataFrame(rows)


# ============================================================
# 10. 输出与主程序
# ============================================================

def prepare_cost_breakdown(par: Q3Parameters, bits: Sequence[int]) -> pd.DataFrame:
    scalar_cost = evaluate_policy(par, bits)[1]
    vec_cost, vec = evaluate_policy_breakdown(par, bits)
    if abs(vec_cost - scalar_cost) >= 1e-10:
        raise AssertionError("成本分解向量与标量评价器不一致。")

    rows = []
    for cat, val in zip(COST_CATEGORIES, vec):
        rows.append({
            "category": cat,
            "category_cn": COST_CN[cat],
            "expected_cost": float(val),
            "share": float(val / scalar_cost if scalar_cost > 0 else 0.0),
        })
    rows.append({
        "category": "TOTAL",
        "category_cn": "总期望成本",
        "expected_cost": scalar_cost,
        "share": 1.0,
    })
    return pd.DataFrame(rows)


def write_summary_text(
    path: str,
    elapsed: float,
    flat_df: pd.DataFrame,
    tree_top: List[Dict],
    validation_df: pd.DataFrame,
    mc_summary: Dict[str, float],
    generic_mc_summary: Optional[Dict[str, float]],
    threshold_df: Optional[pd.DataFrame],
    extreme_df: pd.DataFrame,
) -> None:
    best = flat_df.iloc[0]
    second = flat_df.iloc[1]
    lines = []
    lines.append("2024 高教社杯 B题 问题三 —— Python 求解结果摘要")
    lines.append("=" * 72)
    lines.append(f"运行耗时: {elapsed:.3f} s")
    lines.append(f"静态策略数: {len(flat_df)}")
    lines.append(f"最优16位策略: {best['policy_bits']}")
    lines.append(f"单位期望成本: {best['cost']:.10f} 元")
    lines.append(f"单位期望利润: {best['profit']:.10f} 元")
    lines.append(f"次优策略: {second['policy_bits']}")
    lines.append(f"次优利润: {second['profit']:.10f} 元")
    lines.append(f"最优-次优利润差 Δ: {best['profit'] - second['profit']:.10f} 元")
    lines.append("")
    lines.append("最优策略解释：")
    lines.append("  C1~C8：全部检测")
    lines.append("  H1~H3：全部检测；检测到次品后全部拆解")
    lines.append("  F：不做出厂检测；消费者退回次品后拆解")
    lines.append("")
    lines.append("树形DP交叉检查：")
    lines.append(f"  tree best = {tree_top[0]['policy_bits']}, profit = {tree_top[0]['profit']:.10f}")
    lines.append("")
    lines.append("200,000笔向量化Monte Carlo：")
    lines.append(f"  mean={mc_summary['mean_profit']:.6f}, sd={mc_summary['sample_sd']:.6f}, SE={mc_summary['standard_error']:.6f}")
    lines.append(f"  95%CI=[{mc_summary['ci95_low']:.6f}, {mc_summary['ci95_high']:.6f}]")
    lines.append("")
    if generic_mc_summary is not None:
        lines.append("逐事件Monte Carlo：")
        lines.append(f"  n={generic_mc_summary['n_orders']}, mean={generic_mc_summary['mean_profit']:.6f}")
        lines.append(f"  95%CI=[{generic_mc_summary['ci95_low']:.6f}, {generic_mc_summary['ci95_high']:.6f}]")
        lines.append("")
    lines.append("模型检验：")
    for _, r in validation_df.iterrows():
        lines.append(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['test']}: {r['metric']} | target={r['target']}")
    lines.append("")
    lines.append("极端情形检验：")
    for _, r in extreme_df.iterrows():
        lines.append(f"  [{'PASS' if r['passed'] else 'FAIL'}] {r['case']}: {r['best_policy_bits']}")
    if threshold_df is not None:
        lines.append("")
        lines.append("关键灵敏度阈值：")
        for _, r in threshold_df.iterrows():
            lines.append(f"  {r['parameter']}: actual={r['actual']:.6g}, threshold={r['threshold']:.6g}, baseline={r['baseline_side']}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="2024国赛B题问题三：信息保持型BOM动态规划精确求解")
    parser.add_argument("--outdir", default="../结果输出", help="CSV/摘要输出目录，默认相对脚本目录的 ../结果输出（即 求解代码/第三问/结果输出/），PyCharm 直接运行即可生效")
    parser.add_argument("--mc", type=int, default=200_000, help="向量化Monte Carlo订单数，默认200000")
    parser.add_argument("--generic-mc", type=int, default=10_000, help="逐事件Monte Carlo订单数，默认10000")
    parser.add_argument("--skip-generic-mc", action="store_true", help="跳过逐事件Monte Carlo")
    parser.add_argument("--skip-sensitivity", action="store_true", help="跳过灵敏度分析")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.perf_counter()

    par = default_parameters()
    validate_parameters(par)

    print("[1/8] 读取并校验题目表2参数 ...")
    print("      参数校验通过。")

    print("[2/8] 运行16位平坦精确枚举（65536策略） ...")
    flat_df = flat_enumeration(par)
    flat_df.to_csv(os.path.join(args.outdir, "01_all_strategies.csv"), index=False, encoding="utf-8-sig")
    flat_df.head(10).to_csv(os.path.join(args.outdir, "02_top10_strategies.csv"), index=False, encoding="utf-8-sig")

    best = flat_df.iloc[0]
    best_bits = tuple(int(c) for c in best["policy_bits"])
    opt_row = {"policy_bits": best["policy_bits"], "profit": best["profit"], "cost": best["cost"]}
    opt_row.update(bits_to_dict(best_bits))
    pd.DataFrame([opt_row]).to_csv(os.path.join(args.outdir, "03_optimal_policy.csv"), index=False, encoding="utf-8-sig")

    print(f"      最优策略 = {best['policy_bits']}")
    print(f"      期望成本 = {best['cost']:.10f}")
    print(f"      期望利润 = {best['profit']:.10f}")

    print("[3/8] 运行BOM树候选表动态规划搜索并与平坦枚举交叉检查 ...")
    tree_top = tree_dp_search(par, top_k=10, return_all=False)
    print(f"      Tree-DP best = {tree_top[0]['policy_bits']}, profit={tree_top[0]['profit']:.10f}")

    print("[4/8] 计算最优策略成本分解 ...")
    cost_df = prepare_cost_breakdown(par, best_bits)
    cost_df.to_csv(os.path.join(args.outdir, "04_cost_breakdown.csv"), index=False, encoding="utf-8-sig")

    print(f"[5/8] Monte Carlo独立验证：向量化 {args.mc:,} 笔订单 ...")
    mc_summary, mc_conv = vectorized_mc_for_manual_optimum(par, args.mc, 20260820)
    pd.DataFrame([mc_summary]).to_csv(os.path.join(args.outdir, "06_mc_vectorized_summary.csv"), index=False, encoding="utf-8-sig")
    mc_conv.to_csv(os.path.join(args.outdir, "08_mc_convergence.csv"), index=False, encoding="utf-8-sig")
    print(
        f"      mean={mc_summary['mean_profit']:.6f}, "
        f"95%CI=[{mc_summary['ci95_low']:.6f}, {mc_summary['ci95_high']:.6f}]"
    )

    generic_summary = None
    if not args.skip_generic_mc and args.generic_mc > 0:
        print(f"      逐事件独立模拟 {args.generic_mc:,} 笔订单 ...")
        generic_summary = generic_event_mc(par, best_bits, args.generic_mc, 20260821)
        pd.DataFrame([generic_summary]).to_csv(os.path.join(args.outdir, "07_mc_event_summary.csv"), index=False, encoding="utf-8-sig")
        print(
            f"      event-MC mean={generic_summary['mean_profit']:.6f}, "
            f"95%CI=[{generic_summary['ci95_low']:.6f}, {generic_summary['ci95_high']:.6f}]"
        )

    print("[6/8] 执行解析、算法、随机过程三层模型检验 ...")
    validation_df = run_validations(par, flat_df, tree_top, mc_summary, generic_summary)
    validation_df.to_csv(os.path.join(args.outdir, "05_validation_summary.csv"), index=False, encoding="utf-8-sig")
    if not bool(validation_df["passed"].all()):
        bad = validation_df.loc[~validation_df["passed"]]
        print("      警告：存在未通过检验：")
        print(bad.to_string(index=False))
    else:
        print(f"      {len(validation_df)} 项模型检验全部通过。")

    threshold_df = None
    if not args.skip_sensitivity:
        print("[7/8] 执行经济阈值与灵敏度分析（非机械±5%） ...")
        threshold_df = analytic_thresholds(par)
        threshold_df.to_csv(os.path.join(args.outdir, "09_sensitivity_thresholds.csv"), index=False, encoding="utf-8-sig")

        component_sens = component_fee_sensitivity(par, best_bits)
        component_sens.to_csv(os.path.join(args.outdir, "10_sensitivity_component_fee.csv"), index=False, encoding="utf-8-sig")

        final_phase = final_phase_sensitivity(par, best_bits)
        final_phase.to_csv(os.path.join(args.outdir, "11_sensitivity_final_phase.csv"), index=False, encoding="utf-8-sig")

        local_curves = local_sensitivity_curves(par, best_bits)
        local_curves.to_csv(os.path.join(args.outdir, "12_sensitivity_local_curves.csv"), index=False, encoding="utf-8-sig")

        global_spot = global_sensitivity_spotcheck(par)
        global_spot.to_csv(os.path.join(args.outdir, "13_sensitivity_global_spotcheck.csv"), index=False, encoding="utf-8-sig")
        print("      灵敏度分析完成：解析阈值 + 局部利润差曲线 + 阈值附近全局重优化。")
    else:
        print("[7/8] 已按参数跳过灵敏度分析。")

    print("[8/8] 极端情形合理性检查 ...")
    extreme_df = extreme_case_checks(par)
    extreme_df.to_csv(os.path.join(args.outdir, "14_extreme_case_checks.csv"), index=False, encoding="utf-8-sig")
    if bool(extreme_df["passed"].all()):
        print(f"      {len(extreme_df)} 个极端情形全部符合经济直觉。")
    else:
        print(extreme_df.to_string(index=False))

    elapsed = time.perf_counter() - t0
    write_summary_text(
        os.path.join(args.outdir, "run_summary.txt"),
        elapsed,
        flat_df,
        tree_top,
        validation_df,
        mc_summary,
        generic_summary,
        threshold_df,
        extreme_df,
    )

    print("\n" + "=" * 72)
    print("运行完成")
    print(f"总耗时：{elapsed:.3f} s")
    print(f"结果目录：{os.path.abspath(args.outdir)}")
    print(f"最优策略：{best['policy_bits']}")
    print(f"单位期望成本：{best['cost']:.6f} 元")
    print(f"单位期望利润：{best['profit']:.6f} 元")
    print(f"最优-次优利润差：{best['profit'] - flat_df.iloc[1]['profit']:.6f} 元")
    print("=" * 72)


if __name__ == "__main__":
    main()
