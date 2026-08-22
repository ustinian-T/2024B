"""Q4 灵敏度分析：单参数扫描、切换点定位、二维相图。

题目自身的实际参数跨度，不使用统一 ±5%。
所有切换点通过粗网格定位 + 二分精修得到，绝不硬编码。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Q2_SCENARIOS, Q3_NODES, Q3_REPLACEMENT_LOSS, Q3_SALE_PRICE
from .q2_model import solve_q2
from .q3_model import solve_q3_fast
from .q4_uncertainty import build_interval_table, SampleRecord


# ============================================================
# Q2 单参数扫描
# ============================================================

def q2_nominal_sensitivity() -> pd.DataFrame:
    """基于题目实际跨度做全局重优化。

    每个参数在自己的题目实际区间上扫描（不替换参数用 ±5%）。
    """
    p0 = Q2_SCENARIOS[1]
    specs = {
        "p1": np.linspace(0.05, 0.20, 31),
        "p2": np.linspace(0.05, 0.20, 31),
        "pf": np.linspace(0.05, 0.20, 31),
        "b1": np.linspace(1.0, 8.0, 29),
        "b2": np.linspace(1.0, 8.0, 29),
        "replacement_loss": np.linspace(6.0, 40.0, 35),
        "disassembly_cost": np.linspace(5.0, 40.0, 36),
    }
    rows = []
    for name, grid in specs.items():
        prev_code = None
        for v in grid:
            pp = replace(p0, **{name: float(v)})
            best, rank = solve_q2(pp, topk=3)
            rows.append({
                "parameter": name,
                "value": float(v),
                "policy4": best.code4,
                "policy6": best.code6,
                "profit": best.expected_profit,
                "gap": float(rank.iloc[0]["primary_policy_gap"]),
                "switch_from_previous": prev_code is not None and best.code4 != prev_code,
            })
            prev_code = best.code4
    return pd.DataFrame(rows)


def q2_switch_points(sens_df: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """从 sens_df 中精确定位每个切换：旧策略→新策略的阈值。

    使用粗网格 + 二分法精修到 tol 精度。
    """
    rows = []
    p0 = Q2_SCENARIOS[1]
    for param in sens_df["parameter"].unique():
        g = sens_df[sens_df["parameter"] == param].sort_values("value").reset_index(drop=True)
        codes = g["policy4"].values
        vals = g["value"].values
        for i in range(1, len(g)):
            if codes[i] == codes[i - 1]:
                continue
            # 二分精修
            old_code = codes[i - 1]
            new_code = codes[i]
            lo, hi = float(vals[i - 1]), float(vals[i])
            for _ in range(60):
                mid = (lo + hi) / 2
                pp = replace(p0, **{param: float(mid)})
                best, _ = solve_q2(pp, topk=1)
                if best.code4 == old_code:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < tol:
                    break
            threshold = (lo + hi) / 2
            rows.append({
                "scope": "Q2_S1",
                "parameter": param,
                "left_value": float(vals[i - 1]),
                "right_value": float(vals[i]),
                "old_policy": old_code,
                "new_policy": new_code,
                "refined_threshold": threshold,
            })
    return pd.DataFrame(rows)


# ============================================================
# Q3 单参数扫描
# ============================================================

def q3_nominal_sensitivity() -> pd.DataFrame:
    rows = []
    for pF in np.linspace(0.05, 0.20, 61):
        code, profit, top = solve_q3_fast({"F": float(pF)}, topk=3)
        rows.append({
            "parameter": "pF", "value": float(pF),
            "policy16": code, "profit": profit,
            "gap": float(top.iloc[0]["best_second_gap"]),
        })
    for L in np.linspace(6.0, 40.0, 69):
        code, profit, top = solve_q3_fast(replacement_loss=float(L), topk=3)
        rows.append({
            "parameter": "replacement_loss", "value": float(L),
            "policy16": code, "profit": profit,
            "gap": float(top.iloc[0]["best_second_gap"]),
        })
    return pd.DataFrame(rows)


def q3_switch_points(sens_df: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    rows = []
    for param in sens_df["parameter"].unique():
        g = sens_df[sens_df["parameter"] == param].sort_values("value").reset_index(drop=True)
        codes = g["policy16"].values
        vals = g["value"].values
        for i in range(1, len(g)):
            if codes[i] == codes[i - 1]:
                continue
            old_code = codes[i - 1]
            new_code = codes[i]
            lo, hi = float(vals[i - 1]), float(vals[i])
            for _ in range(60):
                mid = (lo + hi) / 2
                if param == "pF":
                    code, _, _ = solve_q3_fast({"F": float(mid)}, topk=1)
                else:
                    code, _, _ = solve_q3_fast(replacement_loss=float(mid), topk=1)
                if code == old_code:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < tol:
                    break
            threshold = (lo + hi) / 2
            rows.append({
                "scope": "Q3",
                "parameter": param,
                "left_value": float(vals[i - 1]),
                "right_value": float(vals[i]),
                "old_policy": old_code,
                "new_policy": new_code,
                "refined_threshold": threshold,
            })
    return pd.DataFrame(rows)


# ============================================================
# Q3 二维相图（nominal / robust90 / robust95 三种条件）
# ============================================================

def _q3_phase_map(base_quality: Mapping[str, float], pf_range: np.ndarray,
                   loss_range: np.ndarray, scope_label: str) -> pd.DataFrame:
    """对每个 (pF, L) 网格点都重新做完整 65536 搜索；返回策略表 + 首次切换阈值。"""
    base = dict(base_quality)
    rows = []
    for pf in pf_range:
        q = base.copy(); q["F"] = float(pf)
        for L in loss_range:
            code, profit, top = solve_q3_fast(q, replacement_loss=float(L), topk=2)
            rows.append({
                "pF": float(pf),
                "replacement_loss": float(L),
                "policy16": code,
                "profit": profit,
                "gap": float(top.iloc[0]["best_second_gap"]),
                "scope": scope_label,
            })
    df = pd.DataFrame(rows)
    # 计算每个 L 截面下的首次切换 pF
    switch_pts = []
    for L in loss_range:
        sub = df[df["replacement_loss"] == float(L)].sort_values("pF").reset_index(drop=True)
        if len(sub) == 0:
            continue
        first_code = sub.iloc[0]["policy16"]
        for i in range(1, len(sub)):
            if sub.iloc[i]["policy16"] != first_code:
                switch_pts.append({
                    "scope": scope_label,
                    "replacement_loss": float(L),
                    "first_switch_pF": float(sub.iloc[i]["pF"]),
                    "old_policy": first_code,
                    "new_policy": sub.iloc[i]["policy16"],
                })
                break
        else:
            switch_pts.append({
                "scope": scope_label,
                "replacement_loss": float(L),
                "first_switch_pF": np.nan,
                "old_policy": first_code,
                "new_policy": first_code,
            })
    return df, pd.DataFrame(switch_pts)


def q3_phase_map_nominal(
    pf_range: Sequence[float] | None = None,
    loss_range: Sequence[float] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """所有质量参数保持名义值 0.10；对每个 (pF, L) 网格点重做完整 65536 搜索。"""
    if pf_range is None:
        pf_range = np.linspace(0.05, 0.20, 31)
    if loss_range is None:
        loss_range = np.linspace(6.0, 40.0, 35)
    base = {nm: 0.10 for nm in Q3_NODES}
    return _q3_phase_map(base, np.asarray(pf_range), np.asarray(loss_range),
                         scope_label="Q3_nominal")


def q3_phase_map_robust(
    records: Sequence[SampleRecord],
    family_size: int,
    risk_label: str,
    pf_range: Sequence[float] | None = None,
    loss_range: Sequence[float] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """其他质量参数在相应同时上界，pF 沿名义值到上界之间扫描。

    family_size=12, risk=90/95 → 同时上界不同。
    仅使用 quality_id 以 "Q3-" 开头的记录。
    """
    if pf_range is None:
        pf_range = np.linspace(0.05, 0.20, 31)
    if loss_range is None:
        loss_range = np.linspace(6.0, 40.0, 35)
    q3_records = [r for r in records if r.quality_id.startswith("Q3-")]
    if len(q3_records) != family_size:
        raise ValueError(f"Expected {family_size} Q3 records, got {len(q3_records)}")
    tab = build_interval_table(q3_records, family_size=family_size)
    base = {}
    pF_upper = None
    risk_key = int(risk_label)
    for _, row in tab.iterrows():
        nm = row["quality_id"].replace("Q3-", "")
        if nm == "F":
            pF_upper = float(row[f"U_simultaneous_{risk_key}"])
        else:
            base[nm] = float(row[f"U_simultaneous_{risk_key}"])
    if pF_upper is None:
        raise ValueError("Q3 records missing Q3-F for phase map")
    pf_range_arr = np.linspace(0.10, pF_upper, 31)
    return _q3_phase_map(base, pf_range_arr, np.asarray(loss_range),
                         scope_label=f"Q3_robust{risk_label}")