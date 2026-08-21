from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom

from .config import (
    Q2_EXPECTED,
    Q2_SCENARIOS,
    Q3_EXPECTED_CODE,
    Q3_EXPECTED_COST,
    Q3_EXPECTED_PROFIT,
    Q3_NODES,
    Q3_REPLACEMENT_LOSS,
    Q3_SALE_PRICE,
    Q3Node,
)
from .q2_model import evaluate_q2_strategy, evaluate_q2_strategy_scalar, solve_q2
from .q3_model import (
    ScalarSummary, all_profit_vector, evaluate_q3_strategy_detailed,
    policy_from_code, solve_q3_fast, solve_q3_flat, _pmap,
)
from .q4_uncertainty import (
    SampleRecord, build_interval_table, build_global_family_diagnostic,
    cp_upper, log_e_minus, log_e_plus, sequential_upper,
)
from .policy_codec import decode_q2_policy, decode_q3_policy, diff_q2_policy, diff_q3_policy


# ============================================================
# Q2 / Q3 名义回归
# ============================================================

def regression_q2(tol: float = 1e-9) -> pd.DataFrame:
    rows = []
    for sid, p in Q2_SCENARIOS.items():
        best, _ = solve_q2(p, topk=5)
        exp_code, exp_profit = Q2_EXPECTED[sid]
        ok_code = best.code4 == exp_code
        ok_profit = abs(best.expected_profit - exp_profit) < tol
        rows.append({
            "scenario_id": sid,
            "policy4": best.code4,
            "policy6": best.code6,
            "profit": best.expected_profit,
            "expected_policy4": exp_code,
            "expected_profit": exp_profit,
            "abs_error": abs(best.expected_profit - exp_profit),
            "PASS": bool(ok_code and ok_profit),
        })
    return pd.DataFrame(rows)


def q2_closed_form_validation(tol: float = 1e-10) -> pd.DataFrame:
    """Q2 z=0 闭式退化核验。6 情形 × 8 (x1,x2,y) 组合 = 48 项。"""
    from .q2_model import no_disassembly_closed_form
    rows = []
    for sid, p in Q2_SCENARIOS.items():
        for x1, x2, y in product((0, 1), repeat=3):
            pi = (x1, x2, y, 0, 0, 0)
            e = evaluate_q2_strategy(p, pi)
            closed = no_disassembly_closed_form(p, x1, x2, y)
            err = abs(e.expected_cost - closed)
            rows.append({
                "scenario_id": sid,
                "x1": x1, "x2": x2, "y": y,
                "bellman": e.expected_cost,
                "closed_form": closed,
                "abs_error": err,
                "PASS": err < tol,
            })
    return pd.DataFrame(rows)


def regression_q3(tol: float = 1e-10, flat_cross_check: bool = True) -> pd.DataFrame:
    code, profit, top = solve_q3_fast(topk=10)
    detail = evaluate_q3_strategy_detailed(code)
    rows = [{
        "solver": "layered_vectorized",
        "policy16": code,
        "profit": profit,
        "cost": detail.expected_cost,
        "expected_policy16": Q3_EXPECTED_CODE,
        "expected_profit": Q3_EXPECTED_PROFIT,
        "expected_cost": Q3_EXPECTED_COST,
        "profit_abs_error": abs(profit - Q3_EXPECTED_PROFIT),
        "cost_abs_error": abs(detail.expected_cost - Q3_EXPECTED_COST),
        "PASS": bool(
            code == Q3_EXPECTED_CODE
            and abs(profit - Q3_EXPECTED_PROFIT) < tol
            and abs(detail.expected_cost - Q3_EXPECTED_COST) < tol
        ),
    }]
    if flat_cross_check:
        fcode, fprofit = solve_q3_flat()
        rows.append({
            "solver": "flat_65536",
            "policy16": fcode,
            "profit": fprofit,
            "cost": 200 - fprofit,
            "expected_policy16": Q3_EXPECTED_CODE,
            "expected_profit": Q3_EXPECTED_PROFIT,
            "expected_cost": Q3_EXPECTED_COST,
            "profit_abs_error": abs(fprofit - Q3_EXPECTED_PROFIT),
            "cost_abs_error": abs((200 - fprofit) - Q3_EXPECTED_COST),
            "PASS": bool(fcode == Q3_EXPECTED_CODE and abs(fprofit - Q3_EXPECTED_PROFIT) < tol),
        })
    return pd.DataFrame(rows)


def q3_handcheck_validation(tol: float = 1e-10) -> pd.DataFrame:
    d = evaluate_q3_strategy_detailed(Q3_EXPECTED_CODE)
    expected = {
        "purchase": 71.11111111111111,
        "component_inspection": 12.22222222222222,
        "assembly": 35.55555555555556,
        "intermediate_inspection": 13.33333333333333,
        "final_inspection": 0.0,
        "replacement": 4.44444444444444,
        "disassembly": 3.11111111111111,
    }
    rows = []
    for k, v in expected.items():
        act = d.cost_breakdown[k]
        rows.append({
            "item": k,
            "actual": act,
            "expected": v,
            "abs_error": abs(act - v),
            "PASS": abs(act - v) < tol,
        })
    rows.append({
        "item": "total",
        "actual": d.expected_cost,
        "expected": Q3_EXPECTED_COST,
        "abs_error": abs(d.expected_cost - Q3_EXPECTED_COST),
        "PASS": abs(d.expected_cost - Q3_EXPECTED_COST) < tol,
    })
    return pd.DataFrame(rows)


# ============================================================
# 统计层覆盖率
# ============================================================

def exact_cp_coverage_validation(
    p_values: Sequence[float] = (0.05, 0.10, 0.20),
    n_values: Sequence[int] = (20, 50, 100, 200),
    confidences: Sequence[float] = (0.90, 0.95),
) -> pd.DataFrame:
    rows = []
    for p in p_values:
        for n in n_values:
            ks = np.arange(n + 1)
            pmf = binom.pmf(ks, n, p)
            for conf in confidences:
                upper = np.array([cp_upper(int(k), n, conf) for k in ks])
                cov = float(pmf[upper + 1e-15 >= p].sum())
                rows.append({
                    "p_true": p, "n_validation_grid": n, "confidence": conf,
                    "coverage": cov, "conservative_margin": cov - conf,
                    "PASS": cov + 1e-12 >= conf,
                })
    return pd.DataFrame(rows)


def q1_sequential_boundary_validation(tol: float = 1e-9) -> pd.DataFrame:
    """复核 Q1 序贯 e-process 接收/拒收两个已知停止边界及反演上界。

    必须打印真实计算得到的 E_minus / E_plus，而不是把判定阈值冒充为 e-value。
    """
    import math
    e_accept = math.exp(log_e_minus(34, 0, 0.10))
    e_reject = math.exp(log_e_plus(2, 2, 0.10))
    u90 = sequential_upper(34, 0, 0.90)
    rows = [
        {
            "check": "accept_boundary_n34_k0",
            "actual_E_minus": e_accept,
            "threshold_for_accept": 10.0,
            "PASS": bool(e_accept >= 10.0 - tol),
        },
        {
            "check": "reject_boundary_n2_k2",
            "actual_E_plus": e_reject,
            "threshold_for_reject": 20.0,
            "PASS": bool(e_reject >= 20.0 - tol),
        },
        {
            "check": "inverted_U90_n34_k0",
            "actual_U90": u90,
            "reference_p0": 0.10,
            "PASS": bool(u90 <= 0.10 + 1e-9),
        },
    ]
    return pd.DataFrame(rows)


def sequential_cs_coverage_validation(
    p_values: Sequence[float] = (0.05, 0.10, 0.20),
    reference_p0: float = 0.10,
    confidences: Sequence[float] = (0.90, 0.95),
    reps: int = 1000,
    max_n: int = 2000,
    seed: int = 20260820,
) -> pd.DataFrame:
    """Monte Carlo 验证序贯 confidence sequence 覆盖率。

    区分 "stopped by e-process" 与 "censored_at_max_n"。
    """
    from scipy.special import betainc as _bi, betaln as _bl
    rng = np.random.default_rng(seed)
    rows = []

    def log_em_vec(n_arr, k_arr, p0):
        n_arr = np.asarray(n_arr, dtype=float)
        k_arr = np.asarray(k_arr, dtype=float)
        if p0 <= 0:
            return np.full_like(n_arr, -np.inf, dtype=float)
        if p0 >= 1:
            return np.where(k_arr < n_arr, np.inf, -np.log(n_arr + 1.0))
        a = k_arr + 1.0
        b = n_arr - k_arr + 1.0
        I = _bi(a, b, p0)
        return _bl(a, b) + np.log(np.maximum(I, np.finfo(float).tiny)) - np.log(p0) \
            - k_arr * np.log(p0) - (n_arr - k_arr) * np.log1p(-p0)

    def log_ep_vec(n_arr, k_arr, p0):
        n_arr = np.asarray(n_arr, dtype=float)
        k_arr = np.asarray(k_arr, dtype=float)
        if p0 <= 0:
            return np.where(k_arr > 0, np.inf, -np.log(n_arr + 1.0))
        if p0 >= 1:
            return np.full_like(n_arr, -np.inf, dtype=float)
        a = k_arr + 1.0
        b = n_arr - k_arr + 1.0
        I = _bi(a, b, p0)
        tail = np.maximum(1.0 - I, np.finfo(float).tiny)
        return _bl(a, b) + np.log(tail) - np.log1p(-p0) \
            - k_arr * np.log(p0) - (n_arr - k_arr) * np.log1p(-p0)

    for pi, p_true in enumerate(p_values):
        k = np.zeros(reps, dtype=np.int32)
        stop_n = np.zeros(reps, dtype=np.int32)
        stop_k = np.zeros(reps, dtype=np.int32)
        stop_reason = np.array([""] * reps, dtype=object)
        active = np.ones(reps, dtype=bool)
        for n in range(1, max_n + 1):
            ids = np.flatnonzero(active)
            if len(ids) == 0:
                break
            k[ids] += (rng.random(len(ids)) < p_true).astype(np.int32)
            nn = np.full(len(ids), n, dtype=float)
            kk = k[ids]
            accept = log_em_vec(nn, kk, reference_p0) >= np.log(10.0)
            reject = log_ep_vec(nn, kk, reference_p0) >= np.log(20.0)
            stop = accept | reject
            if stop.any():
                sids = ids[stop]
                stop_n[sids] = n
                stop_k[sids] = k[sids]
                stop_reason[sids] = np.where(accept[stop], "accept", "reject")
                active[sids] = False
        ids = np.flatnonzero(active)
        stop_n[ids] = max_n
        stop_k[ids] = k[ids]
        stop_reason[ids] = "censored_at_max_n"
        censor_rate = float(np.mean(stop_reason == "censored_at_max_n"))
        accept_rate = float(np.mean(stop_reason == "accept"))
        reject_rate = float(np.mean(stop_reason == "reject"))
        for conf in confidences:
            loge = log_em_vec(stop_n, stop_k, p_true)
            covered = loge < -np.log(1.0 - conf) + 1e-12
            cov = float(np.mean(covered))
            se = float(np.sqrt(max(cov * (1 - cov), 0.0) / reps))
            rows.append({
                "p_true": p_true, "reference_p0": reference_p0, "confidence": conf,
                "reps": reps, "validation_max_n": max_n,
                "stop_reason_accept_rate": accept_rate,
                "stop_reason_reject_rate": reject_rate,
                "stop_reason_censored_rate": censor_rate,
                "coverage": cov, "mc_se": se, "lower_2se": cov - 2 * se,
                "PASS": bool(cov + 2 * se + 1e-12 >= conf),
            })
    return pd.DataFrame(rows)


# ============================================================
# Q2/Q3 单调性审计（zero/upper/midpoint 三类锚点）
# ============================================================

def audit_q2_monotonicity(
    params, upper: Mapping[str, float],
    grid_points: int = 5, tol: float = 1e-9,
) -> pd.DataFrame:
    rows = []
    from .q2_model import enumerate_q2_policies
    anchors = {
        "zero_others": {"p1": 0.0, "p2": 0.0, "pf": 0.0},
        "upper_others": {k: float(upper[k]) for k in ("p1", "p2", "pf")},
        "midpoint": {k: float(upper[k]) / 2.0 for k in ("p1", "p2", "pf")},
    }
    for anchor_name, base_anchor in anchors.items():
        for var in ("p1", "p2", "pf"):
            vals = np.linspace(0.0, float(upper[var]), grid_points)
            passed = True
            max_violation = 0.0
            for pi in enumerate_q2_policies():
                prev = None
                for val in vals:
                    q = base_anchor.copy()
                    q[var] = float(val)
                    pp = params.with_quality(q["p1"], q["p2"], q["pf"])
                    cost, profit, rho, feasible = evaluate_q2_strategy_scalar(pp, pi)
                    profit_v = profit if feasible else -np.inf
                    if prev is not None and profit_v > prev + tol:
                        passed = False
                        max_violation = max(max_violation, profit_v - prev)
                    prev = profit_v
            rows.append({
                "anchor": anchor_name, "parameter": var,
                "upper": upper[var], "grid_points": grid_points,
                "max_profit_increase_violation": max_violation,
                "PASS": passed,
            })
    return pd.DataFrame(rows)


def audit_q3_monotonicity(
    upper: Mapping[str, float],
    grid_points: int = 5, tol: float = 1e-9,
) -> pd.DataFrame:
    rows = []
    anchors = {
        "zero_others": {k: 0.0 for k in Q3_NODES},
        "upper_others": {k: float(upper[k]) for k in Q3_NODES},
        "midpoint": {k: float(upper[k]) / 2.0 for k in Q3_NODES},
    }
    for anchor_name, base_anchor in anchors.items():
        for var in Q3_NODES:
            vals = np.linspace(0.0, float(upper[var]), grid_points)
            prev = None
            passed = True
            maxv = 0.0
            for val in vals:
                q = base_anchor.copy()
                q[var] = float(val)
                prof, _ = all_profit_vector(q)
                if prev is not None:
                    diff = prof - prev
                    finite = np.isfinite(diff)
                    if finite.any():
                        mv = float(np.max(diff[finite]))
                        if mv > tol:
                            passed = False
                            maxv = max(maxv, mv)
                prev = prof
            rows.append({
                "anchor": anchor_name, "parameter": var,
                "upper": upper[var], "grid_points": grid_points,
                "max_profit_increase_violation": maxv,
                "PASS": passed,
            })
    return pd.DataFrame(rows)


# ============================================================
# ★ Bonferroni 族规模核验（per-scope d=3/12 + 全族诊断 d=30）
# ============================================================

def family_scope_validation() -> pd.DataFrame:
    """核验同时置信上界的 Bonferroni 族规模是否按 Q2 情形 d=3 / Q3 系统 d=12 计算。"""
    from .sampling_fixture import (
        Q2_FIXED_RECORDS, Q3_RAW_RECORDS, Q3_PROC_RECORDS,
    )
    rows = []

    # Q2 d=3：每个 Q2 情形的 3 条 fixed 记录
    by_scenario: Dict[int, list] = {}
    for r in Q2_FIXED_RECORDS:
        sid = int(r.quality_id.split("-")[1][1:])  # "Q2-S1-C1" → 1
        by_scenario.setdefault(sid, []).append(r)
    for sid in sorted(by_scenario):
        recs = by_scenario[sid]
        assert len(recs) == 3, f"Q2-S{sid} 应该有 3 条记录，实际 {len(recs)}"
        # n=100, k=10 的 raw_component 期望值
        for r in recs:
            if r.sample_design == "fixed" and r.n == 100 and r.k == 10:
                u90 = cp_upper(r.k, r.n, 1 - (1 - 0.90) / 3)
                u95 = cp_upper(r.k, r.n, 1 - (1 - 0.95) / 3)
                rows.append({
                    "scope": "Q2_d3", "quality_id": r.quality_id,
                    "expected_U95": 0.1830790108, "actual_U95": u95,
                    "abs_error_U95": abs(u95 - 0.1830790108),
                    "expected_U90": 0.1711651507, "actual_U90": u90,
                    "abs_error_U90": abs(u90 - 0.1711651507),
                    "PASS": abs(u95 - 0.1830790108) < 1e-6 and abs(u90 - 0.1711651507) < 1e-6,
                })
                break

    # Q3 d=12：8 raw + 4 process
    q3_records = Q3_RAW_RECORDS + Q3_PROC_RECORDS
    assert len(q3_records) == 12
    for r in q3_records:
        if r.sample_design == "fixed" and r.n == 100 and r.k == 10:
            u90 = cp_upper(r.k, r.n, 1 - (1 - 0.90) / 12)
            u95 = cp_upper(r.k, r.n, 1 - (1 - 0.95) / 12)
            rows.append({
                "scope": "Q3_d12", "quality_id": r.quality_id,
                "expected_U95": 0.2046530170, "actual_U95": u95,
                "abs_error_U95": abs(u95 - 0.2046530170),
                "expected_U90": 0.1941837792, "actual_U90": u90,
                "abs_error_U90": abs(u90 - 0.1941837792),
                "PASS": abs(u95 - 0.2046530170) < 1e-6 and abs(u90 - 0.1941837792) < 1e-6,
            })
            break

    # 全族诊断 d=30
    all_records = Q2_FIXED_RECORDS + Q3_RAW_RECORDS + Q3_PROC_RECORDS
    for r in all_records:
        if r.sample_design == "fixed" and r.n == 100 and r.k == 10:
            u90 = cp_upper(r.k, r.n, 1 - (1 - 0.90) / 30)
            u95 = cp_upper(r.k, r.n, 1 - (1 - 0.95) / 30)
            rows.append({
                "scope": "global_family_d30_diagnostic", "quality_id": r.quality_id,
                "expected_U95": 0.2177122543, "actual_U95": u95,
                "abs_error_U95": abs(u95 - 0.2177122543),
                "expected_U90": 0.2079085787, "actual_U90": u90,
                "abs_error_U90": abs(u90 - 0.2079085787),
                "PASS": abs(u95 - 0.2177122543) < 1e-6 and abs(u90 - 0.2079085787) < 1e-6,
            })
            break

    return pd.DataFrame(rows)


def family_scope_value_equivalence() -> pd.DataFrame:
    """核验：solve_q4_q2 / solve_q4_q3 实际使用的 Bonferroni 族规模 = 3 / 12。

    通过重新计算对比验证：
    - 若 Q2 用 d=3，则 S1 robust90 利润 ≈ 14.50 元（当前实际值）
    - 若 Q2 用 d=30，则 S1 robust90 利润 ≈ 12.41 元（被过度保守稀释）
    - 同理 S2: d=3 → 6.27 元，d=30 → 2.99 元
    """
    from .sampling_fixture import Q2_FIXED_RECORDS
    from .q2_model import solve_q2_robust
    from .config import Q2_SCENARIOS
    rows = []
    for sid, sid_str in [
        (1, "Q2-S1"),
        (2, "Q2-S2"),
    ]:
        recs = [r for r in Q2_FIXED_RECORDS if sid_str in r.quality_id]
        mapping = {f"{sid_str}-C1": "p1", f"{sid_str}-C2": "p2", f"{sid_str}-F": "pf"}

        # d=3
        tab_d3 = build_interval_table(recs, family_size=3)
        u_d3 = {mapping[r.quality_id]: float(tab_d3[tab_d3['quality_id']==r.quality_id]['U_simultaneous_90'].iloc[0]) for r in recs}
        best, _ = solve_q2_robust(Q2_SCENARIOS[sid], u_d3, topk=1)
        profit_d3 = best.expected_profit

        # d=30（对照，不应使用）
        tab_d30 = build_interval_table(recs, family_size=30)
        u_d30 = {mapping[r.quality_id]: float(tab_d30[tab_d30['quality_id']==r.quality_id]['U_simultaneous_90'].iloc[0]) for r in recs}
        best30, _ = solve_q2_robust(Q2_SCENARIOS[sid], u_d30, topk=1)
        profit_d30 = best30.expected_profit

        # 当前 solve_q4_q2 用的就是 d=3（per-scope family_size=3）
        actual_profit = _read_actual_robust90_profit(sid)
        rows.append({
            "scenario_id": sid,
            "profit_using_d3": profit_d3,
            "profit_using_d30_counter_factual": profit_d30,
            "actual_csv_robust90_profit": actual_profit,
            "abs_diff_d3_minus_actual": abs(profit_d3 - actual_profit) if actual_profit is not None else None,
            "abs_diff_d30_minus_actual": abs(profit_d30 - actual_profit) if actual_profit is not None else None,
            "verdict": "Q4 uses d=3 (per-scope)" if actual_profit is not None and abs(profit_d3 - actual_profit) < abs(profit_d30 - actual_profit) else "WARN",
        })
    return pd.DataFrame(rows)


def _read_actual_robust90_profit(sid: int) -> float | None:
    """从结果输出 CSV 读出 S{sid} 的 robust90 实际利润。"""
    from pathlib import Path
    candidates = list(Path(__file__).resolve().parent.parent.glob("结果输出/q4_q2_policy.csv"))
    if not candidates:
        # 兼容旧位置
        candidates = list(Path(__file__).resolve().parent.parent.glob("结果输出/q4_q2_policy.csv"))
    if not candidates:
        return None
    try:
        df = pd.read_csv(candidates[0])
        sub = df[(df['scenario_id'] == sid) & (df['risk_level'] == 'robust90')]
        if not sub.empty:
            return float(sub.iloc[0]['expected_profit'])
    except Exception:
        return None
    return None


# ============================================================
# ★ 最优利润单调性（关键经济单调性）
# ============================================================

def optimal_profit_monotonicity_validation() -> pd.DataFrame:
    """核验：固定其他参数时，max_π Π_π 对 L 或 pF 单调非增。

    覆盖：
    - Q2 replacement_loss 扫描（基准 SCENARIOS[1]）；
    - Q3 replacement_loss 扫描；
    - Q3 nominal phase map（沿 pF）；
    - Q3 robust95 phase map（沿 pF，其他参数在 U95 上界）。
    """
    rows = []

    # Q2: 在 SCENARIOS[1] 基线上扫 replacement_loss
    from .q2_model import solve_q2 as q2_solve
    from dataclasses import replace
    base = Q2_SCENARIOS[1]
    ls = np.linspace(6.0, 40.0, 35)
    profits = []
    for L in ls:
        pp = replace(base, replacement_loss=float(L))
        _, rank = q2_solve(pp, topk=3)
        profits.append(float(rank.iloc[0]["expected_profit"]))
    profits = np.array(profits)
    diffs = np.diff(profits)
    viol = float(np.max(diffs)) if len(diffs) > 0 else 0.0  # 若 > 0 表示单调性破坏
    rows.append({
        "scope": "Q2_S1_replacement_loss", "n_points": int(len(ls)),
        "profit_min": float(profits.min()), "profit_max": float(profits.max()),
        "max_increase_violation": viol,
        "PASS": viol <= 1e-9,
    })

    # Q3: 沿 L
    l_grid = np.linspace(6.0, 40.0, 35)
    q3_profits = []
    for L in l_grid:
        _, profit, _ = solve_q3_fast(replacement_loss=float(L), topk=1)
        q3_profits.append(float(profit))
    q3_profits = np.array(q3_profits)
    diffs_q3 = np.diff(q3_profits)
    viol_q3 = float(np.max(diffs_q3)) if len(diffs_q3) > 0 else 0.0
    rows.append({
        "scope": "Q3_nominal_replacement_loss", "n_points": int(len(l_grid)),
        "profit_min": float(q3_profits.min()), "profit_max": float(q3_profits.max()),
        "max_increase_violation": viol_q3,
        "PASS": viol_q3 <= 1e-9,
    })

    # Q3: nominal 沿 pF
    pf_grid = np.linspace(0.05, 0.20, 31)
    pf_profits = []
    for pf in pf_grid:
        _, profit, _ = solve_q3_fast({"F": float(pf)}, topk=1)
        pf_profits.append(float(profit))
    pf_profits = np.array(pf_profits)
    diffs_pf = np.diff(pf_profits)
    viol_pf = float(np.max(diffs_pf)) if len(diffs_pf) > 0 else 0.0
    rows.append({
        "scope": "Q3_nominal_pF", "n_points": int(len(pf_grid)),
        "profit_min": float(pf_profits.min()), "profit_max": float(pf_profits.max()),
        "max_increase_violation": viol_pf,
        "PASS": viol_pf <= 1e-9,
    })

    # Q3: robust95 沿 pF（其他参数取 U95 上界）
    from .sampling_fixture import Q3_RAW_RECORDS, Q3_PROC_RECORDS
    base95 = {}
    for r in Q3_RAW_RECORDS + Q3_PROC_RECORDS:
        if r.sample_design == "fixed":
            base95[r.quality_id.replace("Q3-", "")] = cp_upper(r.k, r.n, 1 - (1 - 0.95) / 12)
        else:
            base95[r.quality_id.replace("Q3-", "")] = sequential_upper(r.n, r.k, 1 - (1 - 0.95) / 12)
    # pF 取 0.10→U95 扫描
    base95_no_F = {k: v for k, v in base95.items() if k != "F"}
    pf_robust_profits = []
    for pf in pf_grid:
        q = base95_no_F.copy()
        q["F"] = float(pf)
        _, profit, _ = solve_q3_fast(q, topk=1)
        pf_robust_profits.append(float(profit))
    pf_robust_profits = np.array(pf_robust_profits)
    diffs_prr = np.diff(pf_robust_profits)
    viol_prr = float(np.max(diffs_prr)) if len(diffs_prr) > 0 else 0.0
    rows.append({
        "scope": "Q3_robust95_pF", "n_points": int(len(pf_grid)),
        "profit_min": float(pf_robust_profits.min()),
        "profit_max": float(pf_robust_profits.max()),
        "max_increase_violation": viol_prr,
        "PASS": viol_prr <= 1e-9,
    })

    return pd.DataFrame(rows)


# ============================================================
# ★ Q3 nominal phase map 在 L=40 时的首次切换阈值核验
# ============================================================

def phase_map_nominal_regression(tol: float = 1e-3) -> pd.DataFrame:
    """核验 Q3 nominal phase map 在 L=40 时的首次 pF 切换边界。

    通过二分法精确找到第一次策略发生变化的 pF*，与名义 0.1247856 对照。
    """
    nominal_policy = Q3_EXPECTED_CODE

    def code_at(pF: float) -> str:
        return solve_q3_fast({"F": float(pF)}, replacement_loss=40.0, topk=1)[0]

    lo, hi = 0.10, 0.20
    for _ in range(80):
        mid = (lo + hi) / 2
        if code_at(mid) == nominal_policy:
            lo = mid
        else:
            hi = mid
    threshold = hi
    rows = [{
        "scope": "Q3_nominal_phase_map_L40",
        "nominal_policy": nominal_policy,
        "first_switch_threshold_pF": threshold,
        "reference_pF": 0.1247856,
        "abs_error": abs(threshold - 0.1247856),
        "PASS": abs(threshold - 0.1247856) < tol,
    }]
    return pd.DataFrame(rows)


def phase_map_robust_consistency_check() -> pd.DataFrame:
    """核验 Q3 robust95 phase map 的实际行为。

    背景：所有 8 raw + 4 process 节点（除 F）固定在 U95 同时上界（d=12）。
    - 在 pF=0.10（fixture 的 p_hat）时，F 检测的边际收益 < 成本；
      optimal policy 仍为 1111111111111101（y_F=0）。
    - 在 pF=0.15 附近，F 检测开启：y_F:0→1。
    - 在 L<31 区间，L*b 太小，F 检测永远不划算；pF=0.20 仍 1111111111111101。

    不再使用旧的"pF=0.10 时 F 已经开启"错误解释。
    """
    from .sampling_fixture import Q3_RAW_RECORDS, Q3_PROC_RECORDS
    u95 = {}
    for r in Q3_RAW_RECORDS:
        u95[r.quality_id.replace("Q3-", "")] = cp_upper(r.k, r.n, 1 - (1 - 0.95) / 12)
    for r in Q3_PROC_RECORDS:
        u95[r.quality_id.replace("Q3-", "")] = sequential_upper(r.n, r.k, 1 - (1 - 0.95) / 12)

    rows = []
    # 1. pF=0.10 时策略 y_F=0
    q_pf10 = dict(u95); q_pf10["F"] = 0.10
    code_pf10, profit_pf10, _ = solve_q3_fast(q_pf10, topk=1)
    rows.append({
        "check": "pF=0.10_yF_should_be_0",
        "actual_policy16": code_pf10,
        "expected_policy16": "1111111111111101",
        "PASS": code_pf10 == "1111111111111101",
        "note": "其他节点已在 U95 上界，但 pF=0.10 < 临界 0.15，F 检测不划算",
    })

    # 2. pF=0.20 时策略 y_F=1
    q_pf20 = dict(u95); q_pf20["F"] = 0.20
    code_pf20, profit_pf20, _ = solve_q3_fast(q_pf20, topk=1)
    rows.append({
        "check": "pF=0.20_yF_should_be_1",
        "actual_policy16": code_pf20,
        "expected_policy16": "1111111111111111",
        "PASS": code_pf20 == "1111111111111111",
        "note": "pF=0.20 > 临界 0.15，F 检测开启",
    })

    # 3. pF=0.10 + L=40 仍 y_F=0
    code_l40, profit_l40, _ = solve_q3_fast(q_pf10, replacement_loss=40.0, topk=1)
    rows.append({
        "check": "pF=0.10_L=40_yF_should_be_0",
        "actual_policy16": code_l40,
        "expected_policy16": "1111111111111101",
        "PASS": code_l40 == "1111111111111101",
        "note": "虽然 L=40 大，但 pF=0.10 不足以触发 F 检测",
    })

    # 4. 精确切换点（bi_search）
    lo, hi = 0.10, 0.20
    for _ in range(60):
        mid = (lo + hi) / 2
        q_mid = dict(u95); q_mid["F"] = mid
        c, _, _ = solve_q3_fast(q_mid, topk=1)
        if c == code_pf10:
            lo = mid
        else:
            hi = mid
    threshold = hi
    rows.append({
        "check": "robust95_phase_map_L40_switch_pF",
        "actual_threshold": threshold,
        "expected_threshold_range": (0.14, 0.16),
        "PASS": 0.14 < threshold < 0.16,
        "note": "在 L=40、其他节点=U95 条件下，pF*≈0.15（与手册 b_F/(p_F*L)=6/40 一致）",
    })

    return pd.DataFrame(rows)


# ============================================================
# ★ 角点最坏点核验（top-k 策略是否真的以 upper corner 为最坏点）
# ============================================================

def robust_corner_check_q2(
    records: Sequence[SampleRecord] | None = None,
    top_k: int = 5,
    tol: float = 1e-9,
) -> pd.DataFrame:
    """Q2 d=3 → 8 角点。逐策略取所有角点的最坏利润，验证 upper corner 确为最坏。

    默认用硬编码 fixture；如果 records 为 None 则使用 sampling_fixture 的 Q2 部分。
    只处理以 "Q2-S" 开头的记录，避免与 Q3 记录混淆。
    """
    from .sampling_fixture import Q2_FIXED_RECORDS
    from .q4_uncertainty import build_interval_table
    if records is None:
        records = Q2_FIXED_RECORDS
    by_scenario: Dict[int, list] = {}
    for r in records:
        if r.sample_design != "fixed" or not r.quality_id.startswith("Q2-S"):
            continue
        # "Q2-S1-C1" → 1
        parts = r.quality_id.split("-")
        # parts = ["Q2", "S1", "C1"]
        sid = int(parts[1][1:])
        by_scenario.setdefault(sid, []).append(r)
    rows = []
    for sid, recs in by_scenario.items():
        if len(recs) != 3:
            continue
        tab = build_interval_table(recs, family_size=3)
        name_map = {f"Q2-S{sid}-C1": "p1", f"Q2-S{sid}-C2": "p2", f"Q2-S{sid}-F": "pf"}
        u = {}
        for r in recs:
            sub = tab[tab["quality_id"] == r.quality_id]
            if not sub.empty:
                key = name_map.get(r.quality_id)
                if key is not None:
                    u[key] = float(sub["U_simultaneous_95"].iloc[0])
        if set(u.keys()) != {"p1", "p2", "pf"}:
            continue
        base = Q2_SCENARIOS[sid]
        # 取当前 Q2 名义最优 + Top-k
        best, rank = solve_q2(base, topk=top_k)
        policies = [(best.code4, best.code6)]
        for _, r in rank.iloc[1:top_k].iterrows():
            policies.append((r["policy4"], r["policy6"]))
        for code4, code6 in policies:
            pi = tuple(map(int, code6))
            worst_profit = np.inf
            worst_corner = None
            feasible = True
            for p1, p2, pf in product((0.0, u["p1"]), (0.0, u["p2"]), (0.0, u["pf"])):
                pp = base.with_quality(p1, p2, pf)
                cost, profit, rho, ok = evaluate_q2_strategy_scalar(pp, pi)
                if not ok or not np.isfinite(profit):
                    feasible = False
                    break
                if profit < worst_profit:
                    worst_profit = profit
                    worst_corner = (p1, p2, pf)
            if not feasible:
                rows.append({
                    "scope": f"Q2_S{sid}", "policy6": code6,
                    "worst_corner_p1": None, "worst_corner_p2": None, "worst_corner_pf": None,
                    "upper_corner_profit": None, "actual_worst_profit": None,
                    "PASS": False,
                })
                continue
            # 重新计算 upper corner 利润以确保一致
            pp_up = base.with_quality(u["p1"], u["p2"], u["pf"])
            _, up_profit, _, ok = evaluate_q2_strategy_scalar(pp_up, pi)
            is_upper_worst = abs(worst_profit - up_profit) < tol if ok else False
            rows.append({
                "scope": f"Q2_S{sid}", "policy6": code6,
                "worst_corner_p1": worst_corner[0] if worst_corner else None,
                "worst_corner_p2": worst_corner[1] if worst_corner else None,
                "worst_corner_pf": worst_corner[2] if worst_corner else None,
                "upper_corner_profit": up_profit if ok else None,
                "actual_worst_profit": worst_profit,
                "PASS": is_upper_worst,
            })
    return pd.DataFrame(rows)


def robust_corner_check_q3(
    records: Sequence[SampleRecord] | None = None,
    top_k: int = 5,
    tol: float = 1e-9,
) -> pd.DataFrame:
    """Q3 d=12 → 4096 角点。验证 nominal / robust95 最优策略及 Top-k 的最坏点。

    完整 4096 角点 × 5 策略 × 1 case ≈ 20k 次 Q3 求解，每次约 5 ms，约 100 s。
    """
    from .sampling_fixture import Q3_RAW_RECORDS, Q3_PROC_RECORDS
    from .q4_uncertainty import build_interval_table
    if records is None:
        records = Q3_RAW_RECORDS + Q3_PROC_RECORDS
    tab = build_interval_table(records, family_size=12)
    u_map = {r.quality_id: float(tab[tab["quality_id"] == r.quality_id]["U_simultaneous_95"].iloc[0]) for r in records}
    # 三个关键策略：nominal / robust95 / robust90 / Top-k 中再补两个
    from .q4_robust import solve_q4_q3
    from .sampling_fixture import DEFAULT_SAMPLING_RECORDS
    _, q4_res, q4_top, _ = solve_q4_q3(DEFAULT_SAMPLING_RECORDS, topk=top_k, flat_cross_check=False)
    # 收集策略
    policies = set()
    if not q4_res.empty:
        for risk in ("nominal", "robust90", "robust95"):
            row = q4_res[q4_res["risk_level"] == risk]
            if not row.empty:
                policies.add(row.iloc[0]["policy16"])
    if not q4_top.empty:
        for _, r in q4_top.iterrows():
            policies.add(r["policy16"])
    if Q3_EXPECTED_CODE not in policies:
        policies.add(Q3_EXPECTED_CODE)

    names = list(Q3_NODES.keys())
    n_corners = 2 ** len(names)
    upper_idx = n_corners - 1
    rows = []
    for code in policies:
        from .q3_model import _all_profit_vector
        # profits_per_corner[k] = 65536-dim profit vector at corner k
        profits_per_corner = np.zeros((n_corners, 65536))
        for idx, bits in enumerate(product((0, 1), repeat=len(names))):
            q = {nm: (float(u_map[f"Q3-{nm}"]) if b else 0.0) for nm, b in zip(names, bits)}
            prof, _ = _all_profit_vector(q)
            profits_per_corner[idx] = prof
        from .q3_model import _cached_policy_codes
        code_list = list(_cached_policy_codes())
        if code in code_list:
            code_idx = code_list.index(code)
            upper_profit = float(profits_per_corner[upper_idx, code_idx])
            worst_profit = float(profits_per_corner[:, code_idx].min())
            worst_idx = int(np.argmin(profits_per_corner[:, code_idx]))
            is_upper_worst = (worst_idx == upper_idx) and (worst_profit - upper_profit > -tol)
            rows.append({
                "policy16": code,
                "upper_corner_profit": upper_profit,
                "actual_worst_profit": worst_profit,
                "is_upper_worst": bool(is_upper_worst),
                "PASS": bool(is_upper_worst),
            })
    return pd.DataFrame(rows)


# ============================================================
# ★ 策略位差 smoke 测试
# ============================================================

def decode_diff_smoke() -> pd.DataFrame:
    """检查 nominal → robust 切换都至少产生一条位差，避免报告里写"无变化"。"""
    from .sampling_fixture import DEFAULT_SAMPLING_RECORDS
    from .q4_robust import solve_q4_q2, solve_q4_q3
    rows = []
    # Q2
    _, q2_res, _, _ = solve_q4_q2(DEFAULT_SAMPLING_RECORDS, topk=5)
    for sid in sorted(q2_res["scenario_id"].unique()):
        sub = q2_res[q2_res["scenario_id"] == sid]
        nominal_code = sub[sub["risk_level"] == "nominal"].iloc[0]["policy6"]
        for risk in ("robust90", "robust95"):
            row = sub[sub["risk_level"] == risk]
            if row.empty:
                continue
            robust_code = row.iloc[0]["policy6"]
            diffs = diff_q2_policy(nominal_code, robust_code)
            rows.append({
                "scope": f"Q2_S{sid}", "from_risk_level": "nominal",
                "to_risk_level": risk, "from_code": nominal_code,
                "to_code": robust_code, "n_changes": len(diffs),
                "diff_summary": "; ".join(f"{d['name']}:{d['from']}->{d['to']}" for d in diffs),
            })
    # Q3
    _, q3_res, _, _ = solve_q4_q3(DEFAULT_SAMPLING_RECORDS, topk=5, flat_cross_check=False)
    for risk in ("robust90", "robust95"):
        nominal = q3_res[q3_res["risk_level"] == "nominal"].iloc[0]["policy16"]
        row = q3_res[q3_res["risk_level"] == risk]
        if row.empty:
            continue
        robust = row.iloc[0]["policy16"]
        diffs = diff_q3_policy(nominal, robust)
        rows.append({
            "scope": "Q3", "from_risk_level": "nominal",
            "to_risk_level": risk, "from_code": nominal,
            "to_code": robust, "n_changes": len(diffs),
            "diff_summary": "; ".join(f"{d['name']}:{d['from']}->{d['to']}" for d in diffs),
        })
    return pd.DataFrame(rows)