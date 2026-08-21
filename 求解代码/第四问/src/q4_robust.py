from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Q2_SCENARIOS, Q3_NODES, Q3_REPLACEMENT_LOSS
from .q2_model import solve_q2, solve_q2_robust, solve_q2_robust_corners
from .q3_model import evaluate_q3_strategy_detailed, solve_q3_fast, solve_q3_flat, solve_q3_robust_corners
from .q4_uncertainty import (
    SampleRecord,
    bootstrap_record,
    build_interval_table,
)
from .q4_validation import audit_q2_monotonicity, audit_q3_monotonicity


def _map_q2_quality(records: Sequence[SampleRecord], sid: int) -> Dict[str, SampleRecord]:
    ids = {f"Q2-S{sid}-C1": "p1", f"Q2-S{sid}-C2": "p2", f"Q2-S{sid}-F": "pf"}
    rmap = {r.quality_id: r for r in records}
    missing = [x for x in ids if x not in rmap]
    if missing:
        raise KeyError(f"Q2 scenario {sid} missing records: {missing}")
    return {param: rmap[qid] for qid, param in ids.items()}


def _map_q3_quality(records: Sequence[SampleRecord]) -> Dict[str, SampleRecord]:
    ids = [f"Q3-C{i}" for i in range(1, 9)] + ["Q3-H1", "Q3-H2", "Q3-H3", "Q3-F"]
    rmap = {r.quality_id: r for r in records}
    missing = [x for x in ids if x not in rmap]
    if missing:
        raise KeyError(f"Q3 missing records: {missing}")
    return {qid.replace("Q3-", ""): rmap[qid] for qid in ids}


def _interval_vectors(
    mapped: Mapping[str, SampleRecord],
    family_size: int,
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], Dict[str, float]]:
    """构造同时 90%/95% 上界向量。

    family_size 显式传入 Bonferroni 族元素数：Q2 每情形 3；Q3 系统 12。
    """
    recs = list(mapped.values())
    tab = build_interval_table(recs, family_size=family_size)
    byid = tab.set_index("quality_id")
    inv = {r.quality_id: key for key, r in mapped.items()}
    phat = {inv[qid]: float(row["p_hat"]) for qid, row in byid.iterrows()}
    u90 = {inv[qid]: float(row["U_simultaneous_90"]) for qid, row in byid.iterrows()}
    u95 = {inv[qid]: float(row["U_simultaneous_95"]) for qid, row in byid.iterrows()}
    return tab, phat, u90, u95


def solve_q4_q2(
    records: Sequence[SampleRecord],
    topk: int = 10,
    data_source: str = "fixture",
):
    interval_rows = []
    result_rows = []
    top_rows = []
    mono_rows = []
    for sid, base in Q2_SCENARIOS.items():
        try:
            mapped = _map_q2_quality(records, sid)
        except KeyError:
            continue
        # family_size = 3: Q2 每个情形只有 p1, p2, pf
        tab, phat, u90, u95 = _interval_vectors(mapped, family_size=3)
        tab.insert(0, "scope", f"Q2-S{sid}")
        tab["family_size"] = 3
        tab["data_source"] = data_source
        interval_rows.append(tab)

        # 名义点估计只用于对照，不冒充真实参数。
        pp = base.with_quality(phat["p1"], phat["p2"], phat["pf"])
        best, rank = solve_q2(pp, topk=topk)
        result_rows.append({
            "scenario_id": sid,
            "risk_level": "nominal",
            "robust_method": "point_estimate_reference",
            "policy4": best.code4,
            "policy6": best.code6,
            "expected_cost": best.expected_cost,
            "expected_profit": best.expected_profit,
            "gap": float(rank.iloc[0]["primary_policy_gap"]),
            "family_size": 3,
            "data_source": data_source,
        })
        r2 = rank.copy()
        r2.insert(0, "robust_method", "point_estimate_reference")
        r2.insert(0, "risk_level", "nominal")
        r2.insert(0, "scenario_id", sid)
        top_rows.append(r2)

        for gamma, u, risk in [(0.90, u90, "robust90"), (0.95, u95, "robust95")]:
            m = audit_q2_monotonicity(base, u)
            m.insert(0, "confidence", gamma)
            m.insert(0, "scope", f"Q2-S{sid}")
            mono_rows.append(m)
            mono_ok = bool(m["PASS"].all())
            if mono_ok:
                best, rank = solve_q2_robust(base, u, topk=topk)
                method = "upper_corner_by_monotonicity"
            else:
                # 单调性失败时回退到完整 2^3 盒端点搜索，禁止静默沿用上界捷径。
                best, rank = solve_q2_robust_corners(base, u, topk=topk)
                method = "8_corner_fallback"
            gap = float(rank.iloc[0]["primary_policy_gap"])
            result_rows.append({
                "scenario_id": sid, "risk_level": risk, "robust_method": method,
                "policy4": best.code4, "policy6": best.code6,
                "expected_cost": best.expected_cost,
                "expected_profit": best.expected_profit, "gap": gap,
                "family_size": 3, "data_source": data_source,
            })
            r2 = rank.copy()
            r2.insert(0, "robust_method", method)
            r2.insert(0, "risk_level", risk)
            r2.insert(0, "scenario_id", sid)
            top_rows.append(r2)
    return (
        pd.concat(interval_rows, ignore_index=True) if interval_rows else pd.DataFrame(),
        pd.DataFrame(result_rows),
        pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame(),
        pd.concat(mono_rows, ignore_index=True) if mono_rows else pd.DataFrame(),
    )


def solve_q4_q3(
    records: Sequence[SampleRecord],
    topk: int = 10,
    flat_cross_check: bool = True,
    data_source: str = "fixture",
):
    mapped = _map_q3_quality(records)
    # family_size = 12: Q3 系统 8 raw + 4 process
    tab, phat, u90, u95 = _interval_vectors(mapped, family_size=12)
    tab.insert(0, "scope", "Q3")
    tab["family_size"] = 12
    tab["data_source"] = data_source
    rows = []
    tops = []
    mono = []

    # 名义点估计参考解。
    code, profit, top = solve_q3_fast(phat, topk=topk)
    detail = evaluate_q3_strategy_detailed(code, phat)
    row = {
        "risk_level": "nominal",
        "robust_method": "point_estimate_reference",
        "policy16": code,
        "expected_cost": detail.expected_cost,
        "expected_profit": profit,
        "gap": float(top.iloc[0]["best_second_gap"]),
        "family_size": 12,
        "data_source": data_source,
    }
    if flat_cross_check:
        fcode, fprofit = solve_q3_flat(phat)
        row.update({
            "flat_policy16": fcode, "flat_profit": fprofit,
            "flat_abs_error": abs(fprofit - profit),
            "crosscheck_PASS": bool(fcode == code and abs(fprofit - profit) < 1e-9),
        })
    rows.append(row)
    t = top.copy()
    t.insert(0, "robust_method", "point_estimate_reference")
    t.insert(0, "risk_level", "nominal")
    tops.append(t)

    for gamma, u, risk in [(0.90, u90, "robust90"), (0.95, u95, "robust95")]:
        m = audit_q3_monotonicity(u)
        m.insert(0, "confidence", gamma)
        mono.append(m)
        mono_ok = bool(m["PASS"].all())
        if mono_ok:
            code, profit, top = solve_q3_fast(u, topk=topk)
            method = "upper_corner_by_monotonicity"
        else:
            # 单调性失败时切换到盒端点直接搜索：4096角点 × 65536策略由向量化评价完成。
            code, profit, top = solve_q3_robust_corners(u, topk=topk)
            method = "4096_corner_fallback"
        detail = evaluate_q3_strategy_detailed(code, u if mono_ok else {
            k: float(top.iloc[0].get(f"worst_{k}", u[k])) for k in Q3_NODES
        })
        row = {
            "risk_level": risk,
            "robust_method": method,
            "policy16": code,
            "expected_cost": float(200 - profit),
            "expected_profit": profit,
            "gap": float(top.iloc[0]["best_second_gap"]),
            "family_size": 12,
            "data_source": data_source,
        }
        if flat_cross_check and mono_ok:
            fcode, fprofit = solve_q3_flat(u)
            row.update({
                "flat_policy16": fcode, "flat_profit": fprofit,
                "flat_abs_error": abs(fprofit - profit),
                "crosscheck_PASS": bool(fcode == code and abs(fprofit - profit) < 1e-9),
            })
        elif flat_cross_check:
            row.update({
                "flat_policy16": "not_applicable_corner_robust",
                "flat_profit": np.nan, "flat_abs_error": np.nan,
                "crosscheck_PASS": np.nan,
            })
        rows.append(row)
        t = top.copy()
        t.insert(0, "robust_method", method)
        t.insert(0, "risk_level", risk)
        tops.append(t)
    return tab, pd.DataFrame(rows), pd.concat(tops, ignore_index=True), pd.concat(mono, ignore_index=True)


# ============================================================
# Price of Robustness 分解
# ============================================================

def robustness_decomposition_q2(
    records: Sequence[SampleRecord],
    family_size: int = 3,
    data_source: str = "fixture",
) -> pd.DataFrame:
    """对每个 Q2 情形计算：

    P_NN = Π(π_N, p̂)           名义策略在名义参数
    P_RN = Π(π_R, p̂)           稳健策略在名义参数
    P_NU = Π(π_N, U)            名义策略在 95% 同时上界
    P_RU = Π(π_R, U)            稳健策略在 95% 同时上界

    PoR = P_NN - P_RN            Price of Robustness（名义环境下为安全付出的代价）
    RG  = P_RU - P_NU            Robust Gain（参数真变坏时稳健策略额外保护的利润）
    PDE = P_NN - P_NU            Parameter Deterioration Effect（仅参数恶化造成的利润损失）
    WG  = P_RU                   Worst-case Guarantee（稳健策略最坏利润）
    """
    rows = []
    for sid, base in Q2_SCENARIOS.items():
        try:
            mapped = _map_q2_quality(records, sid)
        except KeyError:
            continue
        tab, phat, u90, u95 = _interval_vectors(mapped, family_size=family_size)
        u95_vec = u95

        # 名义策略：p̂
        pp_hat = base.with_quality(phat["p1"], phat["p2"], phat["pf"])
        best_n, _ = solve_q2(pp_hat, topk=1)
        pi_N_code = best_n.code6

        # 95% 稳健策略：在 U95 上界上重求
        best_R, _ = solve_q2_robust(base, u95_vec, topk=1)
        pi_R_code = best_R.code6

        # 计算 P_RN：把 π_R 放在 p̂ 上
        pp_hat_2 = base.with_quality(phat["p1"], phat["p2"], phat["pf"])
        from .q2_model import evaluate_q2_strategy_scalar
        _, P_RN, _, _ = evaluate_q2_strategy_scalar(pp_hat_2, tuple(map(int, pi_R_code)))
        _, P_NN, _, _ = evaluate_q2_strategy_scalar(pp_hat_2, tuple(map(int, pi_N_code)))

        # 计算 P_NU：把 π_N 放在 U95 上
        pp_up = base.with_quality(u95_vec["p1"], u95_vec["p2"], u95_vec["pf"])
        _, P_NU, _, _ = evaluate_q2_strategy_scalar(pp_up, tuple(map(int, pi_N_code)))
        _, P_RU, _, _ = evaluate_q2_strategy_scalar(pp_up, tuple(map(int, pi_R_code)))

        rows.append({
            "scenario_id": sid,
            "family_size": family_size,
            "data_source": data_source,
            "pi_N_code6": pi_N_code, "pi_R_code6": pi_R_code,
            "P_NN": P_NN, "P_RN": P_RN, "P_NU": P_NU, "P_RU": P_RU,
            "PoR_P_NN_minus_P_RN": P_NN - P_RN,
            "RobustGain_P_RU_minus_P_NU": P_RU - P_NU,
            "PDE_P_NN_minus_P_NU": P_NN - P_NU,
            "WG_P_RU": P_RU,
            "switched_strategy": pi_N_code != pi_R_code,
        })
    return pd.DataFrame(rows)


def robustness_decomposition_q3(
    records: Sequence[SampleRecord],
    family_size: int = 12,
    data_source: str = "fixture",
) -> pd.DataFrame:
    """对 Q3 计算 PoR 分解。"""
    mapped = _map_q3_quality(records)
    tab, phat, u90, u95 = _interval_vectors(mapped, family_size=family_size)

    # 名义策略
    pi_N, profit_N, _ = solve_q3_fast(phat, topk=1)
    # 95% 稳健策略
    pi_R, profit_R, _ = solve_q3_fast(u95, topk=1)

    # 在 p̂ 上 π_R 的利润
    from .q3_model import _all_profit_vector
    profits_at_phat, _ = _all_profit_vector(phat)
    from .q3_model import _cached_policy_codes
    codes = list(_cached_policy_codes())
    P_NN = float(profit_N)
    P_RN = float(profits_at_phat[codes.index(pi_R)])
    # 重新用 all_profit_vector 在 U95 上计算两个策略的利润
    profits_at_U95, _ = _all_profit_vector(u95)
    P_NU = float(profits_at_U95[codes.index(pi_N)])
    P_RU = float(profits_at_U95[codes.index(pi_R)])

    return pd.DataFrame([{
        "scope": "Q3",
        "family_size": family_size,
        "data_source": data_source,
        "pi_N_policy16": pi_N,
        "pi_R_policy16": pi_R,
        "P_NN": P_NN, "P_RN": P_RN, "P_NU": P_NU, "P_RU": P_RU,
        "PoR_P_NN_minus_P_RN": P_NN - P_RN,
        "RobustGain_P_RU_minus_P_NU": P_RU - P_NU,
        "PDE_P_NN_minus_P_NU": P_NN - P_NU,
        "WG_P_RU": P_RU,
        "switched_strategy": pi_N != pi_R,
    }])


# ============================================================
# Bootstrap
# ============================================================

def _bootstrap_once_records(
    records: Sequence[SampleRecord],
    rng: np.random.Generator,
    max_seq_n: int,
) -> Tuple[List[SampleRecord], int]:
    out = []
    cens = 0
    for r in records:
        rr, c = bootstrap_record(r, rng, max_seq_n=max_seq_n)
        out.append(rr)
        cens += int(c)
    return out, cens


def bootstrap_stability_q2(
    records: Sequence[SampleRecord],
    sid: int,
    seed: int = 20260820,
    start_B: int = 64,
    max_B: int = 512,
    max_seq_n: int = 10000,
    data_source: str = "fixture",
) -> pd.DataFrame:
    mapped = _map_q2_quality(records, sid)
    base_records = list(mapped.values())
    base = Q2_SCENARIOS[sid]
    rng = np.random.default_rng(seed + sid)
    counts = {"nominal": Counter(), "robust90": Counter(), "robust95": Counter()}
    profits = {k: [] for k in counts}
    unique = {k: set() for k in counts}
    rows = []
    prev = {}
    total = 0
    censored = 0
    checkpoint = start_B
    while total < max_B:
        add = min(checkpoint - total, max_B - total)
        for _ in range(add):
            boot, c = _bootstrap_once_records(base_records, rng, max_seq_n)
            censored += c
            bm = {r.quality_id: r for r in boot}
            mapped_b = {k: bm[v.quality_id] for k, v in mapped.items()}
            _, ph, u90, u95 = _interval_vectors(mapped_b, family_size=3)
            for risk, q in [("nominal", ph), ("robust90", u90), ("robust95", u95)]:
                if risk == "nominal":
                    best, _ = solve_q2(base.with_quality(q["p1"], q["p2"], q["pf"]), topk=2)
                else:
                    best, _ = solve_q2_robust(base, q, topk=2)
                counts[risk][best.code6] += 1
                unique[risk].add(best.code6)
                profits[risk].append(best.expected_profit)
        total += add
        for risk in counts:
            code, nc = counts[risk].most_common(1)[0]
            rate = nc / total
            se = (rate * (1 - rate) / total) ** 0.5
            arr = np.array(profits[risk])
            rows.append({
                "scope": f"Q2-S{sid}",
                "risk_level": risk,
                "B": total,
                "modal_policy": code,
                "stability_rate": rate,
                "mc_se": se,
                "mean_optimal_profit": float(arr.mean()) if len(arr) > 0 else np.nan,
                "std_optimal_profit": float(arr.std(ddof=1)) if len(arr) > 1 else np.nan,
                "unique_strategy_count": int(len(unique[risk])),
                "sequential_censored_events": censored,
                "data_source": data_source,
            })
        # 收敛：策略与 SE 不再大幅改变
        converged = True
        if total < 2 * start_B:
            converged = False
        if total >= 2 * start_B:
            # 取每 risk 上一轮（再往前 3 条）的快照
            for risk in counts:
                prev_snapshot = next((x for x in reversed(rows[:-3]) if x["risk_level"] == risk), None)
                curr_snapshot = next((x for x in reversed(rows) if x["risk_level"] == risk), None)
                if prev_snapshot is None or curr_snapshot is None:
                    converged = False; break
                if curr_snapshot["modal_policy"] != prev_snapshot["modal_policy"]:
                    converged = False; break
                if abs(curr_snapshot["stability_rate"] - prev_snapshot["stability_rate"]) > 2 * ((curr_snapshot["mc_se"] ** 2 + prev_snapshot["mc_se"] ** 2) ** 0.5):
                    converged = False; break
        if converged:
            break
        checkpoint = min(max_B, max(checkpoint * 2, total + 1))
    return pd.DataFrame(rows)


def bootstrap_stability_q3(
    records: Sequence[SampleRecord],
    seed: int = 20260820,
    start_B: int = 64,
    max_B: int = 512,
    max_seq_n: int = 10000,
    data_source: str = "fixture",
) -> pd.DataFrame:
    mapped = _map_q3_quality(records)
    base_records = list(mapped.values())
    rng = np.random.default_rng(seed + 300)
    counts = {"nominal": Counter(), "robust90": Counter(), "robust95": Counter()}
    profits = {k: [] for k in counts}
    unique = {k: set() for k in counts}
    rows = []
    total = 0
    censored = 0
    checkpoint = start_B
    while total < max_B:
        add = min(checkpoint - total, max_B - total)
        for _ in range(add):
            boot, c = _bootstrap_once_records(base_records, rng, max_seq_n)
            censored += c
            bm = {r.quality_id: r for r in boot}
            mapped_b = {k: bm[v.quality_id] for k, v in mapped.items()}
            _, ph, u90, u95 = _interval_vectors(mapped_b, family_size=12)
            for risk, q in [("nominal", ph), ("robust90", u90), ("robust95", u95)]:
                code, profit, _ = solve_q3_fast(q, topk=2)
                counts[risk][code] += 1
                unique[risk].add(code)
                profits[risk].append(profit)
        total += add
        try:
            for risk in counts:
                code, nc = counts[risk].most_common(1)[0]
                rate = nc / total
                se = (rate * (1 - rate) / total) ** 0.5
                arr = np.array(profits[risk])
                rows.append({
                    "scope": "Q3",
                    "risk_level": risk,
                    "B": total,
                    "modal_policy": code,
                    "stability_rate": rate,
                    "mc_se": se,
                    "mean_optimal_profit": float(arr.mean()) if len(arr) > 0 else np.nan,
                    "std_optimal_profit": float(arr.std(ddof=1)) if len(arr) > 1 else np.nan,
                    "unique_strategy_count": int(len(unique[risk])),
                    "sequential_censored_events": censored,
                    "data_source": data_source,
                })
        except ValueError as e:
            # Bootstrap 出现极端抽到（k/n=1）→ 跳过本轮
            rows.append({"scope": "Q3", "B": total, "status": f"bootstrap batch skipped: {e}",
                         "data_source": data_source})
            checkpoint = min(max_B, max(checkpoint * 2, total + 1))
            continue
        converged = True
        if total < 2 * start_B:
            converged = False
        if total >= 2 * start_B:
            for risk in counts:
                curr = rows[-1] if rows and rows[-1]["risk_level"] == risk else None
                prev_snapshot = next((x for x in reversed(rows[:-3]) if x["risk_level"] == risk), None)
                if curr is None or prev_snapshot is None:
                    converged = False; break
                if curr["modal_policy"] != prev_snapshot["modal_policy"] or abs(curr["stability_rate"] - prev_snapshot["stability_rate"]) > 2 * ((curr["mc_se"] ** 2 + prev_snapshot["mc_se"] ** 2) ** 0.5):
                    converged = False; break
        if converged:
            break
        checkpoint = min(max_B, max(checkpoint * 2, total + 1))
    return pd.DataFrame(rows)


# ============================================================
# 信息充分性曲线
# ============================================================

def information_sufficiency_q2(
    records: Sequence[SampleRecord],
    factors: Sequence[int] = (1, 2, 4, 8, 16),
    family_size: int = 3,
) -> pd.DataFrame:
    rows = []
    for sid, base in Q2_SCENARIOS.items():
        try:
            mapped = _map_q2_quality(records, sid)
        except KeyError:
            continue
        if any(r.sample_design != "fixed" for r in mapped.values()):
            rows.append({"scope": f"Q2-S{sid}", "status": "skipped: sequential records require actual continued sampling"})
            continue
        for factor in factors:
            projected = {}
            for key, r in mapped.items():
                if factor == 1:
                    rr = r
                else:
                    n2 = int(round(r.n * factor))
                    k2 = int(round(r.p_hat * n2))
                    rr = r.with_counts(n2, k2)
                projected[key] = rr
            _, ph, u90, u95 = _interval_vectors(projected, family_size=family_size)
            for risk, q in [("nominal", ph), ("robust90", u90), ("robust95", u95)]:
                if risk == "nominal":
                    best, top = solve_q2(base.with_quality(q["p1"], q["p2"], q["pf"]), topk=3)
                else:
                    best, top = solve_q2_robust(base, q, topk=3)
                rows.append({
                    "scope": f"Q2-S{sid}",
                    "sample_multiplier": factor,
                    "projection_assumption": "future sample fraction equals current p_hat",
                    "risk_level": risk,
                    "policy4": best.code4,
                    "policy6": best.code6,
                    "profit": best.expected_profit,
                    "gap": float(top.iloc[0]["primary_policy_gap"]),
                    "mean_upper_minus_phat": float(np.mean([q[k] - ph[k] for k in ph])),
                    "family_size": family_size,
                })
    return pd.DataFrame(rows)


def information_sufficiency_q3(
    records: Sequence[SampleRecord],
    factors: Sequence[int] = (1, 2, 4, 8, 16),
    family_size: int = 12,
) -> pd.DataFrame:
    mapped = _map_q3_quality(records)
    if any(r.sample_design != "fixed" for r in mapped.values()):
        return pd.DataFrame([{
            "scope": "Q3",
            "status": "skipped: sequential records require actual continued sampling",
            "family_size": family_size,
        }])
    rows = []
    for factor in factors:
        projected = {}
        for key, r in mapped.items():
            if factor == 1:
                rr = r
            else:
                n2 = int(round(r.n * factor))
                k2 = int(round(r.p_hat * n2))
                rr = r.with_counts(n2, k2)
            projected[key] = rr
        _, ph, u90, u95 = _interval_vectors(projected, family_size=family_size)
        for risk, q in [("nominal", ph), ("robust90", u90), ("robust95", u95)]:
            code, profit, top = solve_q3_fast(q, topk=2)
            rows.append({
                "scope": "Q3",
                "sample_multiplier": factor,
                "projection_assumption": "future sample fraction equals current p_hat",
                "risk_level": risk,
                "policy16": code,
                "profit": profit,
                "gap": float(top.iloc[0]["best_second_gap"]),
                "mean_upper_minus_phat": float(np.mean([q[k] - ph[k] for k in ph])),
                "family_size": family_size,
            })
    return pd.DataFrame(rows)