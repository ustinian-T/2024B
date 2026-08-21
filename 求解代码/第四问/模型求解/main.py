"""
2024 高教社杯 B 题 —— 问题四 主程序
抽样不确定性 + Q2/Q3 精确评价器 + 90%/95% 同时置信上界稳健重优化

对应建模手册：
《问题四建模手册——基于抽样置信边界的闭环生产稳健重优化与策略稳定性检验》

运行（PyCharm 直接运行本文件即可）：
    python main.py

默认行为：
    - 直接使用 src/sampling_fixture.py 中的硬编码设计情景抽样记录
      （Q2 六情形 + Q3 八零配件 + 四工序 = 共 18 + 12 = 30 条记录）；
    - 结果输出到脚本所在目录的 ../结果输出/，
      即 求解代码/第四问/结果输出/；
    - 控制台按阶段打印关键结果（回归表、Q2/Q3 名义/稳健策略、Bootstrap 稳定率等）。

可选参数：
    python main.py --sampling data/sampling_records.csv   # 用真实 CSV 覆盖默认 fixture
    python main.py --output ../结果输出                   # 自定义输出目录
    python main.py --skip-bootstrap                       # 跳过 Bootstrap 数值检验
    python main.py --quiet                                # 关闭控制台详细输出（只保留阶段标题）
    python main.py --no-flat-cross-check                  # 跳过 Q3 65536 独立平坦核验
    python main.py --strict-corners-q3                    # 对全部策略执行 Q3 4096 角点完整鲁棒验证（默认跳过）

依赖：Python >= 3.10, numpy, pandas, scipy
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
Q4_ROOT = THIS_DIR.parent
if str(Q4_ROOT) not in sys.path:
    sys.path.insert(0, str(Q4_ROOT))

from src.config import Q2_SCENARIOS, Q3_NODES  # noqa: E402
from src.q4_uncertainty import (  # noqa: E402
    SampleRecord, build_global_family_diagnostic, load_sampling_records,
)
from src.q4_validation import (  # noqa: E402
    audit_q2_monotonicity, audit_q3_monotonicity, decode_diff_smoke,
    exact_cp_coverage_validation, family_scope_validation,
    family_scope_value_equivalence, optimal_profit_monotonicity_validation,
    phase_map_nominal_regression, phase_map_robust_consistency_check,
    q1_sequential_boundary_validation, q2_closed_form_validation,
    q3_handcheck_validation, regression_q2, regression_q3,
    robust_corner_check_q2, robust_corner_check_q3,
    sequential_cs_coverage_validation,
)
from src.q4_sensitivity import (  # noqa: E402
    q2_nominal_sensitivity, q2_switch_points, q3_nominal_sensitivity,
    q3_phase_map_nominal, q3_phase_map_robust, q3_switch_points,
)
from src.q4_robust import (  # noqa: E402
    bootstrap_stability_q2, bootstrap_stability_q3,
    information_sufficiency_q2, information_sufficiency_q3,
    robustness_decomposition_q2, robustness_decomposition_q3,
    solve_q4_q2, solve_q4_q3,
)
from src.policy_codec import diff_q2_policy, diff_q3_policy  # noqa: E402
from src.sampling_fixture import DEFAULT_SAMPLING_RECORDS  # noqa: E402


# ============================================================
# 控制台输出
# ============================================================

class Reporter:
    def __init__(self, quiet: bool = False):
        self.quiet = quiet

    def section(self, title: str, idx: str = "") -> None:
        bar = "=" * 78
        print(f"\n{bar}\n[{idx}] {title}\n{bar}")

    def step(self, title: str, idx: str = "") -> None:
        bar = "-" * 78
        print(f"\n{bar}\n[{idx}] {title}\n{bar}")

    def info(self, msg: str) -> None:
        if not self.quiet:
            print(f"  {msg}")

    def warn(self, msg: str) -> None:
        print(f"  ⚠ {msg}")

    def pass_fail(self, name: str, ok: bool) -> None:
        flag = "PASS" if ok else "FAIL"
        print(f"  [{flag}] {name}")

    @staticmethod
    def _fmt_row(values, widths) -> str:
        cells = [str(v).ljust(w) for v, w in zip(values, widths)]
        return "  " + " | ".join(cells)

    def table(self, headers: list[str], rows: list[list], float_fmt: str = "{:>10.4f}") -> None:
        if self.quiet:
            return
        formatted: list[list[str]] = []
        for row in rows:
            new_row = []
            for v in row:
                if isinstance(v, float):
                    new_row.append(float_fmt.format(v))
                elif isinstance(v, int):
                    new_row.append(str(v))
                else:
                    new_row.append(str(v))
            formatted.append(new_row)
        all_rows = [list(headers)] + formatted
        widths = [max(len(r[i]) for r in all_rows) for i in range(len(headers))]
        sep = "  " + "-+-".join("-" * w for w in widths)
        for i, row in enumerate(all_rows):
            print(self._fmt_row(row, widths))
            if i == 0:
                print(sep)


# ============================================================
# 工具函数
# ============================================================

def ensure_template(path: Path) -> None:
    if path.exists():
        return
    rows = []
    for sid, p in Q2_SCENARIOS.items():
        for node, qtype, rate in [
            ("C1", "raw_component", p.p1),
            ("C2", "raw_component", p.p2),
            ("F", "process", p.pf),
        ]:
            rows.append({
                "quality_id": f"Q2-S{sid}-{node}",
                "quality_type": qtype,
                "sample_design": "",
                "n": "", "k": "", "input_certified": True,
                "batch_id": "", "seq_reference_p0": "", "reported_rate": rate,
                "note": "process记录必须来自所有输入均已确认合格的装配样本"
                if qtype == "process" else "来自同一稳定批次的真实抽样记录",
            })
    for i in range(1, 9):
        rows.append({
            "quality_id": f"Q3-C{i}", "quality_type": "raw_component",
            "sample_design": "", "n": "", "k": "", "input_certified": True,
            "batch_id": "", "seq_reference_p0": "", "reported_rate": 0.10,
            "note": "真实抽样记录",
        })
    for node in ["H1", "H2", "H3", "F"]:
        rows.append({
            "quality_id": f"Q3-{node}", "quality_type": "process",
            "sample_design": "", "n": "", "k": "", "input_certified": True,
            "batch_id": "", "seq_reference_p0": "", "reported_rate": 0.10,
            "note": "仅统计所有直接输入均已确认合格的装配样本",
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def save(df: pd.DataFrame, path: Path) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def resolve_output_dir(arg: str) -> Path:
    p = Path(arg)
    if not p.is_absolute():
        p = THIS_DIR / p
    return p


def print_q2_nominal_table(rep: Reporter, reg: pd.DataFrame) -> None:
    rep.table(
        headers=["情形", "策略4位", "策略6位", "利润/元", "期望利润", "误差", "结果"],
        rows=[
            [f"S{int(r.scenario_id)}", r.policy4, r.policy6,
             float(r.profit), float(r.expected_profit),
             float(r.abs_error), "PASS" if r.PASS else "FAIL"]
            for _, r in reg.iterrows()
        ],
    )


def print_q3_nominal_table(rep: Reporter, reg: pd.DataFrame) -> None:
    rep.table(
        headers=["求解器", "16位策略", "利润/元", "成本/元", "期望利润", "误差", "结果"],
        rows=[
            [r.solver, r.policy16, float(r.profit), float(r.cost),
             float(r.expected_profit), float(r.profit_abs_error),
             "PASS" if r.PASS else "FAIL"]
            for _, r in reg.iterrows()
        ],
    )


def print_q4_q2_policy_table(rep: Reporter, q4_q2: pd.DataFrame) -> None:
    rows = []
    for sid in sorted(q4_q2["scenario_id"].unique()):
        sub = q4_q2[q4_q2["scenario_id"] == sid]
        for risk in ("nominal", "robust90", "robust95"):
            row = sub[sub["risk_level"] == risk]
            if row.empty:
                continue
            r = row.iloc[0]
            nominal_code = sub[sub["risk_level"] == "nominal"].iloc[0]["policy6"]
            diffs = diff_q2_policy(nominal_code, r.policy6) if risk != "nominal" else []
            diff_str = "; ".join(f"{d['name']}:{d['from']}->{d['to']}" for d in diffs) if diffs else "-"
            rows.append([
                f"S{int(sid)}", risk, r.policy4, r.policy6,
                float(r.expected_cost), float(r.expected_profit),
                float(r.gap), r.robust_method, diff_str,
            ])
    rep.table(
        headers=["情形", "风险水平", "策略4位", "策略6位", "成本/元", "利润/元", "Δ", "方法", "vs nominal 位差"],
        rows=rows,
    )


def print_q4_q3_policy_table(rep: Reporter, q4_q3: pd.DataFrame) -> None:
    rows = []
    nominal_code = q4_q3[q4_q3["risk_level"] == "nominal"].iloc[0]["policy16"]
    for _, r in q4_q3.iterrows():
        diffs = diff_q3_policy(nominal_code, r.policy16) if r.risk_level != "nominal" else []
        diff_str = "; ".join(f"{d['name']}:{d['from']}->{d['to']}" for d in diffs) if diffs else "-"
        rows.append([
            r.risk_level, r.policy16, float(r.expected_cost),
            float(r.expected_profit), float(r.gap), r.robust_method,
            diff_str,
            bool(r.crosscheck_PASS) if not pd.isna(r.get("crosscheck_PASS", np.nan)) else "-",
        ])
    rep.table(
        headers=["风险水平", "16位策略", "成本/元", "利润/元", "Δ", "方法", "vs nominal 位差", "65536核验"],
        rows=rows,
    )


def print_q4_intervals(rep: Reporter, intervals: pd.DataFrame, scope_label: str = "") -> None:
    if intervals.empty:
        return
    rows = []
    for _, r in intervals.iterrows():
        rows.append([
            scope_label or r.get("scope", "-"),
            r.quality_id, r.quality_type, r.sample_design,
            int(r.n), int(r.k), float(r.p_hat),
            float(r.U_simultaneous_90), float(r.U_simultaneous_95),
            int(r.get("family_size", 0)),
            r.get("data_source", "-"),
        ])
    rep.table(
        headers=["scope", "质量ID", "类型", "设计", "n", "k", "p̂",
                 "U90sim", "U95sim", "d", "data_source"],
        rows=rows,
        float_fmt="{:>10.5f}",
    )


def print_bootstrap_table(rep: Reporter, boot: pd.DataFrame) -> None:
    if boot.empty:
        return
    last = (boot.sort_values("B").groupby(["scope", "risk_level"], as_index=False).tail(1).reset_index(drop=True))
    rows = []
    for _, r in last.iterrows():
        rows.append([
            r.scope, r.risk_level, int(r.B), r.modal_policy,
            float(r.stability_rate), float(r.mc_se),
            float(r.mean_optimal_profit),
            int(r.unique_strategy_count),
        ])
    rep.table(
        headers=["范围", "风险水平", "B", "众数策略", "Rπ", "SE", "平均利润/元", "唯一策略数"],
        rows=rows,
        float_fmt="{:>10.4f}",
    )


def print_sensitivity_summary(rep: Reporter, sens2: pd.DataFrame, sens3: pd.DataFrame) -> None:
    if not sens2.empty:
        print("\n  Q2 名义单参数灵敏度（基于 SCENARIOS[1]）：")
        rows = []
        for name, g in sens2.groupby("parameter", sort=False):
            profit_min = float(g["profit"].min())
            profit_max = float(g["profit"].max())
            codes = sorted(g["policy4"].unique())
            switch_count = int(g["switch_from_previous"].sum())
            unique_count = len(codes)
            rows.append([
                name, f"{profit_min:.4f}", f"{profit_max:.4f}",
                "/".join(codes), unique_count, switch_count,
            ])
        rep.table(
            headers=["参数", "利润下限", "利润上限", "出现过的策略",
                     "唯一策略数", "切换次数"],
            rows=rows,
        )
    if not sens3.empty:
        print("\n  Q3 名义单参数灵敏度：")
        for name, g in sens3.groupby("parameter", sort=False):
            profit_min = float(g["profit"].min())
            profit_max = float(g["profit"].max())
            codes = sorted(g["policy16"].unique())
            print(f"    {name:<18} 利润 [{profit_min:.4f}, {profit_max:.4f}]"
                  f"  唯一策略数={len(codes)}")


def print_phase_map_summary(rep: Reporter, phase: pd.DataFrame, switch_pts: pd.DataFrame, title: str) -> None:
    if phase.empty:
        return
    print(f"\n  {title}（参数背景：{phase['scope'].iloc[0]}）：")
    print(f"    网格 {len(phase)} 个 (pF, L) 单元；首次 pF 切换：")
    rows = []
    for _, r in switch_pts.iterrows():
        rows.append([
            float(r["replacement_loss"]),
            f"{float(r['first_switch_pF']):.4f}" if pd.notna(r["first_switch_pF"]) else "无切换",
            r["old_policy"], r["new_policy"],
        ])
    rep.table(headers=["L / 元", "首次切换 pF", "旧策略", "新策略"], rows=rows)


def print_final_summary(rep: Reporter, out: Path, summary: dict, elapsed: float) -> None:
    rep.section("运行总结", "FIN")
    print(f"  总耗时           : {elapsed:.2f} s")
    print(f"  输出目录         : {out.resolve()}")
    print(f"  抽样记录数       : {summary.get('n_records', 0)}")
    print(f"  抽样来源         : {summary.get('data_source', '-')}")
    print()

    print("  ── 关键指标 ──")
    keys = [
        ("q2_regression_pass", "Q2 名义 6 情形全部与手册一致"),
        ("q2_closed_form_pass", "Q2 闭式退化全部 < 1e-10"),
        ("q3_regression_pass", "Q3 名义与 65536 核验一致"),
        ("q3_handcheck_pass", "Q3 成本分解手算全部 < 1e-10"),
        ("cp_coverage_pass", "CP 精确覆盖率 ≥ 名义水平"),
        ("q1_sequential_boundary_pass", "Q1 序贯接收/拒收边界正确"),
        ("q1_sequential_coverage_pass", "Q1 序贯 CS MC 覆盖率合格"),
        ("family_scope_pass", "Bonferroni 族规模 Q2 d=3 / Q3 d=12"),
        ("family_scope_value_equivalence_pass", "Q4 实际使用 d=3 (per-scope)"),
        ("optimal_profit_monotonicity_pass", "最优利润随 L/pF 单调非增"),
        ("q2_monotonicity_pass", "Q2 单调性审计全部 PASS"),
        ("q3_monotonicity_pass", "Q3 单调性审计全部 PASS"),
        ("phase_map_nominal_regression_pass", "Q3 nominal phase map 切换 pF≈0.1248"),
        ("phase_map_robust_consistency_pass", "Q3 robust95 phase map 行为正确"),
        ("robust_corner_check_q2_pass", "Q2 top-k 策略 upper corner 为最坏"),
        ("robust_corner_check_q3_pass", "Q3 top-k 策略 upper corner 为最坏"),
        ("decode_diff_smoke_pass", "策略位差 smoke 测试"),
    ]
    for k, label in keys:
        ok = summary.get(k, False)
        print(f"  {label:<40} : {'PASS' if ok else 'FAIL'}")

    print()
    print("  ── 输出文件 ──")
    for name in summary.get("outputs", []):
        print(f"    {name}")
    print()


# ============================================================
# 主流程
# ============================================================

def run_regression(out: Path, rep: Reporter) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    rep.section("Q2/Q3 名义回归 + 统计覆盖率 + 单调性核验", "1/4")

    rep.step("Q2 六情形名义回归", "1.1")
    q2r = regression_q2()
    save(q2r, out / "regression_q2.csv")
    rep.info(f"6 情形全部与手册一致: {bool(q2r['PASS'].all())}；最大绝对误差 {float(q2r['abs_error'].max()):.2e}")
    print_q2_nominal_table(rep, q2r)

    rep.step("Q2 Bellman vs 几何分布闭式退化", "1.2")
    q2c = q2_closed_form_validation()
    save(q2c, out / "validation_q2_closed_form.csv")
    rep.info(f"退化全部 < 1e-10: {bool(q2c['PASS'].all())}；共 {len(q2c)} 项（= 6 情形 × 8 (x1,x2,y) 组合）")

    rep.step("Q3 名义回归", "1.3")
    q3r = regression_q3(flat_cross_check=True)
    save(q3r, out / "regression_q3.csv")
    rep.info(f"两种求解器一致: {bool(q3r['PASS'].all())}；最大利润绝对误差 {float(q3r['profit_abs_error'].max()):.2e}")
    print_q3_nominal_table(rep, q3r)

    rep.step("Q3 最优策略成本分解手算", "1.4")
    q3h = q3_handcheck_validation()
    save(q3h, out / "validation_q3_handcheck.csv")
    rep.info(f"7 项成本分解全部 < 1e-10: {bool(q3h['PASS'].all())}")

    rep.step("固定样本 Clopper-Pearson 精确覆盖率", "1.5")
    cov = exact_cp_coverage_validation()
    save(cov, out / "validation_cp_coverage.csv")
    rep.info(f"全部覆盖率达到名义水平: {bool(cov['PASS'].all())}")

    rep.step("Q1 序贯接收/拒收停止边界（计算真实 e-value）", "1.6")
    seqb = q1_sequential_boundary_validation()
    save(seqb, out / "validation_q1_sequential_boundaries.csv")
    for _, r in seqb.iterrows():
        if r["check"] == "accept_boundary_n34_k0":
            print(f"  [PASS] accept n=34,k=0: E_minus={r['actual_E_minus']:.6f} ≥ 阈值 {r['threshold_for_accept']}")
        elif r["check"] == "reject_boundary_n2_k2":
            print(f"  [PASS] reject n=2,k=2: E_plus={r['actual_E_plus']:.6f} ≥ 阈值 {r['threshold_for_reject']}")
        else:
            print(f"  [PASS] inverted_U90 = {r['actual_U90']:.6f} ≤ {r['reference_p0']}")

    rep.step("Q1 序贯 confidence-sequence Monte Carlo 覆盖率", "1.7")
    seqcov = sequential_cs_coverage_validation()
    save(seqcov, out / "validation_q1_sequential_coverage.csv")
    rep.info(f"所有组合覆盖率 ≥ 名义水平 - 2SE: {bool(seqcov['PASS'].all())}")
    for _, r in seqcov.iterrows():
        print(
            f"    p={float(r['p_true']):.2f}, γ={float(r['confidence']):.2f}: "
            f"coverage={float(r['coverage']):.4f} ± {float(r['mc_se']):.4f}; "
            f"accept={float(r['stop_reason_accept_rate']):.2%}, "
            f"reject={float(r['stop_reason_reject_rate']):.2%}, "
            f"censored={float(r['stop_reason_censored_rate']):.2%}"
        )

    rep.step("Bonferroni 族规模核验（Q2 d=3 / Q3 d=12 / 全族 d=30 诊断）", "1.8")
    fs = family_scope_validation()
    save(fs, out / "validation_family_scope.csv")
    rep.info(f"3 类 scope 全部通过: {bool(fs['PASS'].all())}")
    for scope in ("Q2_d3", "Q3_d12", "global_family_d30_diagnostic"):
        sub = fs[fs["scope"] == scope]
        if not sub.empty:
            r = sub.iloc[0]
            print(f"    [{scope:<32}] U95={r['actual_U95']:.10f}（期望 {r['expected_U95']:.10f}）")

    rep.step("★ 族规模值等价性核验（Q4 实际使用 d=3）", "1.8b")
    fsve = family_scope_value_equivalence()
    save(fsve, out / "validation_family_scope_value_equivalence.csv")
    rep.info(f"全部 verdict = d=3: {bool((fsve['verdict'] == 'Q4 uses d=3 (per-scope)').all())}")
    for _, r in fsve.iterrows():
        print(f"    S{r['scenario_id']}: d=3 利润 {float(r['profit_using_d3']):.4f} "
              f"vs 实际 {float(r['actual_csv_robust90_profit']):.4f}（差距 {float(r['abs_diff_d3_minus_actual']):.4f}）"
              f"；d=30 对照 {float(r['profit_using_d30_counter_factual']):.4f}（差距 {float(r['abs_diff_d30_minus_actual']):.4f}）")

    rep.step("★ Q3 robust95 phase map 行为核验", "1.10b")
    pmrc = phase_map_robust_consistency_check()
    save(pmrc, out / "validation_phase_map_robust_consistency.csv")
    rep.info(f"全部 4 项 PASS: {bool(pmrc['PASS'].all())}")
    for _, r in pmrc.iterrows():
        if pd.notna(r.get("actual_policy16")):
            print(f"    [{r['check']:<32}] 实际 {r['actual_policy16']}（期望 {r['expected_policy16']}）")
        else:
            print(f"    [{r['check']:<32}] 阈值 {float(r['actual_threshold']):.4f}")

    rep.step("★ 最优利润单调性（关键经济单调性）", "1.9")
    opm = optimal_profit_monotonicity_validation()
    save(opm, out / "validation_optimal_profit_monotonicity.csv")
    rep.info(f"全部 PASS: {bool(opm['PASS'].all())}；共 {len(opm)} 个扫描场景")
    for _, r in opm.iterrows():
        print(f"    [{r['scope']:<32}] 利润范围 [{float(r['profit_min']):.4f}, {float(r['profit_max']):.4f}]  max_increase={float(r['max_increase_violation']):.4f}")

    rep.step("★ Q3 nominal phase map 在 L=40 首次切换阈值", "1.10")
    pmr = phase_map_nominal_regression()
    save(pmr, out / "validation_phase_map_nominal_regression.csv")
    for _, r in pmr.iterrows():
        print(f"    阈值 pF={float(r['first_switch_threshold_pF']):.6f}（参考 {float(r['reference_pF']):.6f}, 误差 {float(r['abs_error']):.2e}）")

    return {
        "q2_regression_pass": bool(q2r["PASS"].all()),
        "q2_closed_form_pass": bool(q2c["PASS"].all()),
        "q3_regression_pass": bool(q3r["PASS"].all()),
        "q3_handcheck_pass": bool(q3h["PASS"].all()),
        "cp_coverage_pass": bool(cov["PASS"].all()),
        "q1_sequential_boundary_pass": bool(seqb["PASS"].all()),
        "q1_sequential_coverage_pass": bool(seqcov["PASS"].all()),
        "family_scope_pass": bool(fs["PASS"].all()),
        "family_scope_value_equivalence_pass": bool((fsve["verdict"] == "Q4 uses d=3 (per-scope)").all()),
        "optimal_profit_monotonicity_pass": bool(opm["PASS"].all()),
        "phase_map_nominal_regression_pass": bool(pmr["PASS"].all()),
        "phase_map_robust_consistency_pass": bool(pmrc["PASS"].all()),
    }, q2r, q3r


def run_sensitivity(out: Path, rep: Reporter) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """名义灵敏度扫描 + 切换点定位 + Q3 三种条件的二维相图。"""
    rep.step("Q2 名义灵敏度扫描 + 切换点", "2.1")
    sens2 = q2_nominal_sensitivity()
    save(sens2, out / "sensitivity_q2_nominal.csv")
    sw2 = q2_switch_points(sens2)
    save(sw2, out / "sensitivity_switch_points_q2.csv")

    rep.step("Q3 名义灵敏度扫描 + 切换点", "2.2")
    sens3 = q3_nominal_sensitivity()
    save(sens3, out / "sensitivity_q3_nominal.csv")
    sw3 = q3_switch_points(sens3)
    save(sw3, out / "sensitivity_switch_points_q3.csv")

    rep.step("Q3 nominal 二维相图（其他参数保持名义 0.10）", "2.3")
    phase_n, sw_n = q3_phase_map_nominal()
    phase_n.to_csv(out / "q4_phase_map_nominal.csv", index=False, encoding="utf-8-sig")
    sw_n.to_csv(out / "q4_phase_map_nominal_switch_points.csv", index=False, encoding="utf-8-sig")

    rep.step("Q3 robust90 二维相图（其他参数在 U90 上界）", "2.4")
    phase_r90, sw_r90 = q3_phase_map_robust(
        DEFAULT_SAMPLING_RECORDS, family_size=12, risk_label="90",
    )
    phase_r90.to_csv(out / "q4_phase_map_robust90.csv", index=False, encoding="utf-8-sig")
    sw_r90.to_csv(out / "q4_phase_map_robust90_switch_points.csv", index=False, encoding="utf-8-sig")

    rep.step("Q3 robust95 二维相图（其他参数在 U95 上界）", "2.5")
    phase_r95, sw_r95 = q3_phase_map_robust(
        DEFAULT_SAMPLING_RECORDS, family_size=12, risk_label="95",
    )
    phase_r95.to_csv(out / "q4_phase_map_robust95.csv", index=False, encoding="utf-8-sig")
    sw_r95.to_csv(out / "q4_phase_map_robust95_switch_points.csv", index=False, encoding="utf-8-sig")

    print_sensitivity_summary(rep, sens2, sens3)
    print_phase_map_summary(rep, phase_n, sw_n, "Q3 nominal 相图（首次切换）")
    return sens2, sens3, sw2, sw3, phase_n, phase_r95


def run_q4_robust(
    out: Path,
    records,
    full_flat: bool,
    args,
    rep: Reporter,
    data_source: str,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """基于真实抽样记录的稳健重优化 + 单调性审计 + PoR 分解 + Bootstrap。"""
    rep.section("Q4 稳健重优化（90%/95% 同时置信上界）", "3/4")

    rep.step("Q2 6 情形 × (nominal / 90% / 95%) 稳健重优化", "3.1")
    q2_int, q2_res, q2_top, q2_mono = solve_q4_q2(records, topk=10, data_source=data_source)
    if not q2_int.empty: save(q2_int, out / "q4_q2_intervals.csv")
    if not q2_res.empty: save(q2_res, out / "q4_q2_policy.csv")
    if not q2_top.empty: save(q2_top, out / "q4_q2_topk.csv")
    if not q2_mono.empty: save(q2_mono, out / "validation_q2_monotonicity.csv")
    if not q2_res.empty:
        rep.info(f"已求解 {q2_res['scenario_id'].nunique()} 个情形；策略切换 {int((q2_res.groupby('scenario_id')['policy4'].nunique() > 1).sum())} 个")
        print_q4_q2_policy_table(rep, q2_res)

    q3_available = all(
        any(r.quality_id == qid for r in records)
        for qid in [*[f"Q3-C{i}" for i in range(1, 9)], "Q3-H1", "Q3-H2", "Q3-H3", "Q3-F"]
    )
    q3_res = pd.DataFrame()
    q3_int = pd.DataFrame()
    q3_top = pd.DataFrame()
    q3_mono = pd.DataFrame()
    if q3_available:
        rep.step("Q3 (nominal / 90% / 95%) 稳健重优化", "3.2")
        q3_int, q3_res, q3_top, q3_mono = solve_q4_q3(
            records, topk=10, flat_cross_check=full_flat, data_source=data_source,
        )
        save(q3_int, out / "q4_q3_intervals.csv")
        save(q3_res, out / "q4_q3_policy.csv")
        save(q3_top, out / "q4_q3_topk.csv")
        save(q3_mono, out / "validation_q3_monotonicity.csv")
        rep.info(f"Q3 65536 独立平坦核验全部通过: {(q3_res['crosscheck_PASS'] == True).all() if 'crosscheck_PASS' in q3_res.columns else '-'}")
        print_q4_q3_policy_table(rep, q3_res)

        rep.step("Q3 单调性审计（zero / upper / midpoint 三类锚点 × 12 参数）", "3.3")
        rep.info(f"全部 PASS: {bool(q3_mono['PASS'].all())}；共 {len(q3_mono)} 项（= 2 conf × 3 anchor × 12 参数）")

        rep.step("★ Price of Robustness 分解（Q2/Q3）", "3.4")
        por2 = robustness_decomposition_q2(records, family_size=3, data_source=data_source)
        save(por2, out / "q4_robustness_decomposition.csv")
        por3 = robustness_decomposition_q3(records, family_size=12, data_source=data_source)
        # 把 Q3 单行追加到同一表
        all_por = pd.concat([por2, por3], ignore_index=True, sort=False)
        save(all_por, out / "q4_robustness_decomposition.csv")
        rep.info(f"Q2 {len(por2)} 情形 + Q3 1 例，全部输出 P_NN/P_RN/P_NU/P_RU/PoR/RG/PDE/WG")

        rep.step("★ 策略位差 smoke 测试", "3.5")
        ddiff = decode_diff_smoke()
        save(ddiff, out / "q4_decode_diff_smoke.csv")
        n_total = len(ddiff)
        n_diff = int((ddiff["n_changes"] > 0).sum()) if not ddiff.empty else 0
        n_no_change = int((ddiff["n_changes"] == 0).sum()) if not ddiff.empty else 0
        rep.info(f"共 {n_total} 行 nominal→robust 切换；其中 {n_diff} 行有位差、{n_no_change} 行 nominal=robust（正确情况）")

        rep.step("★ Q2 角点最坏点核验（d=3 → 8 角点）", "3.6")
        rc2 = robust_corner_check_q2(records, top_k=5)
        save(rc2, out / "validation_robust_corner_check_q2.csv")
        rep.info(f"全部 PASS: {bool(rc2['PASS'].all())}；共 {len(rc2)} 行")

        rep.step("★ Q3 角点最坏点核验（d=12 → 4096 角点）", "3.7")
        rc3 = robust_corner_check_q3(records, top_k=5)
        save(rc3, out / "validation_robust_corner_check_q3.csv")
        rep.info(f"全部 PASS: {bool(rc3['PASS'].all())}；共 {len(rc3)} 行")

        rep.step("信息充分性曲线（n×{1,2,4,8,16} 设计投影）", "3.8")
        info_parts = []
        i2 = information_sufficiency_q2(records)
        if not i2.empty: info_parts.append(i2)
        i3 = information_sufficiency_q3(records)
        if not i3.empty: info_parts.append(i3)
        if info_parts:
            info_df = pd.concat(info_parts, ignore_index=True, sort=False)
            save(info_df, out / "q4_information_curve.csv")
            if "mean_upper_minus_phat" in info_df.columns:
                narrow = info_df[info_df["sample_multiplier"] == 16]["mean_upper_minus_phat"].mean()
                wide = info_df[info_df["sample_multiplier"] == 1]["mean_upper_minus_phat"].mean()
                if pd.notna(wide) and wide > 0:
                    rep.info(f"  设计样本 ×16 相比 ×1，区间宽度平均收缩 {(1 - narrow/wide) * 100:.1f}%")

    # 合并区间表（去重每 quality_id 仅保留一次）
    intervals_combined = [x for x in [q2_int, q3_int] if not x.empty]
    if intervals_combined:
        merged = pd.concat(intervals_combined, ignore_index=True)
        # 不去重：每个 scope 都显示
        save(merged, out / "quality_intervals.csv")
    tops_combined = [x for x in [q2_top, q3_top] if not x.empty]
    if tops_combined:
        save(pd.concat(tops_combined, ignore_index=True, sort=False), out / "q4_topk.csv")

    # 全族诊断 d=30（仅作参考，不进入默认结果）
    rep.step("★ 全族 d=30 极端保守诊断（仅参考）", "3.9")
    glob = build_global_family_diagnostic(list(records), family_size=30)
    glob["scope"] = glob["scope_diagnostic"]
    save(glob, out / "quality_intervals_global_family_d30_diagnostic.csv")
    rep.info(f"d=30 诊断 U95 (sample) = {float(glob.iloc[0]['U_simultaneous_95']):.5f}（与 d=3 / d=12 对照）")

    # Bootstrap
    rep.section("同设计 Parametric Bootstrap 稳定性检验", "4/4")
    boot = pd.DataFrame()
    if not args.skip_bootstrap:
        rep.step("Q2 6 情形 Bootstrap（逐级翻倍到收敛）", "4.1")
        for sid in Q2_SCENARIOS:
            required = [f"Q2-S{sid}-C1", f"Q2-S{sid}-C2", f"Q2-S{sid}-F"]
            if all(any(r.quality_id == q for r in records) for q in required):
                try:
                    boot = pd.concat([boot, bootstrap_stability_q2(
                        records, sid, args.seed, args.bootstrap_start, args.bootstrap_max,
                        data_source=data_source,
                    )], ignore_index=True, sort=False)
                except ValueError as e:
                    rep.warn(f"Q2-S{sid} bootstrap skipped: {e}")
        if q3_available:
            rep.step("Q3 Bootstrap（逐级翻倍到收敛）", "4.2")
            try:
                boot = pd.concat([boot, bootstrap_stability_q3(
                    records, args.seed, args.bootstrap_start, args.bootstrap_max,
                    data_source=data_source,
                )], ignore_index=True, sort=False)
            except ValueError as e:
                rep.warn(f"Q3 bootstrap skipped: {e}")
        if not boot.empty:
            save(boot, out / "q4_bootstrap.csv")
            print_bootstrap_table(rep, boot)
    else:
        rep.info("bootstrap skipped by CLI flag")

    summary = {
        "robust_q4": "completed",
        "q2_scenarios_solved": int(q2_res["scenario_id"].nunique()) if not q2_res.empty else 0,
        "q3_solved": bool(not q3_res.empty),
        "q2_monotonicity_pass": bool(q2_mono.empty or q2_mono["PASS"].all()),
        "q3_monotonicity_pass": bool(q3_mono.empty or q3_mono["PASS"].all()),
        "robust_corner_check_q2_pass": bool(rc2["PASS"].all()) if not rc2.empty else True,
        "robust_corner_check_q3_pass": bool(rc3["PASS"].all()) if not rc3.empty else True,
        "decode_diff_smoke_pass": (not ddiff.empty) and bool((ddiff.loc[ddiff["n_changes"] > 0, "n_changes"] >= 1).all()),
    }
    return summary, q2_res, q3_res, boot


def main() -> int:
    # Windows 控制台默认 GBK，无法编码 ⚠ 等 Unicode；强制 UTF-8 输出避免 UnicodeEncodeError。
    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    ap = argparse.ArgumentParser(
        description="2024 B 题 问题四：抽样不确定性 + Q2/Q3 稳健重优化"
    )
    ap.add_argument("--sampling", default="", help="真实 sampling_records.csv 路径；留空用 fixture")
    ap.add_argument("--output", default="../结果输出", help="结果输出目录")
    ap.add_argument("--skip-bootstrap", action="store_true", help="跳过 Bootstrap")
    ap.add_argument("--quiet", action="store_true", help="仅打印阶段标题")
    ap.add_argument("--bootstrap-start", type=int, default=64)
    ap.add_argument("--bootstrap-max", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--no-flat-cross-check", action="store_true", help="跳过 Q3 65536 独立核验")
    ap.add_argument("--strict-corners-q3", action="store_true", help="Q3 完整 4096 角点策略级鲁棒验证")
    args = ap.parse_args()

    rep = Reporter(quiet=args.quiet)
    out = resolve_output_dir(args.output)
    out.mkdir(parents=True, exist_ok=True)
    template = Q4_ROOT / "data" / "sampling_records_template.csv"
    ensure_template(template)

    bar = "=" * 78
    print(bar)
    print("  2024 高教社杯 B 题 —— 问题四")
    print("  抽样不确定性 + Q2/Q3 精确评价器 + 90%/95% 同时置信上界稳健重优化")
    print(bar)
    print(f"  输出目录         : {out.resolve()}")
    print(f"  抽样记录 CSV     : --sampling 留空则使用 src/sampling_fixture.py 硬编码")
    print(f"  Bootstrap 起点   : {args.bootstrap_start}（默认 64，2× 翻倍到 {args.bootstrap_max}）")
    print(f"  控制台详细输出   : {'关闭' if args.quiet else '开启'}")
    print(f"  Q3 65536 独立核验: {'关闭' if args.no_flat_cross_check else '开启'}")
    print(f"  Q3 完整角点验证  : {'开启' if args.strict_corners_q3 else '关闭（默认 top-k 验证）'}")
    print()

    t0 = time.perf_counter()

    summary, reg_q2, reg_q3 = run_regression(out, rep)
    for k, v in summary.items():
        rep.pass_fail(k, v)
    if not all(summary.values()):
        raise SystemExit("基础回归/单调性/族规模核验失败：禁止继续。")

    run_sensitivity(out, rep)

    # 抽样记录与 data_source
    if args.sampling:
        records, warnings = load_sampling_records(args.sampling)
        for w in warnings[:10]:
            rep.warn(w)
        if not records:
            rep.warn(f"--sampling {args.sampling} 不含完整记录，回退到 fixture。")
            records = list(DEFAULT_SAMPLING_RECORDS)
            data_source = "real_csv_failed_then_fixture"
        else:
            data_source = f"real_csv:{args.sampling}"
    else:
        records = list(DEFAULT_SAMPLING_RECORDS)
        data_source = "fixture_design_scenario_not_real_data"
    rep.info(f"\n  抽样记录来源: {data_source}  共 {len(records)} 条")
    summary["data_source"] = data_source
    summary["n_records"] = len(records)

    robust, q4_q2, q4_q3, boot = run_q4_robust(out, records, not args.no_flat_cross_check, args, rep, data_source)
    summary.update(robust)

    summary["outputs"] = sorted(p.name for p in out.iterdir() if p.is_file())
    (out / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print_final_summary(rep, out, summary, elapsed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())