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

# 让 src 包无论从哪个目录运行都能被导入。
# 模型求解/main.py 的同级父目录 = Q4 根，里面有 src/ 包。
THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent
Q4_ROOT = THIS_DIR.parent
if str(Q4_ROOT) not in sys.path:
    sys.path.insert(0, str(Q4_ROOT))

from src.config import Q2_SCENARIOS, Q3_NODES  # noqa: E402
from src.q4_uncertainty import SampleRecord, load_sampling_records  # noqa: E402
from src.q4_validation import (  # noqa: E402
    audit_q2_monotonicity,
    audit_q3_monotonicity,
    exact_cp_coverage_validation,
    q1_sequential_boundary_validation,
    q2_closed_form_validation,
    q3_handcheck_validation,
    regression_q2,
    regression_q3,
    sequential_cs_coverage_validation,
)
from src.q4_sensitivity import (  # noqa: E402
    q2_nominal_sensitivity,
    q3_nominal_sensitivity,
    q3_phase_map,
)
from src.q4_robust import (  # noqa: E402
    bootstrap_stability_q2,
    bootstrap_stability_q3,
    information_sufficiency_q2,
    information_sufficiency_q3,
    solve_q4_q2,
    solve_q4_q3,
)
from src.sampling_fixture import DEFAULT_SAMPLING_RECORDS  # noqa: E402


# ============================================================
# 控制台输出
# ============================================================

class Reporter:
    """统一控制台输出：阶段标题 / 表格 / 摘要。

    --quiet 模式下只输出阶段标题；默认输出表格与数值。
    所有表格列宽自适应。
    """

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
        # 计算列宽：表头 + 数据 + 数值字段格式化
        sample = [r[:] for r in rows]
        formatted: list[list[str]] = []
        for row in sample:
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
        # 表头分隔线
        sep = "  " + "-+-".join("-" * w for w in widths)
        for i, row in enumerate(all_rows):
            print(self._fmt_row(row, widths))
            if i == 0:
                print(sep)


# ============================================================
# 工具函数
# ============================================================

def ensure_template(path: Path) -> None:
    """生成 CSV 模板，方便竞赛现场填写真实抽样记录后用 --sampling 覆盖。"""
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
                "n": "",
                "k": "",
                "input_certified": True,
                "batch_id": "",
                "seq_reference_p0": "",
                "reported_rate": rate,
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
    """统一处理默认输出目录：相对路径以脚本所在目录（模型求解）为基准。"""
    p = Path(arg)
    if not p.is_absolute():
        p = THIS_DIR / p
    return p


def print_q2_nominal_table(rep: Reporter, reg: pd.DataFrame) -> None:
    """Q2 六情形名义回归表（policy、利润、是否与手册一致）。"""
    rep.table(
        headers=["情形", "策略4位", "策略6位", "利润/元", "期望利润", "误差", "结果"],
        rows=[
            [
                f"S{int(r.scenario_id)}",
                r.policy4,
                r.policy6,
                float(r.profit),
                float(r.expected_profit),
                float(r.abs_error),
                "PASS" if r.PASS else "FAIL",
            ]
            for _, r in reg.iterrows()
        ],
    )


def print_q3_nominal_table(rep: Reporter, reg: pd.DataFrame) -> None:
    """Q3 名义回归：DP vs 平坦枚举核验。"""
    rep.table(
        headers=["求解器", "16位策略", "利润/元", "成本/元", "期望利润", "误差", "结果"],
        rows=[
            [
                r.solver,
                r.policy16,
                float(r.profit),
                float(r.cost),
                float(r.expected_profit),
                float(r.profit_abs_error),
                "PASS" if r.PASS else "FAIL",
            ]
            for _, r in reg.iterrows()
        ],
    )


def print_q4_q2_policy_table(rep: Reporter, q4_q2: pd.DataFrame) -> None:
    """Q4 稳健 Q2 策略：每情形 nominal/robust90/robust95。"""
    rows = []
    for sid in sorted(q4_q2["scenario_id"].unique()):
        sub = q4_q2[q4_q2["scenario_id"] == sid]
        for risk in ("nominal", "robust90", "robust95"):
            row = sub[sub["risk_level"] == risk]
            if row.empty:
                continue
            r = row.iloc[0]
            rows.append([
                f"S{int(sid)}",
                risk,
                r.policy4,
                r.policy6,
                float(r.expected_cost),
                float(r.expected_profit),
                float(r.gap),
                r.robust_method,
            ])
    rep.table(
        headers=["情形", "风险水平", "策略4位", "策略6位", "成本/元", "利润/元", "Δ", "方法"],
        rows=rows,
    )


def print_q4_q3_policy_table(rep: Reporter, q4_q3: pd.DataFrame) -> None:
    """Q4 稳健 Q3 策略：nominal/robust90/robust95 + 65536 核验。"""
    rows = []
    for _, r in q4_q3.iterrows():
        rows.append([
            r.risk_level,
            r.policy16,
            float(r.expected_cost),
            float(r.expected_profit),
            float(r.gap),
            r.robust_method,
            bool(r.crosscheck_PASS) if not pd.isna(r.get("crosscheck_PASS", np.nan)) else "-",
        ])
    rep.table(
        headers=["风险水平", "16位策略", "成本/元", "利润/元", "Δ", "方法", "65536核验"],
        rows=rows,
    )


def print_q4_intervals(rep: Reporter, intervals: pd.DataFrame) -> None:
    """抽样区间表：每 quality_id 的 p_hat 与同时 90/95 上界。"""
    if intervals.empty:
        return
    rows = []
    for _, r in intervals.iterrows():
        rows.append([
            r.quality_id,
            r.quality_type,
            r.sample_design,
            int(r.n),
            int(r.k),
            float(r.p_hat),
            float(r.U_simultaneous_90),
            float(r.U_simultaneous_95),
        ])
    rep.table(
        headers=["质量ID", "类型", "设计", "n", "k", "p̂", "U90sim", "U95sim"],
        rows=rows,
        float_fmt="{:>10.5f}",
    )


def print_bootstrap_table(rep: Reporter, boot: pd.DataFrame) -> None:
    """Bootstrap 末轮稳定率表。"""
    if boot.empty:
        return
    # 取每个 (scope, risk_level) 最后一条（B 最大）作为代表。
    last = (
        boot.sort_values("B")
        .groupby(["scope", "risk_level"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    rows = []
    for _, r in last.iterrows():
        rows.append([
            r.scope,
            r.risk_level,
            int(r.B),
            r.modal_policy,
            float(r.stability_rate),
            float(r.mc_se),
            float(r.mean_optimal_profit),
        ])
    rep.table(
        headers=["范围", "风险水平", "B", "众数策略", "Rπ", "SE", "平均利润/元"],
        rows=rows,
        float_fmt="{:>10.4f}",
    )


def print_sensitivity_summary(rep: Reporter, sens2: pd.DataFrame, sens3: pd.DataFrame) -> None:
    """灵敏度扫描：参数 → 出现过的策略集合 + 利润范围。"""
    if not sens2.empty:
        print("\n  Q2 名义单参数灵敏度（基于 SCENARIOS[1]，区间 = 题目实际跨度）：")
        rows = []
        for name, g in sens2.groupby("parameter", sort=False):
            profit_min = float(g["profit"].min())
            profit_max = float(g["profit"].max())
            codes = sorted(g["policy4"].unique())
            switch_count = int(g["switch_from_previous"].sum())
            rows.append([
                name,
                f"{profit_min:.4f}",
                f"{profit_max:.4f}",
                "/".join(codes),
                switch_count,
            ])
        rep.table(
            headers=["参数", "利润下限", "利润上限", "出现过的策略", "切换次数"],
            rows=rows,
        )

    if not sens3.empty:
        print("\n  Q3 名义单参数灵敏度（pF / 调换损失）：")
        for name, g in sens3.groupby("parameter", sort=False):
            profit_min = float(g["profit"].min())
            profit_max = float(g["profit"].max())
            codes = sorted(g["policy16"].unique())
            print(f"    {name:<18} 利润 [{profit_min:.4f}, {profit_max:.4f}]  出现策略数={len(codes)}")


def print_phase_map_summary(rep: Reporter, phase: pd.DataFrame, title: str) -> None:
    """二维相图：每个 pF 子带的最佳策略集合。"""
    if phase.empty:
        return
    print(f"\n  {title}：")
    rows = []
    for pf, g in phase.groupby("pF", sort=True):
        codes = sorted(g["policy16"].unique())
        profit_min = float(g["profit"].min())
        profit_max = float(g["profit"].max())
        rows.append([
            f"{pf:.4f}",
            f"{profit_min:.4f}",
            f"{profit_max:.4f}",
            "/".join(codes[:6]) + ("..." if len(codes) > 6 else ""),
            len(codes),
        ])
    rep.table(
        headers=["pF", "最低利润", "最高利润", "出现策略(前6)", "策略数"],
        rows=rows,
    )


def print_final_summary(
    rep: Reporter,
    out: Path,
    summary: dict,
    elapsed: float,
    reg_q2: pd.DataFrame,
    reg_q3: pd.DataFrame,
    q4_q2: pd.DataFrame,
    q4_q3: pd.DataFrame,
    boot: pd.DataFrame,
) -> None:
    rep.section("运行总结", "FIN")
    print(f"  总耗时           : {elapsed:.2f} s")
    print(f"  输出目录         : {out.resolve()}")
    print(f"  抽样记录数       : {summary.get('n_records', 0)}")
    print(f"  抽样来源         : {summary.get('sampling_source', '-')}")
    print()

    # 关键指标
    print("  ── 关键指标 ──")
    print(f"  Q2 名义 6 情形全部与手册一致  : {'是' if summary.get('q2_regression_pass') else '否'}")
    print(f"  Q2 闭式退化全部 < 1e-10      : {'是' if summary.get('q2_closed_form_pass') else '否'}")
    print(f"  Q3 名义与 65536 核验一致     : {'是' if summary.get('q3_regression_pass') else '否'}")
    print(f"  Q3 成本分解手算全部 < 1e-10  : {'是' if summary.get('q3_handcheck_pass') else '否'}")
    print(f"  CP 精确覆盖率 ≥ 名义水平     : {'是' if summary.get('cp_coverage_pass') else '否'}")
    print(f"  Q1 序贯接收/拒收边界正确     : {'是' if summary.get('q1_sequential_boundary_pass') else '否'}")
    print(f"  Q1 序贯 CS MC 覆盖率合格     : {'是' if summary.get('q1_sequential_coverage_pass') else '否'}")
    print(f"  Q2 稳健单调性审计            : {'PASS' if summary.get('q2_monotonicity_pass') else 'FAIL'}")
    print(f"  Q3 稳健单调性审计            : {'PASS' if summary.get('q3_monotonicity_pass') else 'FAIL'}")
    print()

    # 简短摘要
    print("  ── Q2/Q3 名义最优 ──")
    if not reg_q2.empty:
        for _, r in reg_q2.iterrows():
            print(
                f"    S{int(r.scenario_id):<2} → {r.policy4} (6位 {r.policy6})  "
                f"利润 {float(r.profit):.4f} 元/订单"
            )
    if not reg_q3.empty:
        r = reg_q3.iloc[0]
        print(
            f"    Q3 → {r.policy16}  利润 {float(r.profit):.4f} 元/订单  "
            f"成本 {float(r.cost):.4f} 元/订单"
        )
    print()

    print("  ── Q4 稳健重优化 ──")
    if not q4_q2.empty:
        switch_count = int(
            (q4_q2.groupby("scenario_id")["policy4"].nunique() > 1).sum()
        )
        print(f"    Q2: 6 个情形中 {switch_count} 个在 nominal→95% 区间发生策略切换")
    if not q4_q3.empty:
        codes_q3 = sorted(q4_q3["policy16"].unique())
        print(f"    Q3: 出现过的策略集 = {codes_q3}")
    if not boot.empty:
        last = (
            boot.sort_values("B")
            .groupby(["scope", "risk_level"], as_index=False)
            .tail(1)
        )
        high_risk = last[last["risk_level"] == "robust95"]
        unstable = high_risk[high_risk["stability_rate"] < 0.7]
        print(
            f"    Bootstrap: 95% 稳健层级中稳定率 < 0.7 的范围有 "
            f"{len(unstable)} 个（提示该范围需要更多抽样）"
        )
    print()

    print("  ── 输出文件 ──")
    for name in summary.get("outputs", []):
        print(f"    {name}")
    print()


# ============================================================
# 主流程
# ============================================================

def run_regression(out: Path, rep: Reporter) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """三层确定性回归 + 统计覆盖率检验；这些不依赖真实抽样记录。"""
    rep.section("Q2/Q3 名义回归 + 统计覆盖率检验", "1/4")

    rep.step("Q2 六情形名义回归（应与手册参考值严格一致）", "1.1")
    q2r = regression_q2(); save(q2r, out / "regression_q2.csv")
    rep.info(f"6 情形全部与手册一致: {bool(q2r['PASS'].all())}；最大绝对误差 {float(q2r['abs_error'].max()):.2e}")
    print_q2_nominal_table(rep, q2r)

    rep.step("Q2 Bellman vs 几何分布闭式退化（z=0 必须一致）", "1.2")
    q2c = q2_closed_form_validation(); save(q2c, out / "validation_q2_closed_form.csv")
    rep.info(f"退化全部 < 1e-10: {bool(q2c['PASS'].all())}；最大相对误差 {float(q2c['abs_error'].max()):.2e}")

    rep.step("Q3 名义回归（分层 DP vs 65536 平坦枚举）", "1.3")
    q3r = regression_q3(flat_cross_check=True); save(q3r, out / "regression_q3.csv")
    rep.info(f"两种求解器一致: {bool(q3r['PASS'].all())}；最大利润绝对误差 {float(q3r['profit_abs_error'].max()):.2e}")
    print_q3_nominal_table(rep, q3r)

    rep.step("Q3 最优策略成本分解手算核验", "1.4")
    q3h = q3_handcheck_validation(); save(q3h, out / "validation_q3_handcheck.csv")
    rep.info(f"7 项成本分解全部 < 1e-10: {bool(q3h['PASS'].all())}")

    rep.step("固定样本 Clopper-Pearson 精确覆盖率（应 ≥ 名义水平）", "1.5")
    cov = exact_cp_coverage_validation(); save(cov, out / "validation_cp_coverage.csv")
    rep.info(f"全部覆盖率达到名义水平: {bool(cov['PASS'].all())}；最保守格点余量 {float((cov['coverage'] - cov['confidence']).min()):.4f}")

    rep.step("Q1 序贯接收/拒收停止边界核验", "1.6")
    seqb = q1_sequential_boundary_validation(); save(seqb, out / "validation_q1_sequential_boundaries.csv")
    for _, r in seqb.iterrows():
        rep.pass_fail(f"{r['check']}", bool(r["PASS"]))

    rep.step("Q1 序贯 confidence-sequence Monte Carlo 覆盖率", "1.7")
    seqcov = sequential_cs_coverage_validation(); save(seqcov, out / "validation_q1_sequential_coverage.csv")
    rep.info(f"所有 p_true × confidence 组合覆盖率 ≥ 名义水平 - 2SE: {bool(seqcov['PASS'].all())}")
    for _, r in seqcov.iterrows():
        marker = "PASS" if r["PASS"] else "FAIL"
        print(
            f"    [{marker}] p={float(r['p_true']):.2f}, γ={float(r['confidence']):.2f}: "
            f"coverage={float(r['coverage']):.4f} ± {float(r['mc_se']):.4f}, "
            f"censor_rate={float(r['censor_rate']):.2%}"
        )

    return {
        "q2_regression_pass": bool(q2r["PASS"].all()),
        "q2_closed_form_pass": bool(q2c["PASS"].all()),
        "q3_regression_pass": bool(q3r["PASS"].all()),
        "q3_handcheck_pass": bool(q3h["PASS"].all()),
        "cp_coverage_pass": bool(cov["PASS"].all()),
        "q1_sequential_boundary_pass": bool(seqb["PASS"].all()),
        "q1_sequential_coverage_pass": bool(seqcov["PASS"].all()),
    }, q2r, q3r


def run_sensitivity(out: Path, rep: Reporter) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """名义灵敏度扫描 + Q3 二维相图（不依赖抽样记录）。"""
    rep.step("Q2 名义灵敏度扫描（题目实际参数区间）", "1.8")
    sens2 = q2_nominal_sensitivity(); save(sens2, out / "sensitivity_q2_nominal.csv")

    rep.step("Q3 名义灵敏度扫描（pF、调换损失）", "1.9")
    sens3 = q3_nominal_sensitivity(); save(sens3, out / "sensitivity_q3_nominal.csv")

    rep.step("Q3 名义二维相图（pF × 调换损失）", "1.10")
    phase = q3_phase_map(); save(phase, out / "q4_phase_map_nominal.csv")
    print_sensitivity_summary(rep, sens2, sens3)
    return sens2, sens3, phase


def run_q4_robust(
    out: Path,
    records,
    full_flat: bool,
    args,
    rep: Reporter,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """基于真实抽样记录的稳健重优化 + 单调性审计 + 信息充分性 + Bootstrap。"""
    rep.section("Q4 稳健重优化（90%/95% 同时置信上界）", "2/4")

    # 1) 抽样区间表
    rep.step("抽样区间表（componentwise 与 simultaneous）", "2.1")
    from src.q4_uncertainty import build_interval_table
    intervals = build_interval_table(list(records))
    print_q4_intervals(rep, intervals)

    # 2) Q2 稳健
    rep.step("Q2 6 情形 × (nominal / 90% / 95%) 稳健重优化", "2.2")
    q2_int, q2_res, q2_top, q2_mono = solve_q4_q2(records, topk=10)
    if not q2_int.empty: save(q2_int, out / "q4_q2_intervals.csv")
    if not q2_res.empty: save(q2_res, out / "q4_q2_policy.csv")
    if not q2_top.empty: save(q2_top, out / "q4_q2_topk.csv")
    if not q2_mono.empty: save(q2_mono, out / "validation_q2_monotonicity.csv")
    if not q2_res.empty:
        rep.info(f"已求解 {q2_res['scenario_id'].nunique()} 个情形；策略切换 {int((q2_res.groupby('scenario_id')['policy4'].nunique() > 1).sum())} 个")
        print_q4_q2_policy_table(rep, q2_res)

    # 3) Q3 稳健
    q3_available = all(
        any(r.quality_id == qid for r in records)
        for qid in [*[f"Q3-C{i}" for i in range(1, 9)], "Q3-H1", "Q3-H2", "Q3-H3", "Q3-F"]
    )
    q3_res = pd.DataFrame()
    q3_int = pd.DataFrame()
    q3_top = pd.DataFrame()
    q3_mono = pd.DataFrame()
    if q3_available:
        rep.step("Q3 (nominal / 90% / 95%) 稳健重优化", "2.3")
        q3_int, q3_res, q3_top, q3_mono = solve_q4_q3(
            records, topk=10, flat_cross_check=full_flat
        )
        save(q3_int, out / "q4_q3_intervals.csv")
        save(q3_res, out / "q4_q3_policy.csv")
        save(q3_top, out / "q4_q3_topk.csv")
        save(q3_mono, out / "validation_q3_monotonicity.csv")
        rep.info(f"Q3 65536 独立平坦核验全部通过: {(q3_res['crosscheck_PASS'] == True).all() if 'crosscheck_PASS' in q3_res.columns else '-'}")
        print_q4_q3_policy_table(rep, q3_res)

        rep.step("Q3 单调性审计（zero/upper 双锚点 × 12 参数）", "2.4")
        rep.info(f"全部 PASS: {bool(q3_mono['PASS'].all())}；共 {len(q3_mono)} 项")

        rep.step("信息充分性曲线（n×{1,2,4,8,16} 设计投影）", "2.5")
        info_parts = []
        i2 = information_sufficiency_q2(records)
        if not i2.empty: info_parts.append(i2)
        i3 = information_sufficiency_q3(records)
        if not i3.empty: info_parts.append(i3)
        if info_parts:
            info_df = pd.concat(info_parts, ignore_index=True, sort=False)
            save(info_df, out / "q4_information_curve.csv")
            # 摘要：n×16 vs n×1 的区间宽度比
            if "mean_upper_minus_phat" in info_df.columns:
                narrow = info_df[info_df["sample_multiplier"] == 16]["mean_upper_minus_phat"].mean()
                wide = info_df[info_df["sample_multiplier"] == 1]["mean_upper_minus_phat"].mean()
                if pd.notna(wide) and wide > 0:
                    rep.info(f"  设计样本 ×16 相比 ×1，区间宽度平均收缩 {(1 - narrow/wide) * 100:.1f}%")

        rep.step("Q4 专属二维相图（pF: p̂ → U95sim × 调换损失 6–40）", "2.6")
        frow = q3_int[q3_int["quality_id"] == "Q3-F"].iloc[0]
        pf_grid = np.linspace(float(frow["p_hat"]), float(frow["U_simultaneous_95"]), 31)
        base95 = {
            qid.replace("Q3-", ""): float(row["U_simultaneous_95"])
            for qid, row in q3_int.set_index("quality_id").iterrows()
        }
        phase_q4 = q3_phase_map(base_quality=base95, pf_range=pf_grid, loss_range=np.linspace(6.0, 40.0, 35))
        phase_q4["quality_axis"] = "Q3-F: p_hat -> U95 simultaneous"
        save(phase_q4, out / "q4_phase_map.csv")
        print_phase_map_summary(rep, phase_q4, "Q4 二维相图（pF × 调换损失）")

    # 合并表
    intervals_combined = [x for x in [q2_int, q3_int] if not x.empty]
    if intervals_combined:
        save(pd.concat(intervals_combined, ignore_index=True), out / "quality_intervals.csv")
    tops_combined = [x for x in [q2_top, q3_top] if not x.empty]
    if tops_combined:
        save(pd.concat(tops_combined, ignore_index=True, sort=False), out / "q4_topk.csv")

    # 4) Bootstrap
    rep.section("同设计 Parametric Bootstrap 稳定性检验", "3/4")
    boot = pd.DataFrame()
    if not args.skip_bootstrap:
        rep.step("Q2 6 情形 Bootstrap（逐级翻倍到收敛）", "3.1")
        for sid in Q2_SCENARIOS:
            required = [f"Q2-S{sid}-C1", f"Q2-S{sid}-C2", f"Q2-S{sid}-F"]
            if all(any(r.quality_id == q for r in records) for q in required):
                try:
                    boot = pd.concat(
                        [boot, bootstrap_stability_q2(
                            records, sid, args.seed,
                            args.bootstrap_start, args.bootstrap_max,
                        )],
                        ignore_index=True, sort=False,
                    )
                except ValueError as e:
                    rep.warn(f"Q2-S{sid} bootstrap skipped: {e}")
        if q3_available:
            rep.step("Q3 Bootstrap（逐级翻倍到收敛）", "3.2")
            try:
                boot = pd.concat(
                    [boot, bootstrap_stability_q3(
                        records, args.seed,
                        args.bootstrap_start, args.bootstrap_max,
                    )],
                    ignore_index=True, sort=False,
                )
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
    }
    return summary, q2_res, q3_res, boot


def main() -> int:
    ap = argparse.ArgumentParser(
        description="2024 B 题 问题四：抽样不确定性 + Q2/Q3 稳健重优化"
    )
    ap.add_argument(
        "--sampling",
        default="",
        help=(
            "真实 sampling_records.csv 路径；留空则使用 src/sampling_fixture.py 中的"
            "硬编码设计情景（内部设计情景标注，不是题目真实抽样记录）。"
        ),
    )
    ap.add_argument(
        "--output",
        default="../结果输出",
        help="结果输出目录，默认相对脚本所在目录的 ../结果输出/",
    )
    ap.add_argument("--skip-bootstrap", action="store_true", help="跳过 Bootstrap 数值检验")
    ap.add_argument("--quiet", action="store_true", help="仅打印阶段标题，不打印详细表格")
    ap.add_argument("--bootstrap-start", type=int, default=64, help="Bootstrap 起始批次")
    ap.add_argument("--bootstrap-max", type=int, default=512, help="Bootstrap 数值安全上限")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--no-flat-cross-check", action="store_true", help="跳过 Q3 65536 独立平坦核验")
    args = ap.parse_args()

    rep = Reporter(quiet=args.quiet)
    out = resolve_output_dir(args.output)
    out.mkdir(parents=True, exist_ok=True)
    template = Q4_ROOT / "data" / "sampling_records_template.csv"
    ensure_template(template)

    # Banner
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
    print()

    t0 = time.perf_counter()

    summary, reg_q2, reg_q3 = run_regression(out, rep)
    rep.info("\n  回归检验全部通过：")
    for k, v in summary.items():
        rep.pass_fail(k, v)
    if not all(summary.values()):
        raise SystemExit(
            "基础回归检验失败：请先修复 Q2/Q3 评价器，禁止继续做 Q4 稳健重优化。"
        )

    run_sensitivity(out, rep)

    # 决定抽样记录
    if args.sampling:
        records, warnings = load_sampling_records(args.sampling)
        for w in warnings[:10]:
            rep.warn(w)
        if not records:
            rep.warn(f"--sampling {args.sampling} 不含完整记录，回退到硬编码 fixture。")
            records = list(DEFAULT_SAMPLING_RECORDS)
            sample_source = "fixture_fallback"
        else:
            sample_source = f"csv:{args.sampling}"
    else:
        records = list(DEFAULT_SAMPLING_RECORDS)
        sample_source = "hardcoded_fixture"
    rep.info(f"\n  抽样记录来源: {sample_source}  共 {len(records)} 条")
    summary["sampling_source"] = sample_source
    summary["n_records"] = len(records)

    robust, q4_q2, q4_q3, boot = run_q4_robust(out, records, not args.no_flat_cross_check, args, rep)
    summary.update(robust)

    summary["outputs"] = sorted(p.name for p in out.iterdir() if p.is_file())
    (out / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - t0
    print_final_summary(rep, out, summary, elapsed, reg_q2, reg_q3, q4_q2, q4_q3, boot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())