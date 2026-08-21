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
      即 求解代码/第四问/结果输出/。

可选参数：
    python main.py --sampling data/sampling_records.csv   # 用真实 CSV 覆盖默认 fixture
    python main.py --output ../结果输出                   # 自定义输出目录
    python main.py --skip-bootstrap                       # 跳过 Bootstrap 数值检验
    python main.py --no-flat-cross-check                  # 跳过 Q3 65536 独立平坦核验

依赖：Python >= 3.10, numpy, pandas, scipy
"""

from __future__ import annotations

import argparse
import json
import sys
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


# ============================================================
# 主流程
# ============================================================

def run_regression(out: Path, full_flat: bool = True) -> dict:
    """三层确定性回归 + 统计覆盖率检验；这些不依赖真实抽样记录。"""
    q2r = regression_q2(); save(q2r, out / "regression_q2.csv")
    q2c = q2_closed_form_validation(); save(q2c, out / "validation_q2_closed_form.csv")
    q3r = regression_q3(flat_cross_check=full_flat); save(q3r, out / "regression_q3.csv")
    q3h = q3_handcheck_validation(); save(q3h, out / "validation_q3_handcheck.csv")
    cov = exact_cp_coverage_validation(); save(cov, out / "validation_cp_coverage.csv")
    seqb = q1_sequential_boundary_validation(); save(seqb, out / "validation_q1_sequential_boundaries.csv")
    seqcov = sequential_cs_coverage_validation(); save(seqcov, out / "validation_q1_sequential_coverage.csv")
    sens2 = q2_nominal_sensitivity(); save(sens2, out / "sensitivity_q2_nominal.csv")
    sens3 = q3_nominal_sensitivity(); save(sens3, out / "sensitivity_q3_nominal.csv")
    phase = q3_phase_map(); save(phase, out / "q4_phase_map_nominal.csv")
    return {
        "q2_regression_pass": bool(q2r["PASS"].all()),
        "q2_closed_form_pass": bool(q2c["PASS"].all()),
        "q3_regression_pass": bool(q3r["PASS"].all()),
        "q3_handcheck_pass": bool(q3h["PASS"].all()),
        "cp_coverage_pass": bool(cov["PASS"].all()),
        "q1_sequential_boundary_pass": bool(seqb["PASS"].all()),
        "q1_sequential_coverage_pass": bool(seqcov["PASS"].all()),
    }


def run_q4_robust(out: Path, records, full_flat: bool, args) -> dict:
    """基于真实抽样记录的稳健重优化 + 单调性审计 + 信息充分性 + Bootstrap。"""
    summary: dict = {}

    print("[2/4] Running Q4 robust re-optimization from sampling records...")
    q2_int, q2_res, q2_top, q2_mono = solve_q4_q2(records, topk=10)
    if not q2_int.empty: save(q2_int, out / "q4_q2_intervals.csv")
    if not q2_res.empty: save(q2_res, out / "q4_q2_policy.csv")
    if not q2_top.empty: save(q2_top, out / "q4_q2_topk.csv")
    if not q2_mono.empty: save(q2_mono, out / "validation_q2_monotonicity.csv")

    q3_available = all(
        any(r.quality_id == qid for r in records)
        for qid in [*[f"Q3-C{i}" for i in range(1, 9)], "Q3-H1", "Q3-H2", "Q3-H3", "Q3-F"]
    )
    q3_res = pd.DataFrame()
    q3_int = pd.DataFrame()
    q3_top = pd.DataFrame()
    q3_mono = pd.DataFrame()
    if q3_available:
        q3_int, q3_res, q3_top, q3_mono = solve_q4_q3(
            records, topk=10, flat_cross_check=full_flat
        )
        save(q3_int, out / "q4_q3_intervals.csv")
        save(q3_res, out / "q4_q3_policy.csv")
        save(q3_top, out / "q4_q3_topk.csv")
        save(q3_mono, out / "validation_q3_monotonicity.csv")

        info_parts = []
        i2 = information_sufficiency_q2(records)
        if not i2.empty:
            info_parts.append(i2)
        i3 = information_sufficiency_q3(records)
        if not i3.empty:
            info_parts.append(i3)
        if info_parts:
            save(pd.concat(info_parts, ignore_index=True, sort=False), out / "q4_information_curve.csv")

        # Q4 专属二维相图：F 风险从点估计走到 95% 系统同时上界；L 沿题目已有 6–40 元跨度。
        frow = q3_int[q3_int["quality_id"] == "Q3-F"].iloc[0]
        pf_grid = np.linspace(float(frow["p_hat"]), float(frow["U_simultaneous_95"]), 31)
        base95 = {
            qid.replace("Q3-", ""): float(row["U_simultaneous_95"])
            for qid, row in q3_int.set_index("quality_id").iterrows()
        }
        phase_q4 = q3_phase_map(base_quality=base95, pf_range=pf_grid, loss_range=np.linspace(6.0, 40.0, 35))
        phase_q4["quality_axis"] = "Q3-F: p_hat -> U95 simultaneous"
        save(phase_q4, out / "q4_phase_map.csv")

    # 合并区间表与 Top-k，方便 MATLAB 直接读。
    intervals = [x for x in [q2_int, q3_int] if not x.empty]
    if intervals:
        save(pd.concat(intervals, ignore_index=True), out / "quality_intervals.csv")
    tops = [x for x in [q2_top, q3_top] if not x.empty]
    if tops:
        save(pd.concat(tops, ignore_index=True, sort=False), out / "q4_topk.csv")

    print("[3/4] Running same-design bootstrap stability validation...")
    boot = []
    if not args.skip_bootstrap:
        for sid in Q2_SCENARIOS:
            required = [f"Q2-S{sid}-C1", f"Q2-S{sid}-C2", f"Q2-S{sid}-F"]
            if all(any(r.quality_id == q for r in records) for q in required):
                try:
                    boot.append(
                        bootstrap_stability_q2(
                            records, sid, args.seed,
                            args.bootstrap_start, args.bootstrap_max,
                        )
                    )
                except ValueError as e:
                    boot.append(pd.DataFrame([{
                        "scope": f"Q2-S{sid}",
                        "status": f"bootstrap skipped: {e}",
                    }]))
        if q3_available:
            try:
                boot.append(
                    bootstrap_stability_q3(
                        records, args.seed,
                        args.bootstrap_start, args.bootstrap_max,
                    )
                )
            except ValueError as e:
                boot.append(pd.DataFrame([{
                    "scope": "Q3",
                    "status": f"bootstrap skipped: {e}",
                }]))
        if boot:
            save(pd.concat(boot, ignore_index=True, sort=False), out / "q4_bootstrap.csv")
    else:
        print("  bootstrap skipped by CLI flag")

    summary.update({
        "robust_q4": "completed",
        "q2_scenarios_solved": int(q2_res["scenario_id"].nunique()) if not q2_res.empty else 0,
        "q3_solved": bool(not q3_res.empty),
        "q2_monotonicity_pass": bool(q2_mono.empty or q2_mono["PASS"].all()),
        "q3_monotonicity_pass": bool(q3_mono.empty or q3_mono["PASS"].all()),
    })
    return summary


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
    ap.add_argument("--bootstrap-start", type=int, default=64, help="Bootstrap 起始批次")
    ap.add_argument("--bootstrap-max", type=int, default=512, help="Bootstrap 数值安全上限")
    ap.add_argument("--seed", type=int, default=20260820)
    ap.add_argument("--no-flat-cross-check", action="store_true", help="跳过 Q3 65536 独立平坦核验")
    args = ap.parse_args()

    out = resolve_output_dir(args.output)
    out.mkdir(parents=True, exist_ok=True)
    template = Q4_ROOT / "data" / "sampling_records_template.csv"
    ensure_template(template)

    print(f"[0/4] 输出目录: {out.resolve()}")
    print("[1/4] Running deterministic regression / validation tests...")
    summary = run_regression(out, full_flat=not args.no_flat_cross_check)
    for k, v in summary.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    if not all(summary.values()):
        raise SystemExit(
            "基础回归检验失败：请先修复 Q2/Q3 评价器，禁止继续做 Q4 稳健重优化。"
        )

    # 决定抽样记录：CSV 优先；缺省使用硬编码设计情景。
    if args.sampling:
        records, warnings = load_sampling_records(args.sampling)
        for w in warnings[:10]:
            print("  warning:", w)
        if not records:
            print(f"\n[2/4] --sampling {args.sampling} 不含完整记录，回退到硬编码 fixture。")
            records = list(DEFAULT_SAMPLING_RECORDS)
            sample_source = "fixture_fallback"
        else:
            sample_source = f"csv:{args.sampling}"
    else:
        records = list(DEFAULT_SAMPLING_RECORDS)
        sample_source = "hardcoded_fixture"

    print(f"  抽样记录来源: {sample_source} ({len(records)} 条)")
    summary["sampling_source"] = sample_source
    summary["n_records"] = len(records)

    robust = run_q4_robust(out, records, not args.no_flat_cross_check, args)
    summary.update(robust)

    summary["outputs"] = sorted(p.name for p in out.iterdir() if p.is_file())
    (out / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("[4/4] Finished.")
    print(f"Outputs: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())