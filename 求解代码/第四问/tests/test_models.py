from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.q4_validation import (
    regression_q2,
    q2_closed_form_validation,
    regression_q3,
    q3_handcheck_validation,
    exact_cp_coverage_validation,
    q1_sequential_boundary_validation,
    sequential_cs_coverage_validation,
    family_scope_validation,
    optimal_profit_monotonicity_validation,
    phase_map_nominal_regression,
    robust_corner_check_q2,
    robust_corner_check_q3,
    decode_diff_smoke,
)
from src.policy_codec import (
    decode_q2_policy, decode_q3_policy, diff_q2_policy, diff_q3_policy,
)


def test_q2_regression():
    assert regression_q2()["PASS"].all()


def test_q2_closed():
    df = q2_closed_form_validation()
    assert df["PASS"].all()
    # 6 情形 × 8 (x1,x2,y) 组合 = 48 项；不允许手写 24
    assert len(df) == 48


def test_q3_regression():
    assert regression_q3(flat_cross_check=False)["PASS"].all()


def test_q3_handcheck():
    assert q3_handcheck_validation()["PASS"].all()


def test_cp_coverage():
    assert exact_cp_coverage_validation()["PASS"].all()


def test_q1_seq_boundary():
    assert q1_sequential_boundary_validation()["PASS"].all()


def test_q1_seq_coverage():
    assert sequential_cs_coverage_validation()["PASS"].all()


def test_family_scope():
    """Bonferroni 族规模必须按 Q2 情形 d=3 / Q3 系统 d=12 计算。"""
    df = family_scope_validation()
    assert df["PASS"].all()
    # 必须包含 Q2 d=3 / Q3 d=12 / 全族 d=30 三类
    scopes = set(df["scope"].unique())
    assert "Q2_d3" in scopes
    assert "Q3_d12" in scopes
    assert "global_family_d30_diagnostic" in scopes


def test_optimal_profit_monotonicity():
    """最优利润随 L 或 pF 单调非增。"""
    df = optimal_profit_monotonicity_validation()
    assert df["PASS"].all()


def test_phase_map_nominal_regression():
    """Q3 nominal phase map 在 L=40 时首次切换应在 pF ≈ 0.124786 附近。"""
    df = phase_map_nominal_regression()
    assert df["PASS"].all()


def test_robust_corner_q2():
    """Q2 d=3 只有 8 角点，全部检查 top-k 策略的最坏点。"""
    df = robust_corner_check_q2()
    assert df["PASS"].all()


def test_robust_corner_q3():
    """Q3 d=12 共 4096 角点，检查 top-k 策略是否真的以 upper corner 为最坏点。"""
    df = robust_corner_check_q3()
    assert df["PASS"].all()


def test_policy_decode_q2():
    bits = (1, 0, 0, 1, 1, 1)
    d = decode_q2_policy(bits)
    assert d == {"x1": 1, "x2": 0, "y": 0, "z": 1, "r1": 1, "r2": 1}
    diffs = diff_q2_policy("100111", "110111")
    assert len(diffs) == 1
    assert diffs[0]["name"] == "x2"


def test_policy_decode_q3():
    bits = (1,)*16
    bits_list = list(bits)
    bits_list[14] = 0  # y_F = 0
    d = decode_q3_policy(bits_list)
    assert d["y_F"] == 0
    diffs = diff_q3_policy("1111111111111111", "1111111111111101")
    assert len(diffs) == 1
    assert diffs[0]["name"] == "y_F"


def test_decode_diff_smoke():
    df = decode_diff_smoke()
    # n_changes=0 仅当 nominal 与 robust 策略相同时允许；其他情况必须有 diff
    diff_rows = df[df["n_changes"] > 0]
    no_diff_rows = df[df["n_changes"] == 0]
    assert (no_diff_rows["from_code"] == no_diff_rows["to_code"]).all()
    assert len(diff_rows) >= 1  # 至少要有一个 nominal→robust 切换