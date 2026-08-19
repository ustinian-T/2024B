# -*- coding: utf-8 -*-
"""
2024 高教社杯全国大学生数学建模竞赛 B 题——问题一
精确二项基准模型 + 单侧混合似然比序贯检验（e-process）

严格按照《第一问建模手册——精确二项与序贯检验》的规划实现：
1) Bernoulli/Binomial 数据机制；
2) 固定样本精确二项单侧检验作为基准；
3) 单侧混合似然比 e-process 作为主模型；
4) 首次越界停止；
5) 动态概率递推进行无蒙特卡洛误差的模型检验；
6) 对真实次品率、置信水平、混合密度做灵敏度分析；
7) 不虚构 p1、批量 N、检测误差率等题目未给参数。

依赖：Python >= 3.9, numpy, pandas, scipy
运行：
    python q1_first_question_model.py
可选：
    python q1_first_question_model.py --max-n 20000 --output-dir q1_output
    python q1_first_question_model.py --sample-csv your_sampling.csv

输入抽样 CSV（可选）建议字段：
    batch_id, sample_order, result
其中 result: 次品=1，合格=0。

程序不绘图，只输出后续 MATLAB 绘图/论文制表所需 CSV 与检验报告。
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.special import betainc, betaincc, betaln
from scipy.stats import binom


# ============================================================================
# 0. 全局默认参数：全部来自题目/建模手册，不额外虚构业务参数
# ============================================================================
DEFAULT_P0 = 0.10
DEFAULT_ACCEPT_CONFIDENCE = 0.90   # 90% 信度支持 p <= p0 -> 接收
DEFAULT_REJECT_CONFIDENCE = 0.95   # 95% 信度支持 p > p0  -> 拒收
DEFAULT_MAX_N = 20000
DEFAULT_TOL = 1e-7

# 固定样本基准边界只需生成到较适合制表的范围；不是“推荐固定样本量”
DEFAULT_FIXED_MAX_N = 500

# 主结果代表性真实次品率：0.05/0.10/0.20来自整题典型质量水平，
# 0.08/0.12用于研究距离标称值±2个百分点时的可辨识性。
MAIN_P_VALUES = (0.05, 0.08, 0.10, 0.12, 0.20)

# 灵敏度：重点加密 p0 附近；这些值只用于敏感性，不视为题目给定参数。
P_SENSITIVITY_VALUES = (
    0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.095,
    0.10, 0.105, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
    0.17, 0.18, 0.19, 0.20,
)

# 手册中的关键 n，用于核对临界边界与后续论文制表
FIXED_CHECK_N = (10, 20, 22, 40, 60, 80, 100, 160, 200)
SEQUENTIAL_CHECK_N = (20, 34, 40, 50, 80, 100, 160, 200, 500)


# ============================================================================
# 1. 参数与结果数据结构
# ============================================================================
@dataclass(frozen=True)
class EProcessConfig:
    """e-process 配置。prior_a=prior_b=1 即主模型“均匀混合”。"""

    p0: float = DEFAULT_P0
    accept_confidence: float = DEFAULT_ACCEPT_CONFIDENCE
    reject_confidence: float = DEFAULT_REJECT_CONFIDENCE
    prior_a: float = 1.0
    prior_b: float = 1.0
    mixture_name: str = "uniform"

    @property
    def alpha_accept(self) -> float:
        return 1.0 - self.accept_confidence

    @property
    def alpha_reject(self) -> float:
        return 1.0 - self.reject_confidence

    @property
    def log_accept_threshold(self) -> float:
        # Ville: P(sup E >= 1/alpha) <= alpha
        return math.log(1.0 / self.alpha_accept)

    @property
    def log_reject_threshold(self) -> float:
        return math.log(1.0 / self.alpha_reject)

    def validate(self) -> None:
        if not (0.0 < self.p0 < 1.0):
            raise ValueError("p0 必须位于 (0,1) 内。")
        if not (0.0 < self.accept_confidence < 1.0):
            raise ValueError("accept_confidence 必须位于 (0,1) 内。")
        if not (0.0 < self.reject_confidence < 1.0):
            raise ValueError("reject_confidence 必须位于 (0,1) 内。")
        if self.prior_a <= 0.0 or self.prior_b <= 0.0:
            raise ValueError("Beta 混合参数 prior_a, prior_b 必须为正。")


@dataclass
class RecursionResult:
    true_p: float
    accept_prob: float
    reject_prob: float
    unresolved_prob: float
    stop_prob: float
    expected_tau_if_converged: float
    truncated_expected_n: float
    conditional_expected_n_given_stopped: float
    horizon: int
    converged: bool


# ============================================================================
# 2. 数据预处理：只做与 Bernoulli/序贯抽样有关的质量控制
# ============================================================================
def validate_sampling_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    按手册要求校验真实抽样记录：
    - result 只能为 0/1；
    - 同批次 sample_order 唯一且连续；
    - 一个输入文件允许多个 batch，但每批独立执行序贯规则；
    - result 缺失的记录不纳入当前统计（建议补测后再加入）。
    """
    required = {"batch_id", "sample_order", "result"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"抽样 CSV 缺少字段: {sorted(missing)}")

    out = df.copy()
    # 缺失结果不把它当“合格/次品”，而是移出当前有效样本
    out = out.dropna(subset=["result"]).copy()

    if out.empty:
        raise ValueError("删除 result 缺失记录后，没有可用抽样数据。")

    # 严格二元编码检查
    try:
        out["result"] = out["result"].astype(int)
        out["sample_order"] = out["sample_order"].astype(int)
    except Exception as exc:
        raise ValueError("result/sample_order 必须可转换为整数。") from exc

    invalid = sorted(set(out["result"].unique()).difference({0, 1}))
    if invalid:
        raise ValueError(f"result 只能取 0/1，发现非法值: {invalid}")

    if (out["sample_order"] <= 0).any():
        raise ValueError("sample_order 必须为正整数。")

    # 同批次同序号不可重复
    dup = out.duplicated(subset=["batch_id", "sample_order"], keep=False)
    if dup.any():
        bad = out.loc[dup, ["batch_id", "sample_order"]].head(10)
        raise ValueError(f"发现重复抽样记录：\n{bad.to_string(index=False)}")

    # 序号连续性
    pieces = []
    for batch_id, g in out.groupby("batch_id", sort=False):
        g = g.sort_values("sample_order").copy()
        orders = g["sample_order"].to_numpy()
        if orders.min() != 1:
            raise ValueError(f"批次 {batch_id!r} 的 sample_order 必须从 1 开始。")
        expected = np.arange(1, len(orders) + 1)
        if not np.array_equal(orders, expected):
            raise ValueError(
                f"批次 {batch_id!r} 的 sample_order 不连续。"
                "序贯方法必须保留真实观察顺序。"
            )
        pieces.append(g)

    return pd.concat(pieces, ignore_index=True)


# ============================================================================
# 3. 精确二项固定样本基准模型
# ============================================================================
def exact_binomial_boundaries(
    n: int,
    p0: float = DEFAULT_P0,
    accept_confidence: float = DEFAULT_ACCEPT_CONFIDENCE,
    reject_confidence: float = DEFAULT_REJECT_CONFIDENCE,
) -> Tuple[Optional[int], Optional[int]]:
    """
    返回固定样本精确二项边界 (a_n, r_n)：
      k <= a_n -> 90% 单侧意义下支持接收；
      k >= r_n -> 95% 单侧意义下支持拒收；
      中间 -> 证据不足。

    定义：
      a_n = max{k: P_{p0}(K_n <= k) <= alpha_accept}
      r_n = min{k: P_{p0}(K_n >= k) <= alpha_reject}
    """
    if n <= 0:
        raise ValueError("n 必须为正整数。")
    if not (0 < p0 < 1):
        raise ValueError("p0 必须位于 (0,1)。")

    alpha_a = 1.0 - accept_confidence
    alpha_r = 1.0 - reject_confidence

    ks = np.arange(n + 1)
    lower = binom.cdf(ks, n, p0)
    upper = binom.sf(ks - 1, n, p0)  # P(K >= k)

    a_candidates = ks[lower <= alpha_a]
    r_candidates = ks[upper <= alpha_r]

    a_n = int(a_candidates.max()) if len(a_candidates) else None
    r_n = int(r_candidates.min()) if len(r_candidates) else None
    return a_n, r_n


def fixed_boundary_table(
    max_n: int = DEFAULT_FIXED_MAX_N,
    p0: float = DEFAULT_P0,
    accept_confidence: float = DEFAULT_ACCEPT_CONFIDENCE,
    reject_confidence: float = DEFAULT_REJECT_CONFIDENCE,
) -> pd.DataFrame:
    rows = []
    for n in range(1, max_n + 1):
        a_n, r_n = exact_binomial_boundaries(
            n, p0, accept_confidence, reject_confidence
        )
        accept_tail = (
            float(binom.cdf(a_n, n, p0)) if a_n is not None else math.nan
        )
        reject_tail = (
            float(binom.sf(r_n - 1, n, p0)) if r_n is not None else math.nan
        )
        rows.append(
            {
                "n": n,
                "accept_boundary": a_n,
                "reject_boundary": r_n,
                "accept_tail_prob_at_p0": accept_tail,
                "reject_tail_prob_at_p0": reject_tail,
            }
        )
    return pd.DataFrame(rows)


# ============================================================================
# 4. 主模型：单侧混合似然比 e-process
# ============================================================================
def _log_lower_incomplete_beta(a: float, b: float, x: float) -> float:
    """log B_x(a,b)，使用 regularized betainc + betaln。"""
    reg = float(betainc(a, b, x))
    if reg <= 0.0:
        return -math.inf
    return float(betaln(a, b)) + math.log(reg)


def _log_upper_incomplete_beta(a: float, b: float, x: float) -> float:
    """log [B(a,b)-B_x(a,b)]，使用 betaincc 避免 1-betainc 的消减误差。"""
    reg_c = float(betaincc(a, b, x))
    if reg_c <= 0.0:
        return -math.inf
    return float(betaln(a, b)) + math.log(reg_c)


def log_e_minus(n: int, k: int, config: EProcessConfig) -> float:
    """
    支持 p < p0（接收方向）的对数 e-value。

    采用截断 Beta(prior_a, prior_b) 混合：
    E^-_n = [B_{p0}(k+a,n-k+b) / B_{p0}(a,b)] / L_n(p0)

    主模型 a=b=1 时即手册中的均匀混合。
    """
    config.validate()
    if not (0 <= k <= n):
        raise ValueError("必须满足 0 <= k <= n。")

    p0 = config.p0
    a0, b0 = config.prior_a, config.prior_b

    log_num = _log_lower_incomplete_beta(k + a0, n - k + b0, p0)
    log_mix_norm = _log_lower_incomplete_beta(a0, b0, p0)
    log_l0 = k * math.log(p0) + (n - k) * math.log1p(-p0)
    return log_num - log_mix_norm - log_l0


def log_e_plus(n: int, k: int, config: EProcessConfig) -> float:
    """
    支持 p > p0（拒收方向）的对数 e-value。

    E^+_n = {[B(k+a,n-k+b)-B_{p0}(k+a,n-k+b)] /
             [B(a,b)-B_{p0}(a,b)]} / L_n(p0)
    """
    config.validate()
    if not (0 <= k <= n):
        raise ValueError("必须满足 0 <= k <= n。")

    p0 = config.p0
    a0, b0 = config.prior_a, config.prior_b

    log_num = _log_upper_incomplete_beta(k + a0, n - k + b0, p0)
    log_mix_norm = _log_upper_incomplete_beta(a0, b0, p0)
    log_l0 = k * math.log(p0) + (n - k) * math.log1p(-p0)
    return log_num - log_mix_norm - log_l0


def sequential_decision(n: int, k: int, config: EProcessConfig) -> str:
    """返回 accept / reject / continue。"""
    lm = log_e_minus(n, k, config)
    lp = log_e_plus(n, k, config)

    hit_accept = lm >= config.log_accept_threshold
    hit_reject = lp >= config.log_reject_threshold

    if hit_accept and hit_reject:
        # 在本题基准参数下不会出现；保留防御性检查，避免模糊决策。
        raise RuntimeError(f"状态 (n={n}, k={k}) 同时触发接收与拒收阈值。")
    if hit_accept:
        return "accept"
    if hit_reject:
        return "reject"
    return "continue"


def _accept_boundary_binary(n: int, config: EProcessConfig) -> Optional[int]:
    """E^- 随 k 单调下降；二分求最大满足阈值的 k。"""
    thr = config.log_accept_threshold
    if log_e_minus(n, 0, config) < thr:
        return None

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if log_e_minus(n, mid, config) >= thr:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _reject_boundary_binary(n: int, config: EProcessConfig) -> Optional[int]:
    """E^+ 随 k 单调上升；二分求最小满足阈值的 k。"""
    thr = config.log_reject_threshold
    if log_e_plus(n, n, config) < thr:
        return None

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if log_e_plus(n, mid, config) >= thr:
            hi = mid
        else:
            lo = mid + 1
    return lo


def generate_sequential_boundaries(
    max_n: int,
    config: EProcessConfig,
    check_monotonicity: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 n=1..max_n 的序贯停止边界。

    返回：
      accept[n] = 最大接收 k；若该 n 无接收边界则为 -1
      reject[n] = 最小拒收 k；若该 n 无拒收边界则为 n+1
    """
    config.validate()
    accept = np.full(max_n + 1, -1, dtype=np.int32)
    reject = np.arange(max_n + 1, dtype=np.int32) + 1

    for n in range(1, max_n + 1):
        a_n = _accept_boundary_binary(n, config)
        r_n = _reject_boundary_binary(n, config)
        if a_n is not None:
            accept[n] = a_n
        if r_n is not None:
            reject[n] = r_n

        if accept[n] >= reject[n]:
            raise RuntimeError(
                f"n={n} 时接收/拒收边界重叠: "
                f"a_n={accept[n]}, r_n={reject[n]}"
            )

    if check_monotonicity:
        # 理论上 Bernoulli 族具有单调似然比性质，因此 E^- 随 k 下降、E^+ 随 k 上升。
        # 大 n 的极端尾部 betainc 可能发生机器精度饱和，故不对整个 0..n 做数值差分；
        # 改为直接核验“二分搜索返回的临界点恰好首次/最后越界”，这与决策正确性更直接。
        probe_ns = sorted(set([1, 2, 10, 34, 100, min(500, max_n), max_n]))
        for n in probe_ns:
            if n < 1:
                continue
            a_n = int(accept[n])
            r_n = int(reject[n])
            if a_n >= 0:
                if log_e_minus(n, a_n, config) < config.log_accept_threshold - 1e-10:
                    raise AssertionError(f"n={n} 的接收临界点自身未越界。")
                if a_n + 1 <= n and log_e_minus(n, a_n + 1, config) >= config.log_accept_threshold + 1e-10:
                    raise AssertionError(f"n={n} 的接收临界点不是最大越界 k。")
            if r_n <= n:
                if log_e_plus(n, r_n, config) < config.log_reject_threshold - 1e-10:
                    raise AssertionError(f"n={n} 的拒收临界点自身未越界。")
                if r_n - 1 >= 0 and log_e_plus(n, r_n - 1, config) >= config.log_reject_threshold + 1e-10:
                    raise AssertionError(f"n={n} 的拒收临界点不是最小越界 k。")

        # 对中小 n 额外做全扫描单调性检查，此时特殊函数不会落入极端饱和区。
        for n in sorted(set([2, 10, 34, 100, min(500, max_n)])):
            em = np.array([log_e_minus(n, k, config) for k in range(n + 1)])
            ep = np.array([log_e_plus(n, k, config) for k in range(n + 1)])
            em_f = np.isfinite(em[:-1]) & np.isfinite(em[1:])
            ep_f = np.isfinite(ep[:-1]) & np.isfinite(ep[1:])
            if np.any((em[1:][em_f] - em[:-1][em_f]) > 1e-10):
                raise AssertionError(f"E^- 在 n={n} 上未保持随 k 单调下降。")
            if np.any((ep[1:][ep_f] - ep[:-1][ep_f]) < -1e-10):
                raise AssertionError(f"E^+ 在 n={n} 上未保持随 k 单调上升。")

    return accept, reject


def sequential_boundary_dataframe(
    accept: np.ndarray,
    reject: np.ndarray,
) -> pd.DataFrame:
    max_n = len(accept) - 1
    n = np.arange(1, max_n + 1)
    a = accept[1:].astype(float)
    r = reject[1:].astype(float)
    a[a < 0] = np.nan
    # reject 默认值 n+1 表示无边界
    no_r = r > n
    r[no_r] = np.nan
    return pd.DataFrame(
        {
            "n": n,
            "accept_boundary": a,
            "reject_boundary": r,
            "accept_boundary_rate": a / n,
            "reject_boundary_rate": r / n,
        }
    )


# ============================================================================
# 5. 动态概率递推：精确检验主模型性能，不依赖蒙特卡洛
# ============================================================================
def exact_sequential_recursion(
    true_p: float,
    accept_boundaries: np.ndarray,
    reject_boundaries: np.ndarray,
    tol: float = DEFAULT_TOL,
) -> RecursionResult:
    """
    状态概率递推：
      f_{n+1}(k) = (1-p) f_n(k) + p f_n(k-1)
    只保留尚未越界的状态。

    该递推精确累计：接收概率、拒收概率、未停止概率、停止时间期望。
    """
    if not (0.0 <= true_p <= 1.0):
        raise ValueError("true_p 必须位于 [0,1]。")
    if len(accept_boundaries) != len(reject_boundaries):
        raise ValueError("接收与拒收边界长度不一致。")

    max_n = len(accept_boundaries) - 1
    alive = np.array([1.0], dtype=float)  # n=0, k=0
    lo_k = 0

    p_accept = 0.0
    p_reject = 0.0
    stop_time_mass = 0.0  # sum n * P(tau=n)
    last_n = 0

    for n in range(1, max_n + 1):
        last_n = n

        # 从前一时刻尚未停止的连续 k 区间传播到新时刻
        new = np.zeros(len(alive) + 1, dtype=float)
        new[:-1] += alive * (1.0 - true_p)
        new[1:] += alive * true_p

        new_lo = lo_k
        new_hi = lo_k + len(new) - 1
        a_n = int(accept_boundaries[n])
        r_n = int(reject_boundaries[n])

        # 接收质量
        if a_n >= new_lo:
            end = min(a_n, new_hi) - new_lo + 1
            if end > 0:
                mass = float(new[:end].sum())
                p_accept += mass
                stop_time_mass += n * mass

        # 拒收质量
        if r_n <= new_hi:
            start = max(r_n, new_lo) - new_lo
            if start < len(new):
                mass = float(new[start:].sum())
                p_reject += mass
                stop_time_mass += n * mass

        # 中间带继续抽样
        live_low = max(new_lo, a_n + 1)
        live_high = min(new_hi, r_n - 1)
        if live_low <= live_high:
            s = live_low - new_lo
            e = live_high - new_lo + 1
            alive = new[s:e]
            lo_k = live_low
        else:
            alive = np.empty(0, dtype=float)
            lo_k = 0

        unresolved = float(alive.sum())

        # 质量守恒检查：接收+拒收+继续≈1
        total_mass = p_accept + p_reject + unresolved
        if abs(total_mass - 1.0) > 5e-10:
            raise AssertionError(
                f"概率质量不守恒: n={n}, total={total_mass:.16f}"
            )

        if unresolved < tol:
            break

    unresolved = float(alive.sum())
    stop_prob = p_accept + p_reject
    converged = unresolved < tol

    # 若 residual 已足够小，则 stop_time_mass 可作为 E[tau] 的数值近似。
    # 若 residual 不小，则不声称已得到无限时域 E[tau]；改报 E[min(tau,N)]。
    expected_if_converged = stop_time_mass if converged else math.nan
    truncated_expected = stop_time_mass + last_n * unresolved
    conditional_expected = (
        stop_time_mass / stop_prob if stop_prob > 0.0 else math.nan
    )

    return RecursionResult(
        true_p=true_p,
        accept_prob=p_accept,
        reject_prob=p_reject,
        unresolved_prob=unresolved,
        stop_prob=stop_prob,
        expected_tau_if_converged=expected_if_converged,
        truncated_expected_n=truncated_expected,
        conditional_expected_n_given_stopped=conditional_expected,
        horizon=last_n,
        converged=converged,
    )


def recursion_results_dataframe(results: Iterable[RecursionResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(x) for x in results])


# ============================================================================
# 6. 在线应用：对真实抽样序列逐件更新并首次越界停止
# ============================================================================
def evaluate_sampling_sequence(
    results: Sequence[int],
    config: EProcessConfig,
) -> pd.DataFrame:
    rows = []
    k = 0
    stopped = False
    for n, x in enumerate(results, start=1):
        if x not in (0, 1):
            raise ValueError("抽样结果只能取 0/1。")
        k += int(x)

        lm = log_e_minus(n, k, config)
        lp = log_e_plus(n, k, config)
        decision = sequential_decision(n, k, config)

        rows.append(
            {
                "n": n,
                "result": int(x),
                "k": k,
                "p_hat": k / n,
                "log_e_minus": lm,
                "e_minus": math.exp(lm) if lm < 700 else math.inf,
                "log_e_plus": lp,
                "e_plus": math.exp(lp) if lp < 700 else math.inf,
                "decision": decision,
            }
        )
        if decision != "continue":
            stopped = True
            break

    if not stopped and rows:
        rows[-1]["decision"] = "continue"
    return pd.DataFrame(rows)


def evaluate_sampling_csv(
    csv_path: Path,
    config: EProcessConfig,
    output_dir: Path,
) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    clean = validate_sampling_dataframe(raw)

    summaries = []
    traces = []
    for batch_id, g in clean.groupby("batch_id", sort=False):
        g = g.sort_values("sample_order")
        trace = evaluate_sampling_sequence(g["result"].tolist(), config)
        trace.insert(0, "batch_id", batch_id)
        traces.append(trace)

        last = trace.iloc[-1]
        summaries.append(
            {
                "batch_id": batch_id,
                "used_n": int(last["n"]),
                "k": int(last["k"]),
                "p_hat": float(last["p_hat"]),
                "decision": str(last["decision"]),
                "e_minus": float(last["e_minus"]),
                "e_plus": float(last["e_plus"]),
            }
        )

    trace_df = pd.concat(traces, ignore_index=True)
    trace_df.to_csv(output_dir / "actual_sampling_trace.csv", index=False, encoding="utf-8-sig")
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(output_dir / "actual_sampling_summary.csv", index=False, encoding="utf-8-sig")
    return summary_df


# ============================================================================
# 7. 模型检验：单元测试 + 理论约束的数值核验 + 风险网格
# ============================================================================
def run_unit_tests(config: EProcessConfig) -> List[str]:
    """全部通过则返回文本日志，否则直接抛 AssertionError。"""
    logs: List[str] = []

    # 1) 二项概率和为 1
    for n in (1, 2, 10, 22, 100, 200):
        ks = np.arange(n + 1)
        s = float(binom.pmf(ks, n, config.p0).sum())
        assert abs(s - 1.0) < 1e-12, (n, s)
    logs.append("PASS: 二项分布 PMF 概率和在机器精度内为 1。")

    # 2) 固定样本边界：定义的“最大接收/最小拒收”性质
    for n in FIXED_CHECK_N:
        a_n, r_n = exact_binomial_boundaries(
            n,
            config.p0,
            config.accept_confidence,
            config.reject_confidence,
        )
        if a_n is not None:
            assert binom.cdf(a_n, n, config.p0) <= config.alpha_accept + 1e-14
            if a_n + 1 <= n:
                assert binom.cdf(a_n + 1, n, config.p0) > config.alpha_accept - 1e-14
        if r_n is not None:
            assert binom.sf(r_n - 1, n, config.p0) <= config.alpha_reject + 1e-14
            if r_n - 1 >= 0:
                assert binom.sf(r_n - 2, n, config.p0) > config.alpha_reject - 1e-14
    logs.append("PASS: 精确二项边界满足最大接收/最小拒收定义。")

    # 3) 手册关键极端结果
    assert abs(float(binom.sf(1, 2, 0.10)) - 0.01) < 1e-14
    assert abs(float(binom.cdf(0, 22, 0.10)) - 0.9**22) < 1e-14
    assert float(binom.cdf(0, 22, 0.10)) <= 0.10
    logs.append("PASS: n=2,k=2 的拒收尾概率=0.01；n=22,k=0 的接收尾概率≈0.09848。")

    # 4) 手册固定样本 n=100 关键边界
    a100, r100 = exact_binomial_boundaries(100, 0.10, 0.90, 0.95)
    assert a100 == 5 and r100 == 16, (a100, r100)
    logs.append("PASS: 固定样本 n=100 得到 a_100=5, r_100=16。")

    # 5) e-process 最早停止案例
    assert sequential_decision(34, 0, config) == "accept"
    assert sequential_decision(33, 0, config) == "continue"
    assert sequential_decision(2, 2, config) == "reject"
    logs.append("PASS: 主模型最早接收 n=34,k=0；最早拒收 n=2,k=2。")

    # 6) 对数计算有限性/方向性
    for n, k in ((34, 0), (100, 3), (100, 10), (100, 21), (200, 36)):
        lm = log_e_minus(n, k, config)
        lp = log_e_plus(n, k, config)
        assert not math.isnan(lm) and not math.isnan(lp)
    logs.append("PASS: e-process 关键状态的 log-e 值无 NaN，数值稳定。")

    return logs


def risk_grid_validation(
    accept_boundaries: np.ndarray,
    reject_boundaries: np.ndarray,
    config: EProcessConfig,
    p_grid: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """
    有限时域数值核验：
      p <= p0 时，错误拒收概率不应超过 alpha_reject；
      p >= p0 时，错误接收概率不应超过 alpha_accept。

    理论上无限时域保证来自 Ville 不等式；此表只验证代码实现。
    """
    if p_grid is None:
        p_grid = np.round(np.arange(0.01, 0.20, 0.01), 4)
        p_grid = sorted(set(p_grid.tolist() + [config.p0]))

    rows = []
    for p in p_grid:
        res = exact_sequential_recursion(
            p, accept_boundaries, reject_boundaries, tol=0.0
        )
        if p <= config.p0 + 1e-15:
            risk_type = "wrong_reject"
            risk_value = res.reject_prob
            risk_limit = config.alpha_reject
        else:
            risk_type = "wrong_accept"
            risk_value = res.accept_prob
            risk_limit = config.alpha_accept

        rows.append(
            {
                "true_p": p,
                "risk_type": risk_type,
                "finite_horizon_risk": risk_value,
                "theoretical_limit": risk_limit,
                "pass": bool(risk_value <= risk_limit + 1e-10),
                "unresolved_prob": res.unresolved_prob,
                "horizon": res.horizon,
            }
        )

    df = pd.DataFrame(rows)
    if not bool(df["pass"].all()):
        bad = df.loc[~df["pass"]]
        raise AssertionError(f"风险网格核验失败：\n{bad.to_string(index=False)}")
    return df


# ============================================================================
# 8. 灵敏度分析
# ============================================================================
def sensitivity_true_p(
    accept_boundaries: np.ndarray,
    reject_boundaries: np.ndarray,
    p_values: Sequence[float] = P_SENSITIVITY_VALUES,
) -> pd.DataFrame:
    results = [
        exact_sequential_recursion(p, accept_boundaries, reject_boundaries)
        for p in p_values
    ]
    df = recursion_results_dataframe(results)
    df["distance_to_p0"] = np.abs(df["true_p"] - DEFAULT_P0)
    return df


def _earliest_boundary(boundary: np.ndarray, mode: str) -> Tuple[Optional[int], Optional[int]]:
    for n in range(1, len(boundary)):
        if mode == "accept":
            if boundary[n] >= 0:
                return n, int(boundary[n])
        elif mode == "reject":
            if boundary[n] <= n:
                return n, int(boundary[n])
        else:
            raise ValueError("mode 必须为 accept/reject")
    return None, None


def sensitivity_confidence_levels(
    max_n: int,
    p0: float = DEFAULT_P0,
) -> pd.DataFrame:
    """
    置信水平 one-at-a-time 灵敏度：
    - 接收置信度 88%~92%，拒收保持95%；
    - 拒收置信度 93%~97%，接收保持90%。
    变化范围严格对应手册建议的±1~2个百分点补充分析。
    """
    configs: List[Tuple[str, EProcessConfig]] = []
    for ac in (0.88, 0.89, 0.90, 0.91, 0.92):
        configs.append(
            (
                "accept_confidence",
                EProcessConfig(
                    p0=p0,
                    accept_confidence=ac,
                    reject_confidence=DEFAULT_REJECT_CONFIDENCE,
                    prior_a=1.0,
                    prior_b=1.0,
                    mixture_name="uniform",
                ),
            )
        )
    for rc in (0.93, 0.94, 0.95, 0.96, 0.97):
        # 避免基准配置重复写两次
        if abs(rc - DEFAULT_REJECT_CONFIDENCE) < 1e-15:
            continue
        configs.append(
            (
                "reject_confidence",
                EProcessConfig(
                    p0=p0,
                    accept_confidence=DEFAULT_ACCEPT_CONFIDENCE,
                    reject_confidence=rc,
                    prior_a=1.0,
                    prior_b=1.0,
                    mixture_name="uniform",
                ),
            )
        )

    rows = []
    eval_p = (0.05, 0.08, 0.12, 0.20)
    for varied, cfg in configs:
        acc, rej = generate_sequential_boundaries(max_n, cfg)
        ea_n, ea_k = _earliest_boundary(acc, "accept")
        er_n, er_k = _earliest_boundary(rej, "reject")

        for p in eval_p:
            res = exact_sequential_recursion(p, acc, rej)
            rows.append(
                {
                    "varied_parameter": varied,
                    "accept_confidence": cfg.accept_confidence,
                    "reject_confidence": cfg.reject_confidence,
                    "accept_threshold": 1.0 / cfg.alpha_accept,
                    "reject_threshold": 1.0 / cfg.alpha_reject,
                    "earliest_accept_n": ea_n,
                    "earliest_accept_k": ea_k,
                    "earliest_reject_n": er_n,
                    "earliest_reject_k": er_k,
                    **asdict(res),
                }
            )
    return pd.DataFrame(rows)


def sensitivity_mixture_density(
    max_n: int,
    p0: float = DEFAULT_P0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    模型形式稳健性对照：
    - uniform: Beta(1,1) 截断混合（主模型）；
    - Jeffreys-type: Beta(1/2,1/2) 截断混合（仅稳健性对照）。

    混合密度改变证据积累效率，但两者都由似然比非负混合构造，
    不改变 Ville 型错误率有效性。
    """
    configs = [
        EProcessConfig(
            p0=p0,
            accept_confidence=DEFAULT_ACCEPT_CONFIDENCE,
            reject_confidence=DEFAULT_REJECT_CONFIDENCE,
            prior_a=1.0,
            prior_b=1.0,
            mixture_name="uniform",
        ),
        EProcessConfig(
            p0=p0,
            accept_confidence=DEFAULT_ACCEPT_CONFIDENCE,
            reject_confidence=DEFAULT_REJECT_CONFIDENCE,
            prior_a=0.5,
            prior_b=0.5,
            mixture_name="jeffreys_type",
        ),
    ]

    perf_rows = []
    boundary_rows = []
    for cfg in configs:
        acc, rej = generate_sequential_boundaries(max_n, cfg)
        ea_n, ea_k = _earliest_boundary(acc, "accept")
        er_n, er_k = _earliest_boundary(rej, "reject")

        for n in SEQUENTIAL_CHECK_N:
            if n <= max_n:
                a_n = int(acc[n]) if acc[n] >= 0 else None
                r_n = int(rej[n]) if rej[n] <= n else None
                boundary_rows.append(
                    {
                        "mixture": cfg.mixture_name,
                        "prior_a": cfg.prior_a,
                        "prior_b": cfg.prior_b,
                        "n": n,
                        "accept_boundary": a_n,
                        "reject_boundary": r_n,
                    }
                )

        for p in (0.05, 0.08, 0.12, 0.20):
            res = exact_sequential_recursion(p, acc, rej)
            perf_rows.append(
                {
                    "mixture": cfg.mixture_name,
                    "prior_a": cfg.prior_a,
                    "prior_b": cfg.prior_b,
                    "earliest_accept_n": ea_n,
                    "earliest_accept_k": ea_k,
                    "earliest_reject_n": er_n,
                    "earliest_reject_k": er_k,
                    **asdict(res),
                }
            )

    return pd.DataFrame(perf_rows), pd.DataFrame(boundary_rows)


# ============================================================================
# 9. 可选扩展接口：有限总体/检测误差只在有真实参数时启用
# ============================================================================
def finite_population_note(total_batch_size: Optional[int]) -> str:
    """
    手册要求：题目未给批量 N，因此主模型不能虚构 N。
    这里只给接口提示；若将来获得 N，才比较二项与超几何有限总体模型。
    """
    if total_batch_size is None:
        return "未提供批量 N：主模型保持 Bernoulli/Binomial，不执行超几何修正。"
    if total_batch_size <= 0:
        raise ValueError("total_batch_size 必须为正。")
    return (
        f"已提供批量 N={total_batch_size}。若实际抽样比例 n/N 较大，"
        "应另行以超几何分布重算固定样本边界；本脚本主结果不自动替换，"
        "以避免改变建模手册的基准口径。"
    )


def measurement_error_note(
    sensitivity: Optional[float], specificity: Optional[float]
) -> str:
    """只有企业提供检测灵敏度/特异度时才允许扩展误分类模型。"""
    if sensitivity is None or specificity is None:
        return "未提供检测灵敏度 Se/特异度 Sp：按题意默认检测结果可正确识别，不虚构误检参数。"
    if not (0 < sensitivity <= 1 and 0 < specificity <= 1):
        raise ValueError("Se/Sp 必须位于 (0,1]。")
    return (
        f"检测设备参数 Se={sensitivity:.4f}, Sp={specificity:.4f} 已给出。"
        "此时观测到的次品指示不再等同真实状态，应另建误分类 Bernoulli 模型；"
        "该扩展不纳入当前题目主结果。"
    )


# ============================================================================
# 10. 输出与报告
# ============================================================================
def selected_rows(df: pd.DataFrame, n_values: Sequence[int]) -> pd.DataFrame:
    return df[df["n"].isin(n_values)].copy()


def write_validation_report(
    path: Path,
    unit_logs: Sequence[str],
    main_results: pd.DataFrame,
    risk_df: pd.DataFrame,
    config: EProcessConfig,
    max_n: int,
) -> None:
    lines = []
    lines.append("2024B 问题一——模型检验报告（无绘图）")
    lines.append("=" * 72)
    lines.append("")
    lines.append("一、单元测试")
    lines.extend([f"- {x}" for x in unit_logs])
    lines.append("")

    lines.append("二、理论风险约束")
    lines.append(
        f"- 错误拒收控制：sup_(p<=p0) P(reject) <= {config.alpha_reject:.4f}"
    )
    lines.append(
        f"- 错误接收控制：sup_(p>=p0) P(accept) <= {config.alpha_accept:.4f}"
    )
    lines.append("- 上述为 Ville 不等式给出的无限时域结构性保证；数值递推仅核对代码实现。")
    lines.append("")

    lines.append("三、代表性真实次品率的精确动态递推结果")
    lines.append(main_results.to_string(index=False, float_format=lambda x: f"{x:.10g}"))
    lines.append("")

    lines.append("四、有限时域风险网格核验")
    lines.append(
        f"- 递推 horizon 最大为 {max_n}；全部网格点是否通过理论上限检查："
        f"{bool(risk_df['pass'].all())}"
    )
    lines.append(
        risk_df.to_string(index=False, float_format=lambda x: f"{x:.10g}")
    )
    lines.append("")

    p0_row = main_results[np.isclose(main_results["true_p"], config.p0)]
    if not p0_row.empty:
        row = p0_row.iloc[0]
        lines.append("五、边界 p=p0 的解释")
        lines.append(
            f"- 在 p={config.p0:.2f} 且 n<={max_n} 时："
            f"P(accept)={row['accept_prob']:.6%}, "
            f"P(reject)={row['reject_prob']:.6%}, "
            f"P(continue)={row['unresolved_prob']:.6%}."
        )
        lines.append(
            "- 边界处大量路径继续不是模型失败，而是有限样本无法可靠区分 p 略高/略低于 p0。"
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def print_key_summary(
    fixed_selected: pd.DataFrame,
    seq_selected: pd.DataFrame,
    main_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 88)
    print("2024B 问题一：精确二项 + e-process 序贯检验 运行完成")
    print("=" * 88)

    print("\n[固定样本精确二项边界：关键 n]")
    print(fixed_selected.to_string(index=False))

    print("\n[主序贯模型边界：关键 n]")
    print(seq_selected.to_string(index=False))

    print("\n[动态概率递推：代表性真实 p]")
    cols = [
        "true_p",
        "accept_prob",
        "reject_prob",
        "unresolved_prob",
        "expected_tau_if_converged",
        "truncated_expected_n",
        "horizon",
        "converged",
    ]
    print(main_results[cols].to_string(index=False, float_format=lambda x: f"{x:.8g}"))

    print(f"\n全部 CSV/检验报告已输出到：{output_dir.resolve()}")


# ============================================================================
# 11. 主程序
# ============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="2024B 问题一：精确二项 + 混合似然比 e-process 序贯抽样"
    )
    parser.add_argument(
        "--max-n",
        type=int,
        default=DEFAULT_MAX_N,
        help=f"主递推/主边界最大样本量，默认 {DEFAULT_MAX_N}",
    )
    parser.add_argument(
        "--fixed-max-n",
        type=int,
        default=DEFAULT_FIXED_MAX_N,
        help=f"固定样本边界输出上限，默认 {DEFAULT_FIXED_MAX_N}",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="q1_output",
        help="输出目录，默认脚本所在目录下 q1_output",
    )
    parser.add_argument(
        "--sample-csv",
        type=str,
        default=None,
        help="可选：真实抽样记录 CSV，字段 batch_id,sample_order,result",
    )
    parser.add_argument(
        "--skip-confidence-sensitivity",
        action="store_true",
        help="若只想快速跑主结果，可跳过置信水平灵敏度分析",
    )
    parser.add_argument(
        "--skip-mixture-sensitivity",
        action="store_true",
        help="若只想快速跑主结果，可跳过混合密度稳健性分析",
    )
    args = parser.parse_args()

    if args.max_n < 500:
        raise ValueError("max_n 至少建议 >=500，以覆盖手册关键边界。")
    if args.fixed_max_n < max(FIXED_CHECK_N):
        raise ValueError(f"fixed_max_n 至少应 >= {max(FIXED_CHECK_N)}。")

    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = script_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # A. 主参数：题目给定 10%、90%、95%；均匀混合不引入额外业务 p1
    # ----------------------------------------------------------------------
    config = EProcessConfig()
    config.validate()

    # ----------------------------------------------------------------------
    # B. 单元测试
    # ----------------------------------------------------------------------
    unit_logs = run_unit_tests(config)

    # ----------------------------------------------------------------------
    # C. 固定样本精确二项基准
    # ----------------------------------------------------------------------
    fixed_df = fixed_boundary_table(
        max_n=args.fixed_max_n,
        p0=config.p0,
        accept_confidence=config.accept_confidence,
        reject_confidence=config.reject_confidence,
    )
    fixed_df.to_csv(
        output_dir / "q1_exact_binomial_boundaries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    fixed_selected = selected_rows(fixed_df, FIXED_CHECK_N)
    fixed_selected.to_csv(
        output_dir / "q1_exact_binomial_key_boundaries.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ----------------------------------------------------------------------
    # D. 主 e-process 边界（带单调性自检）
    # ----------------------------------------------------------------------
    accept_b, reject_b = generate_sequential_boundaries(
        args.max_n, config, check_monotonicity=True
    )
    seq_df = sequential_boundary_dataframe(accept_b, reject_b)
    seq_df.to_csv(
        output_dir / "q1_sequential_boundaries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    seq_selected = selected_rows(seq_df, SEQUENTIAL_CHECK_N)
    seq_selected.to_csv(
        output_dir / "q1_sequential_key_boundaries.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 核对手册关键序贯边界
    expected_seq = {
        20: (None, 7),
        34: (0, 10),
        40: (0, 11),
        50: (0, 13),
        80: (2, 18),
        100: (3, 21),
        160: (7, 30),
        200: (10, 36),
        500: (33, 75),
    }
    for n, (ea, er) in expected_seq.items():
        got_a = None if accept_b[n] < 0 else int(accept_b[n])
        got_r = None if reject_b[n] > n else int(reject_b[n])
        assert got_a == ea and got_r == er, (n, got_a, got_r, ea, er)
    unit_logs.append("PASS: 主序贯模型关键边界与建模手册表格完全一致。")

    # ----------------------------------------------------------------------
    # E. 代表性 p 的精确动态递推
    # ----------------------------------------------------------------------
    main_results = recursion_results_dataframe(
        [
            exact_sequential_recursion(p, accept_b, reject_b, tol=DEFAULT_TOL)
            for p in MAIN_P_VALUES
        ]
    )
    main_results.to_csv(
        output_dir / "q1_main_validation_performance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 手册关键性能数值核对（允许极小数值误差；p=0.12在N=20000时 residual~1e-7）
    reference = {
        0.05: (0.996396, 0.003604, 161.30),
        0.08: (0.987595, 0.012405, 1292.23),
        0.12: (0.021583, 0.978417, 2846.44),
        0.20: (0.000544, 0.999456, 103.37),
    }
    if args.max_n >= 20000:
        for p, (pa_ref, pr_ref, en_ref) in reference.items():
            row = main_results[np.isclose(main_results["true_p"], p)].iloc[0]
            assert abs(row["accept_prob"] - pa_ref) < 5e-5
            assert abs(row["reject_prob"] - pr_ref) < 5e-5
            if math.isfinite(float(row["expected_tau_if_converged"])):
                assert abs(row["expected_tau_if_converged"] - en_ref) < 0.2
            else:
                # p=0.12 在 20000 时 residual 约 1e-7，大于默认1e-10；
                # 手册报告的是 stop_time_mass 的高精度近似，因此核对条件期望/截断值数量级即可。
                assert abs(row["conditional_expected_n_given_stopped"] - en_ref) < 1.0
        unit_logs.append("PASS: 代表性 p 的递推结果与建模手册参考数值一致。")

    # ----------------------------------------------------------------------
    # F. 风险网格：核验代码未破坏 5%/10% 理论上限
    # ----------------------------------------------------------------------
    risk_df = risk_grid_validation(accept_b, reject_b, config)
    risk_df.to_csv(
        output_dir / "q1_risk_grid_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unit_logs.append("PASS: 有限时域风险网格全部满足理论错误率上限。")

    # ----------------------------------------------------------------------
    # G. 真实 p 灵敏度：核心敏感因素
    # ----------------------------------------------------------------------
    p_sens_df = sensitivity_true_p(accept_b, reject_b)
    p_sens_df.to_csv(
        output_dir / "q1_sensitivity_true_p.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ----------------------------------------------------------------------
    # H. 置信水平灵敏度（±1~2个百分点）
    # 为控制默认运行时间，最多使用 10000 的灵敏度时域；所有 residual 均显式输出。
    # ----------------------------------------------------------------------
    if not args.skip_confidence_sensitivity:
        sens_max_n = min(args.max_n, 10000)
        conf_df = sensitivity_confidence_levels(sens_max_n, p0=config.p0)
        conf_df.to_csv(
            output_dir / "q1_sensitivity_confidence.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # ----------------------------------------------------------------------
    # I. 混合密度形式稳健性：均匀 vs Jeffreys-type
    # ----------------------------------------------------------------------
    if not args.skip_mixture_sensitivity:
        mix_max_n = min(args.max_n, 10000)
        mix_perf, mix_bounds = sensitivity_mixture_density(mix_max_n, p0=config.p0)
        mix_perf.to_csv(
            output_dir / "q1_sensitivity_mixture_performance.csv",
            index=False,
            encoding="utf-8-sig",
        )
        mix_bounds.to_csv(
            output_dir / "q1_sensitivity_mixture_boundaries.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # ----------------------------------------------------------------------
    # J. 可选：对实际抽样 CSV 在线执行序贯规则
    # ----------------------------------------------------------------------
    if args.sample_csv:
        summary_df = evaluate_sampling_csv(Path(args.sample_csv), config, output_dir)
        print("\n[真实抽样批次判定摘要]")
        print(summary_df.to_string(index=False))

    # ----------------------------------------------------------------------
    # K. 输出建模假设边界提示，防止后续论文错误扩展
    # ----------------------------------------------------------------------
    assumptions_df = pd.DataFrame(
        {
            "item": ["finite_population", "measurement_error"],
            "status": [
                finite_population_note(None),
                measurement_error_note(None, None),
            ],
        }
    )
    assumptions_df.to_csv(
        output_dir / "q1_model_scope_notes.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ----------------------------------------------------------------------
    # L. 文字检验报告
    # ----------------------------------------------------------------------
    write_validation_report(
        output_dir / "q1_validation_report.txt",
        unit_logs,
        main_results,
        risk_df,
        config,
        args.max_n,
    )

    print_key_summary(fixed_selected, seq_selected, main_results, output_dir)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断。", file=sys.stderr)
        sys.exit(130)
