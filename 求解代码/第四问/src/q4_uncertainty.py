from __future__ import annotations

from dataclasses import dataclass, replace
from math import log
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import betainc, betaln
from scipy.stats import beta as beta_dist


@dataclass(frozen=True)
class SampleRecord:
    quality_id: str
    quality_type: str
    sample_design: str
    n: int
    k: int
    input_certified: bool = True
    batch_id: str = ""
    seq_reference_p0: Optional[float] = None
    reported_rate: Optional[float] = None

    @property
    def p_hat(self) -> float:
        return self.k / self.n

    def with_counts(self, n: int, k: int) -> "SampleRecord":
        return replace(self, n=int(n), k=int(k))


def _check_record(r: SampleRecord) -> None:
    if not r.quality_id:
        raise ValueError("quality_id 不能为空")
    if r.sample_design not in {"fixed", "sequential"}:
        raise ValueError(f"{r.quality_id}: sample_design 必须为 fixed/sequential")
    if r.n <= 0 or int(r.n) != r.n:
        raise ValueError(f"{r.quality_id}: n 必须为正整数")
    if r.k < 0 or r.k > r.n or int(r.k) != r.k:
        raise ValueError(f"{r.quality_id}: 必须满足 0<=k<=n")
    if r.quality_type not in {"raw_component", "process"}:
        raise ValueError(f"{r.quality_id}: quality_type 必须为 raw_component/process")
    if r.quality_type == "process" and not r.input_certified:
        raise ValueError(
            f"{r.quality_id}: process 次品率只能使用“所有直接输入均已确认合格”的样本"
        )
    if r.sample_design == "sequential" and r.seq_reference_p0 is not None:
        if not 0 < r.seq_reference_p0 < 1:
            raise ValueError(f"{r.quality_id}: seq_reference_p0 必须位于(0,1)")


def load_sampling_records(path: str | Path) -> Tuple[List[SampleRecord], List[str]]:
    """读取真实抽样记录。空 n/k 行作为模板保留，但不会进入计算。"""
    path = Path(path)
    if not path.exists():
        return [], [f"sampling file not found: {path}"]
    df = pd.read_csv(path)
    required = {"quality_id", "quality_type", "sample_design", "n", "k", "input_certified"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"sampling_records 缺少列: {sorted(missing)}")
    records: List[SampleRecord] = []
    warnings: List[str] = []
    for _, row in df.iterrows():
        if pd.isna(row["n"]) or pd.isna(row["k"]) or not str(row["sample_design"]).strip():
            warnings.append(f"skip incomplete row: {row.get('quality_id', '')}")
            continue
        def opt_float(name: str) -> Optional[float]:
            if name not in df.columns or pd.isna(row[name]) or str(row[name]).strip() == "":
                return None
            return float(row[name])
        val = row["input_certified"]
        if isinstance(val, str):
            cert = val.strip().lower() in {"true", "1", "yes", "y"}
        else:
            cert = bool(val)
        r = SampleRecord(
            quality_id=str(row["quality_id"]).strip(),
            quality_type=str(row["quality_type"]).strip(),
            sample_design=str(row["sample_design"]).strip().lower(),
            n=int(row["n"]),
            k=int(row["k"]),
            input_certified=cert,
            batch_id="" if "batch_id" not in df.columns or pd.isna(row.get("batch_id")) else str(row["batch_id"]),
            seq_reference_p0=opt_float("seq_reference_p0"),
            reported_rate=opt_float("reported_rate"),
        )
        _check_record(r)
        records.append(r)
    ids = [r.quality_id for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("quality_id 必须唯一；若保存逐件记录，请先按 quality_id 聚合为 n,k")
    return records, warnings


def cp_upper(k: int, n: int, confidence: float) -> float:
    """固定样本：Clopper-Pearson 精确单侧上置信限。"""
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0,1)")
    if n <= 0 or not (0 <= k <= n):
        raise ValueError("invalid n,k")
    if k == n:
        return 1.0
    return float(beta_dist.ppf(confidence, k + 1, n - k))


def simultaneous_component_confidence(global_confidence: float, d: int) -> float:
    """Bonferroni/union bound: 1-alpha/d。"""
    if d <= 0:
        raise ValueError("d must be positive")
    alpha = 1.0 - global_confidence
    return 1.0 - alpha / d


def _safe_log_prob(x: float) -> float:
    if x <= 0.0:
        return -np.inf
    return float(np.log(x))


def log_e_minus(n: int, k: int, p0: float) -> float:
    """
    Q1 同一单侧混合似然比 e-process：支持 p < p0 的证据。
    混合密度取 [0,p0] 上均匀分布；不引入额外备择 p1。
    """
    if p0 <= 0.0:
        return -np.inf
    if p0 >= 1.0:
        return np.inf if k < n else -log(n + 1.0)
    a, b = k + 1, n - k + 1
    I = float(betainc(a, b, p0))
    logI = _safe_log_prob(I)
    if not np.isfinite(logI):
        return -np.inf
    return float(
        betaln(a, b)
        + logI
        - log(p0)
        - k * log(p0)
        - (n - k) * np.log1p(-p0)
    )


def log_e_plus(n: int, k: int, p0: float) -> float:
    """Q1 同一单侧混合似然比 e-process：支持 p > p0 的证据。"""
    if p0 <= 0.0:
        return np.inf if k > 0 else -log(n + 1.0)
    if p0 >= 1.0:
        return -np.inf
    a, b = k + 1, n - k + 1
    I = float(betainc(a, b, p0))
    tail = max(0.0, 1.0 - I)
    log_tail = _safe_log_prob(tail)
    if not np.isfinite(log_tail):
        return -np.inf
    return float(
        betaln(a, b)
        + log_tail
        - log(1.0 - p0)
        - k * log(p0)
        - (n - k) * np.log1p(-p0)
    )


def sequential_upper(n: int, k: int, confidence: float) -> float:
    """
    反演 E^- 得到任意停止有效的单侧上置信界：
      U = inf{p0: E^-(p0) >= 1/(1-confidence)}.
    只依赖停止时刻的 n,k；其合法性来自 e-process 的任意停止性质。
    """
    if not (0 < confidence < 1):
        raise ValueError("confidence must be in (0,1)")
    if n <= 0 or not (0 <= k <= n):
        raise ValueError("invalid n,k")
    if k == n:
        return 1.0
    threshold = -log(1.0 - confidence)
    phat = k / n
    lo = max(phat, 1e-12)
    hi = 1.0 - 1e-12

    def f(p: float) -> float:
        return log_e_minus(n, k, p) - threshold

    flo = f(lo)
    if flo >= 0:
        return float(lo)
    fhi = f(hi)
    if not np.isfinite(fhi) or fhi >= 0:
        # 粗网格找首次越界，再 Brent 精修。
        grid = np.linspace(lo, hi, 512)
        prev_x, prev_f = grid[0], flo
        for x in grid[1:]:
            fx = f(float(x))
            if fx >= 0 and prev_f < 0:
                return float(brentq(f, prev_x, float(x), xtol=1e-12, rtol=1e-12, maxiter=200))
            prev_x, prev_f = float(x), fx
        if fhi >= 0:
            return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200))
    return 1.0


def upper_for_record(r: SampleRecord, confidence: float) -> float:
    if r.sample_design == "fixed":
        return cp_upper(r.k, r.n, confidence)
    if r.sample_design == "sequential":
        return sequential_upper(r.n, r.k, confidence)
    raise ValueError(f"unknown sample design: {r.sample_design}")


def build_interval_table(records: Sequence[SampleRecord], global_confidences=(0.90, 0.95)) -> pd.DataFrame:
    """对一个 Q2 情形(d=3)或 Q3 系统(d=12)构造 componentwise 与 simultaneous 上界。"""
    if not records:
        return pd.DataFrame()
    d = len(records)
    rows = []
    for r in records:
        row = {
            "quality_id": r.quality_id,
            "quality_type": r.quality_type,
            "sample_design": r.sample_design,
            "n": r.n,
            "k": r.k,
            "p_hat": r.p_hat,
            "input_certified": r.input_certified,
            "reported_rate": r.reported_rate,
        }
        for gamma in global_confidences:
            key = int(round(100 * gamma))
            row[f"U_component_{key}"] = upper_for_record(r, gamma)
            per_conf = simultaneous_component_confidence(gamma, d)
            row[f"per_parameter_conf_{key}"] = per_conf
            row[f"U_simultaneous_{key}"] = upper_for_record(r, per_conf)
        rows.append(row)
    return pd.DataFrame(rows)


def simulate_q1_stop(
    p_true: float,
    reference_p0: float,
    rng: np.random.Generator,
    accept_confidence: float = 0.90,
    reject_confidence: float = 0.95,
    max_n: int = 10000,
) -> Tuple[int, int, str, bool]:
    """复现 Q1 的 sequential stopping。max_n 只是仿真安全上限，不是模型截断。"""
    log_accept = -log(1.0 - accept_confidence)  # log(10)
    log_reject = -log(1.0 - reject_confidence)  # log(20)
    k = 0
    for n in range(1, max_n + 1):
        k += int(rng.random() < p_true)
        if log_e_minus(n, k, reference_p0) >= log_accept:
            return n, k, "accept", False
        if log_e_plus(n, k, reference_p0) >= log_reject:
            return n, k, "reject", False
    return max_n, k, "censored", True


def bootstrap_record(r: SampleRecord, rng: np.random.Generator, max_seq_n: int = 10000) -> Tuple[SampleRecord, bool]:
    """同抽样设计 parametric bootstrap。返回(新记录, 是否截尾)。"""
    p = r.p_hat
    if r.sample_design == "fixed":
        k_star = int(rng.binomial(r.n, p))
        return r.with_counts(r.n, k_star), False
    if r.seq_reference_p0 is None:
        raise ValueError(
            f"{r.quality_id}: sequential bootstrap 需要 seq_reference_p0，以复现原停止规则；"
            "不能把随机停止样本伪装成固定 n。"
        )
    n_star, k_star, _, censored = simulate_q1_stop(
        p, r.seq_reference_p0, rng, max_n=max_seq_n
    )
    return r.with_counts(n_star, k_star), censored
