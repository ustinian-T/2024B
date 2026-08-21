"""
2024 高教社杯 B 题 —— 问题四：硬编码抽样记录（设计情景）

本模块提供 main.py 默认使用的全部 quality_id 抽样记录。
严格遵循《问题四建模手册》§7.5 的"禁止伪造结果"原则：
- 所有 n、k、sample_design 必须在题面或第一问序贯规则下有意义；
- 由于题面只给出各次品率点估计，未给真实抽样量，本文件按"内部设计情景"标注，
  严禁把它们当成题目事实或最终 Q4 数值结论。

设计情景选择原则：
1. 与表 1 / 表 2 中报点率 reported_rate 严格一致（k/n ≈ reported_rate）；
2. 样本量在题目本身没有给出时取一个能反映"中等不确定性"的代表值：
   - raw_component 取 n=100；
   - process 取 n=200（工序样本更难收集）；
3. 同时覆盖 fixed 与 sequential 两种抽样设计，验证两路统计代码路径：
   - Q2 各情形的 raw_component 与 process 全部用 fixed；
   - Q3 的 8 个 raw_component 用 fixed、4 个 process 用 sequential，
     从而同时演示 sequential_upper() 反演路径。

若竞赛现场提供了真实 sampling_records.csv，仍可通过 main.py --sampling <path> 覆盖。
"""

from __future__ import annotations

from typing import List

from .q4_uncertainty import SampleRecord


# ----------------------------------------------------------------------
# Q2 六情形 × 3 个质量参数 = 18 条记录（每情形 C1/C2/F）
# 全部使用 fixed 设计；样本量 n=100、n=200。
# ----------------------------------------------------------------------

def _q2_fixed(quality_id: str, p_reported: float, n: int) -> SampleRecord:
    k = int(round(p_reported * n))
    return SampleRecord(
        quality_id=quality_id,
        quality_type="raw_component" if not quality_id.endswith("-F") else "process",
        sample_design="fixed",
        n=n,
        k=k,
        input_certified=True,
        batch_id="B-Q2-default",
        reported_rate=p_reported,
    )


Q2_FIXED_RECORDS: List[SampleRecord] = [
    # 情形 1：表 1 第 1 行（C1/C2/F 都是 10%）
    _q2_fixed("Q2-S1-C1", 0.10, 100),
    _q2_fixed("Q2-S1-C2", 0.10, 100),
    _q2_fixed("Q2-S1-F", 0.10, 200),
    # 情形 2：次品率均为 20%
    _q2_fixed("Q2-S2-C1", 0.20, 100),
    _q2_fixed("Q2-S2-C2", 0.20, 100),
    _q2_fixed("Q2-S2-F", 0.20, 200),
    # 情形 3：C1/F=10%, C2=10%
    _q2_fixed("Q2-S3-C1", 0.10, 100),
    _q2_fixed("Q2-S3-C2", 0.10, 100),
    _q2_fixed("Q2-S3-F", 0.10, 200),
    # 情形 4：次品率均为 20%
    _q2_fixed("Q2-S4-C1", 0.20, 100),
    _q2_fixed("Q2-S4-C2", 0.20, 100),
    _q2_fixed("Q2-S4-F", 0.20, 200),
    # 情形 5：C1/F=10%, C2=20%
    _q2_fixed("Q2-S5-C1", 0.10, 100),
    _q2_fixed("Q2-S5-C2", 0.20, 100),
    _q2_fixed("Q2-S5-F", 0.10, 200),
    # 情形 6：次品率均为 5%
    _q2_fixed("Q2-S6-C1", 0.05, 100),
    _q2_fixed("Q2-S6-C2", 0.05, 100),
    _q2_fixed("Q2-S6-F", 0.05, 200),
]


# ----------------------------------------------------------------------
# Q3 实例：12 个质量参数（8 raw_component + 4 process）
# raw_component 取 n=100 fixed；process 取 n=200 sequential。
# sequential 记录必须给出 seq_reference_p0 以供 parametric bootstrap 复现
# 问题一"90%接收/95%拒收"停止规则，本题沿用 p0=0.10 即可。
# ----------------------------------------------------------------------

def _q3_fixed_component(name: str, n: int = 100) -> SampleRecord:
    return SampleRecord(
        quality_id=f"Q3-{name}",
        quality_type="raw_component",
        sample_design="fixed",
        n=n,
        k=int(round(0.10 * n)),
        input_certified=True,
        batch_id="B-Q3-raw",
        reported_rate=0.10,
    )


def _q3_sequential_process(name: str, n: int = 200, p0: float = 0.10) -> SampleRecord:
    return SampleRecord(
        quality_id=f"Q3-{name}",
        quality_type="process",
        sample_design="sequential",
        n=n,
        k=int(round(0.10 * n)),
        input_certified=True,
        batch_id="B-Q3-proc",
        seq_reference_p0=p0,
        reported_rate=0.10,
    )


Q3_RAW_RECORDS: List[SampleRecord] = [_q3_fixed_component(f"C{i}") for i in range(1, 9)]
Q3_PROC_RECORDS: List[SampleRecord] = [
    _q3_sequential_process("H1"),
    _q3_sequential_process("H2"),
    _q3_sequential_process("H3"),
    _q3_sequential_process("F"),
]


# ----------------------------------------------------------------------
# 合并：默认全部硬编码抽样记录
# ----------------------------------------------------------------------

DEFAULT_SAMPLING_RECORDS: List[SampleRecord] = Q2_FIXED_RECORDS + Q3_RAW_RECORDS + Q3_PROC_RECORDS


__all__ = [
    "DEFAULT_SAMPLING_RECORDS",
    "Q2_FIXED_RECORDS",
    "Q3_RAW_RECORDS",
    "Q3_PROC_RECORDS",
]