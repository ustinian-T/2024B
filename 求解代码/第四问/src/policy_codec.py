"""Q2 / Q3 策略位解码器与位差比较器。

题面约束：
- Q2 策略编码 (x1, x2, y, z, r1, r2)，长度 6。
- Q3 策略编码 (x1..x8, y1, y2, y3, z1, z2, z3, y_F, z_F)，长度 16。

位序含义：
- Q2 第 0..1 位：是否检测零配件 1 / 2；
- Q2 第 2 位  ：是否检测成品；
- Q2 第 3 位  ：坏品是否拆解；
- Q2 第 4..5 位：拆解后对回收的零件 1 / 2 是否再检测（z=0 时规范化为 0）。
- Q3 第 0..7 位：是否检测 C1..C8；
- Q3 第 8..10 位：是否检测半成品 H1..H3；
- Q3 第 11..13 位：坏半成品 H1..H3 是否拆解；
- Q3 第 14 位 ：是否检测成品 F；
- Q3 第 15 位 ：坏成品是否拆解。

所有报告中的策略文字解释都必须通过本模块生成，禁止手写"检成品"等含糊说法。
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple


# ============================================================
# Q2 策略
# ============================================================

Q2_BIT_NAMES: Tuple[str, ...] = (
    "x1",       # 检测零配件 1
    "x2",       # 检测零配件 2
    "y",        # 检测成品
    "z",        # 坏品拆解
    "r1",       # 拆解后回收零件 1 检测
    "r2",       # 拆解后回收零件 2 检测
)


def decode_q2_policy(bits) -> Dict[str, int]:
    """把 6 位 0/1 解码为可读名字字典。

    接受 6 元素 list/tuple，或长度为 6 的 "0/1 字符串"。
    z=0 时 r1/r2 必须为 0（policy.canonical() 已经规范化）。
    """
    if isinstance(bits, str):
        if len(bits) != 6 or any(c not in "01" for c in bits):
            raise ValueError(f"Q2 policy 必须是 6 个 0/1，实际={bits}")
        bits_list = [int(c) for c in bits]
    else:
        bits_list = list(bits)
        if len(bits_list) != 6 or any(b not in (0, 1) for b in bits_list):
            raise ValueError(f"Q2 policy 必须是 6 个 0/1，实际={bits}")
    return {name: int(bits_list[i]) for i, name in enumerate(Q2_BIT_NAMES)}


def diff_q2_policy(code_a: str, code_b: str) -> List[Dict[str, int]]:
    """逐位比较两个 Q2 策略；返回 changed bits 列表。

    每个元素形如 {"name": "x2", "from": 0, "to": 1, "decision_cn": "新增零配件 2 检测"}。
    若两策略完全相同返回空列表。
    """
    a = decode_q2_policy(code_a)
    b = decode_q2_policy(code_b)
    changes = []
    for name in Q2_BIT_NAMES:
        if a[name] != b[name]:
            changes.append({
                "name": name,
                "from": int(a[name]),
                "to": int(b[name]),
                "decision_cn": _q2_decision_cn(name, a[name], b[name]),
            })
    return changes


def _q2_decision_cn(name: str, frm: int, to: int) -> str:
    table = {
        "x1": ("取消零配件 1 检测" if frm == 1 and to == 0 else "新增零配件 1 检测"),
        "x2": ("取消零配件 2 检测" if frm == 1 and to == 0 else "新增零配件 2 检测"),
        "y":  ("取消成品出厂检测" if frm == 1 and to == 0 else "新增成品出厂检测"),
        "z":  ("取消坏品拆解回收" if frm == 1 and to == 0 else "新增坏品拆解回收"),
        "r1": ("取消回收零件 1 再检测" if frm == 1 and to == 0 else "新增回收零件 1 再检测"),
        "r2": ("取消回收零件 2 再检测" if frm == 1 and to == 0 else "新增回收零件 2 再检测"),
    }
    return table.get(name, f"{name}: {frm} -> {to}")


# ============================================================
# Q3 策略
# ============================================================

Q3_BIT_NAMES: Tuple[str, ...] = (
    "x_C1", "x_C2", "x_C3", "x_C4", "x_C5", "x_C6", "x_C7", "x_C8",
    "y_H1", "y_H2", "y_H3",
    "z_H1", "z_H2", "z_H3",
    "y_F",  "z_F",
)


def decode_q3_policy(bits) -> Dict[str, int]:
    """把 16 位 0/1 解码为可读名字字典。"""
    if isinstance(bits, str):
        if len(bits) != 16 or any(c not in "01" for c in bits):
            raise ValueError(f"Q3 policy 必须是 16 个 0/1，实际={bits}")
        bits_list = [int(c) for c in bits]
    else:
        bits_list = list(bits)
        if len(bits_list) != 16 or any(b not in (0, 1) for b in bits_list):
            raise ValueError(f"Q3 policy 必须是 16 个 0/1，实际={bits}")
    return {name: int(bits_list[i]) for i, name in enumerate(Q3_BIT_NAMES)}


def diff_q3_policy(code_a: str, code_b: str) -> List[Dict[str, int]]:
    """逐位比较两个 Q3 策略；返回 changed bits 列表。"""
    a = decode_q3_policy(code_a)
    b = decode_q3_policy(code_b)
    changes = []
    for name in Q3_BIT_NAMES:
        if a[name] != b[name]:
            changes.append({
                "name": name,
                "from": int(a[name]),
                "to": int(b[name]),
                "decision_cn": _q3_decision_cn(name, a[name], b[name]),
            })
    return changes


def _q3_decision_cn(name: str, frm: int, to: int) -> str:
    table = {
        "x_C1": ("取消 C1 检测" if frm == 1 and to == 0 else "新增 C1 检测"),
        "x_C2": ("取消 C2 检测" if frm == 1 and to == 0 else "新增 C2 检测"),
        "x_C3": ("取消 C3 检测" if frm == 1 and to == 0 else "新增 C3 检测"),
        "x_C4": ("取消 C4 检测" if frm == 1 and to == 0 else "新增 C4 检测"),
        "x_C5": ("取消 C5 检测" if frm == 1 and to == 0 else "新增 C5 检测"),
        "x_C6": ("取消 C6 检测" if frm == 1 and to == 0 else "新增 C6 检测"),
        "x_C7": ("取消 C7 检测" if frm == 1 and to == 0 else "新增 C7 检测"),
        "x_C8": ("取消 C8 检测" if frm == 1 and to == 0 else "新增 C8 检测"),
        "y_H1": ("取消 H1 出厂检测" if frm == 1 and to == 0 else "新增 H1 出厂检测"),
        "y_H2": ("取消 H2 出厂检测" if frm == 1 and to == 0 else "新增 H2 出厂检测"),
        "y_H3": ("取消 H3 出厂检测" if frm == 1 and to == 0 else "新增 H3 出厂检测"),
        "z_H1": ("取消坏 H1 拆解" if frm == 1 and to == 0 else "新增坏 H1 拆解"),
        "z_H2": ("取消坏 H2 拆解" if frm == 1 and to == 0 else "新增坏 H2 拆解"),
        "z_H3": ("取消坏 H3 拆解" if frm == 1 and to == 0 else "新增坏 H3 拆解"),
        "y_F":  ("取消成品 F 出厂检测" if frm == 1 and to == 0 else "新增成品 F 出厂检测"),
        "z_F":  ("取消坏 F 拆解" if frm == 1 and to == 0 else "新增坏 F 拆解"),
    }
    return table.get(name, f"{name}: {frm} -> {to}")


__all__ = [
    "Q2_BIT_NAMES", "decode_q2_policy", "diff_q2_policy",
    "Q3_BIT_NAMES", "decode_q3_policy", "diff_q3_policy",
]