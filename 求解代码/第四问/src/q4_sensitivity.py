from __future__ import annotations

from dataclasses import replace
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from .config import Q2_SCENARIOS, Q3_NODES, Q3_REPLACEMENT_LOSS
from .q2_model import solve_q2
from .q3_model import solve_q3_fast


def q2_nominal_sensitivity() -> pd.DataFrame:
    """基于题目已出现的实际跨度做全局重优化，不使用统一±5%。"""
    p0=Q2_SCENARIOS[1]
    specs={
        "p1":np.linspace(0.05,0.20,31),
        "p2":np.linspace(0.05,0.20,31),
        "pf":np.linspace(0.05,0.20,31),
        "b1":np.linspace(1.0,8.0,29),
        "b2":np.linspace(1.0,8.0,29),
        "replacement_loss":np.linspace(6.0,40.0,35),
        "disassembly_cost":np.linspace(5.0,40.0,36),
    }
    rows=[]
    for name,grid in specs.items():
        prev_code=None
        for v in grid:
            pp=replace(p0,**{name:float(v)})
            best,rank=solve_q2(pp,topk=3)
            code=best.code4
            rows.append({"parameter":name,"value":float(v),"policy4":code,"policy6":best.code6,"profit":best.expected_profit,"gap":float(rank.iloc[0]["primary_policy_gap"]),"switch_from_previous":prev_code is not None and code!=prev_code})
            prev_code=code
    return pd.DataFrame(rows)


def q3_nominal_sensitivity() -> pd.DataFrame:
    rows=[]
    # Q3层面优先分析 p_F、调换损失、成品拆解费；其余可在论文附录扩展。
    for pF in np.linspace(0.05,0.20,61):
        code,profit,top=solve_q3_fast({"F":float(pF)},topk=3)
        rows.append({"parameter":"pF","value":float(pF),"policy16":code,"profit":profit,"gap":float(top.iloc[0]["best_second_gap"])})
    for L in np.linspace(6.0,40.0,69):
        code,profit,top=solve_q3_fast(replacement_loss=float(L),topk=3)
        rows.append({"parameter":"replacement_loss","value":float(L),"policy16":code,"profit":profit,"gap":float(top.iloc[0]["best_second_gap"])})
    # dF 改成本节点需要临时替换全局表会污染；在主程序中用专用函数做。
    return pd.DataFrame(rows)


def q3_phase_map(
    base_quality: Mapping[str,float] | None = None,
    pf_range: Sequence[float] | None = None,
    loss_range: Sequence[float] | None = None,
) -> pd.DataFrame:
    if pf_range is None:
        pf_range=np.linspace(0.05,0.20,31)
    if loss_range is None:
        loss_range=np.linspace(6.0,40.0,35)
    base=dict(base_quality or {})
    rows=[]
    for pf in pf_range:
        q=base.copy(); q["F"]=float(pf)
        for L in loss_range:
            code,profit,top=solve_q3_fast(q,replacement_loss=float(L),topk=2)
            rows.append({"pF":float(pf),"replacement_loss":float(L),"policy16":code,"profit":profit,"gap":float(top.iloc[0]["best_second_gap"])})
    return pd.DataFrame(rows)
