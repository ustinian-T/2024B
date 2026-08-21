from __future__ import annotations

from itertools import product
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binom

from .config import (
    Q2_EXPECTED,
    Q2_SCENARIOS,
    Q3_EXPECTED_CODE,
    Q3_EXPECTED_COST,
    Q3_EXPECTED_PROFIT,
    Q3_NODES,
)
from .q2_model import evaluate_q2_strategy, evaluate_q2_strategy_scalar, no_disassembly_closed_form, solve_q2
from .q3_model import all_profit_vector, evaluate_q3_strategy_detailed, solve_q3_fast, solve_q3_flat
from .q4_uncertainty import cp_upper, log_e_minus, log_e_plus, sequential_upper


def regression_q2(tol: float = 1e-9) -> pd.DataFrame:
    rows=[]
    for sid,p in Q2_SCENARIOS.items():
        best,rank=solve_q2(p,topk=5)
        exp_code,exp_profit=Q2_EXPECTED[sid]
        ok_code=best.code4==exp_code
        ok_profit=abs(best.expected_profit-exp_profit)<tol
        rows.append({
            "scenario_id":sid,"policy4":best.code4,"policy6":best.code6,
            "profit":best.expected_profit,"expected_policy4":exp_code,
            "expected_profit":exp_profit,"abs_error":abs(best.expected_profit-exp_profit),
            "PASS":bool(ok_code and ok_profit)
        })
    return pd.DataFrame(rows)


def q2_closed_form_validation(tol: float = 1e-10) -> pd.DataFrame:
    rows=[]
    for sid,p in Q2_SCENARIOS.items():
        for x1,x2,y in product((0,1),repeat=3):
            pi=(x1,x2,y,0,0,0)
            e=evaluate_q2_strategy(p,pi)
            closed=no_disassembly_closed_form(p,x1,x2,y)
            err=abs(e.expected_cost-closed)
            rows.append({"scenario_id":sid,"x1":x1,"x2":x2,"y":y,"bellman":e.expected_cost,"closed_form":closed,"abs_error":err,"PASS":err<tol})
    return pd.DataFrame(rows)


def regression_q3(tol: float = 1e-10, flat_cross_check: bool = True) -> pd.DataFrame:
    code,profit,top=solve_q3_fast(topk=10)
    detail=evaluate_q3_strategy_detailed(code)
    rows=[{
        "solver":"layered_vectorized","policy16":code,"profit":profit,"cost":detail.expected_cost,
        "expected_policy16":Q3_EXPECTED_CODE,"expected_profit":Q3_EXPECTED_PROFIT,"expected_cost":Q3_EXPECTED_COST,
        "profit_abs_error":abs(profit-Q3_EXPECTED_PROFIT),"cost_abs_error":abs(detail.expected_cost-Q3_EXPECTED_COST),
        "PASS":bool(code==Q3_EXPECTED_CODE and abs(profit-Q3_EXPECTED_PROFIT)<tol and abs(detail.expected_cost-Q3_EXPECTED_COST)<tol)
    }]
    if flat_cross_check:
        fcode,fprofit=solve_q3_flat()
        rows.append({
            "solver":"flat_65536","policy16":fcode,"profit":fprofit,"cost":200-fprofit,
            "expected_policy16":Q3_EXPECTED_CODE,"expected_profit":Q3_EXPECTED_PROFIT,"expected_cost":Q3_EXPECTED_COST,
            "profit_abs_error":abs(fprofit-Q3_EXPECTED_PROFIT),"cost_abs_error":abs((200-fprofit)-Q3_EXPECTED_COST),
            "PASS":bool(fcode==Q3_EXPECTED_CODE and abs(fprofit-Q3_EXPECTED_PROFIT)<tol)
        })
    return pd.DataFrame(rows)


def q3_handcheck_validation(tol: float = 1e-10) -> pd.DataFrame:
    d=evaluate_q3_strategy_detailed(Q3_EXPECTED_CODE)
    expected={
        "purchase":71.11111111111111,
        "component_inspection":12.22222222222222,
        "assembly":35.55555555555556,
        "intermediate_inspection":13.33333333333333,
        "final_inspection":0.0,
        "replacement":4.44444444444444,
        "disassembly":3.11111111111111,
    }
    rows=[]
    for k,v in expected.items():
        act=d.cost_breakdown[k]
        rows.append({"item":k,"actual":act,"expected":v,"abs_error":abs(act-v),"PASS":abs(act-v)<tol})
    rows.append({"item":"total","actual":d.expected_cost,"expected":Q3_EXPECTED_COST,"abs_error":abs(d.expected_cost-Q3_EXPECTED_COST),"PASS":abs(d.expected_cost-Q3_EXPECTED_COST)<tol})
    return pd.DataFrame(rows)


def exact_cp_coverage_validation(
    p_values: Sequence[float]=(0.05,0.10,0.20),
    n_values: Sequence[int]=(20,50,100,200),
    confidences: Sequence[float]=(0.90,0.95),
) -> pd.DataFrame:
    """不靠MC，直接对K=0..n求和得到CP上界的精确覆盖率。n仅为数值验证网格，不是题目数据。"""
    rows=[]
    for p in p_values:
        for n in n_values:
            ks=np.arange(n+1)
            pmf=binom.pmf(ks,n,p)
            for conf in confidences:
                upper=np.array([cp_upper(int(k),n,conf) for k in ks])
                cov=float(pmf[upper+1e-15>=p].sum())
                rows.append({"p_true":p,"n_validation_grid":n,"confidence":conf,"coverage":cov,"conservative_margin":cov-conf,"PASS":cov+1e-12>=conf})
    return pd.DataFrame(rows)




def sequential_cs_coverage_validation(
    p_values: Sequence[float]=(0.05,0.10,0.20),
    reference_p0: float=0.10,
    confidences: Sequence[float]=(0.90,0.95),
    reps: int=1000,
    max_n: int=2000,
    seed: int=20260820,
) -> pd.DataFrame:
    """
    序贯置信序列覆盖率 Monte Carlo 检验。
    每条 Bernoulli 流严格按第一问90%接收/95%拒收 e-process停止；max_n仅是验证截尾上限。
    对停止点直接检查 E^-(p_true) 是否越过相应上界阈值，等价于检查 p_true<=U_seq。
    """
    from scipy.special import betainc as _bi, betaln as _bl
    rng=np.random.default_rng(seed)
    rows=[]

    def log_em_vec(n_arr,k_arr,p0):
        n_arr=np.asarray(n_arr,dtype=float); k_arr=np.asarray(k_arr,dtype=float)
        if p0<=0: return np.full_like(n_arr,-np.inf,dtype=float)
        if p0>=1: return np.where(k_arr<n_arr,np.inf,-np.log(n_arr+1.0))
        a=k_arr+1.0; b=n_arr-k_arr+1.0
        I=_bi(a,b,p0)
        out=_bl(a,b)+np.log(np.maximum(I,np.finfo(float).tiny))-np.log(p0)-k_arr*np.log(p0)-(n_arr-k_arr)*np.log1p(-p0)
        return out

    def log_ep_vec(n_arr,k_arr,p0):
        n_arr=np.asarray(n_arr,dtype=float); k_arr=np.asarray(k_arr,dtype=float)
        if p0<=0: return np.where(k_arr>0,np.inf,-np.log(n_arr+1.0))
        if p0>=1: return np.full_like(n_arr,-np.inf,dtype=float)
        a=k_arr+1.0; b=n_arr-k_arr+1.0
        I=_bi(a,b,p0); tail=np.maximum(1.0-I,np.finfo(float).tiny)
        return _bl(a,b)+np.log(tail)-np.log1p(-p0)-k_arr*np.log(p0)-(n_arr-k_arr)*np.log1p(-p0)

    for pi,p_true in enumerate(p_values):
        k=np.zeros(reps,dtype=np.int32); stop_n=np.zeros(reps,dtype=np.int32); stop_k=np.zeros(reps,dtype=np.int32)
        active=np.ones(reps,dtype=bool)
        for n in range(1,max_n+1):
            ids=np.flatnonzero(active)
            if len(ids)==0: break
            k[ids]+= (rng.random(len(ids))<p_true).astype(np.int32)
            nn=np.full(len(ids),n,dtype=float); kk=k[ids]
            accept=log_em_vec(nn,kk,reference_p0)>=np.log(10.0)
            reject=log_ep_vec(nn,kk,reference_p0)>=np.log(20.0)
            stop=accept|reject
            if stop.any():
                sids=ids[stop]; stop_n[sids]=n; stop_k[sids]=k[sids]; active[sids]=False
        # 截尾路径在 max_n 时同样可检查任意时刻有效的CS，不把截尾当决策。
        ids=np.flatnonzero(active); stop_n[ids]=max_n; stop_k[ids]=k[ids]
        censor_rate=float(active.mean())
        for conf in confidences:
            loge=log_em_vec(stop_n,stop_k,p_true)
            covered=loge < -np.log(1.0-conf) + 1e-12
            cov=float(np.mean(covered)); se=float(np.sqrt(max(cov*(1-cov),0.0)/reps))
            rows.append({"p_true":p_true,"reference_p0":reference_p0,"confidence":conf,"reps":reps,
                         "validation_max_n":max_n,"censor_rate":censor_rate,"coverage":cov,"mc_se":se,
                         "lower_2se":cov-2*se,"PASS":bool(cov+2*se+1e-12>=conf)})
    return pd.DataFrame(rows)

def q1_sequential_boundary_validation(tol: float=1e-9) -> pd.DataFrame:
    """复核第一问序贯e-process的两个已知停止边界及反演上界。"""
    import math
    e_accept=math.exp(log_e_minus(34,0,0.10))
    e_reject=math.exp(log_e_plus(2,2,0.10))
    u90=sequential_upper(34,0,0.90)
    rows=[
        {"check":"accept_boundary_n34_k0","actual":e_accept,"criterion":10.0,"PASS":bool(e_accept>=10.0-tol)},
        {"check":"reject_boundary_n2_k2","actual":e_reject,"criterion":20.0,"PASS":bool(e_reject>=20.0-tol)},
        {"check":"inverted_U90_n34_k0","actual":u90,"criterion":0.10,"PASS":bool(u90<=0.10+1e-9)},
    ]
    return pd.DataFrame(rows)

def audit_q2_monotonicity(params, upper: Mapping[str,float], grid_points:int=5, tol:float=1e-9)->pd.DataFrame:
    """
    对所有可行 Q2 静态策略做单调性数值审计。
    同时检查“其余质量风险为0”和“其余质量风险取各自上界”两类锚点，
    后者更贴近稳健最坏角点。审计不代替理论证明，但能捕捉实现错误。
    """
    rows=[]
    from .q2_model import enumerate_q2_policies
    for anchor_name in ("zero_others","upper_others"):
        base_anchor={"p1":0.0,"p2":0.0,"pf":0.0} if anchor_name=="zero_others" else {k:float(upper[k]) for k in ("p1","p2","pf")}
        for var in ("p1","p2","pf"):
            vals=np.linspace(0.0,float(upper[var]),grid_points)
            passed=True; max_violation=0.0
            for pi in enumerate_q2_policies():
                prev=None
                for val in vals:
                    q=base_anchor.copy(); q[var]=float(val)
                    pp=params.with_quality(q["p1"],q["p2"],q["pf"])
                    cost,profit,rho,feasible=evaluate_q2_strategy_scalar(pp,pi)
                    profit=profit if feasible else -np.inf
                    if prev is not None and profit>prev+tol:
                        passed=False; max_violation=max(max_violation,profit-prev)
                    prev=profit
            rows.append({"anchor":anchor_name,"parameter":var,"upper":upper[var],"grid_points":grid_points,"max_profit_increase_violation":max_violation,"PASS":passed})
    return pd.DataFrame(rows)


def audit_q3_monotonicity(upper: Mapping[str,float], grid_points:int=5, tol:float=1e-9)->pd.DataFrame:
    """
    对全部65536静态策略做向量化单调性核验；检查两类锚点。
    若任何策略在某质量参数增加时利润反而显著上升，稳健求解不应静默使用上界角点。
    """
    rows=[]
    for anchor_name in ("zero_others","upper_others"):
        base_anchor={k:0.0 for k in Q3_NODES} if anchor_name=="zero_others" else {k:float(upper[k]) for k in Q3_NODES}
        for var in Q3_NODES:
            vals=np.linspace(0.0,float(upper[var]),grid_points)
            prev=None; passed=True; maxv=0.0
            for val in vals:
                q=base_anchor.copy(); q[var]=float(val)
                prof,_=all_profit_vector(q)
                if prev is not None:
                    diff=prof-prev
                    finite=np.isfinite(diff)
                    if finite.any():
                        mv=float(np.max(diff[finite]))
                        if mv>tol:
                            passed=False; maxv=max(maxv,mv)
                prev=prof
            rows.append({"anchor":anchor_name,"parameter":var,"upper":upper[var],"grid_points":grid_points,"max_profit_increase_violation":maxv,"PASS":passed})
    return pd.DataFrame(rows)

