from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Q2_SCENARIOS, Q3_NODES, Q3_REPLACEMENT_LOSS
from .q2_model import solve_q2, solve_q2_robust, solve_q2_robust_corners
from .q3_model import evaluate_q3_strategy_detailed, solve_q3_fast, solve_q3_flat, solve_q3_robust_corners
from .q4_uncertainty import (
    SampleRecord,
    bootstrap_record,
    build_interval_table,
)
from .q4_validation import audit_q2_monotonicity, audit_q3_monotonicity


def _map_q2_quality(records: Sequence[SampleRecord], sid: int) -> Dict[str, SampleRecord]:
    ids={f"Q2-S{sid}-C1":"p1",f"Q2-S{sid}-C2":"p2",f"Q2-S{sid}-F":"pf"}
    rmap={r.quality_id:r for r in records}
    missing=[x for x in ids if x not in rmap]
    if missing:
        raise KeyError(f"Q2 scenario {sid} missing records: {missing}")
    return {param:rmap[qid] for qid,param in ids.items()}


def _map_q3_quality(records: Sequence[SampleRecord]) -> Dict[str, SampleRecord]:
    ids=[f"Q3-C{i}" for i in range(1,9)]+["Q3-H1","Q3-H2","Q3-H3","Q3-F"]
    rmap={r.quality_id:r for r in records}
    missing=[x for x in ids if x not in rmap]
    if missing:
        raise KeyError(f"Q3 missing records: {missing}")
    return {qid.replace("Q3-",""):rmap[qid] for qid in ids}


def _interval_vectors(mapped: Mapping[str,SampleRecord]) -> Tuple[pd.DataFrame, Dict[str,float], Dict[str,float], Dict[str,float]]:
    recs=list(mapped.values())
    tab=build_interval_table(recs)
    byid=tab.set_index("quality_id")
    inv={r.quality_id:key for key,r in mapped.items()}
    phat={inv[qid]:float(row["p_hat"]) for qid,row in byid.iterrows()}
    u90={inv[qid]:float(row["U_simultaneous_90"]) for qid,row in byid.iterrows()}
    u95={inv[qid]:float(row["U_simultaneous_95"]) for qid,row in byid.iterrows()}
    return tab,phat,u90,u95


def solve_q4_q2(records: Sequence[SampleRecord], topk:int=10):
    interval_rows=[]; result_rows=[]; top_rows=[]; mono_rows=[]
    for sid,base in Q2_SCENARIOS.items():
        try:
            mapped=_map_q2_quality(records,sid)
        except KeyError:
            continue
        tab,phat,u90,u95=_interval_vectors(mapped)
        tab.insert(0,"scope",f"Q2-S{sid}"); interval_rows.append(tab)

        # 名义点估计只用于对照，不冒充真实参数。
        pp=base.with_quality(phat["p1"],phat["p2"],phat["pf"])
        best,rank=solve_q2(pp,topk=topk)
        result_rows.append({"scenario_id":sid,"risk_level":"nominal","robust_method":"point_estimate_reference",
                            "policy4":best.code4,"policy6":best.code6,"expected_cost":best.expected_cost,"expected_profit":best.expected_profit,
                            "gap":float(rank.iloc[0]["primary_policy_gap"])})
        r2=rank.copy(); r2.insert(0,"robust_method","point_estimate_reference"); r2.insert(0,"risk_level","nominal"); r2.insert(0,"scenario_id",sid); top_rows.append(r2)

        for gamma,u,risk in [(0.90,u90,"robust90"),(0.95,u95,"robust95")]:
            m=audit_q2_monotonicity(base,u); m.insert(0,"confidence",gamma); m.insert(0,"scope",f"Q2-S{sid}"); mono_rows.append(m)
            mono_ok=bool(m["PASS"].all())
            if mono_ok:
                best,rank=solve_q2_robust(base,u,topk=topk)
                method="upper_corner_by_monotonicity"
            else:
                # 按手册回退到完整 2^3 盒端点搜索，禁止静默沿用上界捷径。
                best,rank=solve_q2_robust_corners(base,u,topk=topk)
                method="8_corner_fallback"
            gap=float(rank.iloc[0]["primary_policy_gap"])
            result_rows.append({"scenario_id":sid,"risk_level":risk,"robust_method":method,
                                "policy4":best.code4,"policy6":best.code6,"expected_cost":best.expected_cost,"expected_profit":best.expected_profit,"gap":gap})
            r2=rank.copy(); r2.insert(0,"robust_method",method); r2.insert(0,"risk_level",risk); r2.insert(0,"scenario_id",sid); top_rows.append(r2)
    return (
        pd.concat(interval_rows,ignore_index=True) if interval_rows else pd.DataFrame(),
        pd.DataFrame(result_rows),
        pd.concat(top_rows,ignore_index=True) if top_rows else pd.DataFrame(),
        pd.concat(mono_rows,ignore_index=True) if mono_rows else pd.DataFrame(),
    )


def solve_q4_q3(records: Sequence[SampleRecord], topk:int=10, flat_cross_check:bool=True):
    mapped=_map_q3_quality(records)
    tab,phat,u90,u95=_interval_vectors(mapped)
    tab.insert(0,"scope","Q3")
    rows=[]; tops=[]; mono=[]

    # 名义点估计参考解。
    code,profit,top=solve_q3_fast(phat,topk=topk)
    detail=evaluate_q3_strategy_detailed(code,phat)
    row={"risk_level":"nominal","robust_method":"point_estimate_reference","policy16":code,
         "expected_cost":detail.expected_cost,"expected_profit":profit,"gap":float(top.iloc[0]["best_second_gap"])}
    if flat_cross_check:
        fcode,fprofit=solve_q3_flat(phat)
        row.update({"flat_policy16":fcode,"flat_profit":fprofit,"flat_abs_error":abs(fprofit-profit),"crosscheck_PASS":bool(fcode==code and abs(fprofit-profit)<1e-9)})
    rows.append(row); t=top.copy(); t.insert(0,"robust_method","point_estimate_reference"); t.insert(0,"risk_level","nominal"); tops.append(t)

    for gamma,u,risk in [(0.90,u90,"robust90"),(0.95,u95,"robust95")]:
        m=audit_q3_monotonicity(u); m.insert(0,"confidence",gamma); mono.append(m)
        mono_ok=bool(m["PASS"].all())
        if mono_ok:
            code,profit,top=solve_q3_fast(u,topk=topk)
            method="upper_corner_by_monotonicity"
        else:
            # 手册要求单调性失败时切换到盒端点直接搜索：4096角点 × 65536策略由向量化评价完成。
            code,profit,top=solve_q3_robust_corners(u,topk=topk)
            method="4096_corner_fallback"
        detail=evaluate_q3_strategy_detailed(code,u if mono_ok else {k:float(top.iloc[0].get(f"worst_{k}",u[k])) for k in Q3_NODES})
        row={"risk_level":risk,"robust_method":method,"policy16":code,"expected_cost":float(200-profit),"expected_profit":profit,"gap":float(top.iloc[0]["best_second_gap"])}
        if flat_cross_check and mono_ok:
            fcode,fprofit=solve_q3_flat(u)
            row.update({"flat_policy16":fcode,"flat_profit":fprofit,"flat_abs_error":abs(fprofit-profit),"crosscheck_PASS":bool(fcode==code and abs(fprofit-profit)<1e-9)})
        elif flat_cross_check:
            row.update({"flat_policy16":"not_applicable_corner_robust","flat_profit":np.nan,"flat_abs_error":np.nan,"crosscheck_PASS":np.nan})
        rows.append(row); t=top.copy(); t.insert(0,"robust_method",method); t.insert(0,"risk_level",risk); tops.append(t)
    return tab,pd.DataFrame(rows),pd.concat(tops,ignore_index=True),pd.concat(mono,ignore_index=True)


def _bootstrap_once_records(records:Sequence[SampleRecord], rng:np.random.Generator, max_seq_n:int)->Tuple[List[SampleRecord],int]:
    out=[]; cens=0
    for r in records:
        rr,c=bootstrap_record(r,rng,max_seq_n=max_seq_n)
        out.append(rr); cens+=int(c)
    return out,cens


def bootstrap_stability_q2(
    records:Sequence[SampleRecord], sid:int, seed:int=20260820, start_B:int=64, max_B:int=512, max_seq_n:int=10000
)->pd.DataFrame:
    mapped=_map_q2_quality(records,sid); base_records=list(mapped.values()); base=Q2_SCENARIOS[sid]
    rng=np.random.default_rng(seed+sid)
    targets=[]; counts={"nominal":Counter(),"robust90":Counter(),"robust95":Counter()}; profits={k:[] for k in counts}
    prev={}; total=0; censored=0
    checkpoint=start_B
    while total<max_B:
        add=min(checkpoint-total,max_B-total)
        for _ in range(add):
            boot,c=_bootstrap_once_records(base_records,rng,max_seq_n); censored+=c
            bm={r.quality_id:r for r in boot}
            mapped_b={k:bm[v.quality_id] for k,v in mapped.items()}
            _,ph,u90,u95=_interval_vectors(mapped_b)
            for risk,q in [("nominal",ph),("robust90",u90),("robust95",u95)]:
                if risk=="nominal": best,_=solve_q2(base.with_quality(q["p1"],q["p2"],q["pf"]),topk=2)
                else: best,_=solve_q2_robust(base,q,topk=2)
                counts[risk][best.code6]+=1; profits[risk].append(best.expected_profit)
        total+=add
        converged=True if total>=2*start_B else False
        for risk in counts:
            code,nc=counts[risk].most_common(1)[0]; rate=nc/total; se=(rate*(1-rate)/total)**0.5
            targets.append({"scope":f"Q2-S{sid}","risk_level":risk,"B":total,"modal_policy":code,"stability_rate":rate,"mc_se":se,"mean_optimal_profit":float(np.mean(profits[risk])),"sequential_censored_events":censored})
            if risk in prev:
                pcode,prate,pse=prev[risk]
                if code!=pcode or abs(rate-prate)>2*((se**2+pse**2)**0.5): converged=False
            else: converged=False
            prev[risk]=(code,rate,se)
        if converged: break
        checkpoint=min(max_B, max(checkpoint*2,total+1))
    return pd.DataFrame(targets)


def bootstrap_stability_q3(
    records:Sequence[SampleRecord], seed:int=20260820, start_B:int=64, max_B:int=512, max_seq_n:int=10000
)->pd.DataFrame:
    mapped=_map_q3_quality(records); base_records=list(mapped.values())
    rng=np.random.default_rng(seed+300)
    counts={"nominal":Counter(),"robust90":Counter(),"robust95":Counter()}; profits={k:[] for k in counts}
    rows=[]; prev={}; total=0; censored=0; checkpoint=start_B
    while total<max_B:
        add=min(checkpoint-total,max_B-total)
        for _ in range(add):
            boot,c=_bootstrap_once_records(base_records,rng,max_seq_n); censored+=c
            bm={r.quality_id:r for r in boot}; mapped_b={k:bm[v.quality_id] for k,v in mapped.items()}
            _,ph,u90,u95=_interval_vectors(mapped_b)
            for risk,q in [("nominal",ph),("robust90",u90),("robust95",u95)]:
                code,profit,_=solve_q3_fast(q,topk=2)
                counts[risk][code]+=1; profits[risk].append(profit)
        total+=add
        converged=True if total>=2*start_B else False
        for risk in counts:
            code,nc=counts[risk].most_common(1)[0]; rate=nc/total; se=(rate*(1-rate)/total)**0.5
            rows.append({"scope":"Q3","risk_level":risk,"B":total,"modal_policy":code,"stability_rate":rate,"mc_se":se,"mean_optimal_profit":float(np.mean(profits[risk])),"sequential_censored_events":censored})
            if risk in prev:
                pcode,prate,pse=prev[risk]
                if code!=pcode or abs(rate-prate)>2*((se**2+pse**2)**0.5): converged=False
            else: converged=False
            prev[risk]=(code,rate,se)
        if converged: break
        checkpoint=min(max_B,max(checkpoint*2,total+1))
    return pd.DataFrame(rows)


def information_sufficiency_q2(records:Sequence[SampleRecord], factors=(1,2,4,8,16))->pd.DataFrame:
    """Q2固定样本的信息量设计投影；只回答继续补测时区间/策略如何收敛，不冒充真实追加观测。"""
    rows=[]
    for sid,base in Q2_SCENARIOS.items():
        try:
            mapped=_map_q2_quality(records,sid)
        except KeyError:
            continue
        if any(r.sample_design!="fixed" for r in mapped.values()):
            rows.append({"scope":f"Q2-S{sid}","status":"skipped: sequential records require actual continued sampling"})
            continue
        for factor in factors:
            projected={}
            for key,r in mapped.items():
                if factor==1: rr=r
                else:
                    n2=int(round(r.n*factor)); k2=int(round(r.p_hat*n2)); rr=r.with_counts(n2,k2)
                projected[key]=rr
            _,ph,u90,u95=_interval_vectors(projected)
            for risk,q in [("nominal",ph),("robust90",u90),("robust95",u95)]:
                if risk=="nominal": best,top=solve_q2(base.with_quality(q["p1"],q["p2"],q["pf"]),topk=3)
                else: best,top=solve_q2_robust(base,q,topk=3)
                rows.append({"scope":f"Q2-S{sid}","sample_multiplier":factor,
                             "projection_assumption":"future sample fraction equals current p_hat","risk_level":risk,
                             "policy4":best.code4,"policy6":best.code6,"profit":best.expected_profit,
                             "gap":float(top.iloc[0]["primary_policy_gap"]),
                             "mean_upper_minus_phat":float(np.mean([q[k]-ph[k] for k in ph]))})
    return pd.DataFrame(rows)


def information_sufficiency_q3(records:Sequence[SampleRecord], factors=(1,2,4,8,16))->pd.DataFrame:
    """固定样本的设计投影：未来样本比例保持当前 p_hat；明确是信息量敏感性，不冒充实际观测。"""
    mapped=_map_q3_quality(records)
    if any(r.sample_design!="fixed" for r in mapped.values()):
        return pd.DataFrame([{"scope":"Q3","status":"skipped: sequential records require actual continued sampling"}])
    rows=[]
    for factor in factors:
        projected={}
        for key,r in mapped.items():
            if factor==1:
                rr=r
            else:
                n2=int(round(r.n*factor)); k2=int(round(r.p_hat*n2)); rr=r.with_counts(n2,k2)
            projected[key]=rr
        _,ph,u90,u95=_interval_vectors(projected)
        for risk,q in [("nominal",ph),("robust90",u90),("robust95",u95)]:
            code,profit,top=solve_q3_fast(q,topk=2)
            rows.append({"scope":"Q3","sample_multiplier":factor,"projection_assumption":"future sample fraction equals current p_hat","risk_level":risk,"policy16":code,"profit":profit,"gap":float(top.iloc[0]["best_second_gap"]),"mean_upper_minus_phat":float(np.mean([q[k]-ph[k] for k in ph]))})
    return pd.DataFrame(rows)
