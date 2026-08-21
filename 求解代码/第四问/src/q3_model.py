from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math
import numpy as np
import pandas as pd

from .config import Q3_NODES, Q3_REPLACEMENT_LOSS, Q3_SALE_PRICE, Q3Node

Q3_CATEGORIES = (
    "purchase",
    "component_inspection",
    "assembly",
    "intermediate_inspection",
    "final_inspection",
    "replacement",
    "disassembly",
)


@dataclass(frozen=True)
class ScalarSummary:
    fresh_cost: float
    q: float
    known: bool
    certified_cost: float
    repair_cost: float
    inspection_cost: float
    kind: str


@dataclass(frozen=True)
class Q3Eval:
    policy: Tuple[int, ...]
    expected_cost: float
    expected_profit: float
    cost_breakdown: Dict[str, float]

    @property
    def code16(self) -> str:
        return "".join(map(str, self.policy))


def policy_from_code(code: str) -> Tuple[int, ...]:
    if len(code) != 16 or any(c not in "01" for c in code):
        raise ValueError("Q3策略编码必须是16位0/1")
    return tuple(map(int, code))


def _pmap(quality: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
    p = {name: node.defect_rate for name, node in Q3_NODES.items()}
    if quality:
        for k, v in quality.items():
            if k not in p:
                raise KeyError(f"unknown Q3 quality node: {k}")
            p[k] = float(v)
    for k, v in p.items():
        if not (0 <= v < 1):
            raise ValueError(f"Q3 {k} defect_rate={v} 不在[0,1)；上界=1时无法保证完成订单")
    return p


def _unknown_recovery_scalar(child: ScalarSummary, parent_g: float) -> float:
    if child.known:
        return 0.0
    denom = 1.0 - parent_g
    if denom <= 1e-15:
        return 0.0
    bad_post = (1.0 - child.q) / denom
    bad_post = min(max(bad_post, 0.0), 1.0)
    return child.inspection_cost + bad_post * child.repair_cost


def _build_half_candidates(
    half_name: str,
    child_names: Tuple[str, ...],
    p: Mapping[str, float],
) -> List[Tuple[Tuple[int, ...], int, int, ScalarSummary]]:
    node = Q3_NODES[half_name]
    out = []
    for xbits in product((0, 1), repeat=len(child_names)):
        children = []
        for x, cname in zip(xbits, child_names):
            c = Q3_NODES[cname]
            pc = p[cname]
            K = (c.purchase_or_assembly_cost + c.inspection_cost) / (1.0 - pc)
            if x:
                s = ScalarSummary(K, 1.0, True, K, K, c.inspection_cost, c.kind)
            else:
                s = ScalarSummary(c.purchase_or_assembly_cost, 1.0 - pc, False, K, K, c.inspection_cost, c.kind)
            children.append(s)
        B = node.purchase_or_assembly_cost + sum(c.fresh_cost for c in children)
        g = (1.0 - p[half_name]) * math.prod(c.q for c in children)
        for y in (0, 1):
            for z in (0, 1):
                if z == 0:
                    K = (B + node.inspection_cost) / g
                    D = K
                else:
                    pp = p[half_name]
                    Kproc = (
                        node.purchase_or_assembly_cost
                        + node.inspection_cost
                        + pp * node.disassembly_cost
                    ) / (1.0 - pp)
                    D = node.disassembly_cost + Kproc
                    D += sum(_unknown_recovery_scalar(c, g) for c in children)
                    K = B + node.inspection_cost + (1.0 - g) * D
                if y:
                    s = ScalarSummary(K, 1.0, True, K, D, node.inspection_cost, node.kind)
                else:
                    s = ScalarSummary(B, g, False, K, D, node.inspection_cost, node.kind)
                out.append((xbits, y, z, s))
    return out


@lru_cache(maxsize=1)
def _cached_policy_codes() -> Tuple[str, ...]:
    """Q3 的 65536 个静态策略编码顺序只由拓扑/枚举顺序决定，与质量参数无关。"""
    A_bits = [(xbits, y, z) for xbits in product((0, 1), repeat=3) for y in (0, 1) for z in (0, 1)]
    B_bits = [(xbits, y, z) for xbits in product((0, 1), repeat=3) for y in (0, 1) for z in (0, 1)]
    C_bits = [(xbits, y, z) for xbits in product((0, 1), repeat=2) for y in (0, 1) for z in (0, 1)]
    codes=[]
    for yF in (0,1):
        for zF in (0,1):
            for a in A_bits:
                for b in B_bits:
                    for c in C_bits:
                        bits = a[0] + b[0] + c[0] + (a[1], b[1], c[1], a[2], b[2], c[2], yF, zF)
                        codes.append("".join(map(str,bits)))
    return tuple(codes)


def _all_profit_vector(
    quality: Optional[Mapping[str, float]] = None,
    replacement_loss: float = Q3_REPLACEMENT_LOSS,
    sale_price: float = Q3_SALE_PRICE,
) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """分层候选 + NumPy 根节点向量化，精确覆盖 2^16 静态策略。"""
    p = _pmap(quality)
    A = _build_half_candidates("H1", ("C1", "C2", "C3"), p)
    B = _build_half_candidates("H2", ("C4", "C5", "C6"), p)
    C = _build_half_candidates("H3", ("C7", "C8"), p)

    def arr(cands, attr):
        return np.array([getattr(x[3], attr) for x in cands], dtype=float)

    cf1, q1, k1, d1 = arr(A, "fresh_cost"), arr(A, "q"), arr(A, "known"), arr(A, "repair_cost")
    cf2, q2, k2, d2 = arr(B, "fresh_cost"), arr(B, "q"), arr(B, "known"), arr(B, "repair_cost")
    cf3, q3, k3, d3 = arr(C, "fresh_cost"), arr(C, "q"), arr(C, "known"), arr(C, "repair_cost")
    k1, k2, k3 = k1.astype(bool), k2.astype(bool), k3.astype(bool)

    F = Q3_NODES["F"]
    pF = p["F"]
    Base = (
        F.purchase_or_assembly_cost
        + cf1[:, None, None]
        + cf2[None, :, None]
        + cf3[None, None, :]
    )
    g = (1.0 - pF) * q1[:, None, None] * q2[None, :, None] * q3[None, None, :]
    fail = 1.0 - g

    # 父节点失败后，未知子节点的坏品后验为 (1-q_child)/(1-g)。
    denom = np.where(fail > 1e-15, fail, 1.0)
    rec1 = np.where(k1[:, None, None], 0.0, Q3_NODES["H1"].inspection_cost + ((1-q1[:,None,None])/denom)*d1[:,None,None])
    rec2 = np.where(k2[None, :, None], 0.0, Q3_NODES["H2"].inspection_cost + ((1-q2[None,:,None])/denom)*d2[None,:,None])
    rec3 = np.where(k3[None, None, :], 0.0, Q3_NODES["H3"].inspection_cost + ((1-q3[None,None,:])/denom)*d3[None,None,:])
    rec1 = np.where(fail > 1e-15, rec1, 0.0)
    rec2 = np.where(fail > 1e-15, rec2, 0.0)
    rec3 = np.where(fail > 1e-15, rec3, 0.0)

    profits = []
    for yF in (0, 1):
        for zF in (0, 1):
            if yF:
                if zF == 0:
                    V = (Base + F.inspection_cost) / g
                else:
                    Kproc = (
                        F.purchase_or_assembly_cost
                        + F.inspection_cost
                        + pF * F.disassembly_cost
                    ) / (1.0 - pF)
                    D = F.disassembly_cost + Kproc + rec1 + rec2 + rec3
                    V = Base + F.inspection_cost + fail * D
            else:
                if zF == 0:
                    V = (Base + fail * replacement_loss) / g
                else:
                    M = (
                        F.purchase_or_assembly_cost
                        + pF * (replacement_loss + F.disassembly_cost)
                    ) / (1.0 - pF)
                    V = Base + fail * (
                        replacement_loss + F.disassembly_cost + rec1 + rec2 + rec3 + M
                    )
            profits.append((sale_price - V).ravel())
    return np.concatenate(profits), _cached_policy_codes()


def solve_q3_fast(
    quality: Optional[Mapping[str, float]] = None,
    replacement_loss: float = Q3_REPLACEMENT_LOSS,
    sale_price: float = Q3_SALE_PRICE,
    topk: int = 10,
) -> Tuple[str, float, pd.DataFrame]:
    profits, codes = _all_profit_vector(quality, replacement_loss, sale_price)
    if not np.isfinite(profits).any():
        raise RuntimeError("Q3 无有限期望成本的静态策略")
    idx = np.argpartition(profits, -min(topk, len(profits)))[-min(topk, len(profits)):]
    idx = idx[np.argsort(profits[idx])[::-1]]
    rows = []
    best_profit = float(profits[idx[0]])
    second = float(profits[idx[1]]) if len(idx) > 1 else np.nan
    for rank, j in enumerate(idx, 1):
        code = codes[int(j)]
        rows.append(
            {
                "rank": rank,
                "policy16": code,
                "expected_profit": float(profits[j]),
                "expected_cost": float(sale_price - profits[j]),
                "gap_from_best": best_profit - float(profits[j]),
                "best_second_gap": best_profit - second if rank == 1 else np.nan,
            }
        )
    return codes[int(idx[0])], best_profit, pd.DataFrame(rows)


def evaluate_q3_strategy_scalar(
    policy: Tuple[int, ...],
    quality: Optional[Mapping[str, float]] = None,
    replacement_loss: float = Q3_REPLACEMENT_LOSS,
    sale_price: float = Q3_SALE_PRICE,
) -> Tuple[float, float]:
    """独立的逐策略标量实现，用于平坦65536交叉核验。"""
    if len(policy) != 16:
        raise ValueError("policy length must be 16")
    p = _pmap(quality)
    xs = policy[:8]
    ys = policy[8:11]
    zs = policy[11:14]
    yF, zF = policy[14], policy[15]
    sm: Dict[str, ScalarSummary] = {}
    for i in range(1, 9):
        name = f"C{i}"
        n = Q3_NODES[name]
        pp = p[name]
        K = (n.purchase_or_assembly_cost + n.inspection_cost) / (1.0 - pp)
        if xs[i - 1]:
            sm[name] = ScalarSummary(K, 1, True, K, K, n.inspection_cost, n.kind)
        else:
            sm[name] = ScalarSummary(n.purchase_or_assembly_cost, 1-pp, False, K, K, n.inspection_cost, n.kind)
    for j, h in enumerate(("H1", "H2", "H3")):
        n = Q3_NODES[h]
        ch = [sm[c] for c in n.children]
        B = n.purchase_or_assembly_cost + sum(c.fresh_cost for c in ch)
        g = (1-p[h]) * math.prod(c.q for c in ch)
        if zs[j] == 0:
            K = (B + n.inspection_cost) / g
            D = K
        else:
            Kproc = (n.purchase_or_assembly_cost+n.inspection_cost+p[h]*n.disassembly_cost)/(1-p[h])
            D = n.disassembly_cost + Kproc + sum(_unknown_recovery_scalar(c, g) for c in ch)
            K = B + n.inspection_cost + (1-g)*D
        if ys[j]:
            sm[h] = ScalarSummary(K,1,True,K,D,n.inspection_cost,n.kind)
        else:
            sm[h] = ScalarSummary(B,g,False,K,D,n.inspection_cost,n.kind)
    F=Q3_NODES["F"]
    ch=[sm[c] for c in F.children]
    B=F.purchase_or_assembly_cost+sum(c.fresh_cost for c in ch)
    g=(1-p["F"])*math.prod(c.q for c in ch)
    if yF:
        if zF==0:
            V=(B+F.inspection_cost)/g
        else:
            Kproc=(F.purchase_or_assembly_cost+F.inspection_cost+p["F"]*F.disassembly_cost)/(1-p["F"])
            D=F.disassembly_cost+Kproc+sum(_unknown_recovery_scalar(c,g) for c in ch)
            V=B+F.inspection_cost+(1-g)*D
    else:
        if zF==0:
            V=(B+(1-g)*replacement_loss)/g
        else:
            M=(F.purchase_or_assembly_cost+p["F"]*(replacement_loss+F.disassembly_cost))/(1-p["F"])
            V=B+(1-g)*(replacement_loss+F.disassembly_cost+sum(_unknown_recovery_scalar(c,g) for c in ch)+M)
    return float(sale_price-V), float(V)


def solve_q3_flat(
    quality: Optional[Mapping[str, float]] = None,
    replacement_loss: float = Q3_REPLACEMENT_LOSS,
    sale_price: float = Q3_SALE_PRICE,
) -> Tuple[str, float]:
    best_code, best_profit = None, -np.inf
    for bits in product((0,1), repeat=16):
        profit, _ = evaluate_q3_strategy_scalar(bits, quality, replacement_loss, sale_price)
        if profit > best_profit:
            best_profit, best_code = profit, "".join(map(str,bits))
    return str(best_code), float(best_profit)


def _vec_zero() -> np.ndarray:
    return np.zeros(len(Q3_CATEGORIES), dtype=float)


def _vec_add(v: np.ndarray, category: str, amount: float) -> np.ndarray:
    out = v.copy()
    out[Q3_CATEGORIES.index(category)] += amount
    return out


def _inspection_category(kind: str) -> str:
    if kind == "component":
        return "component_inspection"
    if kind == "intermediate":
        return "intermediate_inspection"
    return "final_inspection"


@dataclass(frozen=True)
class VecSummary:
    fresh: np.ndarray
    q: float
    known: bool
    K: np.ndarray
    D: np.ndarray
    inspection_cost: float
    kind: str


def evaluate_q3_strategy_detailed(
    code: str,
    quality: Optional[Mapping[str,float]] = None,
    replacement_loss: float = Q3_REPLACEMENT_LOSS,
    sale_price: float = Q3_SALE_PRICE,
) -> Q3Eval:
    """向量奖励版递推，只用于最优/Top策略成本分解。"""
    pi=policy_from_code(code)
    p=_pmap(quality)
    xs=pi[:8]; ys=pi[8:11]; zs=pi[11:14]; yF,zF=pi[14],pi[15]
    sm: Dict[str, VecSummary]={}
    for i in range(1,9):
        name=f"C{i}"; n=Q3_NODES[name]; pp=p[name]
        K=_vec_zero(); K=_vec_add(K,"purchase",n.purchase_or_assembly_cost/(1-pp)); K=_vec_add(K,"component_inspection",n.inspection_cost/(1-pp))
        if xs[i-1]: fresh,q,known=K.copy(),1.0,True
        else:
            fresh=_vec_add(_vec_zero(),"purchase",n.purchase_or_assembly_cost); q,known=1-pp,False
        sm[name]=VecSummary(fresh,q,known,K,K.copy(),n.inspection_cost,n.kind)

    def rec_vec(child:VecSummary,g:float)->np.ndarray:
        if child.known or 1-g<=1e-15:
            return _vec_zero()
        bad=(1-child.q)/(1-g); bad=min(max(bad,0),1)
        r=_vec_add(_vec_zero(),_inspection_category(child.kind),child.inspection_cost)
        return r+bad*child.D

    for j,h in enumerate(("H1","H2","H3")):
        n=Q3_NODES[h]; ch=[sm[c] for c in n.children]
        B=sum((c.fresh for c in ch),_vec_zero()); B=_vec_add(B,"assembly",n.purchase_or_assembly_cost)
        g=(1-p[h])*math.prod(c.q for c in ch)
        bvec=_vec_add(_vec_zero(),"intermediate_inspection",n.inspection_cost)
        if zs[j]==0:
            K=(B+bvec)/g; D=K.copy()
        else:
            Kproc=_vec_zero(); Kproc=_vec_add(Kproc,"assembly",n.purchase_or_assembly_cost); Kproc=_vec_add(Kproc,"intermediate_inspection",n.inspection_cost); Kproc=_vec_add(Kproc,"disassembly",p[h]*n.disassembly_cost); Kproc=Kproc/(1-p[h])
            D=_vec_add(_vec_zero(),"disassembly",n.disassembly_cost)+Kproc+sum((rec_vec(c,g) for c in ch),_vec_zero())
            K=B+bvec+(1-g)*D
        if ys[j]: sm[h]=VecSummary(K.copy(),1,True,K,D,n.inspection_cost,n.kind)
        else: sm[h]=VecSummary(B.copy(),g,False,K,D,n.inspection_cost,n.kind)

    F=Q3_NODES["F"]; ch=[sm[c] for c in F.children]
    B=sum((c.fresh for c in ch),_vec_zero()); B=_vec_add(B,"assembly",F.purchase_or_assembly_cost)
    g=(1-p["F"])*math.prod(c.q for c in ch)
    if yF:
        bvec=_vec_add(_vec_zero(),"final_inspection",F.inspection_cost)
        if zF==0: V=(B+bvec)/g
        else:
            Kproc=_vec_zero(); Kproc=_vec_add(Kproc,"assembly",F.purchase_or_assembly_cost); Kproc=_vec_add(Kproc,"final_inspection",F.inspection_cost); Kproc=_vec_add(Kproc,"disassembly",p["F"]*F.disassembly_cost); Kproc=Kproc/(1-p["F"])
            D=_vec_add(_vec_zero(),"disassembly",F.disassembly_cost)+Kproc+sum((rec_vec(c,g) for c in ch),_vec_zero())
            V=B+bvec+(1-g)*D
    else:
        if zF==0:
            V=B/g
            V=_vec_add(V,"replacement",(1-g)*replacement_loss/g)
        else:
            M=_vec_zero(); M=_vec_add(M,"assembly",F.purchase_or_assembly_cost); M=_vec_add(M,"replacement",p["F"]*replacement_loss); M=_vec_add(M,"disassembly",p["F"]*F.disassembly_cost); M=M/(1-p["F"])
            branch=_vec_zero(); branch=_vec_add(branch,"replacement",replacement_loss); branch=_vec_add(branch,"disassembly",F.disassembly_cost)
            V=B+(1-g)*(branch+sum((rec_vec(c,g) for c in ch),_vec_zero())+M)
    cost=float(V.sum()); br={k:float(v) for k,v in zip(Q3_CATEGORIES,V)}
    return Q3Eval(pi,cost,float(sale_price-cost),br)


def all_profit_vector(quality=None,replacement_loss=Q3_REPLACEMENT_LOSS,sale_price=Q3_SALE_PRICE):
    return _all_profit_vector(quality,replacement_loss,sale_price)


def solve_q3_robust_corners(
    upper: Mapping[str,float],
    replacement_loss: float = Q3_REPLACEMENT_LOSS,
    sale_price: float = Q3_SALE_PRICE,
    topk: int = 10,
) -> Tuple[str,float,pd.DataFrame]:
    """
    单调性审计失败时的端点回退：精确扫描 12维盒的 2^12=4096 个角点，
    并同时维护全部 65536 个静态策略在角点集合上的最坏利润。
    注意：这是建模手册所述“盒内端点”回退；若扩展模型可能存在内点极值，应另加连续数值最小化。
    """
    names=tuple(Q3_NODES.keys())
    if set(upper) != set(names):
        missing=set(names)-set(upper); extra=set(upper)-set(names)
        raise KeyError(f"Q3 robust corner upper keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    if any(not (0 <= float(upper[k]) < 1) for k in names):
        raise RuntimeError("Q3 某质量上界达到1，端点回退无法保证有限交付成本")
    worst=None; worst_corner_index=None; codes=None
    for idx,bits in enumerate(product((0,1),repeat=len(names))):
        q={k:(float(upper[k]) if bit else 0.0) for k,bit in zip(names,bits)}
        prof,cc=all_profit_vector(q,replacement_loss,sale_price)
        if worst is None:
            worst=prof.copy(); worst_corner_index=np.zeros(len(prof),dtype=np.int32); codes=cc
        else:
            mask=prof<worst
            worst[mask]=prof[mask]; worst_corner_index[mask]=idx
    assert worst is not None and codes is not None and worst_corner_index is not None
    kk=min(topk,len(worst)); ids=np.argpartition(worst,-kk)[-kk:]; ids=ids[np.argsort(worst[ids])[::-1]]
    best=float(worst[ids[0]]); second=float(worst[ids[1]]) if len(ids)>1 else np.nan
    rows=[]
    for rank,j in enumerate(ids,1):
        ci=int(worst_corner_index[j]); bits=tuple((ci >> (len(names)-1-i)) & 1 for i in range(len(names)))
        corner={k:(float(upper[k]) if bit else 0.0) for k,bit in zip(names,bits)}
        row={"rank":rank,"policy16":codes[int(j)],"expected_profit":float(worst[j]),"expected_cost":float(sale_price-worst[j]),
             "gap_from_best":best-float(worst[j]),"best_second_gap":best-second if rank==1 else np.nan,"worst_corner_index":ci}
        row.update({f"worst_{k}":corner[k] for k in names}); rows.append(row)
    return codes[int(ids[0])],best,pd.DataFrame(rows)
