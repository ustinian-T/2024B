from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import Q2Params

STATES = ["N"] + list(product(("K", "UG", "UB"), repeat=2))
STATE_INDEX = {s: i for i, s in enumerate(STATES)}
COST_CATEGORIES = (
    "purchase",
    "component_inspection",
    "assembly",
    "final_inspection",
    "replacement",
    "disassembly",
)


@dataclass(frozen=True)
class Q2Eval:
    policy: Tuple[int, int, int, int, int, int]
    expected_cost: float
    expected_profit: float
    spectral_radius: float
    feasible: bool
    cost_breakdown: Dict[str, float]

    @property
    def code6(self) -> str:
        return "".join(map(str, self.policy))

    @property
    def code4(self) -> str:
        x1, x2, y, z, _, _ = self.policy
        return f"{x1}{x2}{y}{z}"


def _validate_probabilities(params: Q2Params) -> None:
    for name in ("p1", "p2", "pf"):
        p = float(getattr(params, name))
        if not (0.0 <= p < 1.0):
            raise ValueError(f"Q2 {name}={p} 不在 [0,1)；若上置信界=1，则稳健完成订单不可保证")


def _component_acquire_breakdown(a: float, b: float, p: float) -> Tuple[float, float]:
    return a / (1.0 - p), b / (1.0 - p)


def build_markov_reward(
    params: Q2Params,
    policy: Tuple[int, int, int, int, int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造暂态 Q、总即时成本 c、分类即时成本 Ccat。"""
    _validate_probabilities(params)
    x1, x2, y, z, r1, r2 = policy
    Q = np.zeros((10, 10), dtype=float)
    Ccat = np.zeros((10, len(COST_CATEGORIES)), dtype=float)
    cat = {name: i for i, name in enumerate(COST_CATEGORIES)}

    # 起始状态 N：首次检测只由 x_i 决定，不能误用回收规则 r_i。
    dist1 = [("K", 1.0)] if x1 else [("UG", 1 - params.p1), ("UB", params.p1)]
    dist2 = [("K", 1.0)] if x2 else [("UG", 1 - params.p2), ("UB", params.p2)]
    if x1:
        pa, ib = _component_acquire_breakdown(params.a1, params.b1, params.p1)
        Ccat[0, cat["purchase"]] += pa
        Ccat[0, cat["component_inspection"]] += ib
    else:
        Ccat[0, cat["purchase"]] += params.a1
    if x2:
        pa, ib = _component_acquire_breakdown(params.a2, params.b2, params.p2)
        Ccat[0, cat["purchase"]] += pa
        Ccat[0, cat["component_inspection"]] += ib
    else:
        Ccat[0, cat["purchase"]] += params.a2
    Ccat[0, cat["assembly"]] += params.assembly_cost
    Ccat[0, cat["final_inspection"]] += y * params.final_inspection_cost

    for s1, pr1 in dist1:
        for s2, pr2 in dist2:
            pr = pr1 * pr2
            all_good = s1 != "UB" and s2 != "UB"
            success = (1.0 - params.pf) if all_good else 0.0
            fail = 1.0 - success
            Ccat[0, cat["replacement"]] += pr * fail * (1 - y) * params.replacement_loss
            Ccat[0, cat["disassembly"]] += pr * fail * z * params.disassembly_cost
            if fail > 0:
                if z:
                    Q[0, STATE_INDEX[(s1, s2)]] += pr * fail
                else:
                    Q[0, 0] += pr * fail

    # 回收状态：r_i 只作用于拆解后的未知件。
    for state in STATES[1:]:
        i = STATE_INDEX[state]
        s = list(state)
        for pos, (ri, p, a, b) in enumerate(
            ((r1, params.p1, params.a1, params.b1), (r2, params.p2, params.a2, params.b2))
        ):
            if s[pos] == "K" or not ri:
                continue
            Ccat[i, cat["component_inspection"]] += b
            if s[pos] == "UB":
                pa, ib = _component_acquire_breakdown(a, b, p)
                Ccat[i, cat["purchase"]] += pa
                Ccat[i, cat["component_inspection"]] += ib
            s[pos] = "K"
        post = tuple(s)
        Ccat[i, cat["assembly"]] += params.assembly_cost
        Ccat[i, cat["final_inspection"]] += y * params.final_inspection_cost
        all_good = post[0] != "UB" and post[1] != "UB"
        success = (1.0 - params.pf) if all_good else 0.0
        fail = 1.0 - success
        Ccat[i, cat["replacement"]] += fail * (1 - y) * params.replacement_loss
        Ccat[i, cat["disassembly"]] += fail * z * params.disassembly_cost
        if fail > 0:
            if z:
                Q[i, STATE_INDEX[post]] += fail
            else:
                Q[i, 0] += fail

    c = Ccat.sum(axis=1)
    return Q, c, Ccat


def evaluate_q2_strategy_scalar(params: Q2Params, policy: Tuple[int, int, int, int, int, int]) -> Tuple[float, float, float, bool]:
    """轻量评价：只解一次总成本，用于64策略搜索/灵敏度。"""
    if len(policy) != 6 or any(x not in (0, 1) for x in policy):
        raise ValueError("Q2 policy 必须是6个0/1")
    if policy[3] == 0 and (policy[4] or policy[5]):
        return np.inf, -np.inf, np.inf, False
    try:
        Q, c, _ = build_markov_reward(params, policy)
    except ValueError:
        return np.inf, -np.inf, np.inf, False
    rho = float(np.max(np.abs(np.linalg.eigvals(Q))))
    if rho >= 1.0 - 1e-12:
        return np.inf, -np.inf, rho, False
    V = np.linalg.solve(np.eye(len(STATES)) - Q, c)
    cost = float(V[0])
    return cost, float(params.sale_price - cost), rho, True


def evaluate_q2_strategy(params: Q2Params, policy: Tuple[int, int, int, int, int, int]) -> Q2Eval:
    if len(policy) != 6 or any(x not in (0, 1) for x in policy):
        raise ValueError("Q2 policy 必须是6个0/1")
    if policy[3] == 0 and (policy[4] or policy[5]):
        return Q2Eval(policy, np.inf, -np.inf, np.inf, False, {})
    try:
        Q, c, Ccat = build_markov_reward(params, policy)
    except ValueError:
        return Q2Eval(policy, np.inf, -np.inf, np.inf, False, {})
    rho = float(np.max(np.abs(np.linalg.eigvals(Q))))
    if rho >= 1.0 - 1e-12:
        return Q2Eval(policy, np.inf, -np.inf, rho, False, {})
    A = np.eye(len(STATES)) - Q
    V = np.linalg.solve(A, c)
    breakdown = {}
    for j, name in enumerate(COST_CATEGORIES):
        vj = np.linalg.solve(A, Ccat[:, j])
        breakdown[name] = float(vj[0])
    expected_cost = float(V[0])
    if abs(sum(breakdown.values()) - expected_cost) > 1e-8:
        raise AssertionError("Q2 cost breakdown 与总成本不一致")
    return Q2Eval(
        policy=policy,
        expected_cost=expected_cost,
        expected_profit=float(params.sale_price - expected_cost),
        spectral_radius=rho,
        feasible=True,
        cost_breakdown=breakdown,
    )


def enumerate_q2_policies() -> Iterable[Tuple[int, int, int, int, int, int]]:
    for pi in product((0, 1), repeat=6):
        if pi[3] == 0 and (pi[4] or pi[5]):
            continue
        yield pi


def solve_q2(params: Q2Params, topk: int = 10) -> Tuple[Q2Eval, pd.DataFrame]:
    scalar_rows = []
    for pi in enumerate_q2_policies():
        cost, profit, rho, feasible = evaluate_q2_strategy_scalar(params, pi)
        if feasible:
            scalar_rows.append((profit, pi, cost, rho))
    if not scalar_rows:
        raise RuntimeError("Q2 无可行吸收策略")
    scalar_rows.sort(key=lambda x: (x[0], "".join(map(str,x[1]))), reverse=True)

    # 四类生产决策去重后的利润间隔。
    best_primary = {}
    for profit,pi,cost,rho in scalar_rows:
        code4=f"{pi[0]}{pi[1]}{pi[2]}{pi[3]}"
        best_primary.setdefault(code4, profit)
    primary_vals=sorted(best_primary.values(),reverse=True)
    primary_gap=primary_vals[0]-primary_vals[1] if len(primary_vals)>1 else np.nan
    full_gap=scalar_rows[0][0]-scalar_rows[1][0] if len(scalar_rows)>1 else np.nan

    # 只对Top-k做分类奖励分解，避免灵敏度/Bootstrap重复解大量线性方程。
    out=[]; detailed=[]
    for rank,(profit,pi,cost,rho) in enumerate(scalar_rows[:topk],1):
        e=evaluate_q2_strategy(params,pi)
        detailed.append(e)
        d={
            "rank":rank,"policy6":e.code6,"policy4":e.code4,
            "x1":pi[0],"x2":pi[1],"y":pi[2],"z":pi[3],"r1":pi[4],"r2":pi[5],
            "expected_cost":e.expected_cost,"expected_profit":e.expected_profit,"spectral_radius":e.spectral_radius,
            "full_policy_gap":full_gap if rank==1 else np.nan,
            "primary_policy_gap":primary_gap if rank==1 else np.nan,
        }
        d.update({f"cost_{k}":v for k,v in e.cost_breakdown.items()})
        out.append(d)
    return detailed[0], pd.DataFrame(out)


def no_disassembly_closed_form(params: Q2Params, x1: int, x2: int, y: int) -> float:
    """问题二式(11)：z=0的几何分布闭式期望，用作强单元测试。"""
    q1 = (1 - x1) * params.p1
    q2 = (1 - x2) * params.p2
    g = (1.0 - q1) * (1.0 - q2) * (1.0 - params.pf)
    c1 = params.a1 if not x1 else (params.a1 + params.b1) / (1.0 - params.p1)
    c2 = params.a2 if not x2 else (params.a2 + params.b2) / (1.0 - params.p2)
    c_try = c1 + c2 + params.assembly_cost + y * params.final_inspection_cost
    return float(c_try / g + (1 - y) * params.replacement_loss * (1.0 - g) / g)


def robust_absorbing_q2(
    params: Q2Params,
    policy: Tuple[int, int, int, int, int, int],
    upper: Dict[str, float],
) -> Tuple[bool, float]:
    """d=3时精确检查盒 [0,U] 的8个角点谱半径。"""
    max_rho = -np.inf
    for p1, p2, pf in product((0.0, upper["p1"]), (0.0, upper["p2"]), (0.0, upper["pf"])):
        if max(p1, p2, pf) >= 1.0:
            return False, np.inf
        pp = params.with_quality(p1, p2, pf)
        e = evaluate_q2_strategy(pp, policy)
        if not e.feasible:
            return False, e.spectral_radius
        max_rho = max(max_rho, e.spectral_radius)
    return max_rho < 1.0 - 1e-12, float(max_rho)


def solve_q2_robust(params: Q2Params, upper: Dict[str, float], topk: int = 10) -> Tuple[Q2Eval, pd.DataFrame]:
    if any(not (0 <= upper[k] < 1) for k in ("p1", "p2", "pf")):
        raise RuntimeError("Q2 某质量上界达到1，无法在该置信包络下保证有限期望交付成本")
    worst_params = params.with_quality(upper["p1"], upper["p2"], upper["pf"])
    scalar=[]
    for pi in enumerate_q2_policies():
        ok,_=robust_absorbing_q2(params,pi,upper)
        if not ok: continue
        cost,profit,rho,feasible=evaluate_q2_strategy_scalar(worst_params,pi)
        if feasible: scalar.append((profit,pi,cost,rho))
    if not scalar:
        raise RuntimeError("Q2 稳健不确定集合内无可行吸收策略")
    scalar.sort(key=lambda x:x[0],reverse=True)
    best_primary={}
    for profit,pi,_,_ in scalar:
        code4=f"{pi[0]}{pi[1]}{pi[2]}{pi[3]}"
        best_primary.setdefault(code4,profit)
    pv=sorted(best_primary.values(),reverse=True)
    primary_gap=pv[0]-pv[1] if len(pv)>1 else np.nan
    full_gap=scalar[0][0]-scalar[1][0] if len(scalar)>1 else np.nan
    data=[]; detailed=[]
    for rank,(profit,pi,cost,rho) in enumerate(scalar[:topk],1):
        e=evaluate_q2_strategy(worst_params,pi); detailed.append(e)
        row={"rank":rank,"policy6":e.code6,"policy4":e.code4,"expected_cost":e.expected_cost,"expected_profit":e.expected_profit,
             "spectral_radius_at_upper":e.spectral_radius,"full_policy_gap":full_gap if rank==1 else np.nan,"primary_policy_gap":primary_gap if rank==1 else np.nan}
        row.update({f"cost_{k}":v for k,v in e.cost_breakdown.items()}); data.append(row)
    return detailed[0], pd.DataFrame(data)



def solve_q2_robust_corners(params: Q2Params, upper: Dict[str, float], topk: int = 10) -> Tuple[Q2Eval, pd.DataFrame]:
    """单调性审计失败时的盒端点回退：对 d=3 的 8 个角点逐策略取最坏利润。"""
    if any(not (0 <= upper[k] < 1) for k in ("p1", "p2", "pf")):
        raise RuntimeError("Q2 某质量上界达到1，端点回退也无法保证有限交付成本")
    corners=[{"p1":p1,"p2":p2,"pf":pf} for p1,p2,pf in product((0.0,upper["p1"]),(0.0,upper["p2"]),(0.0,upper["pf"]))]
    rows=[]
    for pi in enumerate_q2_policies():
        worst_profit=np.inf; worst_corner=None; worst_rho=0.0; feasible=True
        for q in corners:
            pp=params.with_quality(q["p1"],q["p2"],q["pf"])
            cost,profit,rho,ok=evaluate_q2_strategy_scalar(pp,pi)
            if not ok:
                feasible=False; break
            if profit < worst_profit:
                worst_profit=profit; worst_corner=q.copy(); worst_rho=max(worst_rho,rho)
        if feasible:
            rows.append((worst_profit,pi,worst_corner,worst_rho))
    if not rows:
        raise RuntimeError("Q2 端点回退：不确定集合角点上无共同可行策略")
    rows.sort(key=lambda x:x[0],reverse=True)
    best_primary={}
    for profit,pi,_,_ in rows:
        best_primary.setdefault(f"{pi[0]}{pi[1]}{pi[2]}{pi[3]}",profit)
    vals=sorted(best_primary.values(),reverse=True)
    primary_gap=vals[0]-vals[1] if len(vals)>1 else np.nan
    full_gap=rows[0][0]-rows[1][0] if len(rows)>1 else np.nan
    out=[]; best_eval=None
    for rank,(profit,pi,corner,rho) in enumerate(rows[:topk],1):
        pp=params.with_quality(corner["p1"],corner["p2"],corner["pf"])
        e=evaluate_q2_strategy(pp,pi)
        if rank==1: best_eval=e
        row={"rank":rank,"policy6":e.code6,"policy4":e.code4,"expected_cost":e.expected_cost,"expected_profit":profit,
             "spectral_radius_at_worst_corner":e.spectral_radius,"worst_p1":corner["p1"],"worst_p2":corner["p2"],"worst_pf":corner["pf"],
             "full_policy_gap":full_gap if rank==1 else np.nan,"primary_policy_gap":primary_gap if rank==1 else np.nan}
        row.update({f"cost_{k}":v for k,v in e.cost_breakdown.items()}); out.append(row)
    assert best_eval is not None
    return best_eval,pd.DataFrame(out)
