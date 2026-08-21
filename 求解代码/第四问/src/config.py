from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Tuple


@dataclass(frozen=True)
class Q2Params:
    scenario_id: int
    p1: float
    a1: float
    b1: float
    p2: float
    a2: float
    b2: float
    pf: float
    assembly_cost: float
    final_inspection_cost: float
    sale_price: float
    replacement_loss: float
    disassembly_cost: float

    def with_quality(self, p1: float, p2: float, pf: float) -> "Q2Params":
        return replace(self, p1=float(p1), p2=float(p2), pf=float(pf))


Q2_SCENARIOS: Dict[int, Q2Params] = {
    1: Q2Params(1, 0.10, 4, 2, 0.10, 18, 3, 0.10, 6, 3, 56, 6, 5),
    2: Q2Params(2, 0.20, 4, 2, 0.20, 18, 3, 0.20, 6, 3, 56, 6, 5),
    3: Q2Params(3, 0.10, 4, 2, 0.10, 18, 3, 0.10, 6, 3, 56, 30, 5),
    4: Q2Params(4, 0.20, 4, 1, 0.20, 18, 1, 0.20, 6, 2, 56, 30, 5),
    5: Q2Params(5, 0.10, 4, 8, 0.20, 18, 1, 0.10, 6, 2, 56, 10, 5),
    6: Q2Params(6, 0.05, 4, 2, 0.05, 18, 3, 0.05, 6, 3, 56, 10, 40),
}


@dataclass(frozen=True)
class Q3Node:
    name: str
    kind: str  # component / intermediate / final
    defect_rate: float
    purchase_or_assembly_cost: float
    inspection_cost: float
    disassembly_cost: float = 0.0
    children: Tuple[str, ...] = ()


Q3_NODES: Dict[str, Q3Node] = {
    "C1": Q3Node("C1", "component", 0.10, 2, 1),
    "C2": Q3Node("C2", "component", 0.10, 8, 1),
    "C3": Q3Node("C3", "component", 0.10, 12, 2),
    "C4": Q3Node("C4", "component", 0.10, 2, 1),
    "C5": Q3Node("C5", "component", 0.10, 8, 1),
    "C6": Q3Node("C6", "component", 0.10, 12, 2),
    "C7": Q3Node("C7", "component", 0.10, 8, 1),
    "C8": Q3Node("C8", "component", 0.10, 12, 2),
    "H1": Q3Node("H1", "intermediate", 0.10, 8, 4, 6, ("C1", "C2", "C3")),
    "H2": Q3Node("H2", "intermediate", 0.10, 8, 4, 6, ("C4", "C5", "C6")),
    "H3": Q3Node("H3", "intermediate", 0.10, 8, 4, 6, ("C7", "C8")),
    "F": Q3Node("F", "final", 0.10, 8, 6, 10, ("H1", "H2", "H3")),
}

Q3_SALE_PRICE = 200.0
Q3_REPLACEMENT_LOSS = 40.0

Q2_EXPECTED = {
    1: ("1001", 18.84111111111111),
    2: ("1101", 12.0),
    3: ("1011", 16.47444444444444),
    4: ("1111", 14.75),
    5: ("0101", 14.96333333333333),
    6: ("0000", 21.67867036011081),
}

Q3_EXPECTED_CODE = "1111111111111101"
Q3_EXPECTED_COST = 139.77777777777777
Q3_EXPECTED_PROFIT = 60.22222222222222
