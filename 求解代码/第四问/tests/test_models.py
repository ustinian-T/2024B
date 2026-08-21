from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from src.q4_validation import regression_q2,q2_closed_form_validation,regression_q3,q3_handcheck_validation,exact_cp_coverage_validation,q1_sequential_boundary_validation

def test_q2_regression(): assert regression_q2()["PASS"].all()
def test_q2_closed(): assert q2_closed_form_validation()["PASS"].all()
def test_q3_regression(): assert regression_q3(flat_cross_check=False)["PASS"].all()
def test_q3_handcheck(): assert q3_handcheck_validation()["PASS"].all()
def test_cp_coverage(): assert exact_cp_coverage_validation()["PASS"].all()

def test_q1_seq_boundary(): assert q1_sequential_boundary_validation()["PASS"].all()
