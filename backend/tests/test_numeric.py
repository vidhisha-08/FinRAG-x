import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import (ProgramError, execute, extract_numbers, is_numeric_question,  # noqa: E402
                    number_coverage, numbers_in, program_inputs, verify)
from metrics import has_answer, wilson  # noqa: E402

EV = [{"chunk_id": "D1_C1", "text": "Net revenue 2015 1,240.5 2014 1,100.2 (in millions)"}]


# ---------- number parsing
def test_parsing_signs_commas_and_ids():
    got = extract_numbers("(35) and -12.5 and $1,234.50 and 2014-2015 and Q4 [D3_C2]")
    assert [v for v, _ in got] == [-35.0, -12.5, 1234.5, 2014.0, 2015.0]
    assert [d for _, d in got] == [0, 1, 2, 0, 0]


def test_citation_ids_are_not_numbers():
    assert numbers_in("see D3_C2 and [D12_C4]") == set()


# ---------- calculator
def test_percent_change():
    p = {"steps": [{"op": "subtract", "args": [1240.5, 1100.2]},
                   {"op": "divide", "args": ["#0", 1100.2]}], "percent": True}
    assert execute(p) == pytest.approx(12.7522, abs=1e-3)
    assert program_inputs(p) == [1240.5, 1100.2, 1100.2]


def test_average_of_three():
    assert execute({"steps": [{"op": "average", "args": [10, 20, 30]}]}) == 20


def test_empty_program_returns_none():
    assert execute({"steps": []}) is None


@pytest.mark.parametrize("bad", [
    {"steps": [{"op": "divide", "args": [1, 0]}]},
    {"steps": [{"op": "exec", "args": [1, 2]}]},
    {"steps": [{"op": "add", "args": ["#1", 2]}]},
    {"steps": [{"op": "add", "args": ["__import__('os')", 2]}]},
    {"steps": [{"op": "subtract", "args": [1]}]},
])
def test_bad_programs_raise(bad):
    with pytest.raises(ProgramError):
        execute(bad)


# ---------- verifier
PROG = {"steps": [{"op": "subtract", "args": [1240.5, 1100.2]},
                  {"op": "divide", "args": ["#0", 1100.2]}], "percent": True}


def test_verify_accepts_good_answer():
    r = execute(PROG)
    ok = verify("Net revenue grew 12.75% [D1_C1]", EV, PROG, r)
    assert all(ok.values())


def test_verify_rejects_ungrounded_inputs():
    bad = {"steps": [{"op": "subtract", "args": [1300, 1100.2]}], "percent": False}
    c = verify("It rose by 199.8 [D1_C1]", EV, bad, execute(bad))
    assert not c["inputs_grounded"]


def test_verify_rejects_missing_or_fake_citation():
    r = execute(PROG)
    assert not verify("Net revenue grew 12.75%", EV, PROG, r)["citations_valid"]
    assert not verify("Net revenue grew 12.75% [D9_C9]", EV, PROG, r)["citations_valid"]


def test_verify_catches_hallucinated_number_without_program():
    c = verify("Growth was 47.3% [D1_C1]", EV, None, None)
    assert not c["answer_numbers_supported"]
    assert verify("Revenue was 1,240.5 million in 2015 [D1_C1]", EV, None, None)["answer_numbers_supported"]


def test_verify_refuses_insufficient():
    assert not verify("INSUFFICIENT EVIDENCE", EV, None, None)["answered"]


def test_coverage_scaling():
    assert number_coverage("Revenue was $1.2405 billion [D1_C1]", EV) == 1.0
    assert number_coverage("Revenue was 999.9 [D1_C1]", EV) == 0.0


# ---------- scoring
def test_has_answer_rules():
    assert has_answer("The share was 4.36%", 0.0436)               # x100 convention
    assert has_answer("about 14% [D1_C1]", 0.14464)                # integer rounding
    assert has_answer("It decreased by 35 million", -35.0)         # sign written as a word
    assert not has_answer("It was -35", 35.0)                      # wrong sign
    assert not has_answer("It was 2", 0.024)                       # no accidental match
    assert not has_answer("In 2015 the value rose [D3_C2]", 3.0)   # years / ids ignored


def test_wilson_interval():
    lo, hi = wilson(16, 30)
    assert 0.35 < lo < 0.37 and 0.69 < hi < 0.71


# ---------- router
def test_router_uses_word_boundaries():
    assert is_numeric_question("What was the net revenue growth?")
    assert is_numeric_question("What % of leases are due in 2018?")
    assert not is_numeric_question("Describe Cisco's internet strategy")
    assert not is_numeric_question("Who is the consumer segment head?")