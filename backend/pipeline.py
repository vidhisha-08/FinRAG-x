"""FinRAG-X pipeline: retrieve -> (calculate) -> generate -> verify."""
import logging
from collections import Counter, defaultdict

from engine import (MAX_EVIDENCE_CHARS, ProgramError, execute, extract_program, generate,
                    is_numeric_question, number_coverage, verify)
from retrieval import Retriever

log = logging.getLogger("finrag")

R = Retriever()
_BY_PAGE = defaultdict(list)                  # context_id -> chunks (built once, not per query)
for _c in R.chunks:
    _BY_PAGE[_c["context_id"]].append(_c)


def expand_with_page(evidence, max_extra=3):
    """Add more chunks (tables first) from the page that appears most in the top results."""
    if not evidence:
        return evidence
    votes = Counter(c["context_id"] for c in evidence)
    first = {}
    for i, c in enumerate(evidence):
        first.setdefault(c["context_id"], i)
    top = max(votes, key=lambda p: (votes[p], -first[p]))      # most votes, ties go to higher rank
    have = {c["chunk_id"] for c in evidence}
    page = [c for c in _BY_PAGE[top] if c["chunk_id"] not in have]
    page.sort(key=lambda c: c["type"] != "table")              # tables first
    return evidence + page[:max_extra]


def answer_question(q, mode="dense", expand=True, use_calc=True, use_checker=True):
    evidence = R.search(q, k=5, mode=mode)
    if expand:
        evidence = expand_with_page(evidence)

    program, result, calc_error = None, None, None
    if use_calc and is_numeric_question(q):
        try:
            text = "\n\n".join(c["text"][:MAX_EVIDENCE_CHARS] for c in evidence)
            program = extract_program(q, text)
            result = execute(program)                           # None when steps == []
        except ProgramError as e:
            # Only program errors are caught. API/network errors now propagate so the
            # evaluator stops and resumes, instead of silently scoring as "no calculation".
            calc_error = str(e)
            log.warning("calculation failed for %r: %s", q[:60], calc_error)
            program, result = None, None

    answer = generate(q, evidence, f"{result:.6f}" if result is not None else None)
    checks = verify(answer, evidence, program, result, question=q)
    passed = all(checks.values()) if use_checker else True
    # Evidence coverage = share of numbers in the answer traceable to evidence/result.
    # (The old value was the % of checks passed, which carried no extra information.)
    coverage = round(100 * number_coverage(answer, evidence, q, result))

    return {
        "question": q,
        "answer": answer if passed else "Insufficient evidence to answer reliably.",
        "raw_answer": answer,
        "passed": passed,
        "program": program,
        "calc_error": calc_error,
        "result": None if result is None else round(result, 4),
        "evidence": [{"id": c["chunk_id"], "context_id": c["context_id"], "type": c["type"],
                      "text": c["text"][:800]} for c in evidence],
        "checks": checks,
        "evidence_coverage": coverage,
    }