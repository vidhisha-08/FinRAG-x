from collections import Counter
from retrieval import Retriever
from engine import extract_program, execute, generate, verify

R = Retriever()
NUMERIC_WORDS = ("growth", "increase", "decrease", "change", "difference", "ratio",
                 "percent", "%", "total", "sum", "margin", "average", "how much", "net")


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
    page = [c for c in R.chunks if c["context_id"] == top and c["chunk_id"] not in have]
    page.sort(key=lambda c: c["type"] != "table")              # tables first
    return evidence + page[:max_extra]


def answer_question(q, mode="dense", expand=True, use_calc=True, use_checker=True):
    evidence = R.search(q, k=5, mode=mode)
    if expand:
        evidence = expand_with_page(evidence)
    program, result = None, None

    if use_calc and any(w in q.lower() for w in NUMERIC_WORDS):
        try:
            text = "\n\n".join(c["text"][:1000] for c in evidence)
            program = extract_program(q, text)
            result = execute(program) if program.get("op") != "none" else None
        except Exception:
            program, result = None, None

    answer = generate(q, evidence, f"{result:.6f}" if result is not None else "none")
    checks = verify(answer, evidence, program, result)
    passed = all(checks.values()) if use_checker else True

    return {
        "question": q,
        "answer": answer if passed else "Insufficient evidence to answer reliably.",
        "raw_answer": answer,
        "passed": passed,
        "program": program,
        "result": None if result is None else round(result, 4),
        "evidence": [{"id": c["chunk_id"], "type": c["type"], "text": c["text"][:800]} for c in evidence],
        "checks": checks,
        "evidence_coverage": round(100 * sum(checks.values()) / max(len(checks), 1)),
    }