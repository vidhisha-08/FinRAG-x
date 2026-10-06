"""FinRAG-X engine: LLM wrapper, number parsing, safe calculator, verification."""
import hashlib
import json
import logging
import math
import os
import re
import time
from pathlib import Path

from config import MODEL

log = logging.getLogger("finrag")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR = DATA_DIR / "llm_cache"      # add data/llm_cache/ to .gitignore
MAX_EVIDENCE_CHARS = 1500               # tables are long; 1000 chars cut rows off

_client = None
_reasoning_ok = True                    # flips to False if the API rejects reasoning_effort


# --------------------------------------------------------------------------- LLM
def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        key = os.environ.get("LLM_API_KEY", "").strip()
        if not key:
            raise RuntimeError("LLM_API_KEY environment variable is not set")
        _client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
    return _client


def _call(prompt, max_tokens):
    global _reasoning_ok
    extra = ""
    for attempt in range(8):
        kwargs = dict(model=MODEL, messages=[{"role": "user", "content": prompt + extra}],
                      max_tokens=max_tokens, temperature=0)
        if _reasoning_ok:
            kwargs["extra_body"] = {"reasoning_effort": "low"}
        try:
            r = _get_client().chat.completions.create(**kwargs)
        except Exception as e:
            msg = str(e)
            if _reasoning_ok and "reasoning" in msg.lower():
                _reasoning_ok = False               # remember, don't retry it on every call
                continue
            if "tool_use_failed" in msg and attempt < 7:
                extra = "\n\nIMPORTANT: Do not call any tools or functions. Reply in plain text only."
                continue
            if ("429" in msg or "Connection" in msg) and attempt < 7:
                m = re.search(r"try again in (?:(\d+)m)?\s*([\d.]+)s", msg)
                wait = (int(m.group(1) or 0) * 60 + float(m.group(2)) + 2) if m else 20
                print(f"  rate limit, waiting {int(wait)}s ...")
                time.sleep(min(wait, 900))
                continue
            raise
        choice = r.choices[0]
        text = choice.message.content or ""
        # Reasoning models spend max_tokens on hidden reasoning; an empty reply with
        # finish_reason=length is a truncation, NOT an answer. Retry with more room.
        if not text.strip() and choice.finish_reason == "length" and attempt < 7:
            max_tokens = min(max_tokens * 2, 8000)
            log.warning("empty reply (token limit hit); retrying with max_tokens=%d", max_tokens)
            continue
        return text
    return ""


def llm(prompt, max_tokens=1500, json_mode=False, use_cache=True):
    """Temperature-0 call with an on-disk cache so reruns are free and reproducible."""
    key = hashlib.sha256(json.dumps([MODEL, prompt], ensure_ascii=False).encode()).hexdigest()
    path = CACHE_DIR / f"{key}.json"
    text = None
    if use_cache and path.exists():
        text = json.loads(path.read_text(encoding="utf-8"))["text"]
    if text is None:
        text = _call(prompt, max_tokens)
        if use_cache and text.strip():
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": text}), encoding="utf-8")
    if json_mode:
        m = re.search(r"\{.*\}", text, re.S)
        text = m.group(0) if m else text
    return text


# ------------------------------------------------------------------ number parsing
_CITE_RE = re.compile(r"\[[^\]]*\]|\bD\d+_C\d+\b")      # citation ids are not numbers
_NUM_RE = re.compile(
    r"(?<![\w.,])"                                      # skip Q4, FY2015, 10-K style glue
    r"(?P<neg>[-\u2212])?"
    r"(?P<open>\()?\$?"
    r"(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)"
    r"(?P<close>\))?"
)


def extract_numbers(text):
    """Return [(signed_value, decimals_written)]. '(35)' and '-35' are negative."""
    out = []
    for m in _NUM_RE.finditer(_CITE_RE.sub(" ", text or "")):
        raw = m.group("num")
        value = float(raw.replace(",", ""))
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        if m.group("neg") or (m.group("open") and m.group("close")):
            value = -value
        out.append((value, decimals))
    return out


def numbers_in(text):
    """Set of absolute values appearing in text."""
    return {abs(v) for v, _ in extract_numbers(text)}


def is_year(value, decimals):
    return decimals == 0 and 1900 <= abs(value) <= 2100


# ------------------------------------------------------------- question routing
NUMERIC_RE = re.compile(
    r"\b(growth|increase[ds]?|decrease[ds]?|change[ds]?|differences?|ratios?|percent(?:age)?|"
    r"proportion|total|sum|margin|average|how much|net)\b|%", re.I)


def is_numeric_question(q):
    """Word-boundary match (the old substring test fired on 'internet', 'consumer', ...).
    Temporary until the trained question classifier exists."""
    return bool(NUMERIC_RE.search(q))


# ----------------------------------------------------------- safe calculator
class ProgramError(ValueError):
    pass


def _div(x, y):
    if y == 0:
        raise ProgramError("division by zero")
    return x / y


OPS = {
    "add": lambda xs: sum(xs),
    "subtract": lambda xs: xs[0] - xs[1],
    "multiply": lambda xs: math.prod(xs),
    "divide": lambda xs: _div(xs[0], xs[1]),
    "average": lambda xs: sum(xs) / len(xs),
}
ARITY = {"add": (2, 8), "subtract": (2, 2), "multiply": (2, 8), "divide": (2, 2), "average": (2, 8)}
_REF_RE = re.compile(r"#(\d+)")

PROGRAM_PROMPT = """Convert the finance question into a small calculation program. Return JSON only.

Format:
{{"steps": [{{"op": "subtract", "args": [1240.5, 1100.2]}}, {{"op": "divide", "args": ["#0", 1100.2]}}], "percent": true}}

Rules:
- Allowed ops: add, subtract, multiply, divide, average. "#0", "#1" ... refer to the result of an earlier step.
- Copy each input number exactly as printed in the evidence or the question (digits only, no $ or commas). Keep negative numbers negative.
- Percentage change = (new - old) / old, with "percent": true. "What percentage is X of Y" = X / Y with "percent": true. Plain differences, totals, averages and ratios use "percent": false.
- Use numbers from the evidence; numbers stated in the question may also be used.
- If a needed number is missing, or no calculation is needed, return {{"steps": [], "percent": false}}.

Question: {question}
Evidence:
{evidence}"""


def extract_program(question, evidence_text):
    raw = llm(PROGRAM_PROMPT.format(question=question, evidence=evidence_text),
              max_tokens=1500, json_mode=True)
    try:
        prog = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ProgramError(f"model did not return valid JSON: {raw[:120]!r}") from e
    if not isinstance(prog, dict) or not isinstance(prog.get("steps", []), list):
        raise ProgramError("program must be an object with a 'steps' list")
    prog.setdefault("steps", [])
    prog["percent"] = bool(prog.get("percent", False))
    return prog


def _resolve(arg, results):
    if isinstance(arg, bool):
        raise ProgramError(f"bad argument: {arg!r}")
    if isinstance(arg, (int, float)):
        return float(arg)
    if isinstance(arg, str):
        s = arg.strip()
        m = _REF_RE.fullmatch(s)
        if m:
            i = int(m.group(1))
            if i >= len(results):
                raise ProgramError(f"reference {s} used before it exists")
            return results[i]
        nums = extract_numbers(s)
        if len(nums) == 1:
            return nums[0][0]
    raise ProgramError(f"bad argument: {arg!r}")


def execute(program):
    """Run a validated program. Only the ops in OPS exist; nothing is eval'd."""
    steps = (program or {}).get("steps") or []
    if not steps:
        return None
    results = []
    for s in steps:
        if not isinstance(s, dict):
            raise ProgramError("each step must be an object")
        op, args = s.get("op"), s.get("args")
        if op not in OPS or not isinstance(args, list):
            raise ProgramError(f"unsupported step: {s!r}")
        lo, hi = ARITY[op]
        if not lo <= len(args) <= hi:
            raise ProgramError(f"{op} needs {lo}-{hi} arguments")
        results.append(OPS[op]([_resolve(a, results) for a in args]))
    out = results[-1]
    return out * 100 if program.get("percent") else out


def program_inputs(program):
    """The raw numbers the program uses (everything that is not a #n reference)."""
    vals = []
    for s in (program or {}).get("steps") or []:
        for a in s.get("args", []):
            if isinstance(a, str) and _REF_RE.fullmatch(a.strip()):
                continue
            try:
                vals.append(_resolve(a, []))
            except ProgramError:
                pass
    return vals


# --------------------------------------------------------------- generation
def generate(question, evidence, computed=None):
    """computed=None -> the model may calculate itself (baselines). Otherwise it must use `computed`."""
    ev = "\n\n".join(f"[{c['chunk_id']}]\n{c['text'][:MAX_EVIDENCE_CHARS]}" for c in evidence)
    if computed is None:
        calc_note = "If the question needs a calculation, do it step by step using numbers from the evidence."
    else:
        calc_note = f"Computed result (use it exactly, do not recompute): {computed}"
    return llm(f"""Answer using ONLY the evidence below.
After each claim, cite the source id in square brackets, like [D3_C2].
State the final value clearly, rounded sensibly (for example 2 decimals for percentages).
If the evidence does not contain the answer, reply exactly: INSUFFICIENT EVIDENCE.
{calc_note}
Question: {question}
Evidence:
{ev}""", max_tokens=1500)


def direct_answer(question):
    """System A: no retrieval."""
    return llm(f"""Answer the finance question. If a calculation is needed, do it step by step.
State the final value clearly, rounded sensibly.
Question: {question}""", max_tokens=1500)


# ------------------------------------------------------------- verification
_SCALES = (1, 1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9, 100, 0.01)    # units / thousands / millions / percent


def number_coverage(answer, evidence, question="", result=None):
    """Share (0-1) of the numbers in the answer that can be traced to the evidence,
    the question, or the computed result (allowing unit/percent scaling and rounding)."""
    pool = set()
    for c in evidence:
        pool |= numbers_in(c["text"])
    pool |= numbers_in(question)
    if result is not None:
        pool.add(abs(result))
    pool.discard(0.0)
    answer_nums = [(abs(v), d) for v, d in extract_numbers(answer) if not is_year(v, d)]
    if not answer_nums:
        return 1.0
    supported = 0
    for v, d in answer_nums:
        floor = 0.5 * 10 ** (-d) + 1e-9
        if any(abs(v - p * s) <= max(0.005 * p * s, floor) for p in pool for s in _SCALES):
            supported += 1
    return supported / len(answer_nums)


def _result_in_answer(result, answer):
    for n, d in extract_numbers(answer):
        for t in (abs(result), abs(result) * 100):
            tol = min(max(0.006 * t, 0.5 * 10 ** (-d) + 1e-9), 0.05 * t + 1e-9)
            if abs(abs(n) - t) <= tol:
                return True
    return False


def verify(answer, evidence, program=None, result=None, question=""):
    """Every value must be True for the answer to be released."""
    ev_text = " ".join(c["text"] for c in evidence)
    ev_ids = {c["chunk_id"] for c in evidence}
    checks = {}
    checks["answered"] = bool(answer.strip()) and "INSUFFICIENT EVIDENCE" not in answer.upper()
    cited = set(re.findall(r"D\d+_C\d+", answer))
    checks["citations_valid"] = bool(cited) and cited <= ev_ids
    if program and result is not None:
        pool = {round(p, 6) for p in numbers_in(ev_text) | numbers_in(question)}
        inputs = program_inputs(program)
        checks["inputs_grounded"] = bool(inputs) and all(round(abs(v), 6) in pool for v in inputs)
        checks["result_in_answer"] = _result_in_answer(result, answer)
    else:
        # No engine result: any number in the answer is the LLM's own arithmetic or recall.
        # The old verifier let these through unchecked. Require them to appear in the evidence.
        checks["answer_numbers_supported"] = number_coverage(answer, evidence, question) == 1.0
    return checks