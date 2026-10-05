import os
import re
import json
import time
from openai import OpenAI
from config import MODEL

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ["LLM_API_KEY"].strip())


def llm(prompt, max_tokens=800, json_mode=False):
    text = ""
    extra = ""
    for attempt in range(8):
        try:
            try:
                r = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt + extra}],
                    max_tokens=max_tokens,
                    temperature=0,
                    extra_body={"reasoning_effort": "low"},
                )
            except Exception as e:
                if "reasoning" in str(e).lower():       # setting not accepted: retry without it
                    r = client.chat.completions.create(
                        model=MODEL,
                        messages=[{"role": "user", "content": prompt + extra}],
                        max_tokens=max_tokens,
                        temperature=0,
                    )
                else:
                    raise
            text = r.choices[0].message.content or ""
            break
        except Exception as e:
            msg = str(e)
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
    if json_mode:
        m = re.search(r"\{.*\}", text, re.S)
        text = m.group(0) if m else text
    return text

def extract_program(question, evidence_text):
    raw = llm(f"""Use ONLY the evidence. Return JSON only, with exactly these keys:
{{"op":"growth or diff or ratio or sum or percent_of or none","a":0,"b":0,"a_label":"","b_label":"","unit":""}}
a = the later (or top) value, b = the earlier (or bottom) value.
op meanings: growth = (a-b)/b*100, diff = a-b, ratio = a/b, sum = a+b, percent_of = a/b*100.
Copy the numbers exactly as they appear in the evidence (no units, no commas).
Use op "none" if the needed numbers are not in the evidence.
Question: {question}
Evidence:
{evidence_text}""", json_mode=True)
    return json.loads(raw)


def execute(p):
    a, b = float(p["a"]), float(p["b"])
    ops = {
        "growth": lambda: (a - b) / b * 100,
        "diff": lambda: a - b,
        "ratio": lambda: a / b,
        "sum": lambda: a + b,
        "percent_of": lambda: a / b * 100,
    }
    return ops[p["op"]]() if p["op"] in ops else None


def generate(question, evidence, computed):
    ev = "\n\n".join(f"[{c['chunk_id']}]\n{c['text'][:1000]}" for c in evidence)
    return llm(f"""Answer using ONLY the evidence and the computed result below.
After each claim, cite the source id in square brackets, like [D3_C2].
If the evidence does not contain the answer, reply exactly: INSUFFICIENT EVIDENCE.
Question: {question}
Computed result (do not recompute): {computed}
Evidence:
{ev}""")


def numbers_in(text):
    out = set()
    for x in re.findall(r"\d[\d,]*\.?\d*", text):
        x = x.replace(",", "").rstrip(".")
        if x:
            out.add(float(x))
    return out


def verify(answer, evidence, program, result):
    ev_text = " ".join(c["text"] for c in evidence)
    ev_ids = {c["chunk_id"] for c in evidence}
    checks = {}
    cited = set(re.findall(r"D\d+_C\d+", answer))     # finds IDs however the model brackets them
    checks["citations_valid"] = bool(cited) and cited <= ev_ids
    if program and program.get("op") not in (None, "none") and result is not None:
        ev_nums = numbers_in(ev_text)
        checks["inputs_grounded"] = all(float(program[k]) in ev_nums for k in ("a", "b"))
        checks["result_in_answer"] = any(abs(n - abs(result)) <= max(0.006 * abs(result), 0.06)
                                        for n in numbers_in(answer))
    checks["sufficient"] = "INSUFFICIENT EVIDENCE" not in answer.upper()
    return checks