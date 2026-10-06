"""Scoring helpers. No model or network imports, so they are easy to unit-test."""
import math
import re

from engine import extract_numbers, is_year

DECREASE_RE = re.compile(r"\b(decreas|declin|reduc|drop|fell|lower|loss|negative|down)\w*", re.I)
YES_NO_STARTS = ("did", "was", "is", "were", "does", "has", "had", "are", "do", "can")


def to_float(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").replace("$", ""))
    except ValueError:
        return None


def is_yes_no(question, gold):
    first = question.strip().split()[0].lower()
    return gold in (0.0, 1.0) and first in YES_NO_STARTS


def _tolerance(target, decimals):
    """1% relative, widened to the rounding of what was written ('14%' for 14.46), capped at 5%."""
    t = abs(target)
    return max(min(max(0.01 * t, 0.5 * 10 ** (-decimals)), 0.05 * t), 1e-9)


def has_answer(text, gold):
    """True if `text` contains the gold value (as written, or x100 for percentages).
    Sign-aware: -35 does not match +35, except 'decreased by 35' for a negative gold."""
    if not text or gold is None:
        return False
    for n, dec in extract_numbers(text):
        if is_year(n, dec):
            continue
        for t in (gold, gold * 100):
            tol = _tolerance(t, dec)
            if abs(n - t) <= tol:
                return True
            if t < 0 and n > 0 and abs(n + t) <= tol and DECREASE_RE.search(text):
                return True
    return False


def wilson(k, n, z=1.96):
    """95% confidence interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)