"""Conservative source age-clause selection for Improved management only.

Unit conversion is arithmetic. Clinical age bands come only from source text;
no completed-year convention or adult/child boundary is invented.
"""
import math
import re

_NUM = r"(\d+(?:\.\d+)?)"
_UNIT = r"(years?|yrs?|months?|mos?)"
_MARK = re.compile(r"\b(?:ages?|aged)\s+(?=\d|under\b|below\b|over\b|above\b|less\b|at\b)", re.I)
_RANGE = re.compile(rf"{_NUM}\s*[–—-]\s*{_NUM}\s*{_UNIT}\b", re.I)
_LOWER = re.compile(rf"{_NUM}\s*{_UNIT}\s+(?:and|or)\s+(?:older|above)\b", re.I)
_UNDER = re.compile(rf"(?:under|below|less than)\s+{_NUM}\s*{_UNIT}\b", re.I)
_OVER = re.compile(rf"(?:over|above|more than)\s+{_NUM}\s*{_UNIT}\b", re.I)


def patient_age_months(context):
    candidates = []
    for key, factor in (("age_months", 1), ("age_years", 12)):
        value = context.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        converted = float(value) * factor
        if not math.isfinite(converted) or converted < 0:
            return None
        candidates.append(converted)
    if len(candidates) == 2 and not math.isclose(*candidates, rel_tol=1e-9, abs_tol=1e-9):
        return None
    return candidates[0] if candidates else None


def _months(number, unit):
    return float(number) * (12 if unit.casefold().startswith("y") else 1)


def _band(clause):
    marks = list(_MARK.finditer(clause))
    if len(marks) != 1:
        return None
    tail = clause[marks[0].end():]
    if match := _RANGE.match(tail):
        low, high = _months(match[1], match[3]), _months(match[2], match[3])
        return (low, high, True, True) if low <= high else None
    if match := _LOWER.match(tail):
        return _months(match[1], match[2]), math.inf, True, False
    if match := _UNDER.match(tail):
        return 0, _months(match[1], match[2]), True, False
    if match := _OVER.match(tail):
        return _months(match[1], match[2]), math.inf, False, False
    return None


def select_age_clauses(text, age_months):
    result = {"text": text, "status": "no_age_clause", "matched": [], "omitted": []}
    if not isinstance(text, str) or not _MARK.search(text):
        return result
    clauses = [part.strip() for part in re.split(r";|(?<=\.)\s+(?=[A-Z])", text) if part.strip()]
    bands = [_band(clause) for clause in clauses]
    # Do not reattach an unmarked "repeat this dose" or warning to another
    # age band. Its scope cannot be established by removing a neighbouring clause.
    if any(band is None for band in bands):
        result["status"] = "unparsed"
        return result
    if (isinstance(age_months, bool) or not isinstance(age_months, (int, float))
            or not math.isfinite(age_months) or age_months < 0):
        result["status"] = "age_unknown"
        return result
    matched = []
    for clause, (low, high, low_inclusive, high_inclusive) in zip(clauses, bands):
        if (age_months > low or low_inclusive and age_months == low) and (
                age_months < high or high_inclusive and age_months == high):
            matched.append(clause)
    result.update(matched=matched, omitted=[c for c in clauses if c not in matched])
    if len(matched) > 1:
        result["status"] = "overlapping_bands"
    elif not matched:
        # A fractional age between two stated bands is not silently rounded
        # into a completed-year/month cohort. Request source-boundary review.
        between = any(low < age_months for low, _, _, _ in bands) and any(
            high > age_months for _, high, _, _ in bands)
        result.update(status="boundary_review" if between else "excluded", text=text if between else None)
    else:
        result.update(status="matched", text=matched[0])
    return result
