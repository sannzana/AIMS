"""Explicit age units for Improved management; no clinical age-group cutoffs."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
import re

_AGE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(years?|yrs?|y|months?|mos?|m)?\.?\s*", re.I)


def normalize_management_age(patient, context):
    """Bare numbers mean years by the input contract; convert units only.

    Preserve supplied source facts. Conflicting facts need clarification rather
    than silently replacing them. Never infer adult/child or any severity class.
    """
    result = dict(context)
    raw = patient.get("age")
    info = {"status": "missing"}
    if raw is None or raw == "":
        return result, info
    if isinstance(raw, bool):
        return result, {"status": "invalid", "message": "Enter an age with years or months."}
    match = _AGE.fullmatch(str(raw))
    if not match:
        return result, {"status": "invalid", "message": "Enter one age, e.g. 32, 32yr or 32 months."}
    try:
        value = Decimal(match[1])
        unit = (match[2] or "years").casefold()
        months = value if unit.startswith("m") else value * 12
        years = months / 12
        values = {"age_years": float(years), "age_months": float(months)}
        if not all(math.isfinite(v) for v in values.values()):
            raise ValueError("Age is not finite")
    except (InvalidOperation, ValueError, OverflowError):
        return result, {"status": "invalid", "message": "Enter a finite numeric age."}
    for key, expected in values.items():
        existing = context.get(key)
        if existing is not None:
            try:
                agrees = not isinstance(existing, bool) and math.isclose(float(existing), expected, rel_tol=1e-9)
            except (ValueError, TypeError, OverflowError):
                agrees = False
            if not agrees:
                return result, {"status": "conflict", "message": "Age input conflicts with supplied age facts; confirm age and units."}
    for key, value in values.items():
        if result.get(key) is None:
            result[key] = int(value) if value.is_integer() else value
    return result, {"status": "normalized", "unit": "months" if unit.startswith("m") else "years",
                    "defaulted_to_years": match[2] is None, **values}
