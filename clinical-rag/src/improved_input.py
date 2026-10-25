"""Canonical input notation before Improved retrieval and model calls.

Only formatting and explicit units change. Symptoms, findings and clinical
criteria remain untouched. This does not determine diagnosis or severity.
"""
from decimal import Decimal
import re

from management_age import normalize_management_age

_DURATION = re.compile(
    r"\s*(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|days?|d|weeks?|wks?|w|"
    r"months?|mos?|years?|yrs?|y|minutes?|mins?|seconds?|secs?|s)\.?\s*", re.I)


def _number(value):
    text = format(Decimal(str(value)), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def normalize_improved_inputs(patient):
    """Return a copy; bare age means years, bare duration remains unspecified."""
    result = dict(patient)
    _, age = normalize_management_age(patient, {})
    if age["status"] == "normalized":
        unit = age["unit"]
        # Retain the input's exact decimal rather than a converted fraction.
        value = re.match(r"\s*(\d+(?:\.\d+)?)", str(patient["age"]))[1]
        result["age"] = _number(value) + " " + unit
    duration = patient.get("duration")
    match = _DURATION.fullmatch(str(duration)) if duration is not None else None
    if match:
        unit = match[2].casefold()
        if unit.startswith("h"):
            unit = "hours"
        elif unit.startswith("d"):
            unit = "days"
        elif unit.startswith("w"):
            unit = "weeks"
        elif unit.startswith(("mo",)):
            unit = "months"
        elif unit.startswith("mi"):
            unit = "minutes"
        elif unit.startswith(("s",)):
            unit = "seconds"
        else:
            unit = "years"
        result["duration"] = _number(match[1]) + " " + unit
    return result
