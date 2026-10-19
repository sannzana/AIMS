"""Three-valued clinical card applicability evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Any

from schema import Card, Condition


class TruthValue(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


def _get_value(patient: dict[str, Any], field: str) -> tuple[Any, bool]:
    value: Any = patient
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None, False
        value = value[part]
    return value, value is not None


def _is_missing(value: Any, present: bool) -> bool:
    return not present or value is None


def evaluate_condition(condition: Condition, patient: dict[str, Any]) -> TruthValue:
    """Evaluate one condition without treating missing information as false."""
    actual, present = _get_value(patient, condition.field)
    if _is_missing(actual, present):
        return TruthValue.UNKNOWN

    expected = condition.value
    try:
        if condition.op == "==":
            result = actual == expected
        elif condition.op == "!=":
            result = actual != expected
        elif condition.op == "<":
            result = actual < expected
        elif condition.op == "<=":
            result = actual <= expected
        elif condition.op == ">":
            result = actual > expected
        elif condition.op == ">=":
            result = actual >= expected
        elif condition.op == "in":
            result = (
                any(item in expected for item in actual)
                if isinstance(actual, (list, tuple, set))
                else actual in expected
            )
        elif condition.op == "has":
            if expected is None:
                result = bool(actual)
            elif isinstance(actual, (list, tuple, set, str, dict)):
                result = expected in actual
            else:
                result = actual == expected
        else:  # pragma: no cover - constrained by the schema Literal
            return TruthValue.UNKNOWN
    except (TypeError, ValueError):
        return TruthValue.UNKNOWN
    return TruthValue.TRUE if result else TruthValue.FALSE


def _all(values: list[TruthValue]) -> TruthValue:
    if not values:
        return TruthValue.TRUE
    if TruthValue.FALSE in values:
        return TruthValue.FALSE
    if TruthValue.UNKNOWN in values:
        return TruthValue.UNKNOWN
    return TruthValue.TRUE


def _any(values: list[TruthValue]) -> TruthValue:
    if not values:
        return TruthValue.TRUE
    if TruthValue.TRUE in values:
        return TruthValue.TRUE
    if TruthValue.UNKNOWN in values:
        return TruthValue.UNKNOWN
    return TruthValue.FALSE


def _missing_inputs(card: Card, patient: dict[str, Any]) -> list[str]:
    fields = list(card.conditions.required_inputs)
    for condition in (
        card.conditions.all
        + card.conditions.any
        + card.conditions.exceptions
    ):
        if condition.field not in fields:
            fields.append(condition.field)
    return [field for field in fields if _is_missing(*_get_value(patient, field))]


def check_applicability(
    card: Card, patient: dict[str, Any]
) -> dict[str, Any]:
    """Return applicability status and condition-level evidence."""
    all_values = [
        evaluate_condition(condition, patient)
        for condition in card.conditions.all
    ]
    any_values = [
        evaluate_condition(condition, patient)
        for condition in card.conditions.any
    ]
    exception_values = [
        evaluate_condition(condition, patient)
        for condition in card.conditions.exceptions
    ]

    all_result = _all(all_values)
    any_result = _any(any_values)
    # An empty exception list means that no exception exists, unlike an empty
    # conditions.any list, which imposes no restriction.
    exception_result = _any(exception_values) if exception_values else TruthValue.FALSE
    missing = _missing_inputs(card, patient)
    missing_required = [
        field
        for field in card.conditions.required_inputs
        if _is_missing(*_get_value(patient, field))
    ]

    if exception_result is TruthValue.TRUE:
        status = "not_applicable"
    elif (
        all_result is TruthValue.FALSE
        or any_result is TruthValue.FALSE
    ):
        status = "not_applicable"
    elif (
        exception_result is TruthValue.UNKNOWN
        or all_result is TruthValue.UNKNOWN
        or any_result is TruthValue.UNKNOWN
        or missing_required
    ):
        status = "undetermined"
    else:
        status = "applicable"

    met: list[str] = []
    failed: list[str] = []
    for condition, value in zip(
        card.conditions.all + card.conditions.any + card.conditions.exceptions,
        all_values + any_values + exception_values,
    ):
        if value is TruthValue.TRUE:
            met.append(condition.field)
        elif value is TruthValue.FALSE:
            failed.append(condition.field)

    return {
        "status": status,
        "met": met,
        "failed": failed,
        "missing_inputs": missing,
    }
