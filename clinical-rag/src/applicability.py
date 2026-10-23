"""Three-valued clinical card applicability evaluation."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import ValidationError

from schema import Card, Condition


class TruthValue(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


def _condition_from_mapping(value: Any) -> Condition | None:
    if not isinstance(value, dict):
        return None
    field = value.get("field")
    operator = value.get("op", value.get("operator"))
    if not isinstance(field, str) or operator not in {
        "=", "==", "!=", "<", "<=", ">", ">=", "has", "in"
    }:
        return None
    return Condition(
        field=field,
        op="==" if operator == "=" else operator,
        value=value.get("value"),
    )


def _evaluate_tree(tree: Any, patient: dict[str, Any]) -> TruthValue:
    condition = _condition_from_mapping(tree)
    if condition is not None:
        return evaluate_condition(condition, patient)
    if not isinstance(tree, dict):
        return TruthValue.UNKNOWN
    for key, reducer in (("all", _all), ("any", _any), ("exceptions", _any)):
        if key in tree:
            values = tree[key]
            if not isinstance(values, list):
                return TruthValue.UNKNOWN
            result = reducer([_evaluate_tree(item, patient) for item in values])
            if key == "exceptions" and not values:
                return TruthValue.FALSE
            return result
    return TruthValue.UNKNOWN


def _rule_fields(tree: Any) -> list[str]:
    condition = _condition_from_mapping(tree)
    if condition is not None:
        return [condition.field]
    if not isinstance(tree, dict):
        return []
    fields: list[str] = []
    for value in tree.values():
        if isinstance(value, list):
            for item in value:
                fields.extend(_rule_fields(item))
    return fields


def card_rule_applicability(card: Card, patient: dict[str, Any]) -> dict[str, Any] | None:
    """Evaluate only normalized legacy or flat card rules.

    Unsupported source rule shapes return UNKNOWN with no inferred result.
    """
    try:
        conditions = card.conditions_value
    except ValidationError:
        conditions = None
    if conditions is not None:
        normalized = (
            card.model_copy(update={"conditions": conditions})
            if card.structured_rules or card.conditions is None
            else card
        )
        return check_applicability(normalized, patient)

    raw_rules = card.model_dump().get("conditions_decision_logic")
    raw_exceptions = card.model_dump().get("exceptions_contraindications")
    if raw_rules in (None, [], "") and raw_exceptions in (None, [], ""):
        return None
    if not isinstance(raw_rules, list) or not isinstance(raw_exceptions, (list, type(None))):
        return {"status": "undetermined", "met": [], "failed": [], "missing_inputs": []}

    rule_values: list[TruthValue] = []
    fields: list[str] = []
    met: list[str] = []
    failed: list[str] = []
    for rule in raw_rules:
        tree = rule.get("condition") if isinstance(rule, dict) else None
        if tree is None:
            rule_values.append(TruthValue.UNKNOWN)
            continue
        value = _evaluate_tree(tree, patient)
        rule_values.append(value)
        fields.extend(_rule_fields(tree))
        if value is TruthValue.TRUE:
            met.extend(_rule_fields(tree))
        elif value is TruthValue.FALSE:
            failed.extend(_rule_fields(tree))

    exception_values: list[TruthValue] = []
    if raw_exceptions:
        for rule in raw_exceptions:
            tree = rule.get("condition") if isinstance(rule, dict) else None
            value = _evaluate_tree(tree, patient) if tree is not None else TruthValue.UNKNOWN
            exception_values.append(value)
            fields.extend(_rule_fields(tree))
    exception_result = _any(exception_values) if exception_values else TruthValue.FALSE
    if exception_result is TruthValue.TRUE:
        status = "not_applicable"
    elif rule_values and TruthValue.TRUE in rule_values:
        status = "applicable" if exception_result is TruthValue.FALSE else "undetermined"
    elif TruthValue.UNKNOWN in rule_values or exception_result is TruthValue.UNKNOWN:
        status = "undetermined"
    else:
        status = "not_applicable"
    missing = [field for field in dict.fromkeys(fields) if _is_missing(*_get_value(patient, field))]
    return {
        "status": status,
        "met": list(dict.fromkeys(met)),
        "failed": list(dict.fromkeys(failed)),
        "missing_inputs": missing,
    }


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
