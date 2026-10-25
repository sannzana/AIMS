"""Collect the patient facts that the treatment-plan rules need (Improved mode only).

Diagnosis extracts facts only for the case-definition cards it compares. The
fields that decide the treatment branch (dehydration signs, severity signs ...)
live in the *assessment* cards, which diagnosis never offers to the model, so
they were always unknown and every plan card was withheld.

This module asks, once, for exactly those fields: the ones the disease's
classification rules test and that are still unknown. It reuses the same
fail-closed checks as diagnosis (exact quote in the patient text, independent
polarity confirmation, numeric value present in its quote). Nothing is inferred
from absence; anything not confirmed stays unknown. No disease, symptom or
threshold is named here - the fields come from the cards.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from derived_fields import _conditions, classification_rules, field_groups
from fact_verification import (patient_text, verify_boolean_facts, verify_facts,
                               verify_measurement_facts)


def needed_fields(cards: Any, disease: str, known: dict[str, Any]) -> list[str]:
    """Fields tested by the disease's classification rules and not yet known."""
    cards = list(cards)
    rules = classification_rules(cards, disease)
    # only classifications that some treatment card actually depends on
    used = set().union(*(g["labels"] for g in
                         field_groups(cards, disease, {l for _, l, _ in rules}).values()), set())
    names: set[str] = set()
    for _, label, tree in rules:
        if label not in used:
            continue
        for cond in _conditions(tree):
            if isinstance(cond.get("field"), str):
                names.add(cond["field"])
    return sorted(n for n in names if known.get(n) in (None, ""))


def _prompt(patient: dict[str, Any], fields: list[str]) -> str:
    return (
        "Extract patient facts for the listed SOURCE FIELDS from the PATIENT text. "
        "Report a field only when the patient text explicitly states it: "
        "true = stated present, false = stated absent or normal, a number = a "
        "stated measurement. If the text does not say, leave the field out. "
        "Never infer absence from silence. Respect negation. Each quote must be "
        "copied exactly from the patient text. "
        'Return only JSON: {"facts":{"field":{"value":true,"quote":"exact words"}}}. '
        "\nPATIENT:\n" + patient_text(patient)
        + "\nSOURCE FIELDS:\n" + json.dumps(fields)
    )


def extract_management_facts(
    patient: dict[str, Any], cards: Any, disease: str,
    known: dict[str, Any], llm_generate: Callable[[str], str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (verified_facts, diagnostics). Never raises; failure means unknown."""
    fields = needed_fields(list(cards), disease, known)
    info: dict[str, Any] = {"requested_fields": fields, "accepted": [], "rejected": {}}
    if not fields:
        return {}, info
    try:
        raw = llm_generate(_prompt(patient, fields))
        proposals = json.loads(raw).get("facts", {})
        if not isinstance(proposals, dict):
            raise ValueError("facts must be a mapping")
    except Exception as exc:  # LLM/transport/JSON failure => facts stay unknown
        info["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        return {}, info
    allowed = set(fields)
    try:
        facts, rejected = verify_facts(proposals, patient, allowed)
        booleans, bool_rejected = verify_boolean_facts(proposals, patient, allowed, llm_generate)
        numbers, num_rejected = verify_measurement_facts(proposals, patient, allowed, llm_generate)
    except Exception as exc:
        # The shared verifier can receive malformed provider JSON (e.g. a list).
        # Keep this recovery local to Improved management; diagnosis is unchanged.
        info["error"] = f"Fact verification unavailable: {type(exc).__name__}"
        return {}, info
    facts.update(booleans)
    for name in list(facts):  # numeric/boolean values are decided by their own check
        entry = proposals.get(name)
        if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)) \
                and not isinstance(entry.get("value"), bool):
            facts.pop(name, None)
    facts.update(numbers)
    rejected.update(bool_rejected)
    rejected.update(num_rejected)
    facts = {k: v for k, v in facts.items() if k in allowed}
    info["accepted"] = sorted(facts)
    info["rejected"] = {k: v for k, v in rejected.items() if k not in facts}
    return facts, info
