"""Collect source-defined management facts (Improved mode only).

Diagnosis extracts facts only for the case-definition cards it compares. The
fields that decide the treatment branch (dehydration signs, severity signs ...)
live in the *assessment* cards, which diagnosis never offers to the model, so
they were always unknown and every plan card was withheld.

This module asks, once, for missing classification, indication, branch and
restriction fields used by the selected protocol. It reuses the same
fail-closed checks as diagnosis (exact quote in the patient text, independent
polarity confirmation, numeric value present in its quote). Nothing is inferred
from absence; anything not confirmed stays unknown. No disease, symptom or
threshold is named here - the fields come from the cards.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

from derived_fields import _conditions, classification_rules, field_groups
from fact_verification import (patient_text, verify_boolean_facts, verify_facts,
                               verify_measurement_facts, rule_fields)


def needed_fields(cards: Any, disease: str, known: dict[str, Any]) -> list[str]:
    """Missing management fields plus dependencies of consumed classifications."""
    cards = list(cards)
    rules = classification_rules(cards, disease)
    # only classifications that some treatment card actually depends on
    used = set().union(*(g["labels"] for g in
                         field_groups(cards, disease, {l for _, l, _ in rules}).values()), set())
    # Management indications and restrictions matter even without a derived
    # classification. Discover these from source rules, never disease names.
    management = {c.card_id: c for c in cards if c.disease_protocol == disease
                  and c.clinical_category in {"Investigation", "Medication", "Advice"}}
    names: set[str] = rule_fields(management)
    for card in management.values():
        for cond in _conditions(card.model_dump().get("if_else_conditions_and_corresponding_actions")):
            if isinstance(cond.get("field"), str):
                names.add(cond["field"])
    for _, label, tree in rules:
        if label not in used:
            continue
        for cond in _conditions(tree):
            if isinstance(cond.get("field"), str):
                names.add(cond["field"])
    derived_fields = set(field_groups(cards, disease, {l for _, l, _ in rules}))
    derived_labels = {label for _, label, _ in rules}
    return sorted(n for n in names - derived_fields - derived_labels
                  if known.get(n) in (None, ""))


def _prompt(patient: dict[str, Any], fields: list[str]) -> str:
    return (
        "Extract patient facts for the listed SOURCE FIELDS from the PATIENT text. "
        "Report a field only when the patient text explicitly states it: "
        "true = stated present, false = stated absent or normal, a number = a "
        "stated measurement. If the text does not say, leave the field out. "
        "Never infer absence from silence. Respect negation. Each quote must be "
        "copied exactly from the patient text. "
        "Do not derive an age-group boundary, severity class, sign count or "
        "treatment setting from your medical knowledge. Extract a category only "
        "if explicitly reported. Do not convert units or infer negative findings "
        "from a general statement such as no known chronic diseases. "
        'Return only JSON: {"facts":{"field":{"value":true,"quote":"exact words"}}}. '
        "\nPATIENT:\n" + patient_text(patient)
        + "\nSOURCE FIELDS:\n" + json.dumps(fields)
    )


def extract_management_facts(
    patient: dict[str, Any], cards: Any, disease: str,
    known: dict[str, Any], llm_generate: Callable[[str], str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (verified_facts, diagnostics). Never raises; failure means unknown."""
    cards = list(cards)
    fields = needed_fields(cards, disease, known)
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
    # A second model agreeing is not permission to invent a count or a derived
    # classification. Require explicit wording for these source-schema claims.
    labels = [set(label.split("_")) for _, label, _ in classification_rules(cards, disease)]
    source_rejected = {}
    filtered = {}
    for name, entry in proposals.items():
        quote = entry.get("quote", "") if isinstance(entry, dict) else ""
        value = entry.get("value") if isinstance(entry, dict) else None
        words = name.split("_")
        if words[-1:] == ["count"]:
            noun = words[-2].rstrip("s") if len(words) > 1 else "count"
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not isinstance(quote, str)
                    or not re.search(r"\b(?:count|" + re.escape(noun) + r"s?)\b", quote, re.I)):
                source_rejected[name] = "Count must be explicitly reported with its quantity, not computed from symptoms"
                continue
        if len(words) >= 2 and any(set(words) <= label for label in labels):
            if not isinstance(quote, str) or not re.search(r"\b" + r"[\s_-]+".join(map(re.escape, words)) + r"\b", quote, re.I):
                source_rejected[name] = "Classification must be explicitly reported or derived by source rules"
                continue
        filtered[name] = entry
    proposals = filtered
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
    rejected.update(source_rejected)
    facts = {k: v for k, v in facts.items() if k in allowed}
    info["accepted"] = sorted(facts)
    info["rejected"] = {k: v for k, v in rejected.items() if k not in facts}
    return facts, info
