"""Conservative, disease-neutral verification of extracted patient facts."""
from __future__ import annotations
import re
from typing import Any

NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")

def rule_fields(cards: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    def walk(obj):
        if isinstance(obj, dict):
            if isinstance(obj.get("field"), str):
                names.add(obj["field"])
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)
    for card in cards.values():
        data = card.model_dump() if hasattr(card, "model_dump") else vars(card)
        for key in ("conditions_decision_logic", "structured_rules", "exceptions_contraindications", "conditions"):
            walk(data.get(key))
    return names

def patient_text(patient: dict[str, Any]) -> str:
    return "\n".join(str(value) for key, value in patient.items() if not key.startswith("_") and value not in (None, "", [], {}))

def verify_facts(proposals: Any, patient: dict[str, Any], allowed: set[str]):
    """Require exact patient quotation; never treat unknown or conflicting data as true.

    Boolean polarity cannot be safely proven by substring alone, so it remains
    unverified rather than converting negated findings to positive evidence.
    """
    facts, rejected = {}, {}
    text = patient_text(patient).casefold()
    if not isinstance(proposals, dict):
        return facts, {"_facts": "Expected mapping"}
    for name, entry in proposals.items():
        if name not in allowed:
            rejected[name] = "Field absent from evidence rules"
            continue
        if not isinstance(entry, dict):
            rejected[name] = "Missing quoted provenance"
            continue
        quote, value = entry.get("quote"), entry.get("value")
        if not isinstance(quote, str) or len(quote.strip()) < 3 or quote.strip().casefold() not in text:
            rejected[name] = "Quote not in original patient input"
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            rejected[name] = "Polarity requires separate semantic verification"
        elif isinstance(value, (float, int)):
            if any(float(m.group()) == float(value) for m in NUMBER.finditer(quote)):
                facts[name] = value
            else:
                rejected[name] = "Numeric value absent from quote"
        elif isinstance(value, str) and value.strip() and value.casefold() in quote.casefold():
            facts[name] = value
        else:
            rejected[name] = "Unsupported or unproven value"
    return facts, rejected


def verify_boolean_facts(
    proposals: Any,
    patient: dict[str, Any],
    allowed: set[str],
    verify_semantics,
) -> tuple[dict[str, bool], dict[str, str]]:
    """Independently check boolean claims against the original patient narrative.

    A second model pass must agree on meaning, field, quote and polarity.
    Disagreement, missing evidence, negation ambiguity and errors fail closed.
    This is semantic verification, not a keyword or disease-specific rule.
    """
    accepted: dict[str, bool] = {}
    rejected: dict[str, str] = {}
    source = patient_text(patient)
    if not isinstance(proposals, dict):
        return accepted, {"_facts": "Expected mapping"}
    proposed = {}
    for field, entry in proposals.items():
        if field not in allowed or not isinstance(entry, dict):
            continue
        value, quote = entry.get("value"), entry.get("quote")
        if (not isinstance(value, bool) or not isinstance(quote, str)
                or len(quote.strip()) < 3 or quote.strip().casefold() not in source.casefold()):
            continue
        proposed[field] = {"value": value, "quote": quote.strip()}
    if not proposed:
        return accepted, rejected

    import json
    prompt = (
        "You are a separate semantic fact verification pass, NOT a diagnosing "
        "component. Interpret negation scope, temporal context, hedging, "
        "hypotheticals, and contradictory statements. Determine whether each "
        "proposed boolean accurately describes what the patient explicitly "
        "reported. 'No history of' does not mean a current finding is absent "
        "unless the patient's wording establishes that. Missing data is unknown. "
        "Do not infer diagnoses or rely on mere occurrence of a word. "
        "Return only JSON of the form "
        "{\\\"decisions\\\":{\\\"source_field\\\":\\\"confirmed|rejected|uncertain\\\"}}. "
        "Confirm ONLY when both the clinical proposition and polarity are "
        "directly supported by the patient narrative, not merely by a substring. "
        "\\nPATIENT TEXT:\\n" + source +
        "\\nPROPOSED FACTS:\\n" + json.dumps(proposed, ensure_ascii=False)
    )
    try:
        raw = verify_semantics(prompt)
        if not isinstance(raw, str):
            raise ValueError("Verifier must return JSON string")
        answer = json.loads(raw)
        decisions = answer.get("decisions", {})
        if not isinstance(decisions, dict):
            raise ValueError("Invalid decisions")
    except (ValueError, TypeError, KeyError, RuntimeError):
        return accepted, {field: "Semantic verification unavailable" for field in proposed}
    for field, entry in proposed.items():
        if decisions.get(field) == "confirmed":
            accepted[field] = entry["value"]
        else:
            rejected[field] = "Polarity not independently confirmed"
    return accepted, rejected
