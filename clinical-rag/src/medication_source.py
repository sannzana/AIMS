"""Medication items are built from the source card, not from model wording.

The model may only choose WHICH Medication card applies and explain why. Every
structured field (name, dose, route, frequency, duration, instructions) comes
verbatim from the card, so it can never be paraphrased into rejection or
invented. The card's own text is also kept whole as `source_details`, which is
the only faithful form for complex cards (multi-drug, stepwise, table based).

Showing a card's drug is NOT a prescription. Each item carries an eligibility:
  eligible    structured rules are satisfied by known patient facts
  conditional some rule input is unknown, or the card has no structured rules
  ineligible  a rule is definitely not met -> the item is removed
  unassessed  no patient context was supplied (legacy callers)
Unknown inputs never count as met, and never as failed.
"""
from __future__ import annotations

import json
import re
from typing import Any

from applicability import card_rule_applicability

# Card keys that carry each simple field. Only scalar/readable values are used.
FIELD_KEYS = {
    "name": ("medicine", "fluid", "name", "drug"),
    "dose": ("dose", "dosage"),
    "route": ("route",),
    "frequency": ("frequency",),
    "duration": ("duration",),
    "instructions": ("instructions", "instruction", "infusion_note", "note"),
}
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def readable(value: Any, indent: int = 0) -> str:
    """Faithful, line-based rendering of nested card data. Never truncates."""
    pad = "  " * indent
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            text = readable(item, indent + 1)
            if text:
                inline = "\n" not in text
                lines.append(f"{pad}{key}: {text.strip()}" if inline
                             else f"{pad}{key}:\n{text}")
        return "\n".join(lines)
    if isinstance(value, (list, tuple, set)):
        parts = [readable(item, indent + 1) for item in value]
        return "\n".join(f"{pad}- {p.strip()}" if "\n" not in p else p for p in parts if p)
    return str(value)


def simple_fields(card: Any) -> dict[str, str]:
    """Structured fields the card states explicitly; absent means unknown."""
    out: dict[str, str] = {}
    details = card.model_dump().get("medication_details")
    if isinstance(details, dict):
        for field, keys in FIELD_KEYS.items():
            for key in keys:
                text = readable(details.get(key)).strip()
                if text:
                    out[field] = text
                    break
    medication = getattr(card, "medication", None)
    if medication is not None:
        for field in FIELD_KEYS:
            value = getattr(medication, field, None)
            if value not in (None, "") and field not in out:
                out[field] = str(value)
    return out


def source_details(card: Any) -> str:
    raw = card.model_dump()
    text = readable(raw.get("medication_details"))
    if raw.get("table_dependent_rules"):
        text += ("\n(Dose or schedule depends on a table in this card "
                 f"(PDF page(s) {card.pdf_page_numbers}); it is shown in the source, "
                 "not calculated here.)")
    return text.strip()


def _norm(value: Any) -> str:
    return " ".join(re.sub(r"[^0-9a-z]+", " ", str(value).casefold()).split())


def source_text(card: Any) -> str:
    raw = card.model_dump()
    keys = ("medication_details", "medication", "recommended_action",
            "table_dependent_rules", "exceptions_contraindications",
            "if_else_conditions_and_corresponding_actions", "additional_information",
            "exact_supporting_evidence_passages")
    return json.dumps({k: raw.get(k) for k in keys}, ensure_ascii=False, default=str)


def unsupported_figures(text: str, cards: list[Any]) -> set[str]:
    """Numbers written by the model that appear nowhere in the cited cards."""
    known = set(_NUMBER.findall(" ".join(source_text(c) for c in cards)))
    return {n for n in _NUMBER.findall(text or "") if n not in known}


def _restrictions(card: Any) -> list[str]:
    notes = []
    for entry in card.model_dump().get("exceptions_contraindications") or []:
        if isinstance(entry, dict):
            text = entry.get("restriction") or entry.get("interpretation") or entry.get("text")
            if isinstance(text, str) and text.strip():
                notes.append("Source restriction: " + text.strip())
        elif isinstance(entry, str) and entry.strip():
            notes.append("Source restriction: " + entry.strip())
    return notes


def eligibility_for(card: Any, patient_context: dict[str, Any] | None) -> tuple[str, list[str]]:
    notes = _restrictions(card)
    if patient_context is None:
        return "unassessed", notes
    result = card_rule_applicability(card, patient_context)
    if result is None:
        return "conditional", notes + [
            "This card has no structured eligibility rules; check its restrictions."]
    status = result.get("status")
    if status == "applicable":
        return "eligible", notes
    if status == "not_applicable":
        return "ineligible", notes
    missing = result.get("missing_inputs") or []
    extra = ["Eligibility cannot be confirmed yet"
             + (f"; not known: {', '.join(missing[:8])}." if missing else ".")]
    return "conditional", notes + extra


def finalize_medication_item(
    item: Any, cards: dict[str, Any], patient_context: dict[str, Any] | None = None,
) -> tuple[Any | None, str | None]:
    """Return (item or None, rejection reason). Notes go on item.eligibility_notes.

    Mutates `item`: structured fields come only from the primary cited
    Medication card; model figures not found in the sources are replaced.
    """
    cited = [cards[r.card_id] for r in item.evidence if r.card_id in cards]
    medication_cards = [c for c in cited if c.clinical_category == "Medication"]
    if not medication_cards:
        return None, "medication item must cite a Medication card"
    primary = medication_cards[0]
    status, restrictions = eligibility_for(primary, patient_context)
    if status == "ineligible":
        return None, (f"card {primary.card_id} is not applicable to this "
                      "patient's known facts")
    notes: list[str] = []
    fields = simple_fields(primary)
    for field in FIELD_KEYS:
        supplied = getattr(item, field, None)
        if supplied not in (None, "") and _norm(supplied) != _norm(fields.get(field, "")):
            notes.append(f"ignored model-supplied {field}; source value used")
        setattr(item, field, fields.get(field))
    bad = unsupported_figures(item.text, cited) | unsupported_figures(item.reason, cited)
    if bad:
        label = fields.get("name") or getattr(primary, "card_title", None) or primary.card_id
        item.text = f"Guideline option: {label}"
        item.reason = "See the source details; the model's wording was replaced."
        notes.append("model text contained figures not in the source ("
                     + ", ".join(sorted(bad)) + "); replaced")
    item.source_details = source_details(primary) or None
    item.eligibility = status
    item.eligibility_notes = restrictions + notes
    return item, None
