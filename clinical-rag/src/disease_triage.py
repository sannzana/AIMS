"""Source-grounded, disease-agnostic diagnostic triage.

Candidate protocols and criteria come exclusively from the card corpus. A single
LLM call compares patient facts with those criteria. No threshold or symptom
catalogue lives in Python. Invalid/uncertain output fails closed, not to keywords.
"""
from __future__ import annotations

import json
from typing import Any, Callable

from card_roles import assign_role
from llm import generate, LLMError
from fact_verification import rule_fields, verify_facts, verify_boolean_facts, verify_measurement_facts


def _raw(card: Any) -> dict[str, Any]:
    return card.model_dump() if hasattr(card, "model_dump") else vars(card)


def _case_cards(cards: dict[str, Any]) -> list[Any]:
    return [card for card in cards.values()
            if card.clinical_category == "Diagnosis"
            and assign_role(card) == "case_definition"]


def _fields(node: Any) -> set[str]:
    if isinstance(node, dict):
        own = {node["field"]} if isinstance(node.get("field"), str) else set()
        return own.union(*(_fields(v) for v in node.values()))
    if isinstance(node, list):
        return set().union(*(_fields(v) for v in node))
    return set()


def _json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Diagnostic triage must be a JSON object")
    return parsed


def _patient_text(patient: dict[str, Any]) -> str:
    return "\n".join(f"{key}: {value}" for key, value in patient.items()
                     if not key.startswith("_") and value not in (None, "", [], {}))


def triage(patient: dict[str, Any], cards: dict[str, Any],
           llm_generate: Callable[[str], str] = generate) -> dict[str, Any]:
    """Grounded candidate selection. No heuristic fallback on LLM errors.

    Returns names/card IDs compatible with the existing diagnosis pipeline.
    Patient quotations are verified against the literal submitted input.
    Card identity and protocol always come from the corpus, never the model.
    """
    definitions = _case_cards(cards)
    if not definitions:
        return {"status": "no_match", "candidates": [], "features": {},
                "notice": "No source case-definition cards."}
    concise = []
    for c in definitions:
        raw = _raw(c)
        concise.append({
            "id": c.card_id,
            "protocol": c.disease_protocol,
            "criteria": raw.get("relevant_symptoms_or_prerequisites") or "",
            "logic": raw.get("conditions_decision_logic") or [],
        })
    # All source field names are discovered across the entire evidence corpus.
    fields = sorted(rule_fields(cards))
    def build_prompt(batch):
        prompt = (
            "You are an evidence matching component, not a treating clinician. "
            "Return ONLY JSON. Compare ALL patient observations (including negated "
            "findings, counts, lab units, duration, blood pressure) against the "
            "source case definitions. 'No', 'none', 'denies', 'absent', and 'without' "
            "indicate NEGATION, not a positive match. Do not treat missing as false. "
            "A numeric condition must use the patient's measured value and the "
            "operator/threshold in that specific source criterion. Do not invent "
            "thresholds, diagnoses, clinical measurements or patient quotes. "
            "Do not exclude an unrelated diagnosis just because another condition "
            "is present; comorbid diseases can coexist. Do not use severity or "
            "differential cards to introduce new diagnoses. If necessary evidence "
            "is missing, mark uncertain and request information. "
            "Return {\"candidates\":[{\"card_id\":\"id\","
            "\"decision\":\"supported|uncertain|contradicted\","
            "\"patient_quotes\":[\"EXACT substring copied from patient input\"],"
            "\"missing\":[\"needed datum\"]}],"
            "\"facts\":{\"source_rule_field\":{\"value\":number|string|boolean|null,\"quote\":\"exact patient substring\"}}}. "
            "Select only source card IDs from the provided cards. The facts dictionary "
            "may contain only provided field names. Each non-null fact MUST be "
            "an object {\"value\":scalar,\"quote\":\"exact quoted patient substring\"}. "
            "Use null for missing facts. Values must be explicitly stated "
            "in the patient data. Missing values must be null. "
            "\nPATIENT:\n" + _patient_text(patient) +
            "\nAVAILABLE SOURCE FIELDS:\n" + json.dumps(fields) +
            "\nCASE-DEFINITION CARDS:\n" + json.dumps(batch, ensure_ascii=False)
    )
        return prompt
    # A single unbounded clinical corpus can exceed the model request budget.
    # Split case-definition cards by serialized prompt size, never by disease.
    batches = []
    batch = []
    for card in concise:
        trial = batch + [card]
        if batch and len(build_prompt(trial).encode("utf-8")) > 23000:
            batches.append(batch)
            batch = [card]
        else:
            batch = trial
        if len(build_prompt(batch).encode("utf-8")) > 28000:
            return {"status": "unavailable", "candidates": [], "features": {},
                    "notice": "A source case-definition card exceeds the model request budget."}
    if batch:
        batches.append(batch)
    responses = []
    failures = []
    for batch in batches:
        try:
            responses.append(_json_object(llm_generate(build_prompt(batch))))
        except (LLMError, ValueError, TypeError, IndexError, KeyError) as exc:
            failures.append(type(exc).__name__ + ": " + str(exc)[:160])
    if not responses:
        return {"status": "unavailable", "candidates": [], "features": {},
                "notice": "Diagnostic model calls failed: " + "; ".join(failures[:2])}
    response = {"candidates": [], "facts": {}}
    for result in responses:
        entries = result.get("candidates", [])
        if isinstance(entries, list):
            response["candidates"].extend(entries)
        values = result.get("facts", {})
        if isinstance(values, dict):
            response["facts"].update(values)
    proposals = response.get("facts", {})
    facts, rejected_facts = verify_facts(proposals, patient, set(fields))
    boolean_facts, boolean_rejected = verify_boolean_facts(
        proposals, patient, set(fields), llm_generate
    )
    facts.update(boolean_facts)
    rejected_facts.update(boolean_rejected)
    measurement_facts, measurement_rejected = verify_measurement_facts(
        proposals, patient, set(fields), llm_generate
    )
    # Replace unverified numeric values with separately checked measurements.
    for field, entry in (proposals.items() if isinstance(proposals, dict) else []):
        if isinstance(entry, dict) and isinstance(entry.get("value"), (int, float)) and not isinstance(entry.get("value"), bool):
            facts.pop(field, None)
    facts.update(measurement_facts)
    rejected_facts.update(measurement_rejected)
    allowed = {c.card_id: c for c in definitions}
    observed = _patient_text(patient).casefold()
    supported = []
    uncertain = []
    for entry in response.get("candidates", []):
        if not isinstance(entry, dict) or entry.get("card_id") not in allowed:
            continue
        raw_quotes = entry.get("patient_quotes", [])
        if not isinstance(raw_quotes, list):
            continue
        quotes = [q.strip() for q in raw_quotes
                  if isinstance(q, str) and len(q.strip()) >= 3
                  and q.strip().casefold() in observed]
        if len(quotes) != len(raw_quotes):
            continue
        decision = entry.get("decision")
        card = allowed[entry["card_id"]]
        item = {"disease": card.disease_protocol, "card_id": card.card_id,
                "matched_features": quotes, "missing": entry.get("missing", [])}
        if decision == "supported" and quotes:
            supported.append(item)
        elif decision == "uncertain":
            uncertain.append(item)
    # Preserve disease diversity, not multiple cards of the same protocol.
    def unique(values):
        found = {}
        for value in values:
            found.setdefault(value["disease"], value)
        return list(found.values())
    candidates = unique(supported)
    if not candidates:
        candidates = unique(uncertain)
    status = ("clear" if len(candidates) == 1 and supported else
              "ambiguous" if candidates else "no_match")
    return {"status": status, "candidates": candidates, "features": facts,
            "rejected_facts": rejected_facts,
            "notice": None if supported else "Insufficient verified diagnostic evidence."}
