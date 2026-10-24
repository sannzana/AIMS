"""Source-grounded, disease-agnostic diagnostic triage.

Candidate protocols and criteria come exclusively from the card corpus. A single
LLM call compares patient facts with those criteria. No threshold or symptom
catalogue lives in Python. Invalid/uncertain output fails closed, not to keywords.
"""
from __future__ import annotations

import json
import re
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
           llm_generate: Callable[[str], str] = generate,
           candidate_search: Callable[[dict[str, Any], dict[str, Any]], list[str]] | None = None) -> dict[str, Any]:
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
    # Retrieve relevant case definitions first; avoid overflowing the LLM with the entire corpus.
    # Retrieval discovers candidates; the model decides clinical fit using source evidence.
    try:
        if candidate_search is not None:
            selected_ids = set(candidate_search(patient, cards))
        else:
            from retrieve import Retriever, clinical_query
            engine = Retriever(cards=cards)
            hits = engine.search(
                clinical_query(patient), category="Diagnosis",
                roles={"case_definition"}, n=8,
            )
            selected_ids = {hit["card_id"] for hit in hits}
        shortlist = [entry for entry in concise if entry["id"] in selected_ids]
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        return {"status": "unavailable", "candidates": [], "features": {},
                "notice": "Evidence retrieval unavailable: " + str(exc)[:180]}
    if not shortlist:
        return {"status": "no_match", "candidates": [], "features": {},
                "notice": "No case-definition evidence retrieved."}

    # Candidate selection should not request hundreds of unrelated fact fields.
    # Keep fact extraction optional; evidence scoring is the primary task.
    def build_prompt(batch):
        return (
            "Match the patient's described clinical meaning to the provided case "
            "definitions. Return JSON only. Negated observations are NOT positives; "
            "missing findings are unknown, never false. Respect NEGATION scope. Avoid unsupported diagnosis. "
            "Respect source criteria and preserve possible comorbidities. "
            'JSON format: {"candidates":[{"card_id":"ID",'
            '"decision":"supported|uncertain|contradicted",'
            '"patient_quotes":["exact source quote"],"missing":[]}],'
            '"facts":{"field":{"value":null,"quote":"exact patient quote"}}}. '
            "Facts must come from source field names; for a real observation "
            "replace null with its explicit scalar value. Do not invent data. "
            "\\nPATIENT:\\n" + _patient_text(patient)
            + "\\nSOURCE FIELDS:\\n" + json.dumps(sorted(_fields(batch)))
            + "\\nCASE CARDS:\\n" + json.dumps(batch, ensure_ascii=False)
        )

    batches = []
    current = []
    for card in shortlist:
        trial = current + [card]
        if current and len(build_prompt(trial).encode("utf-8")) > 20000:
            batches.append(current)
            current = [card]
        else:
            current = trial
        if len(build_prompt(current).encode("utf-8")) > 27000:
            return {"status": "unavailable", "candidates": [], "features": {},
                    "notice": "Single source card exceeds model context budget."}
    if current:
        batches.append(current)
    responses, failures = [], []
    for batch in batches:
        try:
            responses.append(_json_object(llm_generate(build_prompt(batch))))
        except (LLMError, ValueError, TypeError, IndexError, KeyError) as exc:
            failures.append(type(exc).__name__ + ": " + str(exc)[:160])
    if not responses:
        return {"status": "unavailable", "candidates": [], "features": {},
                "notice": "Diagnostic model unavailable: " + "; ".join(failures[:2])}
    response = {"candidates": [], "facts": {}}
    for result in responses:
        if isinstance(result.get("candidates"), list):
            response["candidates"].extend(result["candidates"])
        if isinstance(result.get("facts"), dict):
            response["facts"].update(result["facts"])
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
    allowed = {c.card_id: c for c in definitions if c.card_id in selected_ids}
    observed = _patient_text(patient).casefold()
    supported = []
    uncertain = []
    decision_trace = {
        "model_decisions": {"supported": 0, "uncertain": 0,
                            "contradicted": 0, "invalid": 0},
        "rejected_candidates": [],
        "accepted_supported_card_ids": [],
        "accepted_uncertain_card_ids": [],
    }
    diagnostic_rejections = {"unknown_card": 0, "invalid_quotes": 0, "unusable_decision": 0}
    for entry in response.get("candidates", []):
        decision = entry.get("decision") if isinstance(entry, dict) else None
        if decision in decision_trace["model_decisions"] and decision != "invalid":
            decision_trace["model_decisions"][decision] += 1
        else:
            decision_trace["model_decisions"]["invalid"] += 1
        if not isinstance(entry, dict) or entry.get("card_id") not in allowed:
            diagnostic_rejections["unknown_card"] += 1
            decision_trace["rejected_candidates"].append({"reason": "unknown_card"})
            continue
        raw_quotes = entry.get("patient_quotes", [])
        if not isinstance(raw_quotes, list):
            diagnostic_rejections["invalid_quotes"] += 1
            decision_trace["rejected_candidates"].append({"card_id": entry["card_id"], "reason": "invalid_quote_type"})
            continue
        # Validate quotes independently. Normalize case and punctuation, but
        # preserve token boundaries so "rash" cannot match inside "crash".
        patient_source = _patient_text(patient)

        def normalize_quote(value: str) -> str:
            return " ".join(re.findall(r"[a-z0-9]+|[\u0980-\u09ff]+", value.casefold()))

        normalized_patient = f" {normalize_quote(patient_source)} "
        quotes = []
        invalid_quote_count = 0
        negated_quote_count = 0
        for raw_quote in raw_quotes:
            if not isinstance(raw_quote, str) or len(raw_quote.strip()) < 3:
                invalid_quote_count += 1
                continue
            quote = raw_quote.strip()
            normalized_quote = normalize_quote(quote)
            if not normalized_quote or f" {normalized_quote} " not in normalized_patient:
                invalid_quote_count += 1
                continue

            # Check the local source clause before the quoted occurrence. This
            # is intentionally conservative; it is not a general clinical NLP
            # negation parser.
            negated = False
            for occurrence in re.finditer(re.escape(quote), patient_source, re.I):
                prefix = re.split(r"[.!?;\n]", patient_source[:occurrence.start()])[-1].casefold()
                cue = re.search(
                    r"\b(?:no|not|without|denies|denied|negative for|absent|never)\b",
                    prefix,
                )
                if cue:
                    after_cue = prefix[cue.end():]
                    if not re.search(r"\b(?:but|however|except|although|yet)\b", after_cue):
                        negated = True
                suffix = re.split(
                    r"[.!?;\n]", patient_source[occurrence.end():], maxsplit=1
                )[0].casefold()
                if re.search(r"\b(?:absent|negative)\b|(?:নেই|নাই|না)", suffix):
                    negated = True
                break
            if negated:
                negated_quote_count += 1
                continue
            quotes.append(quote)

        if invalid_quote_count:
            diagnostic_rejections["invalid_quotes"] += invalid_quote_count
        if invalid_quote_count or negated_quote_count:
            decision_trace["rejected_candidates"].append({
                "card_id": entry["card_id"], "reason": "partial_quote_rejection",
                "valid_count": len(quotes), "invalid_count": invalid_quote_count,
                "negated_count": negated_quote_count, "submitted_count": len(raw_quotes)
            })
        decision = entry.get("decision")
        card = allowed[entry["card_id"]]
        item = {"disease": card.disease_protocol, "card_id": card.card_id,
                "matched_features": quotes, "missing": entry.get("missing", [])}
        if decision == "supported" and quotes:
            supported.append(item)
            decision_trace["accepted_supported_card_ids"].append(card.card_id)
        elif decision == "uncertain":
            uncertain.append(item)
            decision_trace["accepted_uncertain_card_ids"].append(card.card_id)
        else:
            diagnostic_rejections["unusable_decision"] += 1
            # If quote validation already explains why there is no support,
            # avoid a duplicate rejection entry for the same candidate.
            if not invalid_quote_count and not negated_quote_count:
                decision_trace["rejected_candidates"].append({
                    "card_id": card.card_id,
                    "reason": "unsupported_decision_or_no_quotes"
                })
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
              "ambiguous" if candidates else "unavailable" if failures else "no_match")
    diagnostics = {"retrieved_card_ids": sorted(selected_ids),
                   "case_definition_count": len(definitions),
                   "shortlist_count": len(shortlist),
                   "llm_batches": len(batches), "successful_llm_batches": len(responses),
                   "failed_llm_batches": len(failures),
                   "model_candidate_count": len(response.get("candidates", [])),
                   "rejections": diagnostic_rejections,
                   "decision_trace": decision_trace,
                   "verified_fact_count": len(facts),
                   "rejected_fact_count": len(rejected_facts)}
    return {"status": status, "candidates": candidates, "features": facts,
            "diagnostics": diagnostics, "rejected_facts": rejected_facts,
            "notice": None if supported else ("Model batch failed: " + "; ".join(failures[:2]) if failures else "No supported candidate returned by the model.")}
