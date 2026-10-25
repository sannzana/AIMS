"""Source-bound instruction scopes for Improved management only.

Clinical mappings live in version-bound data, not disease-specific Python
branches. Missing/stale mappings remain visible and pending review.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import re

from applicability import TruthValue, _evaluate_tree, _rule_fields

SOURCE_KEYS = ("conditions_decision_logic", "if_else_conditions_and_corresponding_actions",
               "exceptions_contraindications", "relevant_symptoms_or_prerequisites", "medication_details")


def source_digest(card):
    raw = card.model_dump()
    return hashlib.sha256(json.dumps({k: raw.get(k) for k in SOURCE_KEYS},
                                     sort_keys=True, ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=1)
def scope_records():
    try:
        return tuple(json.loads(line) for line in Path(__file__).with_name(
            "management_exception_scopes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, ValueError, TypeError):
        return ()


@lru_cache(maxsize=1)
def prerequisite_records():
    try:
        return tuple(json.loads(line) for line in Path(__file__).with_name(
            "management_prerequisites.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    except (OSError, ValueError, TypeError):
        return ()


def prerequisite_policies(card):
    raw = card.model_dump()
    ids = {rule.get("rule_id") for key in ("conditions_decision_logic", "if_else_conditions_and_corresponding_actions")
           for rule in raw.get(key) or [] if isinstance(rule, dict)}
    result = []
    for record in prerequisite_records():
        if not isinstance(record, dict) or record.get("card_id") != card.card_id:
            continue
        scope = record.get("scope_rule_ids")
        valid = (record.get("source_digest") == source_digest(card)
                 and isinstance(scope, list) and bool(scope) and all(isinstance(rule, str) for rule in scope) and set(scope) <= ids
                 and isinstance(record.get("source_quote"), str)
                 and bool(record["source_quote"].strip()) and record["source_quote"] == raw.get("relevant_symptoms_or_prerequisites")
                 and isinstance(record.get("condition"), dict))
        result.append({"exception": {"condition": record.get("condition"),
                                     "interpretation": "Source prerequisite: " + str(record.get("source_quote", ""))},
                       "kind": "prerequisite" if valid else "scope_pending", "scope": scope if valid else None,
                       "source_quote": str(raw.get("relevant_symptoms_or_prerequisites") or ""), "medication_terms": [], "index": None})
    return result


def policies(card, records=None):
    raw = card.model_dump()
    records = scope_records() if records is None else records
    ids = {rule.get("rule_id") for key in ("conditions_decision_logic", "if_else_conditions_and_corresponding_actions")
           for rule in raw.get(key) or [] if isinstance(rule, dict)}
    result = []
    for index, original in enumerate(raw.get("exceptions_contraindications") or []):
        exception = original if isinstance(original, dict) else {"restriction": str(original)}
        record = next((r for r in records if isinstance(r, dict) and r.get("card_id") == card.card_id
                       and r.get("exception_index") == index), None)
        valid = bool(record and record.get("source_digest") == source_digest(card)
                     and record.get("kind") in {"restriction", "interpretation", "guidance"}
                     and isinstance(record.get("scope_rule_ids"), list)
                     and all(isinstance(rule_id, str) for rule_id in record["scope_rule_ids"])
                     and set(record["scope_rule_ids"]) <= ids
                     and isinstance(record.get("source_quote"), str)
                     and bool(record["source_quote"].strip())
                     and isinstance(record.get("medication_terms", []), list)
                     and all(isinstance(term, str) and term.casefold() in json.dumps(exception, ensure_ascii=False).casefold()
                             for term in record.get("medication_terms", []))
                     and any(record["source_quote"] == value for value in exception.values() if isinstance(value, str)))
        explicit = exception.get("related_rule_ids")
        if not isinstance(explicit, list) or not all(isinstance(rule_id, str) for rule_id in explicit):
            explicit = None
        result.append({"exception": exception, "kind": record["kind"] if valid else "scope_pending",
                       "scope": record["scope_rule_ids"] if valid else explicit if isinstance(explicit, list) else None,
                       "source_quote": record["source_quote"] if valid else exception.get("interpretation") or exception.get("restriction") or str(original),
                       "medication_terms": record.get("medication_terms", []) if valid else [],
                       "index": index})
    return result


def instruction_safety(card, rule_id, context, records=None):
    relevant = [policy for policy in policies(card, records) + prerequisite_policies(card)
                if policy["scope"] is None or rule_id in policy["scope"] or not policy["scope"]]
    restrictions, notes, missing = [], [], []
    restricted = pending = False
    for policy in relevant:
        exception = policy["exception"]
        if policy["kind"] in {"interpretation", "guidance"}:
            notes.append(policy["source_quote"])
            continue
        restrictions.append(exception)
        if policy["kind"] == "scope_pending":
            pending = True
            missing.append("source_restriction_scope_review")
            continue
        state = _evaluate_tree(exception.get("condition"), context)
        if policy["kind"] == "prerequisite":
            state = TruthValue.FALSE if state is TruthValue.TRUE else TruthValue.TRUE if state is TruthValue.FALSE else TruthValue.UNKNOWN
        restricted |= state is TruthValue.TRUE
        pending |= state is TruthValue.UNKNOWN
        if state is TruthValue.UNKNOWN:
            missing.extend(field for field in _rule_fields(exception.get("condition")) if context.get(field) is None)
    return {"restricted": restricted, "pending": pending, "missing": sorted(set(missing)),
            "restrictions": restrictions, "notes": list(dict.fromkeys(notes))}


def applicability_from_options(options):
    if any(o["status"] == "supported" for o in options):
        status = "applicable"
    elif any(o["status"] in {"conditional", "restricted"} for o in options):
        status = "undetermined"
    else:
        status = "not_applicable"
    return {"status": status, "met": [], "failed": [],
            "missing_inputs": sorted({f for o in options if o["status"] == "conditional" for f in o["missing_inputs"]})}


def validation_views(cards, options, context):
    """Keep legacy validators conservative without reapplying blanket exceptions.

    Mixed cards stay review-only at whole-card level. The instruction-level
    source view remains available; no sibling's regimen is authorised by a match.
    These ephemeral views never replace source cards or source traceability.
    """
    from schema import Conditions, Condition
    result, facts = {}, dict(context)
    facts["_management_blocked_instruction"] = False
    for card_id, card in cards.items():
        entries = [entry for entry in options if entry["card_id"] == card_id]
        if entries and all(entry["status"] == "supported" for entry in entries):
            condition = Conditions()
        elif entries and all(entry["status"] in {"not_applicable", "restricted"} for entry in entries):
            condition = Conditions(all=[Condition(field="_management_blocked_instruction", op="==", value=True)])
        else:
            fields = sorted({f for entry in entries if entry["status"] == "conditional" for f in entry["missing_inputs"]})
            condition = Conditions(required_inputs=fields or ["instruction_specific_source_review"])
        result[card_id] = card.model_copy(update={"conditions": condition, "structured_rules": None,
                                                "exceptions_contraindications": []})
    return result, facts


def safety_notices(cards, disease, context, medicines, selected_ids):
    """Mention matching source terms without inferring drug classes or use.

    This is not a drug-interaction database. A mention (even a negated one) is
    reported as a mention, never as proof that a drug is being taken.
    """
    tokens = {word.casefold().rstrip("s") for word in re.findall(r"\w+", str(medicines or ""))}
    result = []
    for card in cards:
        if card.disease_protocol != disease or card.clinical_category not in {"Medication", "Investigation", "Advice"}:
            continue
        for policy in policies(card) + prerequisite_policies(card):
            state = _evaluate_tree(policy["exception"].get("condition"), context)
            if policy["kind"] == "prerequisite":
                state = TruthValue.FALSE if state is TruthValue.TRUE else TruthValue.TRUE if state is TruthValue.FALSE else TruthValue.UNKNOWN
            mentioned = any(set(re.findall(r"\w+", term.casefold().rstrip("s"))) <= tokens
                            for term in policy["medication_terms"] if term)
            if card.card_id not in selected_ids and not mentioned and not (
                    policy["kind"] == "restriction" and state is TruthValue.TRUE):
                continue
            if policy["kind"] in {"restriction", "prerequisite"}:
                status = "restriction_applies" if state is TruthValue.TRUE else "conditional" if state is TruthValue.UNKNOWN else "not_triggered"
            else:
                status = policy["kind"]
            result.append({"card_id": card.card_id, "kind": policy["kind"], "status": status,
                           "scope_rule_ids": policy["scope"], "text": policy["source_quote"],
                           "medicine_mentioned": mentioned, "condition": policy["exception"].get("condition"),
                           "source_population": policy["exception"].get("population"),
                           "pdf_pages": card.pdf_page_numbers})
    return result
