"""Optional, management-only source questions and offline reassessment.

No retrieval, generation, diagnosis mutation, or inferred clinical thresholds.
Raw source labels are deliberately retained rather than translated into invented
clinical definitions. Answers are user-supplied, not verified clinician facts.
"""
from copy import deepcopy
import math

from applicability import _get_value, _evaluate_tree, TruthValue
from derived_fields import classification_rules, field_groups, _conditions
from management_evidence import prerequisite_state, rule_options, review_summary, source_context
from management_presentation import split_conditional_sources
from management_safety import policies, prerequisite_policies, safety_notices

# Units describe source field names, never clinical cutoffs or age groups.
_UNITS = {"_years": "years", "_months": "months", "_kg": "kg", "_mg_dl": "mg/dL",
          "_mmol_l": "mmol/L", "_celsius": "°C"}


def _assess(cards, disease, facts):
    from derived_fields import derive_classifications, blocked_cards
    derived, trace = derive_classifications(cards.values(), disease, facts)
    context, values, resolved = source_context(cards.values(), disease, {**facts, **derived}, trace)
    values.update({field: {label: [] for label in entry["labels"]}
                   for field, entry in trace.get("fields", {}).items()})
    return context, values, resolved, derived, trace, blocked_cards(trace)


def questions(output, cards, asked=()):
    """At most five missing, answerable source facts for the current stage."""
    if not output.get("_clarification_context") or not output.get("working_diagnosis"):
        return []
    facts = output["_clarification_context"]
    disease = output["working_diagnosis"]
    context, values, _, _, trace, _ = _assess(cards, disease, facts)
    classification = classification_rules(cards.values(), disease)
    groups = field_groups(cards.values(), disease, {label for _, label, _ in classification})
    derived_names = set(groups) | {label for _, label, _ in classification} | set(values)
    candidates = {}

    def collect(tree, card, rule_id, priority):
        for clause in _conditions(tree):
            field = clause.get("field")
            if (not isinstance(field, str) or field in asked or field in derived_names
                    or field.startswith("all_") or field.endswith("_count")):
                continue
            value, present = _get_value(context, field)
            if present and value is not None and value != "":
                continue
            expected = clause.get("value")
            unit = next((u for suffix, u in _UNITS.items() if field.endswith(suffix)), None)
            kind = "boolean" if isinstance(expected, bool) else "number" if (
                isinstance(expected, (int, float)) and unit) else None
            # Free text and source-assigned enums need reviewed labels/options first.
            if kind is None:
                continue
            entry = candidates.setdefault(field, {"field": field, "label": field.replace("_", " "),
                "kind": kind, "unit": unit, "priority": priority, "sources": []})
            if entry["kind"] != kind:
                entry["ambiguous"] = True
            entry["priority"] = min(entry["priority"], priority)
            ref = {"card_id": card.card_id, "rule_id": rule_id, "pdf_pages": card.pdf_page_numbers}
            if ref not in entry["sources"]:
                entry["sources"].append(ref)

    primary, additional = split_conditional_sources(output.get("source_review_options", []),
                                           output.get("management_diagnostics", {}))
    # A partially matched instruction can still yield a useful question. This
    # does not promote it to a recommendation or change presentation grouping.
    partial = [entry for entry in additional if any(
        rule.get("status") == "conditional" and any(
            _evaluate_tree(clause, context) is TruthValue.TRUE
            for clause in _conditions(rule.get("condition")))
        for rule in entry.get("conditional_rules", []))]
    for entry in primary + partial:
        card = cards.get(entry["card_id"])
        if not card or card.clinical_category not in {"Medication", "Investigation"}:
            continue  # Later reassessment/discharge questions are a separate stage.
        for rule in entry.get("conditional_rules", []):
            if (rule.get("status") != "conditional" or rule.get("unsupported_rule")
                    or any(g["status"] != "met" for g in rule.get("prerequisites", []))):
                continue
            for policy in policies(card) + prerequisite_policies(card):
                if policy["kind"] not in {"restriction", "prerequisite"}:
                    continue
                if policy["scope"] and rule["rule_id"] in policy["scope"]:
                    collect(policy["exception"].get("condition"), card, rule["rule_id"],
                            0 if policy["kind"] == "restriction" else 1)
            collect(rule.get("condition"), card, rule["rule_id"], 3)
    # When the pathway is undecided, ask source assessment findings instead of
    # asking a user to invent the classification or a computed sign count.
    unknown_labels = {label for group in trace.get("fields", {}).values()
                      if group.get("status") == "undecided" for label in group.get("unknown_labels", [])}
    for cid, label, tree in classification:
        if label in unknown_labels:
            collect(tree, cards[cid], None, 2)
    return sorted((q for q in candidates.values() if not q.get("ambiguous")),
                  key=lambda q: (q["priority"], -len(q["sources"]), q["field"]))[:5]


def apply_answers(output, cards, answers, offered):
    """Return a new source-only view. Empty/unknown answers are exact no-ops."""
    allowed = {q["field"]: q for q in offered}
    facts = dict(output.get("_clarification_context", {}))
    accepted = {}
    for field, value in answers.items():
        if value is None:
            continue
        if field not in allowed:
            raise ValueError("Answer was not offered by this source question set.")
        question = allowed[field]
        if question["kind"] == "boolean":
            valid = isinstance(value, bool)
        else:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        if not valid:
            raise ValueError("Invalid answer type or measurement.")
        previous, present = _get_value(facts, field)
        if present and previous is not None and previous != value:
            raise ValueError("Answer conflicts with existing information; correct the original information first.")
        accepted[field] = value
    if not accepted:
        return deepcopy(output)
    facts.update(accepted)
    disease = output["working_diagnosis"]
    context, values, resolved, derived, trace, blocked = _assess(cards, disease, facts)
    original_ids = {e["card_id"] for e in output.get("source_review_options", [])}
    options = []
    for card in cards.values():
        if card.disease_protocol != disease or card.clinical_category not in {"Medication", "Investigation", "Advice"}:
            continue
        gates = prerequisite_state(card, context, values)
        rules = rule_options(card, context, gates)
        if card.card_id in original_ids or (card.card_id not in blocked and any(r["status"] == "supported" for r in rules)):
            options.extend(rules)
    result = deepcopy(output)
    result["_clarification_context"] = facts
    result["source_review_options"] = review_summary(options)
    result["_clarification_answers"] = {**output.get("_clarification_answers", {}), **accepted}
    result["_clarification_provenance"] = {**output.get("_clarification_provenance", {}), **{
        field: {"origin": "user-supplied", "sources": allowed[field]["sources"],
                "unit": allowed[field]["unit"]} for field in accepted}}
    # Never mix previously generated prose with a newly assessed source plan.
    for key in ("investigations", "medication", "advice"):
        result[key] = []
    result["notice"] = "Updated source guidance from user-supplied answers; no new LLM generation was run. Clinician review required."
    result["missing_info"] = []
    result["evidence_card_ids"] = sorted({o["card_id"] for o in options if o["status"] != "not_applicable"})
    result["safety_notices"] = safety_notices(cards.values(), disease, context, facts.get("medicines"), set(result["evidence_card_ids"]))
    diagnostics = result["management_diagnostics"]
    # Original request errors remain in the untouched original result, not as
    # the status of this offline reassessment.
    result["management_diagnostics"] = {
        "derived_classification": {"fields": derived, **trace}, "blocked_card_ids": sorted(blocked),
        "resolved_source_context": resolved, "age_normalization": diagnostics.get("age_normalization", {}),
        "clarification": {"answered_fields": sorted(result["_clarification_answers"]), "llm_calls": 0},
        "source_rule_outcomes": [{k: o[k] for k in ("card_id", "rule_id", "status", "missing_inputs", "unsupported_rule")} for o in options]}
    return result
