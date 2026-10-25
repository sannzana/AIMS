"""Source-driven management evidence utilities, used only by Improved mode.

Unknown is preserved at instruction level. No disease, plan or age boundary is
encoded here. Source options are review information, never prescriptions.
"""
from __future__ import annotations

import re
from typing import Any

from applicability import TruthValue, _evaluate_tree, _rule_fields, _get_value

CATEGORIES = ("Investigation", "Medication", "Advice")


def condition_text(tree):
    """Readable source logic without changing its meaning or thresholds."""
    if not isinstance(tree, dict):
        return "Eligibility needs source review."
    for key, joiner in (("all", " AND "), ("any", " OR ")):
        if isinstance(tree.get(key), list):
            return "(" + joiner.join(condition_text(t) for t in tree[key]) + ")"
    if isinstance(tree.get("field"), str):
        name = tree["field"].replace("_", " ")
        value = tree.get("value")
        op = tree.get("operator", tree.get("op", "="))
        if isinstance(value, bool) and op in ("=", "=="):
            return name + (": present/true" if value else ": absent/false")
        return f"{name} {op} {value}"
    return "Source-specific condition/table needs manual review."


def source_context(cards, disease, context, trace):
    """Resolve scalar pathway outputs explicitly assigned by diagnosis rules."""
    context = dict(context)
    for label, state in trace.get("labels", {}).items():
        if state.get("value") in ("TRUE", "FALSE") and label not in context:
            context[label] = state["value"] == "TRUE"
    candidates = {}
    for card in cards:
        if card.disease_protocol != disease or card.clinical_category != "Diagnosis":
            continue
        for branch in card.model_dump().get("if_else_conditions_and_corresponding_actions") or []:
            if not isinstance(branch, dict) or not isinstance(branch.get("then"), dict):
                continue
            for key, value in branch["then"].items():
                if key in {"action", "result", "next_card", "source_page", "rule_id"}:
                    continue
                if isinstance(value, str):
                    candidates.setdefault(key, {}).setdefault(value, []).append(branch.get("if"))
    resolved = {}
    for field, values in candidates.items():
        matches = [value for value, trees in values.items()
                   if any(_evaluate_tree(tree, context) is TruthValue.TRUE for tree in trees)]
        if len(matches) == 1 and field not in context:
            context[field] = matches[0]
            resolved[field] = matches[0]
    return context, candidates, resolved


def prerequisite_state(card, context, source_values):
    """Bind an explicit source prerequisite to an enum assigned by source rules.

    Match only the prerequisite/classification text, never the title or an
    action/result mentioning a possible future transition. Ambiguous references
    stay unresolved instead of selecting a pathway.
    """
    raw = card.model_dump()
    text = " ".join(str(raw.get(k) or "") for k in (
        "relevant_symptoms_or_prerequisites", "clinical_classification_presentation"))
    gates = []
    for field, values in source_values.items():
        from derived_fields import _conditions
        explicit = {cond["value"] for cond in _conditions(raw.get("conditions_decision_logic"))
                    if cond.get("field") == field and cond.get("operator", cond.get("op")) in ("=", "==")
                    and isinstance(cond.get("value"), str) and cond["value"] in values}
        if explicit:
            actual = context.get(field)
            gates.append({"field": field, "required_value": " OR ".join(sorted(explicit)),
                          "status": "unknown" if actual is None else "met" if actual in explicit else "not_met"})
            continue
        mentioned = []
        for value in values:
            match = re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text, re.I)
            if not match:
                continue
            clause = re.split(r"[.;\n]", text[:match.start()])[-1]
            if re.search(r"\b(?:not|without|avoid|unless|except)\b", clause, re.I):
                continue
            if re.search(r"\b(?:during|receiving|on|assigned to|requires|requiring|managed with)\s*$", clause, re.I):
                mentioned.append(value)
        if len(mentioned) != 1:
            continue
        actual = context.get(field)
        state = "unknown" if actual is None else "met" if actual == mentioned[0] else "not_met"
        gates.append({"field": field, "required_value": mentioned[0], "status": state})
    return gates


def rule_options(card, context, gates=()):
    """Expose each independent source instruction and its own AND/OR outcome."""
    raw = card.model_dump()
    rules = [(r, "condition", "result") for r in raw.get("conditions_decision_logic") or []
             if isinstance(r, dict)]
    rules += [(r, "if", "then") for r in raw.get("if_else_conditions_and_corresponding_actions") or []
              if isinstance(r, dict)]
    from management_safety import instruction_safety
    options = []
    if not rules:
        rules = [({"result": raw.get("recommended_action")}, "condition", "result")]
    for rule, condition_key, action_key in rules:
        tree = rule.get(condition_key)
        status = _evaluate_tree(tree, context)
        safety = instruction_safety(card, rule.get("rule_id"), context)
        fields = list(dict.fromkeys(_rule_fields(tree)))
        missing = [f for f in fields if _get_value(context, f)[1] is False
                   or _get_value(context, f)[0] is None]
        restricted = safety["restricted"] and status is not TruthValue.FALSE
        if any(g["status"] == "not_met" for g in gates):
            status = TruthValue.FALSE
            restricted = False
        elif safety["pending"] or any(g["status"] == "unknown" for g in gates):
            if status is not TruthValue.FALSE:
                status = TruthValue.UNKNOWN
        missing += [g["field"] for g in gates if g["status"] == "unknown"]
        missing += safety["missing"]
        result = rule.get(action_key)
        if isinstance(result, dict):
            result = result.get("result") or result.get("action")
        selection = None
        if isinstance(result, str):
            from management_clauses import patient_age_months, select_age_clauses
            selection = select_age_clauses(result, patient_age_months(context))
            if selection["status"] == "excluded":
                status, result = TruthValue.FALSE, None
            elif selection["status"] == "matched":
                result = selection["text"]
            elif selection["status"] != "no_age_clause" and status is not TruthValue.FALSE:
                status = TruthValue.UNKNOWN
                missing.append("age_months" if selection["status"] == "age_unknown" else "source_age_clause_review")
        unsupported = tree is not None and not _rule_fields(tree)
        options.append({
            "card_id": card.card_id, "title": card.card_title,
            "category": card.clinical_category, "rule_id": rule.get("rule_id"),
            "status": "restricted" if restricted and status is not TruthValue.FALSE else {
                TruthValue.TRUE: "supported", TruthValue.FALSE: "not_applicable", TruthValue.UNKNOWN: "conditional"}[status],
            "source_instruction": result if isinstance(result, str) else None,
            "age_clause_selection": selection if selection and selection["status"] != "no_age_clause" else None,
            "condition": tree, "restrictions": safety["restrictions"], "safety_notes": safety["notes"],
            "missing_inputs": sorted(set(missing)), "prerequisites": list(gates),
            "matched_fields": [f for f in _rule_fields(tree)
                               if _get_value(context, f)[1]],
            "unsupported_rule": unsupported,
            "evidence": [{"card_id": card.card_id,
                          "pdf_page": rule.get("source_page") or card.pdf_page_numbers[0]}],
        })
    return options


def pack_evidence(prepared, patient, prompt_builder, context_text, limit=19000):
    """Reserve one slot per category before filling extras, count full prompt.

    Instruction-level projections preserve conditions and restrictions; raw
    complex medication schedules remain in the original source, not in a
    truncated or fabricated dose.
    """
    groups = {category: [entry for entry in prepared if entry["clinical_category"] == category]
              for category in CATEGORIES}
    def project(entry):
        keys = ("card_id", "card_title", "disease_protocol", "clinical_category",
                "pdf_page_numbers", "source_pdf_title", "relevant_symptoms_or_prerequisites",
                "exceptions_contraindications", "applicability", "patient_rule_options")
        compact = {k: entry[k] for k in keys if k in entry and k != "patient_rule_options"}
        compact["patient_rule_options"] = [
            {k: option[k] for k in ("rule_id", "status", "source_instruction", "age_clause_selection", "condition",
                                    "missing_inputs", "prerequisites", "unsupported_rule", "safety_notes")}
            for option in entry.get("patient_rule_options", [])]
        return compact
    # Compact every candidate consistently rather than silently deleting advice.
    groups = {category: [project(entry) for entry in entries]
              for category, entries in groups.items()}
    selected, omitted = [], []
    depth = 0
    while any(depth < len(entries) for entries in groups.values()):
        for category in CATEGORIES:
            if depth >= len(groups[category]):
                continue
            entry = groups[category][depth]
            trial = selected + [entry]
            if len((prompt_builder(patient, trial) + context_text).encode("utf-8")) <= limit:
                selected.append(entry)
            else:
                omitted.append(entry["card_id"])
        depth += 1
    return selected, omitted


def review_summary(options):
    """One bounded review entry per card, preserving branches without dosing."""
    grouped = {}
    for option in options:
        if option["status"] == "not_applicable":
            continue
        grouped.setdefault(option["card_id"], []).append(option)
    result = []
    for entries in grouped.values():
        first = entries[0]
        result.append({
            "card_id": first["card_id"], "title": first["title"],
            "category": first["category"], "evidence": first["evidence"],
            "supported_rules": [r for r in entries if r["status"] == "supported"],
            "conditional_rules": [r for r in entries if r["status"] in {"conditional", "restricted"}],
            "missing_inputs": sorted({f for r in entries for f in r["missing_inputs"]}),
        })
    return result
