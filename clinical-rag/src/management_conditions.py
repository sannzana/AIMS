"""Narrow, source-driven refinements for Improved instruction eligibility.

No disease names, clinical cutoffs or plan names are encoded here. Text that
cannot be interpreted conservatively remains conditional, not permission.
"""
import re

from applicability import TruthValue, _evaluate_tree


def boolean_source_fields(cards):
    from derived_fields import _conditions
    return {condition["field"] for card in cards
            for condition in _conditions(card.model_dump())
            if isinstance(condition.get("field"), str) and isinstance(condition.get("value"), bool)}


def explicit_boolean_prerequisites(card, context, fields):
    """Recognize explicit conjunctive prerequisites using source field names.

    This intentionally does not infer synonyms, negate missing observations,
    or turn alternative indications (OR) into global requirements.
    """
    text = str(card.model_dump().get("relevant_symptoms_or_prerequisites") or "")
    guards = []
    for field in sorted(fields):
        words = [w for w in field.split("_") if w not in {"patient", "present", "is", "has"}]
        if len(words) < 2:
            continue
        # A small word gap permits source wording with an object between the
        # same predicate words; no clinical synonym or threshold is added.
        predicate = r"\b" + r"(?:\W+\w+){0,2}\W+".join(map(re.escape, words)) + r"\b"
        match = re.search(predicate, text, re.I)
        if not match:
            continue
        before = re.split(r"[.;]", text[:match.start()])[-1]
        if re.search(r"\b(?:not|without|unless|except|or)\b", before, re.I):
            continue
        if not re.search(r"\b(?:and|once|only if|provided that)\s+(?:the patient\s+)?$", before, re.I):
            continue
        value = context.get(field)
        guards.append({"field": field, "required_value": True,
                       "status": "met" if value is True else "not_met" if value is False else "unknown",
                       "source_quote": text[match.start():match.end()]})
    return guards


_UNIT = r"(years?|yrs?|months?|mos?)"


def _age_clause(clause, context):
    """Interpret only explicit age prefixes, using numbers stated in the source."""
    patterns = [
        (rf"^\s*(?:for\s+)?ages?\s+(\d+(?:\.\d+)?)\s*[–—-]\s*(\d+(?:\.\d+)?)\s+{_UNIT}\b", "range"),
        (rf"^\s*(?:for\s+)?ages?\s+(\d+(?:\.\d+)?)\s+{_UNIT}\s+and\s+(older|younger)\b", "bound"),
        (rf"^\s*(?:for\s+)?(under|over|below|above)\s+(\d+(?:\.\d+)?)\s+{_UNIT}\b", "strict"),
    ]
    for pattern, kind in patterns:
        match = re.search(pattern, clause, re.I)
        if not match:
            continue
        if kind == "range":
            lower, upper, unit = match.groups()
            op_values = ((">=", float(lower)), ("<=", float(upper)))
        elif kind == "bound":
            value, unit, direction = match.groups()
            op_values = ((">=" if direction.casefold() == "older" else "<=", float(value)),)
        else:
            direction, value, unit = match.groups()
            op_values = (("<" if direction.casefold() in {"under", "below"} else ">", float(value)),)
        field = "age_months" if unit.casefold().startswith("mo") else "age_years"
        tree = {"all": [{"field": field, "operator": op, "value": value} for op, value in op_values]}
        return _evaluate_tree(tree, context), field
    return None, None


def age_specific_instruction(text, context):
    """Keep only matching semicolon-separated explicit age clauses, verbatim."""
    if not isinstance(text, str):
        return text, False, []
    clauses = text.split(";")
    outcomes = [_age_clause(clause, context) for clause in clauses]
    if not any(state is not None for state, _ in outcomes):
        return text, False, []
    # Mixed unsupported prose is not silently discarded or reinterpreted.
    if any(state is None for state, _ in outcomes):
        return text, True, [field for state, field in outcomes if state is TruthValue.UNKNOWN]
    kept = [clause.strip() for clause, (state, _) in zip(clauses, outcomes) if state is TruthValue.TRUE]
    unknown = [field for state, field in outcomes if state is TruthValue.UNKNOWN]
    return "; ".join(kept) or None, bool(unknown), unknown


def monitoring_option(card, context, gates):
    """Retain an explicit ongoing monitoring sentence for a matched pathway.

    Future transition/discharge conditions do not withhold current monitoring.
    This only quotes an imperative already in the source with its pathway met.
    """
    if card.clinical_category not in {"Advice", "Investigation"} or not gates:
        return None
    if any(g["status"] != "met" for g in gates):
        return None
    raw = card.model_dump()
    for sentence in re.split(r"(?<=[.!?])\s+", str(raw.get("recommended_action") or "")):
        if not re.search(r"^(?:(?:During|For)\b[^,]*,\s*)?(?:reassess|monitor|review|record|observe|check)\b", sentence, re.I):
            continue
        if re.search(r"\b(?:if|when|once|unless|until|only)\b", sentence, re.I):
            continue
        return {"card_id": card.card_id, "title": card.card_title,
                "category": card.clinical_category, "rule_id": "source-monitoring",
                "status": "supported", "source_instruction": sentence,
                "condition": {"all": [{"field": g["field"], "operator": "=", "value": context[g["field"]]}
                                      for g in gates]},
                "restrictions": [], "missing_inputs": [], "prerequisites": list(gates),
                "matched_fields": [g["field"] for g in gates], "unsupported_rule": False,
                "evidence": [{"card_id": card.card_id, "pdf_page": card.pdf_page_numbers[0]}]}
    return None


def current_instruction_evidence(options, context):
    return any(option["status"] == "supported" for option in options) or any(
        _evaluate_tree(option["condition"], context) is TruthValue.TRUE
        for option in options if option["status"] == "conditional" and not option["unsupported_rule"])


def prioritize_reviews(entries, context):
    for entry in entries:
        # Positive instruction evidence, not shared protocol membership or a
        # missing lab result, establishes relevance to the current assessment.
        entry["current"] = current_instruction_evidence(entry["supported_rules"] + entry["conditional_rules"], context)
    return entries


def review_requirements(entry):
    """Deduplicate questions without conflating UNKNOWN with true or false."""
    from management_evidence import condition_text
    lines = []
    for rule in entry["conditional_rules"]:
        tree = rule["condition"]
        if rule["unsupported_rule"]:
            lines.append("A source rule/table needs manual review; a matching row has not been established.")
        elif isinstance(tree, dict) and isinstance(tree.get("field"), str) and isinstance(tree.get("value"), bool):
            lines.append("Confirm whether " + tree["field"].replace("_", " ") + ".")
        elif tree:
            lines.append("Applies if: " + condition_text(tree))
        for gate in rule["prerequisites"]:
            if gate["status"] == "unknown":
                if gate["required_value"] is True:
                    lines.append("Confirm whether " + gate["field"].replace("_", " ") + ".")
                else:
                    lines.append(f"Confirm {gate['field'].replace('_', ' ')}: {gate['required_value']}.")
        for restriction in rule["restrictions"]:
            if isinstance(restriction, dict):
                text = restriction.get("interpretation") or restriction.get("action") or condition_text(restriction.get("condition"))
            else:
                text = str(restriction)
            lines.append("Restriction: " + text)
    return list(dict.fromkeys(lines))
