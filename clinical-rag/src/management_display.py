"""Read-only instruction cards for Improved management.

Only supported rules enter these groups. Labels, explicit cross-references and
source details never create a clinical match or select an unassessed table row.
"""
import json
import re


def _raw(card):
    return card.model_dump() if card else {}


def _single_medicine(card):
    details = _raw(card).get("medication_details")
    if not isinstance(details, dict) or any(isinstance(v, (dict, list)) for v in details.values()):
        return None
    names = [details[k] for k in ("medicine", "fluid") if isinstance(details.get(k), str) and details[k].strip()]
    return names[0] if len(names) == 1 else None


def _mentions(text, name):
    # Prefer a printed acronym or full source name; no drug/class synonym map.
    names = [name.split("(")[0].strip()] + re.findall(r"\(([A-Z][A-Z0-9]+)\)", name)
    return any(re.search(r"(?<!\w)" + re.escape(n) + r"(?!\w)", text, re.I) for n in names if n)


def readable_instruction(rule, card):
    text = rule.get("source_instruction") or ""
    if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", text):
        # Restore only abbreviations printed in this source, not a clinical map.
        printed = set(re.findall(r"\b[A-Z][A-Z0-9]+\b", json.dumps(_raw(card))))
        words = text.split("_")
        text = " ".join(word.upper() if word.upper() in printed else word for word in words)
    age_label = None
    selection = rule.get("age_clause_selection") or {}
    if selection.get("status") == "matched":
        # Remove a display prefix only after the existing source age selector
        # matched it. Keep that exact prefix as a visible caption.
        from management_clauses import _RANGE, _LOWER, _UNDER, _OVER
        prefix = re.match(r"^\s*(?:for\s+)?(?:ages?|aged)\s+", text, re.I)
        if prefix:
            for pattern in (_RANGE, _LOWER, _UNDER, _OVER):
                band = pattern.match(text[prefix.end():])
                if band:
                    end = prefix.end() + band.end()
                    rest = text[end:].lstrip(" ,")
                    if re.match(r"give\b", rest, re.I):
                        age_label, text = text[:end].strip(), rest
                    break
    return {"text": text[:1].upper() + text[1:], "age_label": age_label}


def _branch(card, rule_id):
    for branch in _raw(card).get("if_else_conditions_and_corresponding_actions") or []:
        if isinstance(branch, dict) and branch.get("rule_id") == rule_id:
            return branch.get("then") if isinstance(branch.get("then"), dict) else {}
    return {}


def instruction_groups(matches, cards, added=()):
    """Group same-card instructions or explicitly linked matching medicine rules.

    A multi-drug card is never assigned a card-wide medicine label or dose.
    Cross-card merging requires a source next-step reference AND the action's
    mention of that target medicine. All original rules/citations are retained.
    """
    matches = [r for r in matches if r.get("status") == "supported" and r.get("source_instruction")]
    supported_ids = {r["card_id"] for r in matches}
    groups = {}
    for rule in matches:
        card = cards.get(rule["card_id"])
        text = rule.get("source_instruction") or ""
        target = None
        step = str(_branch(card, rule.get("rule_id")).get("next_step") or "")
        links = [cid for cid in supported_ids if cid != rule["card_id"] and re.search(
            r"(?<!\w)" + re.escape(cid) + r"(?!\w)", step)]
        if len(links) == 1:
            name = _single_medicine(cards.get(links[0]))
            if name and _mentions(text.replace("_", " "), name):
                target = links[0]
        anchor = target or rule["card_id"]
        anchor_card = cards.get(anchor)
        medicine = _single_medicine(anchor_card)
        # Detached numeric doses may use a single-medicine card label. Other
        # instructions need an actual mention; an ORS action on an IV-fluid card
        # must never inherit its fluid's name.
        dose_only = bool(re.match(r"^(?:give\s+)?\d", text, re.I))
        selected_age = (rule.get("age_clause_selection") or {}).get("status") == "matched"
        use_name = bool(medicine and (target or dose_only or selected_age or _mentions(text.replace("_", " "), medicine)))
        label = medicine if use_name else rule.get("title") or getattr(card, "card_title", "Source guidance")
        key = (anchor, label)
        group = groups.setdefault(key, {"label": label, "rules": [], "lines": [], "updated": False,
                                        "timing": None})
        group["rules"].append(rule)
        line = readable_instruction(rule, card)
        identity = (rule["card_id"], rule.get("rule_id"), text)
        line["updated"] = identity in added
        if not any(old["text"] == line["text"] and old["age_label"] == line["age_label"] for old in group["lines"]):
            group["lines"].append(line)
        elif line["updated"]:
            for old in group["lines"]:
                if old["text"] == line["text"] and old["age_label"] == line["age_label"]:
                    old["updated"] = True
        group["updated"] |= line["updated"]
        # Timing for an age-selected amount is printed on its single-medicine
        # source card. Do not copy card-wide dose/duration or multi-drug details.
        if use_name and selected_age:
            frequency = _raw(anchor_card).get("medication_details", {}).get("frequency")
            if isinstance(frequency, str):
                group["timing"] = frequency
    return list(groups.values())


def render_instruction_groups(st, matches, cards, added=()):
    from management_evidence import condition_text
    for group in instruction_groups(matches, cards, added):
        with st.container(border=True):
            st.markdown("**" + group["label"] + "**")
            if group["updated"]:
                st.caption("🟢 Updated after your answers · source match")
            for line in group["lines"]:
                if line["updated"]:
                    st.success(line["text"])
                else:
                    st.write(line["text"])
                if line["age_label"]:
                    st.caption("Matched source age band: " + line["age_label"])
            if group["timing"]:
                st.caption("Timing from source: " + group["timing"])
            refs = list(dict.fromkeys((r["card_id"], e["pdf_page"]) for r in group["rules"] for e in r["evidence"]))
            st.caption("Source: " + "; ".join(f"{cid} — PDF page {page}" for cid, page in refs))
            with st.expander("Matched source conditions & original wording"):
                for rule in group["rules"]:
                    st.caption(f"{rule['card_id']} / {rule.get('rule_id')}")
                    st.write(condition_text(rule.get("condition")))
                    st.write("Original source instruction: " + (rule.get("source_instruction") or ""))
                    for gate in rule.get("prerequisites", []):
                        st.write(f"Requires {gate['field'].replace('_', ' ')}: {gate['required_value']} ({gate['status']})")
                    card = cards.get(rule["card_id"])
                    if card and _raw(card).get("relevant_symptoms_or_prerequisites"):
                        st.write("Source context: " + str(_raw(card)["relevant_symptoms_or_prerequisites"]))
            with st.expander("Full source reference — not a patient-specific plan"):
                st.caption("This original reference may include other age bands and unresolved conditions. Only the matching guidance above was selected.")
                reference_key = "source_reference_" + "|".join(f"{r['card_id']}:{r.get('rule_id')}" for r in group["rules"])
                if st.toggle("Show original source text", key=reference_key):
                    for cid in dict.fromkeys(r["card_id"] for r in group["rules"]):
                        raw = _raw(cards.get(cid))
                        st.write(raw.get("card_title") or cid)
                        if raw.get("recommended_action"):
                            st.write(raw["recommended_action"])


def render_plan_reference(st, output, cards, pathway):
    """Explicit reference view of matched source cards, not another plan selection."""
    ids = list(dict.fromkeys(entry["card_id"] for entry in output.get("source_review_options", [])
                             if entry.get("supported_rules")))
    with st.expander(f"{pathway} — view source details"):
        st.caption("Reference only: these original cards may contain other ages and unresolved conditions. "
                   "Use the patient-matched guidance below for the current assessment. This is not the full guideline PDF.")
        if st.toggle("Show selected plan source cards", key=f"plan_source_reference_{id(output)}"):
            for cid in ids:
                raw = _raw(cards.get(cid))
                with st.container(border=True):
                    st.markdown("**" + str(raw.get("card_title") or cid) + "**")
                    st.caption(f"Source: {cid} — PDF pages {raw.get('pdf_page_numbers', [])}")
                    if raw.get("recommended_action"):
                        st.write(raw["recommended_action"])
                    with st.expander("Complete card data — original conditions and tables"):
                        st.json(raw)
