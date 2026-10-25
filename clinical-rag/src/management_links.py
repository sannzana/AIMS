"""Link Improved generated items to individual supported source instructions."""
import json


def instruction_links(raw, response, options, allowed):
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1])
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return {}, []
    if not isinstance(payload, dict):
        return {}, []
    index = {(o["card_id"], o["rule_id"]): o for o in options if o["card_id"] in allowed}
    linked, rejected = {}, []
    for section, category in (("investigations", "Investigation"), ("medication", "Medication"), ("advice", "Advice")):
        raw_items = payload.get(section, [])
        if not isinstance(raw_items, list):
            continue
        for item in getattr(response, section):
            evidence = [ref.model_dump() for ref in item.evidence]
            candidates = [entry for entry in raw_items if isinstance(entry, dict)
                          and entry.get("text") == item.text and entry.get("evidence") == evidence]
            if len(candidates) != 1 or not isinstance(candidates[0].get("source_rules"), list):
                continue
            refs = candidates[0]["source_rules"]
            selected = []
            for ref in refs:
                if (not isinstance(ref, dict) or not isinstance(ref.get("card_id"), str)
                        or not isinstance(ref.get("rule_id"), str)):
                    break
                option = index.get((ref.get("card_id"), ref.get("rule_id")))
                if (not option or option["category"] != category or option["status"] != "supported"
                        or not option["source_instruction"]):
                    break
                if not any(actual in evidence for actual in option["evidence"]):
                    break
                selected.append(option)
            if (refs and len(selected) == len(refs)
                    and {o["card_id"] for o in selected} == {ref.card_id for ref in item.evidence}):
                linked[id(item)] = selected
            elif refs:
                rejected.append(section + ": instruction references not supported by eligible source rules")
    return linked, rejected


def apply_instruction_links(output, validated, linked):
    for section in ("investigations", "medication", "advice"):
        for model, item in zip(getattr(validated, section), output.get(section, [])):
            options = linked.get(id(model))
            if not options:
                continue
            # The model chooses/groups instructions. The confirmed text remains
            # the exact applicable source clause, not another branch in the card.
            item["text"] = " ".join(dict.fromkeys(o["source_instruction"] for o in options))
            item["reason"] = "The cited source instruction matches the known patient facts and prerequisites."
            item["review_required"] = False
            item["source_rules"] = [{"card_id": o["card_id"], "rule_id": o["rule_id"]} for o in options]
            if section == "medication":
                # Never copy the whole card's alternative regimens into one
                # instruction. The selected verbatim clause is shown above.
                for field in ("name", "dose", "route", "frequency", "duration", "instructions", "source_details"):
                    item[field] = None
                item["eligibility"] = "eligible"
