"""Display grouping only; never evaluates or changes clinical eligibility."""


def _fields(tree):
    if isinstance(tree, dict):
        result = {tree["field"]} if isinstance(tree.get("field"), str) else set()
        for value in tree.values():
            result.update(_fields(value))
        return result
    if isinstance(tree, list):
        return set().union(*(_fields(value) for value in tree), set())
    return set()


def split_conditional_sources(entries, diagnostics):
    """Use existing match/selection metadata, retaining every source entry.

    Unknown complications and unsupported table rows stay available in a compact
    additional-material section. This grouping does not approve any instruction.
    """
    blocked = set(diagnostics.get("blocked_card_ids", []))
    primary, additional = [], []
    for entry in entries:
        pending = entry.get("conditional_rules", [])
        if not pending:
            continue
        indication_known = any(
            bool(_fields(rule.get("condition")))
            and _fields(rule.get("condition")) <= set(rule.get("matched_fields", []))
            for rule in pending if not rule.get("unsupported_rule"))
        pathway_known = not any(rule.get("unsupported_rule") for rule in pending) and any(
            gate.get("status") == "met" for rule in pending for gate in rule.get("prerequisites", []))
        relevant = bool(entry.get("supported_rules")) or indication_known or pathway_known or entry["card_id"] in blocked
        (primary if relevant else additional).append(entry)
    return primary, additional


def conditional_message(text):
    prefix = "Not shown until "
    if isinstance(text, str) and text.startswith(prefix):
        return ("Not selected for the main recommendation until " + text[len(prefix):]
                + " Conditional source options remain available below.")
    return text
