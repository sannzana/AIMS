"""Derive classification fields (e.g. dehydration class) from source cards.

Management cards test a field such as ``dehydration_classification =
some_dehydration`` that no patient form supplies. The label is defined by an
assessment card: a rule with ``action: classify_<label>`` and a condition tree.

Everything here comes from the cards, nothing from card IDs or disease names:

* Dependencies. A card may say "if <label> = false then next_card: X". Labels
  defined in card X are then evaluated only once <label> is known to be false.
  If <label> is TRUE they are ruled out; if it is UNKNOWN they stay UNKNOWN.
* Exclusive groups. Labels that other cards compare against the *same field*
  form one group (only one value can hold). Two TRUE labels in a group are a
  source conflict: it is reported and nothing is chosen.
* Labels in no group and no dependency are independent and reported separately.

UNKNOWN is never turned into a guess, and a field the patient/clinician already
supplied is never overwritten.
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from applicability import TruthValue, _evaluate_tree
from schema import Card

PREFIX = "classify_"
NEXT_KEYS = ("next_card",)
_CARD_REF = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")


def _walk(node: Any) -> Iterable[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _conditions(node: Any) -> Iterable[dict[str, Any]]:
    for item in _walk(node):
        if "field" in item and "value" in item:
            yield item


def classification_rules(cards: Iterable[Card], disease: str) -> list[tuple[str, str, Any]]:
    """[(card_id, label, condition_tree)] defined by the disease's Diagnosis cards."""
    found = []
    for card in cards:
        if card.disease_protocol != disease or card.clinical_category != "Diagnosis":
            continue
        for rule in card.model_dump().get("conditions_decision_logic") or []:
            action = rule.get("action") if isinstance(rule, dict) else None
            if isinstance(action, str) and action.startswith(PREFIX) and rule.get("condition"):
                found.append((card.card_id, action[len(PREFIX):], rule["condition"]))
    return found


def _referenced_card_ids(value: Any, card_ids: set[str]) -> set[str]:
    if isinstance(value, str):
        return {ref for ref in _CARD_REF.findall(value) if ref in card_ids}
    if isinstance(value, dict):
        return set().union(*(_referenced_card_ids(item, card_ids) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_referenced_card_ids(item, card_ids) for item in value))
    return set()


def dependencies(
    cards: Iterable[Card], disease: str, labels: set[str]
) -> dict[str, tuple[str, TruthValue]]:
    """card_id -> (gating label, required truth) from source card links."""
    cards = list(cards)
    card_ids = {card.card_id for card in cards}
    gates: dict[str, tuple[str, TruthValue]] = {}
    for card in cards:
        if card.disease_protocol != disease:
            continue
        raw = card.model_dump()
        for rule in raw.get("if_else_conditions_and_corresponding_actions") or []:
            cond = rule.get("if") if isinstance(rule, dict) else None
            then = rule.get("then") if isinstance(rule, dict) else None
            if not (isinstance(cond, dict) and isinstance(then, dict)):
                continue
            target = next((then[k] for k in NEXT_KEYS if isinstance(then.get(k), str)), None)
            if target and cond.get("field") in labels and isinstance(cond.get("value"), bool):
                gates[target] = (
                    cond["field"],
                    TruthValue.TRUE if cond["value"] else TruthValue.FALSE,
                )
        for rule in raw.get("conditions_decision_logic") or []:
            if not isinstance(rule, dict):
                continue
            condition_labels = {
                cond["value"]
                for cond in _conditions(rule.get("condition"))
                if cond.get("value") in labels
            }
            for target in _referenced_card_ids(rule, card_ids) | _referenced_card_ids(
                raw.get("if_else_conditions_and_corresponding_actions"), card_ids
            ):
                if target == card.card_id or len(condition_labels) != 1:
                    continue
                gates.setdefault(target, (next(iter(condition_labels)), TruthValue.TRUE))
    return gates


def field_groups(cards: Iterable[Card], disease: str, labels: set[str]) -> dict[str, dict[str, set[str]]]:
    """field -> {'labels': {...}, 'cards': {card ids that test it}} from non-Diagnosis cards."""
    groups: dict[str, dict[str, set[str]]] = {}
    for card in cards:
        if card.disease_protocol != disease or card.clinical_category == "Diagnosis":
            continue
        raw = card.model_dump()
        for key in ("conditions_decision_logic", "if_else_conditions_and_corresponding_actions"):
            for cond in _conditions(raw.get(key)):
                if cond.get("value") in labels and isinstance(cond.get("field"), str):
                    g = groups.setdefault(cond["field"], {"labels": set(), "cards": set()})
                    g["labels"].add(cond["value"])
                    g["cards"].add(card.card_id)
    return groups


def consuming_fields(cards: Iterable[Card], disease: str, labels: set[str]) -> dict[str, set[str]]:
    return {f: g["cards"] for f, g in field_groups(cards, disease, labels).items()}


def derive_classifications(
    cards: Iterable[Card], disease: str, context: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (derived_fields, trace).

    trace['fields'][field] = {'status': decided|conflict|undecided|supplied,
                              'value', 'true_labels', 'unknown_labels',
                              'blocked_cards'}
    """
    cards = list(cards)
    rules = classification_rules(cards, disease)
    labels = {label for _, label, _ in rules}
    trace: dict[str, Any] = {
        "labels": {}, "fields": {}, "conflicts": [], "dependencies": {},
        "blocked_cards": {},
    }
    if not labels:
        return {}, trace
    gates = dependencies(cards, disease, labels)
    trace["dependencies"] = {
        card: f"{label}={required is TruthValue.TRUE}"
        for card, (label, required) in gates.items()
    }

    raw_state: dict[str, TruthValue] = {
        label: _evaluate_tree(tree, context) for _, label, tree in rules}
    card_of = {label: card_id for card_id, label, _ in rules}
    groups = field_groups(cards, disease, labels)
    state: dict[str, TruthValue] = {}
    for field, group in groups.items():
        supplied = context.get(field)
        if supplied not in (None, ""):
            for label in group["labels"]:
                state[label] = (
                    TruthValue.TRUE if label == supplied else TruthValue.FALSE
                )
    notes: dict[str, str] = {}

    visiting: list[str] = []
    cycle_labels: set[str] = set()

    def resolve(label: str) -> TruthValue:
        if label in state:
            return state[label]
        if label in visiting:
            cycle_labels.update(visiting[visiting.index(label):])
            return TruthValue.UNKNOWN
        visiting.append(label)
        value = raw_state[label]
        gate = gates.get(card_of[label])
        if gate and gate[0] in raw_state:
            gate_label, wanted = gate
            gate_value = resolve(gate_label)
            if gate_value == wanted:
                pass
            elif gate_value == TruthValue.UNKNOWN:
                value = TruthValue.UNKNOWN
                notes[label] = f"waits for {gate_label} to be {wanted is TruthValue.TRUE}"
            else:
                value = TruthValue.FALSE
                notes[label] = f"ruled out because {gate_label} is {gate_value.value}"
        visiting.pop()
        state[label] = value
        return value

    for card_id, label, _ in rules:
        value = resolve(label)
        trace["labels"][label] = {"card_id": card_id, "value": value.value,
                                  **({"note": notes[label]} if label in notes else {})}
    for label in cycle_labels:
        state[label] = TruthValue.UNKNOWN
        trace["labels"][label]["value"] = TruthValue.UNKNOWN.value
        trace["labels"][label]["note"] = (
            "cyclic classification dependency: "
            + " -> ".join(sorted(cycle_labels))
        )
    card_by_id = {card.card_id: card for card in cards}
    for card_id, (label, wanted) in gates.items():
        if (card_by_id.get(card_id)
                and card_by_id[card_id].clinical_category != "Diagnosis"
                and label in state and state[label] != wanted):
            trace["blocked_cards"][card_id] = label

    derived: dict[str, Any] = {}
    for field, group in groups.items():
        entry: dict[str, Any] = {"labels": sorted(group["labels"]),
                                 "blocked_cards": sorted(group["cards"])}
        trace["fields"][field] = entry
        if context.get(field) not in (None, ""):
            entry.update(status="supplied", value=context[field])
            entry["blocked_cards"] = []
            continue
        true = sorted(l for l in group["labels"] if state.get(l) == TruthValue.TRUE)
        unknown = sorted(l for l in group["labels"] if state.get(l) == TruthValue.UNKNOWN)
        entry.update(true_labels=true, unknown_labels=unknown)
        if len(true) > 1:
            entry["status"] = "conflict"
            trace["conflicts"].append(
                f"{field}: source rules give {', '.join(true)} at the same time")
        elif len(true) == 1:
            entry.update(status="decided", value=true[0], blocked_cards=[])
            derived[field] = true[0]
        else:
            entry["status"] = "undecided"
    return derived, trace


def blocked_cards(trace: dict[str, Any]) -> dict[str, str]:
    """card_id -> field whose classification is not settled."""
    out: dict[str, str] = {}
    for field, entry in trace.get("fields", {}).items():
        for card_id in entry.get("blocked_cards", []):
            out[card_id] = field
    out.update(trace.get("blocked_cards", {}))
    return out
