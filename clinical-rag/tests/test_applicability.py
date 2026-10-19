from applicability import check_applicability
from schema import Card


def make_card(**conditions) -> Card:
    return Card(
        card_id="test",
        disease="dengue",
        category="diagnosis",
        conditions=conditions,
        source={"title": "Protocol"},
    )


def test_all_conditions_true() -> None:
    card = make_card(all=[
        {"field": "age", "op": ">=", "value": 18},
        {"field": "sex", "op": "==", "value": "female"},
    ])
    result = check_applicability(card, {"age": 30, "sex": "female"})
    assert result["status"] == "applicable"
    assert result["met"] == ["age", "sex"]


def test_one_all_condition_false() -> None:
    card = make_card(all=[{"field": "age", "op": ">=", "value": 18}])
    result = check_applicability(card, {"age": 12})
    assert result["status"] == "not_applicable"
    assert result["failed"] == ["age"]


def test_missing_patient_information_is_unknown() -> None:
    card = make_card(all=[{"field": "fever", "op": "==", "value": True}])
    result = check_applicability(card, {})
    assert result["status"] == "undetermined"
    assert result["missing_inputs"] == ["fever"]


def test_any_condition_true() -> None:
    card = make_card(any=[
        {"field": "fever", "op": "==", "value": True},
        {"field": "rash", "op": "==", "value": True},
    ])
    assert check_applicability(card, {"rash": True})["status"] == "applicable"


def test_any_condition_false() -> None:
    card = make_card(any=[
        {"field": "fever", "op": "==", "value": True},
        {"field": "rash", "op": "==", "value": True},
    ])
    result = check_applicability(card, {"fever": False, "rash": False})
    assert result["status"] == "not_applicable"


def test_empty_all_and_any_impose_no_restriction() -> None:
    card = make_card(all=[], any=[])
    assert check_applicability(card, {})["status"] == "applicable"


def test_exception_true_makes_card_not_applicable() -> None:
    card = make_card(exceptions=[{"field": "pregnant", "op": "==", "value": True}])
    result = check_applicability(card, {"pregnant": True})
    assert result["status"] == "not_applicable"
    assert result["met"] == ["pregnant"]


def test_exception_unknown_makes_card_undetermined() -> None:
    card = make_card(exceptions=[{"field": "pregnant", "op": "==", "value": True}])
    result = check_applicability(card, {})
    assert result["status"] == "undetermined"
    assert result["missing_inputs"] == ["pregnant"]


def test_numeric_comparisons() -> None:
    card = make_card(all=[
        {"field": "age", "op": ">", "value": 10},
        {"field": "weight", "op": "<=", "value": 70},
    ])
    assert check_applicability(card, {"age": 11, "weight": 70})["status"] == "applicable"


def test_list_membership() -> None:
    card = make_card(all=[
        {"field": "symptoms", "op": "has", "value": "vomiting"},
        {"field": "sex", "op": "in", "value": ["female", "male"]},
    ])
    assert check_applicability(
        card, {"symptoms": ["fever", "vomiting"], "sex": "female"}
    )["status"] == "applicable"


def test_required_inputs_missing() -> None:
    card = make_card(required_inputs=["age", "history"])
    result = check_applicability(card, {"age": 20})
    assert result["status"] == "undetermined"
    assert result["missing_inputs"] == ["history"]
