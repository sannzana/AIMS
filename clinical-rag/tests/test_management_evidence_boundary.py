"""Disease- and category-agnostic evidence boundaries for management output."""
from types import SimpleNamespace
from recommend import _enforce_management_evidence_boundary

def _item(*ids):
    return SimpleNamespace(evidence=[SimpleNamespace(card_id=cid) for cid in ids])

def _response():
    return SimpleNamespace(investigations=[], medication=[], advice=[])

def _card(disease, category):
    return SimpleNamespace(disease_protocol=disease, clinical_category=category)

def test_rejects_other_protocol_even_when_reference_is_real():
    cards={"a":_card("Protocol A","Investigation"),
           "b":_card("Protocol B","Investigation")}
    output=_response()
    output.investigations=[_item("a"),_item("b")]
    filtered,rejected=_enforce_management_evidence_boundary(output,cards,"Protocol A")
    assert len(filtered.investigations)==1
    assert filtered.investigations[0].evidence[0].card_id=="a"
    assert len(rejected)==1

def test_rejects_category_mismatch_and_missing_reference():
    cards={"a":_card("Protocol A","Advice")}
    output=_response()
    output.medication=[_item("a"),_item("unknown")]
    filtered,rejected=_enforce_management_evidence_boundary(output,cards,"Protocol A")
    assert filtered.medication==[]
    assert len(rejected)==2

def test_rejects_mixed_citations_and_preserves_correct_items():
    cards={"a":_card("Protocol A","Medication"),
           "b":_card("Protocol B","Medication")}
    output=_response()
    output.medication=[_item("a","b"),_item("a")]
    filtered,rejected=_enforce_management_evidence_boundary(output,cards,"Protocol A")
    assert len(filtered.medication)==1
    assert len(rejected)==1
