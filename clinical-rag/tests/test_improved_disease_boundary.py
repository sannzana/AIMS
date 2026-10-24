"""Improved management cannot silently use retrieval candidates as diagnoses."""
import app
from recommend import recommend_management


class CardStub:
    def __init__(self, protocol):
        self.disease_protocol = protocol


def test_diagnosis_citations_control_working_disease():
    cards = {"CHO-001": CardStub("Cholera"), "CHI-026": CardStub("Chikungunya")}
    result = {"probable_diagnosis": [
        {"text": "Suspected cholera", "evidence": [{"card_id": "CHO-001", "pdf_page": 5}]},
        {"text": "Uncertain", "evidence": [
            {"card_id": "CHO-001", "pdf_page": 5},
            {"card_id": "CHI-026", "pdf_page": 17},
        ]},
    ]}
    assert app.management_disease_choices(result, cards) == ["Cholera"]


def test_no_diagnosis_blocks_improved_management():
    result = recommend_management({}, mode="improved", cards={}, diseases=None)
    assert "one selected disease" in result["notice"]
    assert not result["medication"] and not result["investigations"]


def test_multiple_diseases_block_improved_management():
    result = recommend_management({}, mode="improved", cards={},
                                  diseases=["Cholera", "Chikungunya"])
    assert result["medication"] == []
    assert result["advice"] == []
