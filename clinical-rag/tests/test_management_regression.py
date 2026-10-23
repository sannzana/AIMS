import json
from pathlib import Path
from build_index import load_cards
from retrieve import _category_relevance
from recommend import _structured_patient_context

CARDS = {c.card_id: c for c in load_cards(Path(__file__).parents[1]/'data'/'cards.jsonl')}
CASE = {'symptoms':'High fever, severe joint pain in both wrists knees ankles, headache and rash',
        'duration':'3 days','age':32,'history':'No known chronic diseases',
        'clinical_findings':'Swollen and tender joints; no reported bleeding'}

def test_ordinary_chikungunya_medication_ranks_before_severe_complications():
    def rank(cid, similarity):
        return _category_relevance({'similarity_score':similarity}, CARDS[cid], CASE, 'Chikungunya')[0]
    # Despite a modest semantic advantage for the serious-complications card.
    assert rank('CHI-014', .65) > rank('CHI-018', .80)
    assert rank('CHI-014', .65) > rank('CHI-017', .78)

def test_explicit_duration_and_fever_feed_applicable_rules():
    normalized = _structured_patient_context(CASE)
    assert normalized['days_since_illness_onset'] == 3
    assert normalized['patient_has_fever'] is True
    assert 'days_since_fever_onset' not in normalized

def test_no_negated_fever_inference():
    assert 'patient_has_fever' not in _structured_patient_context({'symptoms':'no fever'})

def test_real_corpus_contains_critical_three_investigation_and_treatment_cards():
    assert all(cid in CARDS for cid in ('CHI-014','CHI-022','CHI-023','CHI-026'))
    assert len(CARDS) == 108
