"""Source-rule-aware retrieval must not turn similarity into clinical eligibility."""
import json
from pathlib import Path

import retrieve
from build_index import load_cards
from recommend import recommend_management

CARDS = {card.card_id: card for card in load_cards(Path('data/cards.jsonl'))}


def test_applicable_rule_beats_unknown_medication_similarity(monkeypatch):
    # The old top-two retrieval was filled by unrelated glucose medication.
    # With verified severe classification the matching fluid-plan source card
    # must be considered even when its embedding score ranks lower.
    pools = {
        'Medication': ['CHO-020', 'CHO-021', 'CHO-004', 'CHO-006'],
        'Advice': ['CHO-012'],
        'Investigation': [],
    }
    seen = {}

    def search(_patient, category, *, top_k=2, **_kw):
        seen[category] = top_k
        return [{'card_id': cid} for cid in pools[category][:top_k]]

    monkeypatch.setattr(retrieve, 'retrieve_category', search)

    def model(_prompt):
        return json.dumps({'medication': [{
            'text': 'Source-guided treatment pathway.',
            'reason': 'The verified classification matches the source rule.',
            'evidence': [{'card_id': 'CHO-004',
                          'pdf_page': CARDS['CHO-004'].pdf_page_numbers[0]}],
        }], 'missing_info': []})

    result = recommend_management(
        {'age': 34, '_structured_facts': {'absent_or_weak_pulse': True}},
        cards=CARDS, diseases=['Cholera'], llm_generate=model,
    )
    diagnostics = result['management_diagnostics']
    assert seen['Medication'] >= 4
    assert 'CHO-004' in diagnostics['prompt_card_ids']
    assert diagnostics['retrieved_card_ids_by_category']['Medication'][0] == 'CHO-004'
    assert result['medication']


def test_unknown_indication_has_visible_conditional_warning(monkeypatch):
    monkeypatch.setattr(
        retrieve, 'retrieve_category',
        lambda _patient, category, **_kw: [{'card_id': cid} for cid in
            {'Medication': ['CHO-020'], 'Advice': ['CHO-012'],
             'Investigation': []}[category]],
    )
    def model(_prompt):
        return json.dumps({
            'medication': [{'text': 'Administer IV dextrose.',
                            'reason': 'Source regimen.',
                            'evidence': [{'card_id': 'CHO-020', 'pdf_page': 13}]}],
            'advice': [{'text': 'Discharge now.', 'reason': 'Source criteria.',
                        'evidence': [{'card_id': 'CHO-012', 'pdf_page': 5}]}],
            'missing_info': [],
        })
    result = recommend_management({'age': 34}, cards=CARDS, diseases=['Cholera'],
                                  llm_generate=model)
    for section in ('medication', 'advice'):
        for item in result[section]:
            assert item['text'].startswith('CONDITIONAL GUIDELINE INFORMATION ONLY')
            assert 'Conditional source guidance' in item['reason']


def test_conditional_regimens_do_not_expose_patient_specific_doses(monkeypatch):
    """The LLM cannot turn unknown indications into executable dose directions."""
    monkeypatch.setattr(
        retrieve, 'retrieve_category',
        lambda _patient, category, **_kw: [
            {'card_id': cid} for cid in {
                'Medication': ['CHO-020', 'CHO-021'],
                'Advice': ['CHO-012'], 'Investigation': [],
            }[category]
        ],
    )

    def model(_prompt):
        return json.dumps({
            'medication': [
                {'text': 'Administer IV dextrose 50% 25–50 mL.',
                 'reason': 'Treat low glucose.',
                 'evidence': [{'card_id': 'CHO-020', 'pdf_page': 13}]},
                {'text': 'Administer IV dextrose 10% 5 mL/kg.',
                 'reason': 'Treat low glucose.',
                 'evidence': [{'card_id': 'CHO-021', 'pdf_page': 13}]},
            ],
            'advice': [{
                'text': 'Discharge now.',
                'reason': 'Discharge criteria.',
                'evidence': [{'card_id': 'CHO-012', 'pdf_page': 5}],
            }],
        })

    result = recommend_management({'age': 24}, cards=CARDS,
                                  diseases=['Cholera'], llm_generate=model)
    assert len(result['medication']) == 2
    for item in result['medication']:
        assert item['text'].startswith('CONDITIONAL GUIDELINE INFORMATION ONLY')
        assert 'Administer' not in item['text']
        assert item['dose'] is None
        assert item['route'] is None
        assert item['source_details'] is None
        assert item['evidence']
    assert result['advice']
    assert 'Discharge now' not in result['advice'][0]['text']
    assert 'not established' in result['advice'][0]['text']
