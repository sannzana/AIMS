"""Display-only readability: preserve source logic, warnings and raw results."""
import copy
from pathlib import Path

from clinical_readability import _condition_html
from test_management_presentation import _result


def test_nested_logic_thresholds_and_negation_are_preserved_without_identifiers():
    tree = {"all": [{"field": "age_months", "operator": ">=", "value": 12}, {"any": [
        {"field": "awake_and_alert", "operator": "=", "value": False},
        {"field": "dehydration_classification", "operator": "=", "value": "severe_dehydration_or_shock"}]}]}
    before = copy.deepcopy(tree)
    html = _condition_html(tree)
    assert 'ALL of these (AND)' in html and 'ANY of these (OR)' in html
    assert 'age months' in html and '12' in html
    assert 'absent/false' in html
    assert 'severe dehydration or shock' in html
    assert '_' not in html
    assert tree == before


def test_source_values_are_escaped_not_executable_html():
    html = _condition_html({"field": "<script>alert(1)</script>", "operator": "=", "value": "<img src=x onerror=alert(1)>"})
    assert '<script>' not in html and '<img' not in html
    assert '&lt;script&gt;' in html and '&lt;img' in html


def test_shared_requires_is_shown_once_but_each_condition_set_is_retained(monkeypatch):
    from streamlit.testing.v1 import AppTest
    result = _result(monkeypatch, {"absent_or_weak_pulse": True})
    entry = next(e for e in result['source_review_options'] if e['card_id'] == 'CHO-010')
    before = copy.deepcopy(entry)
    at = AppTest.from_string('''import streamlit as st
from clinical_readability import render_conditional_options
render_conditional_options(st, [st.session_state['entry']])''')
    at.session_state['entry'] = entry
    at.run()
    assert not at.exception
    text = '\n'.join(x.value for x in at.markdown)
    assert text.count('<strong>Requires treatment plan:</strong>') == 1
    assert text.count('Condition set ') == len(entry['conditional_rules'])
    assert 'ALL of these (AND)' in text and 'ANY of these (OR)' in text
    assert 'Information needed' in text
    assert 'improved pulse strength' in text
    assert at.session_state['entry'] == before


def test_restriction_warning_remains_prominent():
    from streamlit.testing.v1 import AppTest
    entry = {'title': 'Source', 'card_id': 'RENAMED', 'missing_inputs': [], 'conditional_rules': [{
        'rule_id': 'R', 'status': 'restricted', 'condition': {'field': 'marker', 'operator': '=', 'value': True},
        'prerequisites': [], 'restrictions': [], 'unsupported_rule': False}]}
    at = AppTest.from_string('''import streamlit as st
from clinical_readability import render_conditional_options
render_conditional_options(st, [st.session_state['entry']])''')
    at.session_state['entry'] = entry
    at.run()
    assert not at.exception
    assert any('not approved' in warning.value for warning in at.warning)


def test_improved_diagnostics_move_to_bottom_without_mutating_results(monkeypatch):
    from streamlit.testing.v1 import AppTest
    result = _result(monkeypatch, {'absent_or_weak_pulse': True})
    diagnosis = {'probable_diagnosis': [{'text': 'Existing candidate', 'reason': 'Existing reason',
        'evidence': [{'card_id': 'CHO-001', 'pdf_page': 5}]}], 'evidence_card_ids': ['CHO-001']}
    before = copy.deepcopy((diagnosis, result))
    ranking = {'status': 'ambiguous', 'diagnostics': {'retrieved_card_ids': ['CHO-001']}}
    at = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'))
    for key, value in {'case_patient': {'age': '37 years'}, 'case_mode': 'improved', 'case_diseases': ['Cholera'],
        'working_disease': 'Cholera', 'case_management_requested': True, 'diagnosis_ranking': ranking,
        'case_results': {'Diagnosis': diagnosis, 'Management': result}}.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    labels = [e.label for e in at.expander]
    technical = next(i for i, label in enumerate(labels) if 'Technical diagnostics' in label)
    assert technical > next(i for i, label in enumerate(labels) if 'Management: source cards' in label)
    assert technical > labels.index('Improve treatment with additional information')
    assert any('Diagnostic pipeline diagnostics' in label for label in labels[technical:])
    assert any('Management pipeline diagnostics' in label for label in labels[technical:])
    text = '\n'.join(x.value for x in at.markdown)
    assert 'severe_dehydration_or_shock' not in text
    assert 'severe dehydration or shock' in text
    assert 'Existing candidate' in text
    assert 'font-size:.78rem' in text
    assert (at.session_state['case_results']['Diagnosis'], at.session_state['case_results']['Management']) == before
    assert at.session_state['diagnosis_ranking'] == ranking
