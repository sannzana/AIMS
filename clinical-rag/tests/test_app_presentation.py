from pathlib import Path
from streamlit.testing.v1 import AppTest
from app_presentation import reset_case_selection


def test_case_selection_reset_preserves_original_patient_and_uses_new_widget():
    state = {'working_disease': 'Old candidate', 'case_revision': 2,
             'management_clarification': {'old': True}, 'case_patient': {'age': 32}}
    reset_case_selection(state)
    assert state['case_revision'] == 3
    assert 'working_disease' not in state
    assert 'management_clarification' not in state
    assert state['case_patient'] == {'age': 32}


def test_selector_does_not_reuse_previous_case_and_clears_old_management():
    def script():
        import streamlit as st
        from app_presentation import choose_working_diagnosis, reset_case_selection
        if 'case_results' not in st.session_state:
            st.session_state['case_results'] = {'Management': {'old': True}}
            st.session_state['working_disease'] = 'Old candidate'
            st.session_state['case_management_requested'] = True
        if st.button('New case'):
            reset_case_selection(st.session_state)
        choose_working_diagnosis(st, ['Old candidate', 'New candidate'])
    at = AppTest.from_function(script).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.selectbox[0].value is None
    assert at.session_state['working_disease'] is None
    assert not at.session_state['case_management_requested']
    assert 'Management' not in at.session_state['case_results']
    at.selectbox[0].select('New candidate').run()
    assert at.session_state['working_disease'] == 'New candidate'


def test_landing_screen_has_red_disclaimer_and_collapsed_validation():
    at = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py')).run()
    assert not at.exception
    assert any('Prototype only' in error.value for error in at.error)
    assert not any('142 issue' in warning.value for warning in at.warning)
    assert not any('Loaded 108 cards' in caption.value for caption in at.caption)
    assert any(e.label == 'Evidence validation — technical details' for e in at.expander)


def test_diagnosis_cards_have_distinct_colors_and_escape_patient_text():
    def script():
        import streamlit as st
        from app_presentation import diagnosis_card
        diagnosis_card(st, {'text': 'Suspected candidate', 'reason': '<script>test</script>'})
        diagnosis_card(st, {'text': 'Possible candidate'}, differential=True)
    at = AppTest.from_function(script).run()
    assert not at.exception
    text = '\n'.join(m.value for m in at.markdown)
    assert '#e8f1ff' in text and '#fff4df' in text
    assert '&lt;script&gt;' in text and '<script>' not in text
