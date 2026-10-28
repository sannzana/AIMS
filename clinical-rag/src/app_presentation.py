"""App styling and case-specific selection state; no clinical decisions."""
APP_TITLE = "AIMScribe : Detect & Treat"


def style(st):
    st.markdown("""<style>
    .st-key-working_diagnosis_picker [data-baseweb="select"] > div {
        background: #e8f1ff; border: 2px solid #2563eb; color: #102f64;
        border-radius: 9px; min-height: 48px;
    }
    .st-key-working_diagnosis_picker [data-baseweb="select"] span {
        color: #102f64;
    }
    </style>""", unsafe_allow_html=True)


def reset_case_selection(state):
    state['case_revision'] = state.get('case_revision', 0) + 1
    state['case_management_requested'] = False
    state.get('case_results', {}).pop('Management', None)
    state.pop('working_disease', None)
    state.pop('management_clarification', None)


def choose_working_diagnosis(st, candidates):
    revision = st.session_state.get('case_revision', 0)
    widget_key = f'working_disease_case_{revision}'
    if widget_key not in st.session_state:
        previous = st.session_state.get('working_disease')
        st.session_state[widget_key] = previous if previous in candidates else (
            candidates[0] if len(candidates) == 1 else None)
    with st.container(key='working_diagnosis_picker'):
        selected = st.selectbox(
            'Working diagnosis for management (clinician selection)', candidates,
            index=None, placeholder='Select a working diagnosis', key=widget_key,
        )
    previous = st.session_state.get('working_disease')
    if selected != previous:
        st.session_state['case_management_requested'] = False
        st.session_state.get('case_results', {}).pop('Management', None)
        st.session_state.pop('management_clarification', None)
    st.session_state['working_disease'] = selected


def validation_details(st, report):
    with st.expander('Evidence validation — technical details', expanded=False):
        st.caption(report)
