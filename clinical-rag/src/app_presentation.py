"""App styling and case-specific selection state; no clinical decisions."""
APP_TITLE = "AIMScribe : Detect & Treat"


def style(st):
    st.markdown("""<style>
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: #e8f1ff !important; border: 2px solid #2563eb !important; color: #102f64 !important;
        border-radius: 9px; min-height: 48px;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] span {
        color: #102f64 !important;
    }
    [class*="st-key-management_questions_panel"] {
        border: 2px solid #2563eb !important; border-radius: 12px;
        padding: 1rem; background-color: rgba(37,99,235,.07);
    }
    </style>""", unsafe_allow_html=True)


def diagnosis_card(st, item, *, differential=False):
    from html import escape
    background, border, label = ('#fff4df', '#b86b00', 'Possible differential') if differential else (
        '#e8f1ff', '#2563eb', 'Source-backed candidate')
    title = escape(str(item.get('text', 'Diagnosis candidate')))
    reason = escape(str(item.get('reason', '')))
    refs = '; '.join(f"{r['card_id']} — PDF page {r['pdf_page']}" for r in item.get('evidence', []))
    st.markdown(
        f'<div style="background:{background};border:1px solid {border};border-left:6px solid {border};'
        'border-radius:10px;padding:16px 18px;margin:12px 0;color:#172b4d">'
        f'<div style="font-size:.8rem;font-weight:600;color:{border}">{label} · not confirmed</div>'
        f'<div style="font-size:1.1rem;font-weight:700;margin:6px 0">{title}</div>'
        + (f'<div style="line-height:1.6">Reason: {reason}</div>' if reason else '')
        + (f'<div style="font-size:.78rem;color:#526174;margin-top:10px">Source: {escape(refs)}</div>' if refs else '')
        + '</div>', unsafe_allow_html=True)


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
