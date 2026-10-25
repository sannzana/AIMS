"""Streamlit adapter loaded only in the Improved management branch."""
import streamlit as st

from management_clarification import questions, apply_answers


def _state(original):
    state = st.session_state.get("management_clarification")
    if not state or state["original"] is not original:
        state = {"original": original, "view": original, "asked": set(), "round": 0}
        st.session_state["management_clarification"] = state
    return state


def current_view(original):
    return _state(original)["view"]


def render_questions(original, cards):
    state = _state(original)
    if st.session_state.get("working_disease") != original.get("working_diagnosis"):
        return
    offered = questions(state["view"], cards, state["asked"])
    with st.expander("Improve treatment with additional information"):
        st.caption("Optional source-based questions. Answer only if known; Unknown or Skip keeps existing guidance. "
                   "Answers affect management only, not the original diagnosis. Source field labels may require clinician interpretation.")
        if state["view"] is not original:
            st.info("Answers applied: " + "; ".join(
                f"{field.replace('_', ' ')}: {value}" for field, value in
                state["view"].get("_clarification_answers", {}).items()))
            st.caption("Original generated management is retained below; it has not been regenerated.")
            with st.expander("Original management result — before additional answers"):
                for key in ("notice", "investigations", "medication", "advice"):
                    if original.get(key):
                        st.write({key: original[key]})
        if offered:
            with st.form(f"management_questions_{state['round']}"):
                answers = {}
                for q in offered:
                    field = q["field"]
                    key = f"management_answer_{id(original)}_{state['round']}_{field}"
                    if q["kind"] == "boolean":
                        choice = st.selectbox(q["label"], ("Unknown", "Yes", "No"), key=key)
                        answers[field] = {"Unknown": None, "Yes": True, "No": False}[choice]
                    else:
                        raw = st.text_input(f"{q['label']} ({q['unit']}; leave blank if unknown)", key=key)
                        answers[field] = raw.strip() or None
                    st.caption("Source: " + "; ".join(f"{s['card_id']} / {s['rule_id'] or 'classification assessment'} / pages {s['pdf_pages']}" for s in q["sources"]))
                    st.caption({0: "Checks a source safety restriction.", 1: "Checks a prerequisite for the current treatment.",
                                2: "Helps assess the source classification for management.",
                                3: "Checks an unresolved condition of a partially matched instruction."}[q["priority"]])
                apply = st.form_submit_button("Apply answers")
                skip = st.form_submit_button("Skip these questions")
            if apply or skip:
                try:
                    if apply:
                        for q in offered:
                            if q["kind"] == "number" and answers[q["field"]] is not None:
                                answers[q["field"]] = float(answers[q["field"]])
                        if any(value is not None for value in answers.values()):
                            state["view"] = apply_answers(state["view"], cards, answers, offered)
                    state["asked"].update(q["field"] for q in offered)
                    state["round"] += 1
                    st.rerun()
                except ValueError as exc:
                    st.error(str(exc))
        else:
            st.caption("No further source-backed questions are available for this initial management step. "
                       "Unresolved tables, later reassessment, and discharge criteria still require clinician review.")
        if state["asked"] and st.button("Reset additional answers", key="reset_management_answers"):
            st.session_state.pop("management_clarification", None)
            st.rerun()
