"""Small Streamlit interface for the clinical card retrieval prototype."""

from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from applicability import card_rule_applicability
from build_index import load_cards
from recommend import RecommendationError, recommend, recommend_section, recommend_management
from retrieve import retrieve
from schema import Card
from validate_cards import validate_cards

CARDS_PATH = ROOT / "data" / "cards.jsonl"
PAGES_PATH = ROOT / "data" / "pages.json"
INDEX_PATH = ROOT / "data" / "index"


def load_all_cards(path: Path = CARDS_PATH) -> dict[str, Card]:
    """Load the complete corpus without applying verification filtering."""
    return {card.card_id: card for card in load_cards(path)}


def patient_from_inputs(
    symptoms: str,
    findings: str,
    measurements: str,
    duration: str,
    age: str,
    sex: str,
    history: str,
    medicines: str,
    allergies: str,
) -> dict[str, Any]:
    """Build the patient shape accepted by the existing retrieval module."""
    patient: dict[str, Any] = {
        "symptoms": symptoms.strip(),
        "duration": duration.strip(),
        "age": int(age) if age.strip().isdigit() else age.strip(),
        "sex": sex.strip(),
        "history": history.strip(),
        "medicines": medicines.strip(),
        "allergies": allergies.strip(),
    }
    if findings.strip():
        patient["clinical_findings"] = findings.strip()
    if measurements.strip():
        patient["measurements"] = measurements.strip()
    return {key: value for key, value in patient.items() if value not in ("", None)}


def applicability_for(card: Card, patient: dict[str, Any]) -> dict[str, Any] | None:
    """Evaluate structured rules when present, while preserving unruled cards."""
    return card_rule_applicability(card, patient)


def validation_summary() -> tuple[int, int]:
    """Return card and validation-error counts without exposing validator logs."""
    with contextlib.redirect_stdout(io.StringIO()):
        return validate_cards(CARDS_PATH, PAGES_PATH)


def recommendation_for_patient(
    patient: dict[str, Any],
    mode: str,
    top_k: int,
    cards: dict[str, Card],
    *,
    recommend_fn: Any = recommend,
) -> dict[str, Any]:
    """Generate a validated recommendation through the existing engine."""
    return recommend_fn(
        patient,
        mode=mode,
        top_k=top_k,
        cards=cards,
        persist_dir=INDEX_PATH,
    )


def management_disease_choices(diagnosis: dict[str, Any], cards: dict[str, Card]) -> list[str]:
    """Use validated diagnosis evidence, never unrelated retrieval candidates."""
    choices: list[str] = []
    for item in diagnosis.get("probable_diagnosis", []):
        protocols = {
            cards[ref["card_id"]].disease_protocol
            for ref in item.get("evidence", [])
            if isinstance(ref, dict) and ref.get("card_id") in cards
        }
        # Mixed-protocol evidence needs clinician clarification, not auto-selection.
        if len(protocols) == 1:
            disease = next(iter(protocols))
            if disease not in choices:
                choices.append(disease)
    return choices


def _render_recommendation_item(st: Any, item: dict[str, Any]) -> None:
    st.markdown(f"**{item.get('text', 'Unspecified recommendation')}**")
    if item.get("reason"):
        st.write(f"Reason: {item['reason']}")
    if item.get("evidence"):
        st.write("Evidence references:")
        for reference in item["evidence"]:
            st.write(f"- {reference['card_id']} — PDF page {reference['pdf_page']}")


def _render_recommendations(st: Any, result: dict[str, Any]) -> None:
    sections = (
        ("Probable diagnoses", "probable_diagnosis"),
        ("Investigations", "investigations"),
        ("Medication planning", "medication"),
        ("Advice", "advice"),
    )
    for title, key in sections:
        st.subheader(title)
        items = result.get(key, [])
        if not items:
            st.info("None returned.")
            continue
        for item in items:
            _render_recommendation_item(st, item)
            if key == "medication":
                details = {
                    field: item.get(field)
                    for field in (
                        "name",
                        "dose",
                        "route",
                        "frequency",
                        "duration",
                        "instructions",
                    )
                    if item.get(field) is not None
                }
                if details:
                    st.json(details)

    st.subheader("Missing information")
    missing_info = result.get("missing_info", [])
    if missing_info:
        for item in missing_info:
            st.warning(item)
    else:
        st.info("No additional missing information was returned.")


def _streamlit() -> Any:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - depends on deployment extras
        raise RuntimeError(
            "Streamlit is not installed. Run `python -m pip install -r requirements.txt`."
        ) from exc
    return st


def _render_card(st: Any, card: Card, result: dict[str, Any], patient: dict[str, Any]) -> None:
    with st.expander(f"{card.card_id} — {card.disease_protocol}"):
        st.write(f"**Category:** {card.clinical_category}")
        st.write(f"**Source:** {card.source_pdf_title}")
        st.write(f"**PDF pages:** {card.pdf_page_numbers}")
        st.write(f"**Retrieval score:** {result.get('similarity_score', 'n/a')}")
        applicability = applicability_for(card, patient)
        if applicability is None:
            st.info("Applicability: UNKNOWN — this card has no structured rules.")
        else:
            status = {
                "applicable": "TRUE",
                "not_applicable": "FALSE",
                "undetermined": "UNKNOWN",
            }.get(applicability["status"], "UNKNOWN")
            st.write(f"**Applicability:** {status}")
            if applicability["missing_inputs"]:
                st.warning(
                    "Missing clinical inputs: "
                    + ", ".join(applicability["missing_inputs"])
                )
        st.markdown("**Evidence passages**")
        for evidence in card.exact_supporting_evidence_passages:
            st.markdown(f"- **Page {evidence.pdf_page}:** {evidence.passage}")
        if result.get("safety_critical"):
            st.warning("Safety-critical evidence: review contraindications and precautions.")


def main() -> None:
    st = _streamlit()
    st.set_page_config(page_title="Clinical NLP + RAG", layout="wide")
    st.title("Clinical NLP + RAG prototype")
    st.warning(
        "Prototype only. Retrieved evidence is not medical advice. "
        "Diagnoses are not confirmed and every recommendation requires clinician review."
    )

    try:
        cards = load_all_cards()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not load clinical cards: {exc}")
        return
    st.caption(f"Loaded {len(cards)} cards from data/cards.jsonl; no verification filtering applied in the app.")

    try:
        card_count, validation_errors = validation_summary()
    except (OSError, json.JSONDecodeError) as exc:
        st.warning(f"Evidence validation could not run: {exc}")
    else:
        if validation_errors:
            st.warning(
                f"Evidence validation reports {validation_errors} issue(s) across "
                f"{card_count} cards. Manual source verification is still required."
            )
        else:
            st.info("Evidence validation reported no text/page errors; manual verification is still required.")

    with st.form("patient_case"):
        symptoms = st.text_area("Symptoms", placeholder="Fever, headache, joint pain")
        findings = st.text_area("Clinical findings", placeholder="Exam findings and relevant observations")
        measurements = st.text_area("Relevant measurements", placeholder="BP, temperature, SpO2, laboratory values")
        duration = st.text_input("Duration")
        age = st.text_input("Age")
        sex = st.text_input("Sex")
        history = st.text_area("History")
        medicines = st.text_area("Current medicines")
        allergies = st.text_area("Allergies")
        mode = st.radio(
            "Recommendation mode",
            options=("baseline", "improved"),
            format_func=lambda value: value.title(),
            horizontal=True,
        )
        top_k = st.number_input("Number of cards", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("Generate recommendation")

    if submitted:
        st.session_state["case_patient"] = patient_from_inputs(
            symptoms, findings, measurements, duration, age, sex, history, medicines, allergies
        )
        st.session_state["case_mode"] = mode
        st.session_state["case_results"] = {}
        st.session_state.pop("diagnosis_ranking", None)
        st.session_state["case_diseases"] = []
        st.session_state["case_requested_stage"] = "Diagnosis"
        st.session_state["case_management_requested"] = False
        st.session_state.pop("working_disease", None)

    if "case_patient" not in st.session_state:
        return
    patient = st.session_state["case_patient"]
    st.subheader("Patient information")
    st.json(patient)

    # Initial diagnosis runs automatically after case submission.
    if st.session_state.pop("case_requested_stage", None) == "Diagnosis":
        try:
            from retrieve import retrieve_category
            if st.session_state.get("case_mode") == "improved":
                from patient_features import rank_diseases
                ranking = rank_diseases(patient, cards)
                st.session_state["diagnosis_ranking"] = ranking
                names = [item["disease"] for item in ranking["candidates"]]
            else:
                diagnosis_hits = retrieve_category(patient, "Diagnosis", top_k=6,
                                                   persist_dir=INDEX_PATH, cards=cards)
                names = []
                for hit in diagnosis_hits:
                    card = cards.get(hit["card_id"])
                    name = card.disease_protocol if card else None
                    if name in ("Dengue", "Chikungunya", "Cholera", "Hypertension") and name not in names:
                        names.append(name)
            with st.spinner("Reviewing diagnostic evidence..."):
                if st.session_state.get("case_mode") == "improved" and not names:
                    diagnosis = {"probable_diagnosis": [], "missing_info": [
                        "Insufficient matching case-definition evidence; request more clinical findings."
                    ], "evidence_card_ids": [], "notice": "No clinically supported disease candidate."}
                else:
                    diagnosis = recommend_section(
                        patient, "Diagnosis", mode=st.session_state.get("case_mode", "improved"),
                        cards=cards, persist_dir=INDEX_PATH, diseases=names[:2] or None,
                        diagnosis_card_ids=([item["card_id"] for item in ranking["candidates"]]
                                            if st.session_state.get("case_mode") == "improved"
                                            else None))
                st.session_state["case_results"]["Diagnosis"] = diagnosis
                st.session_state["case_diseases"] = (
                    management_disease_choices(diagnosis, cards)
                    if st.session_state.get("case_mode") == "improved"
                    else names[:2]
                )
        except (RecommendationError, ImportError, OSError, RuntimeError, ValueError) as exc:
            st.error(f"Diagnosis unavailable: {exc}")

    if "Diagnosis" in st.session_state.get("case_results", {}):
        output = st.session_state["case_results"]["Diagnosis"]
        st.subheader("Probable diagnosis")
        if st.session_state.get("case_mode") == "improved":
            ranking = st.session_state.get("diagnosis_ranking", {})
            st.caption("Clinical candidate status: " + ranking.get("status", "unknown") +
                       " (heuristic ranking, not a diagnosis probability)")
        if output.get("notice"):
            st.warning(output["notice"])
        for item in output.get("probable_diagnosis", []):
            _render_recommendation_item(st, item)
        if output.get("evidence_card_ids"):
            with st.expander("Diagnosis: source cards"):
                for cid in output["evidence_card_ids"]:
                    card = cards.get(cid)
                    if card:
                        st.write(f"{cid} — {getattr(card, 'card_title', '')} ({card.disease_protocol}), pages {card.pdf_page_numbers}")
        if st.session_state.get("case_mode") == "improved":
            candidates = st.session_state.get("case_diseases", [])
            if candidates:
                st.selectbox(
                    "Working diagnosis for management (clinician selection)",
                    candidates, index=0 if len(candidates) == 1 else None,
                    placeholder="Select a working diagnosis",
                    key="working_disease",
                )
            else:
                st.warning("No diagnosis with single-protocol evidence. Management is blocked pending review.")
            can_generate = bool(st.session_state.get("working_disease") in candidates)
        else:
            can_generate = True
        if st.button("Generate Treatment & Management Plan", type="primary",
                     key="generate_management", disabled=not can_generate):
            st.session_state["case_management_requested"] = True

    if st.session_state.get("case_management_requested"):
        if "Management" not in st.session_state["case_results"]:
            diseases = st.session_state.get("case_diseases") or None
            if st.session_state.get("case_mode") == "improved":
                chosen = st.session_state.get("working_disease")
                if chosen not in (diseases or []):
                    st.error("Choose a supported working diagnosis before management.")
                    return
                diseases = [chosen]
            with st.spinner("Generating evidence-grounded management plan..."):
                st.session_state["case_results"]["Management"] = recommend_management(
                    patient, mode=st.session_state.get("case_mode", "improved"),
                    cards=cards, persist_dir=INDEX_PATH, diseases=diseases)
        output = st.session_state["case_results"]["Management"]
        st.subheader("Treatment & Management Plan")
        if output.get("notice"):
            st.warning(output["notice"])
        for section, key in (("Investigations", "investigations"), ("Treatment", "medication"), ("Advice", "advice")):
            st.markdown(f"### {section}")
            items = output.get(key, [])
            if not items:
                st.info("No validated recommendation available for this section; clinician review required.")
            for item in items:
                _render_recommendation_item(st, item)
                if key == "medication":
                    details = {x: item[x] for x in ("name", "dose", "route", "frequency", "duration", "instructions") if item.get(x)}
                    if details:
                        st.json(details)
        with st.expander("Management: source cards"):
            for cid in output.get("evidence_card_ids", []):
                card = cards.get(cid)
                if card:
                    st.write(f"{cid} — {getattr(card, 'card_title', '')} ({card.disease_protocol}), pages {card.pdf_page_numbers}")
    st.caption("Educational prototype only; all recommendations need qualified clinical review.")


if __name__ == "__main__":
    main()
