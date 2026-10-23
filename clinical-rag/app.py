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

from applicability import check_applicability
from build_index import load_cards
from retrieve import retrieve
from schema import Card, Conditions
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
    conditions = card.conditions_value
    if conditions is None:
        return None
    if card.structured_rules:
        card = card.model_copy(update={"conditions": Conditions.model_validate(conditions)})
    return check_applicability(card, patient)


def validation_summary() -> tuple[int, int]:
    """Return card and validation-error counts without exposing validator logs."""
    with contextlib.redirect_stdout(io.StringIO()):
        return validate_cards(CARDS_PATH, PAGES_PATH)


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
        top_k = st.number_input("Number of cards", min_value=1, max_value=20, value=5)
        submitted = st.form_submit_button("Retrieve guideline evidence")

    if not submitted:
        return
    patient = patient_from_inputs(
        symptoms, findings, measurements, duration, age, sex, history, medicines, allergies
    )
    st.subheader("Patient information")
    st.json(patient)

    try:
        results = retrieve(patient, top_k=int(top_k), persist_dir=INDEX_PATH)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        st.error(
            "The local vector index or an optional retrieval dependency is unavailable. "
            f"Build it with `python src/build_index.py --rebuild`. Details: {exc}"
        )
        return
    if not results:
        st.warning("No guideline cards were retrieved.")
        return

    st.subheader("Retrieved evidence")
    for result in results:
        card = cards.get(result.get("card_id"))
        if card is None:
            st.error(f"Retrieved card {result.get('card_id')} is missing from the loaded corpus.")
            continue
        _render_card(st, card, result, patient)

    st.subheader("Next step")
    st.info(
        "No unsupported recommendation is generated by this interface. "
        "Use the cited evidence and applicability status for clinician review."
    )


if __name__ == "__main__":
    main()
