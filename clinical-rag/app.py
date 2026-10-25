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
    for item in (
        diagnosis.get("probable_diagnosis", [])
        + diagnosis.get("uncertain_differentials", [])
    ):
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



def diagnosis_from_triage(ranking: dict[str, Any], cards: dict[str, Card]) -> dict[str, Any]:
    """Render source-backed candidate hypotheses without another LLM request.

    A triage candidate is NOT a confirmed diagnosis. Only a supported candidate
    with an existing case-definition card and literal patient quotes qualifies.
    """
    from card_roles import assign_role
    items = []
    uncertain = []
    for candidate in ranking.get("candidates", []):
        if not isinstance(candidate, dict):
            continue
        card = cards.get(candidate.get("card_id"))
        quotes = candidate.get("matched_features")
        if (card is None or card.clinical_category != "Diagnosis"
                or assign_role(card) != "case_definition"
                or card.disease_protocol != candidate.get("disease")
                or not isinstance(quotes, list) or not quotes):
            continue
        pages = card.pdf_page_numbers or []
        if not pages:
            continue
        item = {
            "text": "Suspected " + card.disease_protocol + " (source-backed clinical candidate; not confirmed)",
            "reason": "Patient-reported evidence: " + "; ".join(str(x) for x in quotes[:4]),
            "evidence": [{"card_id": card.card_id, "pdf_page": pages[0]}],
        }
        if candidate.get("decision") == "uncertain":
            uncertain.append({
                **item,
                "text": "Possible " + card.disease_protocol + " (conditional differential; not confirmed)",
                "reason": item["reason"] + ". Further evaluation is needed.",
            })
        else:
            items.append(item)
    return {
        "probable_diagnosis": items, "investigations": [], "medication": [], "advice": [],
        "uncertain_differentials": uncertain,
        "missing_info": [str(x) for c in ranking.get("candidates", [])
                         for x in (c.get("missing") or []) if isinstance(x, str)][:8],
        "evidence_card_ids": [
            x["evidence"][0]["card_id"] for x in items + uncertain
        ],
        "stage": "Diagnosis",
        "notice": None if items or uncertain else ranking.get("notice", "No verified source-backed candidate."),
    }


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
        ("Possible differentials", "uncertain_differentials"),
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
                from disease_triage import triage
                ranking = triage(patient, cards)
                st.session_state["diagnosis_ranking"] = ranking
                st.session_state["structured_patient_facts"] = ranking.get("features", {})
                names = [item["disease"] for item in ranking["candidates"]]
            else:
                diagnosis_hits = retrieve_category(patient, "Diagnosis", top_k=6,
                                                   persist_dir=INDEX_PATH, cards=cards)
                names = []
                for hit in diagnosis_hits:
                    card = cards.get(hit["card_id"])
                    name = card.disease_protocol if card else None
                    if name and name not in names:
                        names.append(name)
            with st.spinner("Reviewing diagnostic evidence..."):
                if st.session_state.get("case_mode") == "improved" and not names:
                    diagnosis = {"probable_diagnosis": [], "missing_info": [
                        "Diagnostic selection did not complete; check candidate status and retrieval/model diagnostics."
                    ], "evidence_card_ids": [], "notice": ranking.get("notice") or "No supported candidate after source review."}
                elif st.session_state.get("case_mode") == "improved":
                    diagnosis = diagnosis_from_triage(ranking, cards)
                else:
                    diagnosis = recommend_section(
                        patient, "Diagnosis", mode="baseline",
                        cards=cards, persist_dir=INDEX_PATH, diseases=names[:2] or None)
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
                       " (source-grounded triage status, not a diagnosis probability)")
        if output.get("notice"):
            st.warning(output["notice"])
        if st.session_state.get("case_mode") == "improved":
            with st.expander("Diagnostic pipeline diagnostics (no patient text)"):
                st.json(st.session_state.get("diagnosis_ranking", {}).get("diagnostics", {}))
        for item in output.get("probable_diagnosis", []):
            _render_recommendation_item(st, item)
        for item in output.get("uncertain_differentials", []):
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
                st.warning("No source-backed candidate is available for clinician selection.")
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
                    st.error("Choose a source-backed working diagnosis before management.")
                    return
                diseases = [chosen]
            with st.spinner("Generating evidence-grounded management plan..."):
                st.session_state["case_results"]["Management"] = recommend_management(
                    {**patient, "_structured_facts": st.session_state.get("structured_patient_facts", {})}
                    if st.session_state.get("case_mode") == "improved" else patient,
                    mode=st.session_state.get("case_mode", "improved"),
                    cards=cards, persist_dir=INDEX_PATH, diseases=diseases)
        output = st.session_state["case_results"]["Management"]
        st.subheader("Treatment & Management Plan")
        if output.get("notice"):
            st.warning(output["notice"])
        if output.get("missing_info"):
            with st.expander("Missing management information"):
                for detail in output["missing_info"]:
                    st.write(f"- {detail}")
        if output.get("management_diagnostics"):
            with st.expander("Management pipeline diagnostics (no patient text)"):
                st.json(output["management_diagnostics"])
        for section, key in (("Investigations", "investigations"), ("Treatment", "medication"), ("Advice", "advice")):
            st.markdown(f"### {section}")
            items = output.get(key, [])
            improved = st.session_state.get("case_mode") == "improved"
            if improved:
                from management_evidence import condition_text
                items = [item for item in items if not item.get("review_required")]
            if not items:
                st.info("No patient-specific recommendation confirmed; see source matches and information needed below."
                        if improved else "No validated recommendation available for this section; clinician review required.")
            for item in items:
                _render_recommendation_item(st, item)
                if key == "medication":
                    details = {x: item[x] for x in ("name", "dose", "route", "frequency", "duration", "instructions") if item.get(x)}
                    if details:
                        st.json(details)
            if improved:
                category = {"investigations": "Investigation", "medication": "Medication", "advice": "Advice"}[key]
                reviews = [entry for entry in output.get("source_review_options", [])
                           if entry["category"] == category]
                matches = [rule for entry in reviews for rule in entry["supported_rules"]
                           if rule.get("source_instruction")]
                if matches:
                    st.write("Source instructions matching known facts — clinician review required:")
                    for rule in matches:
                        st.write(rule["source_instruction"])
                        st.caption(f"{rule['card_id']} | Rule: {rule.get('rule_id') or 'source guidance'}")
                        with st.expander("Matched source condition"):
                            st.write(condition_text(rule["condition"]))
                unresolved = [entry for entry in reviews if entry["conditional_rules"]]
                if unresolved:
                    with st.expander("Conditional source options — information needed before use"):
                        st.write("These are source options for assessment, not instructions to start treatment.")
                        for entry in unresolved:
                            st.write(f"{entry['card_id']} — {entry['title']}")
                            for rule in entry["conditional_rules"]:
                                if rule["missing_inputs"]:
                                    st.write("Check: " + ", ".join(f.replace("_", " ") for f in rule["missing_inputs"]))
                                if rule["unsupported_rule"]:
                                    st.write("This source rule/table cannot yet be evaluated; do not assume a matching row.")
                                if rule["condition"]:
                                    st.write("Applies if: " + condition_text(rule["condition"]))
                                if rule["prerequisites"]:
                                    for gate in rule["prerequisites"]:
                                        st.write(f"Requires {gate['field'].replace('_', ' ')}: {gate['required_value']}")
                                if rule["restrictions"]:
                                    st.write("Source restrictions:")
                                    for restriction in rule["restrictions"]:
                                        if isinstance(restriction, dict):
                                            st.write(restriction.get("interpretation") or restriction.get("action") or
                                                     condition_text(restriction.get("condition")))
                                        else:
                                            st.write(str(restriction))
        with st.expander("Management: source cards"):
            for cid in output.get("evidence_card_ids", []):
                card = cards.get(cid)
                if card:
                    st.write(f"{cid} — {getattr(card, 'card_title', '')} ({card.disease_protocol}), pages {card.pdf_page_numbers}")
    st.caption("Educational prototype only; all recommendations need qualified clinical review.")


if __name__ == "__main__":
    main()
