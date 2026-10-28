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


def _render_recommendation_item(st: Any, item: dict[str, Any], *, subtle_sources: bool = False) -> None:
    st.markdown(f"**{item.get('text', 'Unspecified recommendation')}**")
    if item.get("reason"):
        st.write(f"Reason: {item['reason']}")
    if item.get("evidence"):
        if subtle_sources:
            from clinical_readability import muted
            muted(st, "Source: " + "; ".join(f"{r['card_id']} — PDF page {r['pdf_page']}" for r in item["evidence"]))
        else:
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
    from app_presentation import APP_TITLE, style, validation_details
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    style(st)
    st.title(APP_TITLE)
    st.error(
        "Prototype only. Retrieved evidence is not medical advice. "
        "Diagnoses are not confirmed and every recommendation requires clinician review."
    )

    from deployment import configure_runtime, ensure_index
    # Community Cloud supplies settings through Secrets; local .env loading is
    # retained. No clinical processing or model prompts are changed here.
    try:
        settings = dict(st.secrets)
    except FileNotFoundError:
        settings = {}
    configure_runtime(ROOT, settings)
    try:
        with st.spinner("Preparing local evidence search (first startup may take a few minutes)…"):
            st.cache_resource(show_spinner=False)(ensure_index)(ROOT)
    except Exception:
        st.error("Evidence search could not start. Check deployment logs, model download access, and index storage.")
        import logging
        logging.getLogger(__name__).exception("Evidence index startup failed")
        return

    try:
        cards = load_all_cards()
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not load clinical cards: {exc}")
        return
    try:
        card_count, validation_errors = validation_summary()
    except (OSError, json.JSONDecodeError) as exc:
        validation_report = f"Evidence validation could not run: {exc}"
    else:
        if validation_errors:
            validation_report = (
                f"Evidence validation reports {validation_errors} issue(s) across "
                f"{card_count} cards. Manual source verification is still required."
            )
        else:
            validation_report = "Evidence validation reported no text/page errors; manual verification is still required."

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
        from app_presentation import reset_case_selection
        reset_case_selection(st.session_state)

    if "case_patient" not in st.session_state:
        validation_details(st, validation_report)
        return
    patient = st.session_state["case_patient"]
    if st.session_state.get("case_mode") == "improved":
        from improved_input import normalize_improved_inputs
        patient = normalize_improved_inputs(patient)
        st.session_state["case_patient"] = patient
    with st.expander("Patient information — view submitted details", expanded=False):
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
        for item in output.get("probable_diagnosis", []):
            from app_presentation import diagnosis_card
            diagnosis_card(st, item)
        for item in output.get("uncertain_differentials", []):
            from app_presentation import diagnosis_card
            diagnosis_card(st, item, differential=True)
        if output.get("evidence_card_ids"):
            with st.expander(":gray[Diagnosis: source cards]" if st.session_state.get("case_mode") == "improved" else "Diagnosis: source cards"):
                for cid in output["evidence_card_ids"]:
                    card = cards.get(cid)
                    if card:
                        text = f"{cid} — {getattr(card, 'card_title', '')} ({card.disease_protocol}), pages {card.pdf_page_numbers}"
                        if st.session_state.get("case_mode") == "improved":
                            from clinical_readability import muted
                            muted(st, text)
                        else:
                            st.write(text)
        if st.session_state.get("case_mode") == "improved":
            candidates = st.session_state.get("case_diseases", [])
            if candidates:
                from app_presentation import choose_working_diagnosis
                choose_working_diagnosis(st, candidates)
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
        original_management = output
        if st.session_state.get("case_mode") == "improved":
            from management_clarification_ui import current_view
            output = current_view(original_management)
        st.subheader("Treatment & Management Plan")
        guidance_updates = {"added": set(), "removed": []}
        if st.session_state.get("case_mode") == "improved" and output.get("_clarification_answers"):
            from management_presentation import guidance_changes
            guidance_updates = guidance_changes(original_management, output)
            with st.container(border=True):
                st.markdown("#### What changed after your answers")
                st.info("Answers applied to source rules. Diagnosis is unchanged; clinician review is required.")
                cols = st.columns(3)
                cols[0].metric("Answers applied", len(output["_clarification_answers"]))
                cols[1].metric("New source matches", len(guidance_updates["added"]))
                cols[2].metric("No longer matched", len(guidance_updates["removed"]))
                if not guidance_updates["added"] and not guidance_updates["removed"]:
                    st.caption("Your answers were saved; no supported source instruction changed.")
                if guidance_updates["removed"]:
                    with st.expander("Previously matched guidance — no longer a current source match"):
                        st.warning("These previous matches are not part of the updated matching guidance. Review the current conditions and restrictions below.")
                        for rule in guidance_updates["removed"]:
                            st.write(rule["source_instruction"])
                            st.caption(f"Previous source: {rule['card_id']} / {rule.get('rule_id')}")
        if output.get("notice"):
            st.warning(output["notice"])
        extraction_failed = (st.session_state.get("case_mode") == "improved" and bool(
            output.get("management_diagnostics", {}).get("management_fact_extraction", {}).get("error")))
        if extraction_failed:
            st.warning("Patient-fact extraction did not complete because the provider request failed. "
                       "Unassessed plans are reference material, not treatment recommendations. "
                       "This does not mean the patient omitted all the listed findings.")
        if output.get("missing_info") and not extraction_failed:
            with st.expander("Missing management information"):
                for detail in output["missing_info"]:
                    if st.session_state.get("case_mode") == "improved":
                        from management_presentation import conditional_message
                        detail = conditional_message(detail)
                    st.write(f"- {detail}")
        if output.get("management_diagnostics") and st.session_state.get("case_mode") != "improved":
            with st.expander("Management pipeline diagnostics (no patient text)"):
                st.json(output["management_diagnostics"])
        if st.session_state.get("case_mode") == "improved":
            pathway = output.get("management_diagnostics", {}).get("resolved_source_context", {}).get("treatment_plan")
            if isinstance(pathway, str) and pathway:
                st.info("Source-selected management pathway: " + pathway)
                from management_display import render_plan_reference
                render_plan_reference(st, output, cards, pathway)
            age_info = output.get("management_diagnostics", {}).get("age_normalization", {})
            if age_info.get("status") == "normalized":
                unit = age_info["unit"]
                value = age_info["age_months" if unit == "months" else "age_years"]
                st.caption(f"Age interpreted for management: {value:g} {unit}.")
            notices = output.get("safety_notices", [])
            immediate = [notice for notice in notices if notice["status"] == "restriction_applies"
                         or notice["medicine_mentioned"]]
            if immediate:
                st.markdown("### Source medication safety checks")
                for notice in immediate:
                    label = ("Restriction applies to the scoped source instruction" if notice["status"] == "restriction_applies"
                             else "Source condition not triggered; overall suitability is not established" if notice["status"] == "not_triggered"
                             else "Safety review required")
                    st.warning(label + ": " + notice["text"])
                    if notice.get("source_population"):
                        st.caption("Source population: " + str(notice["source_population"]) + "; membership is not assumed for this patient.")
                    if notice["medicine_mentioned"]:
                        st.caption("A source medication term was mentioned in the input; this is not proof of current use or a drug-interaction check.")
                    st.caption(f"Source: {notice['card_id']} — PDF pages {notice['pdf_pages']}")
            other_notes = [notice for notice in notices if notice not in immediate]
            if other_notes:
                with st.expander("Source safety and interpretation notes"):
                    for notice in other_notes:
                        st.write(notice["text"])
                        if notice.get("source_population"):
                            st.caption("Source population: " + str(notice["source_population"]) + "; membership is not assumed for this patient.")
                        if notice["status"] == "scope_pending":
                            st.caption("Restriction scope needs review; it has not been ignored or treated as permission.")
        for section, key in (("Investigations", "investigations"), ("Treatment", "medication"), ("Advice", "advice")):
            st.markdown(f"### {section}")
            items = output.get(key, [])
            improved = st.session_state.get("case_mode") == "improved"
            if improved:
                from management_evidence import condition_text
                items = [item for item in items if not item.get("review_required")]
            if not items and not improved:
                st.info("No validated recommendation available for this section; clinician review required.")
            for item in items:
                _render_recommendation_item(st, item, subtle_sources=st.session_state.get("case_mode") == "improved")
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
                    st.info("Matching source guidance — clinician review required:")
                    from management_display import render_instruction_groups
                    render_instruction_groups(st, matches, cards, guidance_updates["added"])
                from management_presentation import split_conditional_sources
                unresolved, additional = split_conditional_sources(reviews, output.get("management_diagnostics", {}))
                if not items and not matches:
                    st.info("More information is needed; review the conditional options below."
                            if unresolved else "No matching source guidance established; additional source material is available below."
                            if additional else "No matching source guidance available for this section.")
                if unresolved:
                    with st.expander("Conditional source options — information needed before use"):
                        st.warning("Conditional — assess the missing requirements before use.")
                        from clinical_readability import render_conditional_options
                        render_conditional_options(st, unresolved, cards)
                if additional:
                    with st.expander("Additional source material — applicability not established"):
                        st.info("Reference only — not a selected treatment recommendation.")
                        st.write("Other source references, not selected treatment recommendations. Open an entry to review its unresolved conditions.")
                        for entry in additional:
                            with st.expander(f"{entry['title']} — {entry['card_id']}"):
                                from management_display import render_source_doses
                                render_source_doses(st, cards.get(entry['card_id']))
                                if entry.get("missing_inputs"):
                                    st.caption("Not established: " + ", ".join(field.replace("_", " ") for field in entry["missing_inputs"]))
                                if any(rule.get("unsupported_rule") for rule in entry["conditional_rules"]):
                                    st.caption("Not automatically evaluated: this source table/rule requires manual review. Additional answers alone do not enable automatic row selection; no row or dose is assumed.")
        with st.expander(":gray[Management: source cards]" if st.session_state.get("case_mode") == "improved" else "Management: source cards"):
            for cid in output.get("evidence_card_ids", []):
                card = cards.get(cid)
                if card:
                    text = f"{cid} — {getattr(card, 'card_title', '')} ({card.disease_protocol}), pages {card.pdf_page_numbers}"
                    if st.session_state.get("case_mode") == "improved":
                        from clinical_readability import muted
                        muted(st, text)
                    else:
                        st.write(text)
        if st.session_state.get("case_mode") == "improved":
            from management_clarification_ui import render_questions
            render_questions(original_management, cards)
    if st.session_state.get("case_mode") == "improved":
        diagnostic_debug = st.session_state.get("diagnosis_ranking", {}).get("diagnostics", {})
        management_debug = output.get("management_diagnostics", {}) if st.session_state.get("case_management_requested") else {}
        if diagnostic_debug or management_debug:
            with st.expander(":gray[Technical diagnostics] — no patient text"):
                st.caption("Optional technical details; clinical guidance is shown above.")
                if diagnostic_debug:
                    with st.expander(":gray[Diagnostic pipeline diagnostics]"):
                        st.json(diagnostic_debug)
                if management_debug:
                    with st.expander(":gray[Management pipeline diagnostics]"):
                        st.json(management_debug)
    validation_details(st, validation_report)
    st.caption("Educational prototype only; all recommendations need qualified clinical review.")


if __name__ == "__main__":
    main()
