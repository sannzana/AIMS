"""Evidence-grounded recommendation generation for baseline and improved modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from applicability import card_rule_applicability
from build_index import load_cards
from llm import LLMError, generate
from retrieve import retrieve, QUERY_FIELDS
from schema import Card

EXCLUDE_FROM_PROMPT = ("image", "image_related_info")


class EvidenceReference(BaseModel):
    card_id: str
    pdf_page: int


class RecommendationItem(BaseModel):
    text: str
    reason: str
    evidence: list[EvidenceReference]


class MedicationRecommendation(RecommendationItem):
    name: str | None = None
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    instructions: str | None = None
    eligibility: str | None = None
    eligibility_notes: list[str] = Field(default_factory=list)
    source_details: str | None = None


class RecommendationResponse(BaseModel):
    probable_diagnosis: list[RecommendationItem] = Field(default_factory=list)
    investigations: list[RecommendationItem] = Field(default_factory=list)
    medication: list[MedicationRecommendation] = Field(default_factory=list)
    advice: list[RecommendationItem] = Field(default_factory=list)
    missing_info: list[str] = Field(default_factory=list)


class RecommendationError(RuntimeError):
    """Raised when generated recommendations are not safe to use."""


class MalformedRecommendationError(RecommendationError):
    """Raised when the model response is not JSON."""


class RecommendationSchemaError(RecommendationError):
    """Raised when JSON does not match the recommendation schema."""


class UnsupportedEvidenceError(RecommendationError):
    """Raised when a recommendation cites unavailable evidence."""


def _validation_diagnostics(error: ValidationError) -> str:
    """Return field paths and error types without exposing values or input data."""
    diagnostics = []
    for item in error.errors():
        location = item.get("loc", ())
        path = ".".join(str(part) for part in location) or "$"
        error_type = str(item.get("type", "validation_error"))
        diagnostics.append(f"{path} ({error_type})")
    return "; ".join(diagnostics) or "$ (validation_error)"


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _parse_response(raw: str) -> RecommendationResponse:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedRecommendationError(
            "LLM output was not valid JSON"
        ) from exc
    try:
        if hasattr(RecommendationResponse, "model_validate"):
            return RecommendationResponse.model_validate(payload)
        return RecommendationResponse.parse_obj(payload)
    except ValidationError as exc:
        raise RecommendationSchemaError(
            "LLM JSON did not match the recommendation schema; "
            f"validation errors: {_validation_diagnostics(exc)}"
        ) from exc
    except TypeError as exc:
        raise RecommendationSchemaError(
            "LLM JSON did not match the recommendation schema; "
            "validation errors: $ (type_error)"
        ) from exc


def _parse_response_partial(
    raw: str,
) -> tuple[RecommendationResponse, list[str]]:
    """Parse valid items independently so one unsafe item cannot hide others."""
    # Accept harmless Markdown fences while keeping JSON schema strict.
    # Never guess missing clinical fields or silently repair a malformed response.
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            raw = "\n".join(lines[1:-1]).strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedRecommendationError("LLM output was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RecommendationSchemaError(
            "LLM JSON did not match the recommendation schema; "
            "validation errors: $ (dict_type)"
        )
    rejected: list[str] = []
    sections: dict[str, list[Any]] = {}
    models = {
        "probable_diagnosis": RecommendationItem,
        "investigations": RecommendationItem,
        "medication": MedicationRecommendation,
        "advice": RecommendationItem,
    }
    for section, model in models.items():
        raw_items = payload.get(section, [])
        if not isinstance(raw_items, list):
            rejected.append(f"{section} (list_type)")
            sections[section] = []
            continue
        valid_items = []
        for index, raw_item in enumerate(raw_items):
            try:
                valid_items.append(
                    model.model_validate(raw_item)
                    if hasattr(model, "model_validate")
                    else model.parse_obj(raw_item)
                )
            except (ValidationError, TypeError) as exc:
                if isinstance(exc, ValidationError):
                    details = "; ".join(
                        f"{section}.{index}.{part}"
                        if not part.startswith("$")
                        else f"{section}.{index}"
                        for part in _validation_diagnostics(exc).split("; ")
                    )
                else:
                    details = f"{section}.{index} (type_error)"
                rejected.append(details)
        sections[section] = valid_items
    missing_info = payload.get("missing_info", [])
    if not isinstance(missing_info, list) or not all(
        isinstance(item, str) for item in missing_info
    ):
        rejected.append("missing_info (list/string_type)")
        missing_info = []
    response = RecommendationResponse(
        probable_diagnosis=sections["probable_diagnosis"],
        investigations=sections["investigations"],
        medication=sections["medication"],
        advice=sections["advice"],
        missing_info=missing_info,
    )
    return response, rejected


def _card_pages(card: Card) -> set[int]:
    return set(card.pdf_page_numbers) | {
        item.pdf_page for item in card.exact_supporting_evidence_passages
    }


def _contains_source_field(value: Any, field: str, expected: Any) -> bool:
    aliases = {
        "name": ("name",),
        "dose": ("dose", "dosage", "amount", "maximum"),
        "route": ("route",),
        "frequency": ("frequency", "times", "interval"),
        "duration": ("duration", "day", "hour"),
        "instructions": ("instruction", "restriction", "avoid", "indication"),
    }
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).casefold()
            if field == "name" and key_text == str(expected).casefold():
                return True
            if any(alias in key_text for alias in aliases[field]):
                if isinstance(child, (dict, list, tuple, set)):
                    if _contains_source_field(child, field, expected):
                        return True
                elif child == expected:
                    return True
            if _contains_source_field(child, field, expected):
                return True
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_contains_source_field(item, field, expected) for item in value)
    return value == expected


def _medication_field_supported(card: Card, field: str, value: Any) -> bool:
    medication = card.medication
    if medication is not None and getattr(medication, field) == value:
        return True
    details = card.model_dump().get("medication_details")
    if not isinstance(details, dict):
        return False
    return _contains_source_field(details, field, value)


def _validate_output(
    response: RecommendationResponse, cards: dict[str, Card], *,
    patient_context: dict[str, Any] | None = None, source_medication: bool = False,
) -> tuple[RecommendationResponse, list[str]]:
    items = (
        response.probable_diagnosis
        + response.investigations
        + response.medication
        + response.advice
    )
    if not items and not response.missing_info:
        raise RecommendationSchemaError(
            "Recommendation JSON must contain recommendations or missing_info"
        )
    accepted: dict[str, list[Any]] = {
        "probable_diagnosis": [],
        "investigations": [],
        "medication": [],
        "advice": [],
    }
    rejected: list[str] = []
    for section in accepted:
        for index, item in enumerate(getattr(response, section)):
            reason: str | None = None
            if not item.evidence:
                reason = "missing evidence"
            else:
                for reference in item.evidence:
                    card = cards.get(reference.card_id)
                    if card is None:
                        reason = f"unsupported evidence card {reference.card_id}"
                        break
                    if reference.pdf_page not in _card_pages(card):
                        reason = (
                            f"unsupported evidence page {reference.pdf_page} "
                            f"for card {reference.card_id}"
                        )
                        break
            if reason is None and source_medication and isinstance(item, MedicationRecommendation):
                from medication_source import finalize_medication_item
                finalized, reason = finalize_medication_item(item, cards, patient_context)
                if finalized is not None:
                    item = finalized
            elif reason is None and isinstance(item, MedicationRecommendation):
                supplied = {
                    name: getattr(item, name)
                    for name in (
                        "name", "dose", "route", "frequency", "duration", "instructions"
                    )
                    if getattr(item, name) is not None
                }
                sources = [cards[reference.card_id] for reference in item.evidence]
                for name, value in supplied.items():
                    if not any(_medication_field_supported(card, name, value) for card in sources):
                        reason = f"unsupported medication field {name}"
                        break
            if reason is None:
                accepted[section].append(item)
            else:
                rejected.append(f"{section}.{index}: {reason}")
    validated = RecommendationResponse(
        probable_diagnosis=accepted["probable_diagnosis"],
        investigations=accepted["investigations"],
        medication=accepted["medication"],
        advice=accepted["advice"],
        missing_info=response.missing_info,
    )
    if not any(accepted.values()) and not validated.missing_info:
        raise RecommendationSchemaError(
            "No recommendation passed evidence and medication safety validation"
        )
    return validated, rejected


def _attach_traceability(
    output: dict[str, Any], response: RecommendationResponse, cards: dict[str, Card]
) -> None:
    for key in ("probable_diagnosis", "investigations", "medication", "advice"):
        for item in output.get(key, []):
            details = []
            for reference in item.get("evidence", []):
                card = cards[reference["card_id"]]
                passages = [
                    evidence.passage
                    for evidence in card.exact_supporting_evidence_passages
                    if evidence.pdf_page == reference["pdf_page"]
                ]
                details.append(
                    {
                        "card_id": reference["card_id"],
                        "pdf_page": reference["pdf_page"],
                        "source_title": card.source_pdf_title,
                        "passages": passages,
                    }
                )
            item["evidence_details"] = details


def _prompt(patient: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    patient_context = {
        field: patient.get(field)
        for field in (
            *QUERY_FIELDS,
            "clinical_findings",
            "measurements",
            "medicines",
            "allergies",
        )
    }
    schema = {
        "probable_diagnosis": [
            {
                "text": "string",
                "reason": "string",
                "evidence": [{"card_id": "exact retrieved card ID", "pdf_page": 1}],
            }
        ],
        "investigations": [
            {
                "text": "string",
                "reason": "string",
                "evidence": [{"card_id": "exact retrieved card ID", "pdf_page": 1}],
            }
        ],
        "medication": [
            {
                "text": "string",
                "reason": "string",
                "evidence": [{"card_id": "exact retrieved card ID", "pdf_page": 1}],
                "name": "optional string",
                "dose": "optional string",
                "route": "optional string",
                "frequency": "optional string",
                "duration": "optional string",
                "instructions": "optional string",
            }
        ],
        "advice": [
            {
                "text": "string",
                "reason": "string",
                "evidence": [{"card_id": "exact retrieved card ID", "pdf_page": 1}],
            }
        ],
        "missing_info": ["string"],
    }
    return (
        "Return only one JSON object matching this complete schema; do not use "
        "Markdown fences or prose. All five top-level keys are required. "
        f"Schema: {json.dumps(schema, sort_keys=True)}. "
        "Every recommendation item must contain all three required fields: "
        "text (a non-empty recommendation string), reason (a non-empty source-grounded "
        "explanation), and evidence (a non-empty array of references). The medication "
        "array uses exactly the same required text, reason, and evidence fields; do "
        "not omit medication.0.text or medication.0.reason. Medication may additionally "
        "contain only name, dose, route, frequency, duration, and instructions; omit "
        "unsupported optional fields. "
        "Every evidence reference must use an exact card_id and a valid pdf_page "
        "from the supplied retrieved cards. missing_info must be an array of "
        "strings, including missing clinical inputs or uncertainty. "
        "Treat diagnoses as probable, never confirmed. Use only supplied evidence; "
        "never invent medication fields or doses. Every recommendation needs a "
        "source-supported reason and evidence references. "
        "List missing clinical information honestly. "
        f"Patient context: {json.dumps(patient_context, sort_keys=True)}. "
        f"Evidence: {json.dumps(evidence, default=str, sort_keys=True)}. "
    )


def _prompt_card(card: Card) -> dict[str, Any]:
    """Compact but source-faithful clinical context; never send the entire card.

    Retain conditional branches, exclusions and medication details in full; prune
    only descriptive/meta fields and redundant long evidence passages.
    """
    raw = card.model_dump()
    keep = (
        "card_id", "card_title", "disease_protocol", "clinical_category",
        "clinical_classification_presentation", "source_pdf_title",
        "source_version_edition", "pdf_page_numbers",
        "relevant_symptoms_or_prerequisites", "recommended_action",
        "conditions", "structured_rules", "conditions_decision_logic", "if_else_conditions_and_corresponding_actions",
        "exceptions_contraindications", "table_dependent_rules",
        "exclusions_or_missing_information", "exclusions_or_missing_info", "medication_details", "medication",
    )
    compact = {
        key: raw[key] for key in keep
        if raw.get(key) not in (None, "", [], {})
    }
    # Preserve future source-derived clinical fields without bringing in huge
    # metadata blobs. Never forward images, binary data, or entire documents.
    for key, value in raw.items():
        if (
            key not in compact
            and key not in ("image", "image_related_info", "additional_information")
            and ("clinical" in key or "contraindication" in key)
            and value not in (None, "", [], {})
            and len(json.dumps(value, default=str)) <= 1000
        ):
            compact[key] = value
    # Keep exact quotations including page numbers when relevant. A card's
    # whole PDF transcript is not needed in the generation prompt. The full
    # set remains in cards.jsonl for evidence traceability after generation.
    passages = raw.get("exact_supporting_evidence_passages") or []
    if passages:
        compact["exact_supporting_evidence_passages"] = [
            {"pdf_page": item["pdf_page"], "passage": item["passage"]}
            for item in passages[:3]
        ]
    return compact


def recommend(
    patient: dict[str, Any],
    mode: str = "baseline",
    *,
    top_k: int = 5,
    retrieve_fn: Callable[..., list[dict[str, Any]]] = retrieve,
    llm_generate: Callable[[str], str] = generate,
    cards: dict[str, Card] | None = None,
    cards_path: Path | None = None,
    persist_dir: Path = Path("data/index"),
) -> dict[str, Any]:
    if mode not in {"baseline", "improved"}:
        raise ValueError("mode must be 'baseline' or 'improved'")
    if cards is None:
        path = cards_path or (
            Path("data/sample_cards.jsonl")
            if persist_dir.name == "dev_index"
            else Path("data/cards.jsonl")
        )
        cards = {card.card_id: card for card in load_cards(path)}
    if retrieve_fn is retrieve:
        retrieved = retrieve_fn(
            patient=patient,
            top_k=top_k,
            persist_dir=persist_dir,
            cards=cards,
        )
    else:
        retrieved = retrieve_fn(patient=patient, top_k=top_k, persist_dir=persist_dir)
    if not retrieved:
        return _dump(RecommendationResponse(missing_info=["No evidence cards were retrieved."]))
    evidence: list[dict[str, Any]] = []
    for result in retrieved:
        card = cards.get(result.get("card_id"))
        if card is None:
            raise RecommendationError(f"Retrieved card is missing from corpus: {result.get('card_id')}")
        entry = _prompt_card(card)
        for field in (
            "disease_role",
            "safety_critical",
        ):
            if result.get(field) is not None:
                entry[field] = result[field]
        applicability = card_rule_applicability(card, patient)
        if applicability is not None:
            if applicability["status"] == "not_applicable":
                if mode == "improved":
                    continue
            entry["applicability"] = applicability
        evidence.append(entry)
    if not evidence:
        return _dump(RecommendationResponse(
            missing_info=["All retrieved cards were not applicable."]
        ))
    prompt = _prompt(patient, evidence)
    if len(prompt.encode("utf-8")) > 24000:
        raise RecommendationError(
            "Clinical evidence request exceeds the safe size limit. "
            "Reduce selected cards or request one clinical section; "
            "no evidence or safety rules were silently truncated."
        )
    try:
        response, schema_rejections = _parse_response_partial(
            llm_generate(prompt)
        )
    except LLMError as exc:
        raise RecommendationError(str(exc)) from exc
    if not (
        response.probable_diagnosis
        or response.investigations
        or response.medication
        or response.advice
        or response.missing_info
    ) and schema_rejections:
        raise RecommendationSchemaError(
            "LLM JSON did not match the recommendation schema; "
            "validation errors: " + "; ".join(schema_rejections)
        )
    response, safety_rejections = _validate_output(
        response, cards, source_medication=(mode == "improved"),
        patient_context=_structured_patient_context(patient))
    output = _dump(response)
    output["mode"] = mode
    output["dev_only"] = persist_dir.name == "dev_index"
    output["retrieval_summary"] = [
        {
            "card_id": result.get("card_id"),
            "category": cards[result["card_id"]].clinical_category,
            "selection_reason": result.get("selection_reason", "semantic relevance"),
            "safety_critical": bool(result.get("safety_critical", False)),
        }
        for result in retrieved
        if result.get("card_id") in cards
    ]
    output["retrieval_coverage"] = retrieved[0].get(
        "coverage",
        {
            "candidate_diseases": [],
            "available_categories": [],
            "selected_categories": [],
            "missing_categories": [],
            "complete": True,
            "incomplete_reason": None,
        },
    )
    rejections = schema_rejections + safety_rejections
    if rejections:
        output["incomplete_output"] = True
        output["rejected_recommendations"] = rejections
    _attach_traceability(output, response, cards)
    if output["dev_only"]:
        output["notice"] = "DEV ONLY: software testing output; not for patient care."
    return output


def _structured_patient_context(patient: dict[str, Any]) -> dict[str, Any]:
    """Source-schema facts are extracted once at triage, never via disease regex."""
    facts = patient.get("_structured_facts") or {}
    return {**patient, **facts} if isinstance(facts, dict) else dict(patient)


STAGE_TO_KEY = {
    "Diagnosis": "probable_diagnosis", "Investigations": "investigations",
    "Treatment": "medication", "Advice": "advice",
}


def recommend_section(patient: dict[str, Any], stage: str, *, mode: str = "improved",
                      cards: dict[str, Card], persist_dir: Path = Path("data/index"),
                      diseases: list[str] | None = None,
                      llm_generate: Callable[[str], str] = generate,
                      diagnosis_card_ids: list[str] | None = None) -> dict[str, Any]:
    """One independently validated clinical section per bounded LLM request."""
    from retrieve import retrieve_category
    if stage not in STAGE_TO_KEY:
        raise ValueError(f"Unsupported clinical stage: {stage}")
    category = "Medication" if stage == "Treatment" else ("Diagnosis" if stage == "Diagnosis" else stage[:-1] if stage == "Investigations" else stage)
    requested = 4 if stage == "Diagnosis" else (4 if stage == "Investigations" else 3)
    if mode == "improved" and stage == "Diagnosis" and diagnosis_card_ids is not None:
        # Disease ranking already read case definitions; no second vector search.
        items = [{"card_id": cid} for cid in diagnosis_card_ids if cid in cards]
    else:
        items = retrieve_category(patient, category, diseases=diseases,
                                  top_k=requested, persist_dir=persist_dir, cards=cards)
    if mode == "improved" and stage == "Diagnosis":
        from card_roles import assign_role
        items = [hit for hit in items if hit.get("card_id") in cards
                 and assign_role(cards[hit["card_id"]]) == "case_definition"]
    # Cross-protocol safety information remains visible during each stage.
    selected = [cards[x["card_id"]] for x in items if x.get("card_id") in cards]
    # Diagnosis was already generated; management stages only need relevant
    # management evidence and a bounded safety/differential reference.
    # Safety guideline CHI-026 applies to suspected Chikungunya pending dengue exclusion.
    # Include it only where this protocol is a candidate, not for every patient.
    selected = list({c.card_id: c for c in selected}.values())
    if not selected:
        return {**{x: [] for x in STAGE_TO_KEY.values()}, "missing_info": [f"No {stage} evidence retrieved."], "stage": stage}
    key = STAGE_TO_KEY[stage]
    evidence = []
    for card in selected:
        state = card_rule_applicability(card, _structured_patient_context(patient))
        if mode == "improved" and state and state.get("status") == "not_applicable":
            continue
        entry = _prompt_card(card)
        if state is not None:
            entry["applicability"] = state
        evidence.append(entry)
    if not evidence:
        return {**{x: [] for x in STAGE_TO_KEY.values()}, "missing_info": ["Evidence prerequisites not met; clinical review required."], "stage": stage}
    # Bounded prompt: never cut contraindications, condition branches or passages midway.
    # If it overflows, move complete cards into the next request rather than truncating.
    groups = []
    current = []
    for card in evidence:
        trial = current + [card]
        if len(_prompt(patient, trial).encode("utf-8")) > 22000 and current:
            groups.append(current)
            current = [card]
        else:
            current = trial
    if current:
        groups.append(current)
    responses = []
    errors = []
    allowed = {c.card_id: c for c in selected}
    for group in groups:
        prompt = _prompt(patient, group)
        if len(prompt.encode("utf-8")) > 22000:
            errors.append("A single clinical card exceeds the request budget; manual review required.")
            continue
        prompt += (f" ONLY produce items in {key}; all other recommendation arrays MUST be empty. "
                   "Retain relevant safety cautions, source conditions and missing patient inputs. "
                   "Do not claim a medication is appropriate when required data are unknown.")
        try:
            response, issues = _parse_response_partial(llm_generate(prompt))
            group_cards = {x["card_id"]: allowed[x["card_id"]] for x in group}
            for other_key in STAGE_TO_KEY.values():
                if other_key != key:
                    setattr(response, other_key, [])
            response, rejected = _validate_output(
                response, group_cards, source_medication=(mode == "improved"),
                patient_context=_structured_patient_context(patient))
            if mode == "improved" and stage == "Diagnosis":
                # A diagnosis must cite a single eligible disease protocol.
                retained = []
                for item in response.probable_diagnosis:
                    protocols = {group_cards[ref.card_id].disease_protocol
                                 for ref in item.evidence if ref.card_id in group_cards}
                    if len(protocols) == 1 and (not diseases or next(iter(protocols)) in diseases):
                        # The model may describe shock/severity as the disease.
                        # Use the verified card's disease protocol as the display
                        # label; complications can still be discussed separately.
                        protocol = next(iter(protocols))
                        # Preserve the model explanation; protocol comes from citations.
                        retained.append(item)
                    else:
                        rejected.append("Diagnosis lacked eligible single-disease evidence")
                response.probable_diagnosis = retained
            output = _dump(response)
            _attach_traceability(output, response, group_cards)
            responses.append(output)
            errors.extend(issues + rejected)
        except (LLMError, RecommendationError, ValueError) as exc:
            errors.append(str(exc))
    result = {x: [] for x in STAGE_TO_KEY.values()}
    result["missing_info"] = []
    for response in responses:
        result[key].extend(response[key])
        result["missing_info"].extend(response.get("missing_info", []))
    result["stage"] = stage
    result["evidence_card_ids"] = [x["card_id"] for x in evidence]
    if errors:
        result["notice"] = "Some generated items were rejected or unavailable; clinician review needed. " + "; ".join(errors[:4])
    if not result[key] and not result["missing_info"]:
        result["missing_info"] = ["No validated recommendation for this section; clinical assessment required."]
    return result


def _enforce_management_evidence_boundary(
    response: RecommendationResponse,
    allowed_cards: dict[str, Card],
    disease: str,
    patient_context: dict[str, Any] | None = None,
) -> tuple[RecommendationResponse, list[str]]:
    """Validate each item against its own clinical category and chosen protocol.

    Generic safety invariant; never compares disease names or clinical thresholds
    against a fixed list. No LLM-generated item survives without eligible evidence.
    """
    expected = {
        "investigations": "Investigation",
        "medication": "Medication",
        "advice": "Advice",
    }
    rejected: list[str] = []
    for section, category in expected.items():
        accepted = []
        for index, item in enumerate(getattr(response, section)):
            citations = [allowed_cards.get(ref.card_id) for ref in item.evidence]
            if not citations or any(
                card is None or card.disease_protocol != disease
                or card.clinical_category != category
                for card in citations
            ):
                rejected.append(
                    f"{section}.{index}: cited evidence does not match "
                    f"selected protocol and {category} category"
                )
            else:
                applicability = [
                    card_rule_applicability(card, patient_context or {})
                    if hasattr(card, "model_dump") or hasattr(card, "conditions_value")
                    else None
                    for card in citations
                ]
                if any(state and state.get("status") == "not_applicable"
                       for state in applicability):
                    rejected.append(
                        f"{section}.{index}: cited source conditions are contradicted "
                        "by the known patient facts"
                    )
                    continue
                uncertain = []
                for reference, card, state in zip(item.evidence, citations, applicability):
                    if state is None:
                        uncertain.append(
                            f"{reference.card_id}: source eligibility is not structured"
                        )
                    elif state.get("status") == "undetermined":
                        missing = state.get("missing_inputs") or []
                        uncertain.append(
                            f"{reference.card_id}: eligibility is unknown"
                            + (f"; not known: {', '.join(missing[:8])}" if missing else "")
                        )
                if uncertain and section in {"medication", "advice"}:
                    # An unknown prerequisite is NOT permission to reproduce the
                    # model's complete treatment regimen or discharge order.
                    # Keep a discoverable, cited conditional option instead;
                    # the full source still exists in the card corpus.
                    label = ", ".join(dict.fromkeys(getattr(card, "card_title", "cited source") for card in citations))
                    item.text = (
                        "CONDITIONAL GUIDELINE INFORMATION ONLY — clinical "
                        "eligibility not established. Source option for review: "
                        + label + ". No patient-specific dosing or discharge "
                        "instruction is authorised."
                    )
                    item.reason = (
                        "Conditional source guidance; required source conditions are unresolved. "
                        "Assess the missing inputs before deciding whether "
                        "this option is appropriate."
                    )
                    if isinstance(item, MedicationRecommendation):
                        item.eligibility = "conditional"
                        # Even source-authentic doses may refer to another age,
                        # weight or treatment branch. Do not show them as a
                        # patient regimen before eligibility is established.
                        for field in ("dose", "route", "frequency", "duration", "instructions"):
                            setattr(item, field, None)
                        item.source_details = None
                        for note in uncertain:
                            if note not in item.eligibility_notes:
                                item.eligibility_notes.append(note)
                    missing_info = getattr(response, "missing_info", None)
                    if isinstance(missing_info, list):
                        for note in uncertain:
                            if note not in missing_info:
                                missing_info.append(note)
                accepted.append(item)
        setattr(response, section, accepted)
    return response, rejected


def recommend_management(patient: dict[str, Any], *, mode: str = "improved",
                         cards: dict[str, Card], persist_dir: Path = Path("data/index"),
                         diseases: list[str] | None = None,
                         llm_generate: Callable[[str], str] = generate) -> dict[str, Any]:
    """Produce one grounded management response from category-specific cards."""
    from retrieve import retrieve_category
    if mode == "improved" and (not diseases or len(diseases) != 1):
        return {
            **{key: [] for key in STAGE_TO_KEY.values()},
            "missing_info": ["Select exactly one supported working diagnosis for management."],
            "evidence_card_ids": [],
            "stage": "Management",
            "notice": "Improved management requires one selected disease.",
        }
    chosen = []
    retrieval_trace: dict[str, list[str]] = {}
    ctx = _structured_patient_context(patient)
    derived: dict[str, Any] = {}
    derived_trace: dict[str, Any] = {}
    blocked: dict[str, str] = {}
    fact_info: dict[str, Any] = {}
    source_values: dict[str, Any] = {}
    source_options: list[dict[str, Any]] = []
    if mode == "improved" and diseases:
        from management_age import normalize_management_age
        ctx, age_info = normalize_management_age(patient, ctx)
        from derived_fields import blocked_cards, derive_classifications
        from management_facts import extract_management_facts
        from management_evidence import source_context, prerequisite_state, rule_options
        new_facts, fact_info = extract_management_facts(
            patient, cards.values(), diseases[0], ctx, llm_generate)
        ctx = {**ctx, **new_facts}
        derived, derived_trace = derive_classifications(cards.values(), diseases[0], ctx)
        ctx = {**ctx, **derived}
        ctx, source_values, resolved = source_context(cards.values(), diseases[0], ctx, derived_trace)
        source_values.update({field: {label: [] for label in entry["labels"]}
                              for field, entry in derived_trace.get("fields", {}).items()})
        blocked = blocked_cards(derived_trace)
    # Retrieve a broader pool, then use the card's *actual* source-rule
    # applicability to rank candidates. Similarity alone previously exhausted
    # the two-card budget with unrelated, eligibility-unknown medication cards.
    # This changes evidence discovery, never the rule-based eligibility gate.
    category_limits = (("Investigation", 3), ("Medication", 4), ("Advice", 3)) if mode == "improved" else (
        ("Investigation", 2), ("Medication", 2), ("Advice", 2))
    for category, count in category_limits:
        spare = sum(1 for cid in blocked if cards[cid].clinical_category == category) if blocked else 0
        pool_size = max(10, count + spare) if mode == "improved" else count + spare
        hits = retrieve_category(patient, category, diseases=diseases,
                                 top_k=pool_size, persist_dir=persist_dir, cards=cards)
        viable = []
        for hit in hits:
            card = cards.get(hit.get("card_id"))
            if mode == "improved" and card and card.disease_protocol == diseases[0]:
                gates = prerequisite_state(card, ctx, source_values)
                options = rule_options(card, ctx, gates)
                source_options.extend(options)
                if any(g["status"] == "not_met" for g in gates):
                    continue
            if (card is None or card.card_id in blocked
                    or (mode == "improved" and diseases and card.disease_protocol != diseases[0])):
                continue
            status = card_rule_applicability(card, ctx)
            if mode == "improved":
                if options and all(o["status"] == "not_applicable" for o in options):
                    continue
            elif status and status.get("status") == "not_applicable":
                continue
            # Keep retrieval ordering among candidates with equal rule support.
            # A proven applicable source rule outranks an unknown indication,
            # but unknown cards remain candidates rather than being discarded.
            rank = 0 if status and status.get("status") == "applicable" else 1
            if mode == "improved" and rank:
                from applicability import TruthValue, _evaluate_tree
                # A matching indication with unresolved safety checks ranks
                # ahead of unrelated unknown complications; it is not authorised.
                rank = 1 if any(_evaluate_tree(o["condition"], ctx) is TruthValue.TRUE
                                for o in options) else 2
            viable.append((rank, hit))
        viable.sort(key=lambda entry: entry[0])
        hits = [hit for _, hit in viable[:count]]
        retrieval_trace[category] = [
            str(hit.get("card_id")) for hit in hits if hit.get("card_id")
        ]
        chosen.extend(
            cards[hit["card_id"]] for hit in hits
            if hit.get("card_id") in cards
            and (mode != "improved" or cards[hit["card_id"]].disease_protocol == diseases[0])
        )
    chosen = list({card.card_id: card for card in chosen}.values())
    extra_ids: list[str] = []
    for field_name, entry in derived_trace.get("fields", {}).items():
        if entry.get("status") not in ("decided", "supplied"):
            continue
        # A settled classification selects its cards by rule, not by similarity.
        from derived_fields import consuming_fields, classification_rules
        rules = classification_rules(cards.values(), diseases[0])
        group = consuming_fields(cards.values(), diseases[0], {l for _, l, _ in rules})
        for card_id in sorted(group.get(field_name, ())):
            card = cards.get(card_id)
            verdict = card_rule_applicability(card, ctx) if card else None
            if (card and card_id not in {c.card_id for c in chosen}
                    and verdict and verdict.get("status") == "applicable"
                    and len(extra_ids) < 3):
                chosen.append(card)
                extra_ids.append(card_id)
    result = {key: [] for key in STAGE_TO_KEY.values()}
    result.update({
        "missing_info": [], "evidence_card_ids": [], "stage": "Management",
        "management_diagnostics": {
            "retrieved_card_ids_by_category": retrieval_trace,
            "chosen_card_ids": [card.card_id for card in chosen],
            "not_applicable_card_ids": [],
            "applicability_by_card": {},
            "prompt_card_ids": [],
            "schema_rejections": [],
            "validation_rejections": [],
            "generated_item_counts": {},
            "derived_classification": {"fields": derived, **derived_trace},
            "added_by_rule": extra_ids,
        },
    })
    result["management_diagnostics"]["blocked_card_ids"] = sorted(blocked)
    if mode == "improved":
        result["management_diagnostics"]["age_normalization"] = age_info
        if age_info.get("message"):
            result["missing_info"].append(age_info["message"])
        result["management_diagnostics"]["management_fact_extraction"] = fact_info
        result["management_diagnostics"]["resolved_source_context"] = resolved
        result["working_diagnosis"] = diseases[0]
    for field_name, entry in derived_trace.get("fields", {}).items():
        if entry.get("status") == "undecided":
            cards_txt = ", ".join(
                f"{cid} ({cards[cid].card_title})" for cid in entry["blocked_cards"])
            needed: list[str] = []
            for label in entry.get("unknown_labels", []):
                assess = cards.get(derived_trace["labels"][label]["card_id"])
                verdict = card_rule_applicability(assess, ctx) if assess else None
                needed += [m for m in (verdict or {}).get("missing_inputs", []) if m not in needed]
            result["missing_info"].append(
                f"Not shown until '{field_name}' is settled by the source rules: {cards_txt}. "
                + ("Needed: " + ", ".join(needed[:10]) + "." if needed else ""))
    for conflict in derived_trace.get("conflicts", []):
        result["missing_info"].append("Source rules conflict - " + conflict
                                      + "; no plan chosen, clinical review needed.")
    prepared = []
    for card in chosen:
        applicability = card_rule_applicability(card, ctx)
        result["management_diagnostics"]["applicability_by_card"][card.card_id] = (
            applicability.get("status", "unknown") if applicability else "no_structured_rule"
        )
        if mode == "improved":
            options = rule_options(card, ctx, prerequisite_state(card, ctx, source_values))
            if options and all(o["status"] == "not_applicable" for o in options):
                result["management_diagnostics"]["not_applicable_card_ids"].append(card.card_id)
                continue
        entry = _prompt_card(card)
        # Retain all source rules and restrictions; one quoted passage per card
        # is sufficient for a short prompt. Original evidence remains in corpus.
        if entry.get("exact_supporting_evidence_passages"):
            entry["exact_supporting_evidence_passages"] = entry["exact_supporting_evidence_passages"][:1]
        if applicability:
            entry["applicability"] = {"status": applicability.get("status"),
                                       "missing_inputs": applicability.get("missing_inputs", [])[:6]}
        if mode == "improved":
            options = rule_options(card, ctx, prerequisite_state(card, ctx, source_values))
            source_options.extend(options)
            entry["patient_rule_options"] = options
        prepared.append(entry)
    selected = []
    if mode == "improved":
        from management_evidence import pack_evidence, review_summary
        facts_for_prompt = {key: value for key, value in ctx.items()
                            if not key.startswith("_") and key not in {
                                *QUERY_FIELDS, "clinical_findings", "measurements", "medicines", "allergies"}}
        def generation_prompt(case, entries):
            references = {key: [
                {"card_id": e["card_id"], "pdf_pages": e.get("pdf_page_numbers", [])}
                for e in entries if e["clinical_category"] == category]
                for key, category in (("investigations", "Investigation"),
                                      ("medication", "Medication"), ("advice", "Advice"))}
            return _prompt(case, entries) + (
                " IMPROVED MANAGEMENT ONLY. Working diagnosis is probable, not confirmed: "
                + json.dumps(diseases[0]) + ". Verified or supplied facts and source-derived context: "
                + json.dumps(facts_for_prompt, default=str) + ". "
                "Use patient_rule_options at instruction level. Supported rules may inform "
                "recommendations; do not import an unrelated instruction from the same card. "
                "AND requires all clauses; OR requires one supported alternative. Missing "
                "facts are unknown, never negative. Conditional rules are review options, "
                "not instructions to act now. State their exact missing prerequisites. "
                "Never supply a patient dose from a table whose row or safety checks are "
                "unresolved. Do not guess age-group boundaries or treatment setting. "
                "Give relevant advice as well as investigations and treatment when evidence "
                "supports it. Do not prescribe unrelated complications merely because their "
                "cards share the protocol. Set probable_diagnosis to []. "
                "Write concise items, group related information, and avoid repeating the "
                "same instruction or generic disclaimer across items. Missing conditions "
                "are questions to clarify, not patient findings. "
                "Each citation must match its section's permitted references and page: "
                + json.dumps(references))
        selected, omitted = pack_evidence(prepared, patient, generation_prompt, "")
        unique_options = {(o["card_id"], o["rule_id"], json.dumps(o["condition"], sort_keys=True)): o
                          for o in source_options}
        result["source_review_options"] = review_summary(unique_options.values())
        result["management_diagnostics"]["source_rule_outcomes"] = [
            {k: option[k] for k in ("card_id", "rule_id", "status", "missing_inputs", "unsupported_rule")}
            for option in unique_options.values()]
        result["management_diagnostics"]["omitted_for_prompt_budget"] = omitted
        result["management_diagnostics"]["prompt_categories"] = sorted({
            e["clinical_category"] for e in selected})
    else:
        for entry in prepared:
            if len(_prompt(patient, selected + [entry]).encode("utf-8")) <= 19000:
                selected.append(entry)
    result["evidence_card_ids"] = [x["card_id"] for x in selected]
    result["management_diagnostics"]["prompt_card_ids"] = result["evidence_card_ids"]
    if not selected:
        result["notice"] = "No management evidence fits the request; clinical review needed."
        result["management_diagnostics"]["generated_item_counts"] = {
            "investigations": 0, "medication": 0, "advice": 0
        }
        return result
    # The permitted evidence set is source-derived, not disease-name hardcoded.
    evidence_by_section = {
        section: [
            {"card_id": entry["card_id"], "pdf_pages": entry.get("pdf_page_numbers", [])}
            for entry in selected if entry.get("clinical_category") == category
        ]
        for section, category in (
            ("investigations", "Investigation"),
            ("medication", "Medication"),
            ("advice", "Advice"),
        )
    }
    prompt = _prompt(patient, selected) + (
        " This is management for a PROBABLE, unconfirmed diagnosis. "
        " Only output investigations citing Investigation cards, medication citing "
        "Medication cards, and advice citing Advice cards. EACH citation must "
        "belong to the EXACT corresponding section's permitted card IDs AND its "
        "provided PDF page. Never use Advice or Diagnosis cards as investigation "
        "evidence, even when their passages mention tests. If no eligible card "
        "exists, use an empty array and note the evidence gap in missing_info. "
        " Every medication item MUST contain text, reason and evidence, just "
        "like investigations and advice. Optional name, dose, route, frequency, "
        "duration or instructions can only be specified when directly supported. "
        " Never prescribe when prerequisites are unknown. "
        " Do not assume a particular rehydration plan or severity classification "
        "without verifying its own source rules and patient evidence. "
        " Set probable_diagnosis to []. "
        " PERMITTED SECTION-SPECIFIC REFERENCES: "
        + json.dumps(evidence_by_section, ensure_ascii=False)
    )
    allowed = {x["card_id"]: cards[x["card_id"]] for x in selected}
    if mode == "improved":
        prompt = generation_prompt(patient, selected)
        result["management_diagnostics"]["prompt_bytes"] = len(prompt.encode("utf-8"))
    try:
        raw_response = llm_generate(prompt)
        response, errors = _parse_response_partial(raw_response)
        result["management_diagnostics"]["schema_rejections"] = errors[:20]
        response.probable_diagnosis = []
        validated, rejected = _validate_output(
            response, allowed, source_medication=(mode == "improved"),
            patient_context=ctx)
        if mode == "improved":
            validated, boundary_rejections = _enforce_management_evidence_boundary(
                validated, allowed, diseases[0], patient_context=ctx
            )
            rejected.extend(boundary_rejections)
        result["management_diagnostics"]["validation_rejections"] = rejected[:20]
        output = _dump(validated)
        _attach_traceability(output, validated, allowed)
        if mode == "improved":
            # Keep the established response arrays compatible, but mark review
            # options explicitly so the UI cannot present them as actions now.
            conditional_ids = {o["card_id"] for o in source_options
                               if o["status"] != "supported"}
            for section in ("investigations", "medication", "advice"):
                for item in output.get(section, []):
                    refs = {r["card_id"] for r in item.get("evidence", [])}
                    conditional = bool(refs & conditional_ids) or (
                        section in {"medication", "advice"}
                        and item.get("text", "").startswith("CONDITIONAL GUIDELINE"))
                    item["review_required"] = conditional
                    if conditional and section == "medication":
                        for field in ("dose", "route", "frequency", "duration", "instructions", "source_details"):
                            item[field] = None
                        item["eligibility"] = "conditional"
        own_missing = list(result["missing_info"])
        result.update(output)
        result["missing_info"] = own_missing + [
            m for m in result.get("missing_info", []) if m not in own_missing]
        result["management_diagnostics"]["generated_item_counts"] = {
            "investigations": len(result.get("investigations", [])),
            "medication": len(result.get("medication", [])),
            "advice": len(result.get("advice", [])),
        }
        if errors or rejected:
            result["notice"] = "Rejected unsupported items: " + "; ".join((errors + rejected)[:4])
    except (LLMError, RecommendationError, ValueError) as exc:
        result["notice"] = f"Management generation unavailable: {exc}"
        result["management_diagnostics"]["error_type"] = type(exc).__name__
        result["management_diagnostics"]["error_message"] = str(exc)[:240]
    return result
