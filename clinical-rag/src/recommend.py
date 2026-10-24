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
    response: RecommendationResponse, cards: dict[str, Card]
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
            if reason is None and isinstance(item, MedicationRecommendation):
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
    response, safety_rejections = _validate_output(response, cards)
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
    """Promote explicit observed features into rule fields; never infer absent signs."""
    import re
    enriched = dict(patient)
    duration = str(patient.get("duration", ""))
    match = re.fullmatch(r"\s*(\d+)\s*(?:day|days|d)\s*", duration, re.I)
    if match:
        days = int(match.group(1))
        enriched.setdefault("days_since_illness_onset", days)
        enriched.setdefault("illness_day", days)
        # Fever onset is not necessarily illness onset. Only map when explicitly
        # described as fever duration, or leave separate fever-onset field unknown.
    symptoms = str(patient.get("symptoms", "")).lower()
    if "fever" in symptoms and "no fever" not in symptoms and "without fever" not in symptoms:
        enriched.setdefault("patient_has_fever", True)
        enriched.setdefault("fever_present", True)
    return enriched


STAGE_TO_KEY = {
    "Diagnosis": "probable_diagnosis", "Investigations": "investigations",
    "Treatment": "medication", "Advice": "advice",
}


def recommend_section(patient: dict[str, Any], stage: str, *, mode: str = "improved",
                      cards: dict[str, Card], persist_dir: Path = Path("data/index"),
                      diseases: list[str] | None = None,
                      llm_generate: Callable[[str], str] = generate) -> dict[str, Any]:
    """One independently validated clinical section per bounded LLM request."""
    from retrieve import retrieve_category
    if stage not in STAGE_TO_KEY:
        raise ValueError(f"Unsupported clinical stage: {stage}")
    category = "Medication" if stage == "Treatment" else ("Diagnosis" if stage == "Diagnosis" else stage[:-1] if stage == "Investigations" else stage)
    requested = 4 if stage == "Diagnosis" else (4 if stage == "Investigations" else 3)
    items = retrieve_category(patient, category, diseases=diseases,
                              top_k=requested, persist_dir=persist_dir, cards=cards)
    # Cross-protocol safety information remains visible during each stage.
    selected = [cards[x["card_id"]] for x in items if x.get("card_id") in cards]
    # Diagnosis was already generated; management stages only need relevant
    # management evidence and a bounded safety/differential reference.
    # Safety guideline CHI-026 applies to suspected Chikungunya pending dengue exclusion.
    # Include it only where this protocol is a candidate, not for every patient.
    if diseases and "Chikungunya" in diseases and "CHI-026" in cards:
        selected.append(cards["CHI-026"])
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
            response, rejected = _validate_output(response, group_cards)
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


def recommend_management(patient: dict[str, Any], *, mode: str = "improved",
                         cards: dict[str, Card], persist_dir: Path = Path("data/index"),
                         diseases: list[str] | None = None,
                         llm_generate: Callable[[str], str] = generate) -> dict[str, Any]:
    """Produce one grounded management response from category-specific cards."""
    from retrieve import retrieve_category
    chosen = []
    for category, count in (("Investigation", 2), ("Medication", 2), ("Advice", 2)):
        hits = retrieve_category(patient, category, diseases=diseases,
                                 top_k=count, persist_dir=persist_dir, cards=cards)
        chosen.extend(cards[hit["card_id"]] for hit in hits if hit.get("card_id") in cards)
    if diseases and "Chikungunya" in diseases and "CHI-026" in cards:
        chosen.append(cards["CHI-026"])
    chosen = list({card.card_id: card for card in chosen}.values())
    result = {key: [] for key in STAGE_TO_KEY.values()}
    result.update({"missing_info": [], "evidence_card_ids": [], "stage": "Management"})
    prepared = []
    ctx = _structured_patient_context(patient)
    for card in chosen:
        applicability = card_rule_applicability(card, ctx)
        if mode == "improved" and applicability and applicability.get("status") == "not_applicable":
            continue
        entry = _prompt_card(card)
        # Retain all source rules and restrictions; one quoted passage per card
        # is sufficient for a short prompt. Original evidence remains in corpus.
        if entry.get("exact_supporting_evidence_passages"):
            entry["exact_supporting_evidence_passages"] = entry["exact_supporting_evidence_passages"][:1]
        if applicability:
            entry["applicability"] = {"status": applicability.get("status"),
                                       "missing_inputs": applicability.get("missing_inputs", [])[:6]}
        prepared.append(entry)
    selected = []
    for entry in prepared:
        if len(_prompt(patient, selected + [entry]).encode("utf-8")) <= 19000:
            selected.append(entry)
    result["evidence_card_ids"] = [x["card_id"] for x in selected]
    if not selected:
        result["notice"] = "No management evidence fits the request; clinical review needed."
        return result
    prompt = _prompt(patient, selected) + (
        " This is a management plan for a PROBABLE, unconfirmed diagnosis. "
        "Set probable_diagnosis to []. Provide source-supported investigations "
        "with purpose/timing, medication options with restrictions, and advice "
        "in ONE JSON response. Keep all necessary safety conditions, never "
        "invent doses, and do not prescribe when prerequisites are unknown."
    )
    allowed = {x["card_id"]: cards[x["card_id"]] for x in selected}
    try:
        response, errors = _parse_response_partial(llm_generate(prompt))
        response.probable_diagnosis = []
        validated, rejected = _validate_output(response, allowed)
        output = _dump(validated)
        _attach_traceability(output, validated, allowed)
        result.update(output)
        if errors or rejected:
            result["notice"] = "Rejected unsupported items: " + "; ".join((errors + rejected)[:4])
    except (LLMError, RecommendationError, ValueError) as exc:
        result["notice"] = f"Management generation unavailable: {exc}"
    return result
