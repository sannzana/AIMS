"""Evidence-grounded recommendation generation for baseline and improved modes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError

from applicability import check_applicability
from build_index import load_cards
from llm import LLMError, generate
from retrieve import retrieve, QUERY_FIELDS
from schema import Card, Conditions

EXCLUDE_FROM_PROMPT = ("image", "image_related_info")


class EvidenceReference(BaseModel):
    card_id: str
    pdf_page: int


class RecommendationItem(BaseModel):
    text: str
    reason: str
    evidence: list[EvidenceReference] = Field(default_factory=list)


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


def _dump(model: BaseModel) -> dict[str, Any]:
    return model.model_dump() if hasattr(model, "model_dump") else model.dict()


def _parse_response(raw: str) -> RecommendationResponse:
    try:
        payload = json.loads(raw)
        if hasattr(RecommendationResponse, "model_validate"):
            return RecommendationResponse.model_validate(payload)
        return RecommendationResponse.parse_obj(payload)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        raise RecommendationError("LLM output was not valid recommendation JSON") from exc


def _card_pages(card: Card) -> set[int]:
    return set(card.pdf_page_numbers) | {
        item.pdf_page for item in card.exact_supporting_evidence_passages
    }


def _validate_output(response: RecommendationResponse, cards: dict[str, Card]) -> None:
    items = (
        response.probable_diagnosis
        + response.investigations
        + response.medication
        + response.advice
    )
    if not items and not response.missing_info:
        raise RecommendationError(
            "Recommendation JSON must contain recommendations or missing_info"
        )
    for item in items:
        if not item.evidence:
            raise RecommendationError(
                "Every recommendation must contain at least one evidence reference"
            )
        for reference in item.evidence:
            card = cards.get(reference.card_id)
            if card is None:
                raise RecommendationError(
                    f"Unsupported evidence card_id: {reference.card_id}"
                )
            if reference.pdf_page not in _card_pages(card):
                raise RecommendationError(
                    f"Unsupported evidence page {reference.pdf_page} "
                    f"for card {reference.card_id}"
                )
        if isinstance(item, MedicationRecommendation):
            supplied = {
                name: getattr(item, name)
                for name in ("name", "dose", "route", "frequency", "duration", "instructions")
                if getattr(item, name) is not None
            }
            sources = [
                cards[reference.card_id]
                for reference in item.evidence
            ]
            for name, value in supplied.items():
                supported = False
                for card in sources:
                    medication = card.medication
                    if medication is not None and getattr(medication, name) == value:
                        supported = True
                    details = card.model_dump().get("medication_details")
                    if isinstance(details, dict) and details.get(name) == value:
                        supported = True
                if not supported:
                    raise RecommendationError(
                        f"Medication field {name} is not supported by the cited card"
                    )


def _prompt(patient: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    patient_context = {
        field: patient.get(field)
        for field in (*QUERY_FIELDS, "medicines", "allergies")
    }
    return (
        "Return only JSON matching the requested recommendation schema. "
        "Treat diagnoses as probable, never confirmed. Use only supplied evidence; "
        "never invent medication fields or doses. Every recommendation needs a "
        "source-supported reason and evidence references with card_id and pdf_page. "
        "List missing clinical information honestly. "
        f"Patient context: {json.dumps(patient_context, sort_keys=True)}. "
        f"Evidence: {json.dumps(evidence, default=str, sort_keys=True)}. "
        "Required top-level keys: probable_diagnosis, investigations, medication, "
        "advice, missing_info."
    )


def _prompt_card(card: Card) -> dict[str, Any]:
    raw = card.model_dump()
    return {
        key: value
        for key, value in raw.items()
        if key not in EXCLUDE_FROM_PROMPT
        and value is not None
        and value != ""
        and value != []
        and value != {}
    }


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
    retrieved = retrieve_fn(patient=patient, top_k=top_k, persist_dir=persist_dir)
    if not retrieved:
        return _dump(RecommendationResponse(missing_info=["No evidence cards were retrieved."]))

    if cards is None:
        path = cards_path or (
            Path("data/sample_cards.jsonl")
            if persist_dir.name == "dev_index"
            else Path("data/cards.jsonl")
        )
        cards = {card.card_id: card for card in load_cards(path)}
    evidence: list[dict[str, Any]] = []
    for result in retrieved:
        card = cards.get(result.get("card_id"))
        if card is None:
            raise RecommendationError(f"Retrieved card is missing from corpus: {result.get('card_id')}")
        entry = _prompt_card(card)
        entry["similarity_score"] = result.get("similarity_score")
        if mode == "improved" and (card.structured_rules or card.conditions is not None):
            applicability_card = (
                card.model_copy(
                    update={
                        "conditions": Conditions.model_validate(card.structured_rules)
                    }
                )
                if card.structured_rules
                else card
            )
            applicability = check_applicability(applicability_card, patient)
            if applicability["status"] == "not_applicable":
                continue
            entry["applicability"] = applicability
        evidence.append(entry)
    if not evidence:
        return _dump(RecommendationResponse(
            missing_info=["All retrieved cards were not applicable."]
        ))
    try:
        response = _parse_response(llm_generate(_prompt(patient, evidence)))
    except LLMError as exc:
        raise RecommendationError(str(exc)) from exc
    _validate_output(response, cards)
    output = _dump(response)
    output["mode"] = mode
    output["dev_only"] = persist_dir.name == "dev_index"
    if output["dev_only"]:
        output["notice"] = "DEV ONLY: software testing output; not for patient care."
    return output
