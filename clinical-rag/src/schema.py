"""Flexible Schema v1.1 models for evidence cards."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


ConditionOperator = Literal["==", "!=", "<", "<=", ">", ">=", "has", "in"]
CLINICAL_CATEGORY_VALUES = (
    "Diagnosis",
    "Investigation",
    "Medication",
    "Advice",
    "Reference-only / descriptive information",
    "Assessment",
)
ClinicalCategory = Literal[
    "Diagnosis",
    "Investigation",
    "Medication",
    "Advice",
    "Reference-only / descriptive information",
    "Assessment",
]


class Condition(BaseModel):
    field: str
    op: ConditionOperator
    value: Any = None

    class Config:
        extra = "forbid"


class Conditions(BaseModel):
    all: list[Condition] = Field(default_factory=list)
    any: list[Condition] = Field(default_factory=list)
    exceptions: list[Condition] = Field(default_factory=list)
    required_inputs: list[str] = Field(default_factory=list)

    class Config:
        extra = "forbid"


class Evidence(BaseModel):
    pdf_page: int
    passage: str

    class Config:
        extra = "allow"


class Medication(BaseModel):
    name: str | None = None
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    instructions: str | None = None

    class Config:
        extra = "allow"


class Source(BaseModel):
    title: str
    version: str | None = None
    pdf_pages: list[int] = Field(default_factory=list)

    class Config:
        extra = "allow"


class Card(BaseModel):
    schema_version: Literal["1.1"] = "1.1"
    card_id: str = Field(...)
    disease_protocol: str = Field(...)
    clinical_category: ClinicalCategory = Field(...)
    source_pdf_title: str = Field(...)
    source_version_edition: str | None = Field(...)
    pdf_page_numbers: list[int] = Field(...)
    exact_supporting_evidence_passages: list[Evidence] = Field(...)
    verified: bool = Field(...)

    structured_rules: dict[str, Any] | None = None

    # Legacy runtime fields remain optional so old tests and callers can be
    # loaded while all new cards are normalized to the flat core fields.
    disease: str | None = None
    category: str | None = None
    conditions: Conditions | None = None
    action: str | None = None
    medication: Medication | None = None
    exclusions_or_missing_info: str | None = None
    source: Source | None = None
    evidence: list[Evidence] | None = None

    class Config:
        extra = "allow"

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_input(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        legacy = any(
            key in payload
            for key in ("disease", "category", "source", "evidence", "action")
        )
        if not legacy:
            return payload
        payload.setdefault("disease_protocol", payload.get("disease"))
        if "clinical_category" not in payload:
            legacy_category = payload.get("category")
            payload["clinical_category"] = {
                "diagnosis": "Diagnosis",
                "investigation": "Investigation",
                "medication": "Medication",
                "advice": "Advice",
            }.get(legacy_category, legacy_category)
        source = payload.get("source") or {}
        if "source_pdf_title" not in payload:
            payload["source_pdf_title"] = source.get("title")
        if "source_version_edition" not in payload:
            payload["source_version_edition"] = source.get("version")
        if "pdf_page_numbers" not in payload:
            payload["pdf_page_numbers"] = source.get("pdf_pages", [])
        if "exact_supporting_evidence_passages" not in payload:
            payload["exact_supporting_evidence_passages"] = payload.get(
                "evidence", []
            )
        if "verified" not in payload:
            payload["verified"] = False
        return payload

    @property
    def disease_value(self) -> str:
        return self.disease_protocol

    @property
    def category_value(self) -> str:
        return self.clinical_category

    @property
    def source_value(self) -> Source:
        return self.source or Source(
            title=self.source_pdf_title,
            version=self.source_version_edition,
            pdf_pages=self.pdf_page_numbers,
        )

    @property
    def evidence_value(self) -> list[Evidence]:
        return self.evidence or self.exact_supporting_evidence_passages

    @property
    def conditions_value(self) -> Conditions | None:
        if self.structured_rules:
            return Conditions.model_validate(self.structured_rules)
        return self.conditions
