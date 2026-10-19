"""Pydantic models for protocol evidence cards."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


ConditionOperator = Literal["==", "!=", "<", "<=", ">", ">=", "has", "in"]
CardCategory = Literal["diagnosis", "investigation", "medication", "advice"]


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


class Medication(BaseModel):
    name: str | None = None
    dose: str | None = None
    route: str | None = None
    frequency: str | None = None
    duration: str | None = None
    instructions: str | None = None

    class Config:
        extra = "forbid"


class Source(BaseModel):
    title: str
    version: str | None = None
    pdf_pages: list[int] = Field(default_factory=list)

    class Config:
        extra = "forbid"


class Evidence(BaseModel):
    pdf_page: int
    passage: str

    class Config:
        extra = "forbid"


class Card(BaseModel):
    schema_version: Literal["1.1"] = "1.1"
    card_id: str
    disease: str
    category: CardCategory
    conditions: Conditions = Field(default_factory=Conditions)
    action: str | None = None
    medication: Medication | None = None
    exclusions_or_missing_info: str | None = None
    source: Source
    evidence: list[Evidence] = Field(default_factory=list)
    verified: bool = False

    class Config:
        extra = "forbid"
