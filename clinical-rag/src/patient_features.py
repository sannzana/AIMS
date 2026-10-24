"""Patient data access without disease-specific rules or thresholds."""
from __future__ import annotations
from typing import Any


def extract_patient_features(patient: dict[str, Any]) -> dict[str, Any]:
    """Return facts produced by source-schema triage; never infer a clinical fact."""
    facts = patient.get("_structured_facts")
    return dict(facts) if isinstance(facts, dict) else {}


def diagnostic_query(patient: dict[str, Any]) -> str:
    """Preserve observations for semantic retrieval without lexical diagnosis."""
    return ". ".join(str(patient[k]).strip() for k in ("symptoms", "clinical_findings", "history", "measurements") if patient.get(k) not in (None, "", [], {}))
