"""Retrieve a small, category-aware evidence set from a local Chroma index."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

from build_index import load_cards


QUERY_FIELDS = (
    "symptoms",
    "duration",
    "age",
    "sex",
    "history",
    "clinical_findings",
    "measurements",
)
CATEGORIES = ("Diagnosis", "Investigation", "Medication", "Advice")
STAGE1_MAX_CARDS = 8
MAX_CANDIDATE_DISEASES = 3
CATEGORY_INTENTS = {
    "Diagnosis": "diagnostic differential and probable diagnosis",
    "Investigation": "tests investigations laboratory examination and monitoring",
    "Medication": "medication treatment dose route and drug safety",
    "Advice": "management advice follow-up warning signs and precautions",
}
SAFETY_INTENT = (
    "contraindications warnings exceptions precautions danger signs avoid medicines "
    "treatment safety and differential diagnosis"
)


def _embedding_values(values: Any) -> Any:
    return values.tolist() if hasattr(values, "tolist") else values


def patient_query(patient: dict[str, Any]) -> str:
    """Build a query from patient data; disease is intentionally ignored."""
    parts = []
    for field in QUERY_FIELDS:
        value = patient.get(field)
        if value not in (None, "", [], {}):
            parts.append(f"{field}: {value}")
    return "; ".join(parts)


def normalize_disease_name(value: Any) -> str:
    """Normalize labels for grouping without changing the displayed source label."""
    return " ".join(str(value or "").casefold().split())


def _readable(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, dict):
        return "; ".join(
            f"{key}: {text}"
            for key, item in sorted(value.items())
            if (text := _readable(item))
        )
    if isinstance(value, (list, tuple, set)):
        return "; ".join(text for item in value if (text := _readable(item)))
    return str(value)


def _card_safety(card: Any) -> tuple[bool, str]:
    data = card.model_dump() if hasattr(card, "model_dump") else dict(card)
    fields = (
        "exceptions_contraindications",
        "exclusions_or_missing_information",
        "recommended_action",
        "additional_information",
    )
    text = " ".join(_readable(data.get(field)) for field in fields)
    markers = (
        "contraindicat",
        "warning",
        "precaution",
        "avoid",
        "not use",
        "risk",
        "hemorrhage",
        "bleeding",
        "until dengue has been ruled out",
        "exception",
        "urgent",
        "referral",
    )
    return any(marker in text.lower() for marker in markers), text


def _default_cards(persist_dir: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / ("sample_cards.jsonl" if persist_dir.name == "dev_index" else "cards.jsonl")
    try:
        return {card.card_id: card for card in load_cards(path)}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _query(
    collection: Any,
    embedder: Any,
    text: str,
    n_results: int,
    *,
    category: str | None = None,
    disease: str | None = None,
) -> list[dict[str, Any]]:
    embedding = _embedding_values(embedder.encode([text]))
    kwargs = {
        "query_embeddings": embedding,
        "n_results": n_results,
        "include": ["metadatas", "distances"],
    }
    if category is not None and disease is not None:
        kwargs["where"] = {
            "$and": [
                {"clinical_category": category},
                {"disease_protocol": disease},
            ]
        }
    elif category is not None:
        kwargs["where"] = {"clinical_category": category}
    elif disease is not None:
        kwargs["where"] = {"disease_protocol": disease}
    try:
        result = collection.query(**kwargs)
    except (KeyError, TypeError, ValueError):
        # Older/local test collections may not support Chroma's where argument.
        kwargs.pop("where", None)
        result = collection.query(**kwargs)
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    candidates = []
    for metadata, distance in zip(metadatas, distances):
        if category is not None and metadata.get("clinical_category") != category:
            continue
        if disease is not None and metadata.get("disease_protocol") != disease:
            continue
        candidates.append(
            {
                "card_id": metadata.get("card_id"),
                "similarity_score": 1.0 - float(distance),
                "source": {
                    "title": metadata.get("source_title"),
                    "version": metadata.get("source_version") or None,
                    "pdf_pages": json.loads(metadata.get("source_pdf_pages", "[]")),
                },
                "metadata": metadata,
            }
        )
    return candidates


def _card_search_text(card: Any) -> str:
    if card is None:
        return ""
    data = card.model_dump() if hasattr(card, "model_dump") else dict(card)
    keys = (
        "disease_protocol",
        "clinical_classification_presentation",
        "recommended_action",
        "conditions_decision_logic",
        "exceptions_contraindications",
        "exact_supporting_evidence_passages",
    )
    return " ".join(_readable(data.get(key)) for key in keys).lower()


def _tokens(value: Any) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{2,}", str(value).casefold())
        if token not in {"the", "and", "with", "from", "for", "that", "this"}
    }


def _diagnostic_rank(result: dict[str, Any], patient_text: str, cards: Mapping[str, Any]) -> tuple[int, float]:
    overlap = len(_tokens(patient_text) & _tokens(_card_search_text(cards.get(result.get("card_id")))))
    return overlap, result["similarity_score"]


def _stage1_candidates(
    diagnosis_results: list[dict[str, Any]],
    cards: Mapping[str, Any],
) -> tuple[list[str], dict[str, str]]:
    """Choose diseases from ranked diagnosis evidence, never as confirmation."""
    if not any(result["metadata"].get("disease_protocol") for result in diagnosis_results):
        return [], {}
    # Only single-protocol labels are eligible as disease candidates.
    # A composite differential label is evidence, not a distinct disease.
    known_diseases = sorted({
        str(card.disease_protocol).strip()
        for card in cards.values()
        if getattr(card, "disease_protocol", None)
        and "," not in str(card.disease_protocol)
        and " and " not in str(card.disease_protocol).casefold()
    }, key=lambda value: (-len(value), normalize_disease_name(value)))
    grouped: dict[str, dict[str, Any]] = {}
    for result in diagnosis_results[:STAGE1_MAX_CARDS]:
        disease = result["metadata"].get("disease_protocol")
        key = normalize_disease_name(disease)
        if disease and key and disease in known_diseases:
            item = grouped.setdefault(
                key,
                {
                    "display": str(disease).strip(),
                    "count": 0,
                    "max_score": float("-inf"),
                    "max_overlap": 0,
                    "overlap_count": 0,
                    "categories": set(),
                },
            )
            item["count"] += 1
            item["max_score"] = max(item["max_score"], result["similarity_score"])
            overlap = int(result.get("lexical_overlap", 0))
            item["max_overlap"] = max(item["max_overlap"], overlap)
            item["overlap_count"] += overlap
            item["categories"].add(result["metadata"].get("clinical_category"))

    for result in diagnosis_results[:STAGE1_MAX_CARDS]:
        text = _card_search_text(cards.get(result.get("card_id")))
        if not any(
            marker in text
            for marker in ("differential", "compared", "alternative", "until dengue")
        ):
            continue
        for disease in known_diseases:
            key = normalize_disease_name(disease)
            if key and key in text and key not in grouped:
                grouped[key] = {
                    "display": disease,
                    "count": 1,
                    "max_score": result["similarity_score"],
                    "max_overlap": int(result.get("lexical_overlap", 0)),
                    "overlap_count": int(result.get("lexical_overlap", 0)),
                    "categories": {"Diagnosis"},
                    "alternative": True,
                }
    ranked = sorted(
        grouped.values(),
        key=lambda item: (
            bool(item.get("alternative", False)),
            -item["max_overlap"],
            -item["overlap_count"],
            -item["count"],
            -item["max_score"],
            normalize_disease_name(item["display"]),
                )
    )
    selected = ranked[:MAX_CANDIDATE_DISEASES]
    diseases = [item["display"] for item in selected]
    reasons = {
        item["display"]: (
            "Alternative disease named in Stage 1 differential/safety evidence"
            if item.get("alternative")
            else "Stage 1 diagnosis/differential evidence"
        )
        for item in selected
    }
    return diseases, reasons


def _coverage_summary(
    selected: list[dict[str, Any]],
    candidate_diseases: list[str],
    available_categories: set[str],
) -> dict[str, Any]:
    present_categories = {
        item["metadata"].get("clinical_category")
        for item in selected
    }
    required = ["Diagnosis", "Investigation", "Medication", "Advice"]
    missing = [category for category in required if category not in present_categories]
    return {
        "candidate_diseases": candidate_diseases,
        "available_categories": sorted(available_categories),
        "selected_categories": sorted(present_categories - {None}),
        "missing_categories": missing,
        "complete": not missing,
        "incomplete_reason": (
            "selection budget"
            if any(category in available_categories for category in missing)
            else "no disease-filtered evidence available"
        ) if missing else None,
    }


def retrieve(
    patient: dict[str, Any],
    top_k: int = 5,
    *,
    persist_dir: Path = Path("data/index"),
    model_name: str = "all-MiniLM-L6-v2",
    embedder: Any = None,
    client: Any = None,
    collection_name: str | None = None,
    cards: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Select evidence in two stages: diseases, then purpose-specific cards.

    Similarity scores rank evidence only; they are not clinical probabilities.
    """
    if top_k <= 0:
        return []
    if collection_name is None:
        collection_name = "clinical_cards_dev" if persist_dir.name == "dev_index" else "clinical_cards"
    if embedder is None:
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(model_name)
    if client is None:
        import chromadb

        client = chromadb.PersistentClient(path=str(persist_dir))
    collection = client.get_collection(collection_name)
    cards_by_id = dict(cards) if cards is not None else _default_cards(persist_dir)
    base_query = patient_query(patient) or "clinical protocol evidence"
    pool_size = max(top_k * 3, STAGE1_MAX_CARDS)
    stage1 = _query(
        collection,
        embedder,
        f"{base_query}; diagnostic differential and alternative diseases",
        pool_size,
        category="Diagnosis",
    )
    stage1.sort(
        key=lambda result: _diagnostic_rank(result, base_query, cards_by_id),
        reverse=True,
    )
    for result in stage1:
        result["lexical_overlap"] = _diagnostic_rank(
            result, base_query, cards_by_id
        )[0]
    candidate_diseases, disease_reasons = _stage1_candidates(stage1, cards_by_id)
    if not candidate_diseases:
        # Compatibility path for older indexes/test doubles without disease
        # metadata. It does not make a disease claim.
        lanes = [
            (
                "fallback category coverage",
                _query(
                    collection,
                    embedder,
                    f"{base_query}; clinical protocol evidence",
                    pool_size,
                ),
            )
        ]
        candidate_diseases = []
        disease_reasons = {}
    else:
        lanes = []

    for disease in candidate_diseases:
        for category in CATEGORIES:
            lanes.append(
                (
                    f"{disease} {category.lower()} evidence",
                    _query(
                        collection,
                        embedder,
                        f"{base_query}; {disease}; {CATEGORY_INTENTS[category]}",
                        pool_size,
                        category=category,
                        disease=disease,
                    ),
                )
            )
        lanes.append(
            (
                f"{disease} safety and differential precautions",
                _query(
                    collection,
                    embedder,
                    f"{base_query}; {disease}; {SAFETY_INTENT}",
                    pool_size,
                    disease=disease,
                ),
            )
        )
    # Older indexes may not have disease metadata; preserve their broad
    # category behavior rather than silently returning no evidence.
    if not any(lane_results for _, lane_results in lanes):
        lanes = [
            (
                "fallback category coverage",
                _query(collection, embedder, base_query, pool_size),
            )
        ]

    merged: dict[str, dict[str, Any]] = {}
    for reason, lane_results in lanes:
        for result in lane_results:
            card_id = result.get("card_id")
            if not card_id:
                continue
            metadata = result["metadata"]
            card = cards_by_id.get(card_id)
            card_safety, safety_text = _card_safety(card) if card is not None else (False, "")
            indexed_safety = str(metadata.get("safety_critical", "")).lower() == "true"
            candidate = dict(result)
            candidate["safety_critical"] = indexed_safety or card_safety
            candidate["safety_text"] = safety_text or metadata.get("safety_text", "")
            candidate["selection_reasons"] = [reason]
            if candidate["safety_critical"] and "safety-critical" not in reason:
                candidate["selection_reasons"].append(
                    "safety-critical and differential review"
                )
            candidate["candidate_diseases"] = candidate_diseases
            candidate["disease_role"] = (
                "primary"
                if candidate_diseases
                and metadata.get("disease_protocol") == candidate_diseases[0]
                else "alternative"
            )
            current = merged.get(card_id)
            if current is None:
                merged[card_id] = candidate
            else:
                current["selection_reasons"].append(reason)
                current["similarity_score"] = max(
                    current["similarity_score"], candidate["similarity_score"]
                )
                current["safety_critical"] = (
                    current["safety_critical"] or candidate["safety_critical"]
                )
                if candidate["safety_text"]:
                    current["safety_text"] = candidate["safety_text"]

    # Chroma ranking is used for ordering only. No similarity cutoff is
    # interpreted as a diagnosis or clinical probability.
    candidates = list(merged.values())
    candidates.sort(
        key=lambda item: (
            bool(item["safety_critical"]),
            item["similarity_score"],
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    available_categories = {
        item["metadata"].get("clinical_category")
        for item in candidates
        if item["metadata"].get("clinical_category")
    }
    selected_ids: set[str] = set()

    def add(item: dict[str, Any]) -> None:
        if len(selected) < top_k and item["card_id"] not in selected_ids:
            selected.append(item)
            selected_ids.add(item["card_id"])

    # First preserve diagnostic representation for each candidate disease.
    for disease in candidate_diseases:
        for item in candidates:
            if (
                item["metadata"].get("disease_protocol") == disease
                and item["metadata"].get("clinical_category") == "Diagnosis"
            ):
                add(item)
                break

    # Preserve category coverage while keeping diagnostic alternatives visible.
    # At least one investigation/medication/advice when such evidence exists.
    for category in ("Investigation", "Medication", "Advice"):
        for disease in candidate_diseases:
            if len(selected) >= top_k:
                break
            for item in candidates:
                if (
                    item["metadata"].get("disease_protocol") == disease
                    and item["metadata"].get("clinical_category") == category
                ):
                    add(item)
                    break
            if category in {x["metadata"].get("clinical_category") for x in selected}:
                break
    # Safety card is reserved if capacity remains after core clinical coverage.
    for disease in candidate_diseases:
        for item in candidates:
            if item["metadata"].get("disease_protocol") == disease and item["safety_critical"]:
                add(item)
                break
    for item in candidates:
        if item["metadata"].get("clinical_category") in CATEGORIES or item["safety_critical"]:
            add(item)
    # Fallback collections can lack disease metadata. When an explicit
    # diagnosis evidence item exists, don't append an unrelated protocol's
    # medication solely because its vector similarity happens to be high.
    if not candidate_diseases:
        diagnosed = {
            item["metadata"].get("disease_protocol")
            for item in selected
            if item["metadata"].get("clinical_category") == "Diagnosis"
        }
        if diagnosed:
            selected = [
                item for item in selected
                if item["metadata"].get("disease_protocol") in diagnosed
            ]
    for item in selected:
        item["selection_reason"] = "; ".join(dict.fromkeys(item.pop("selection_reasons")))
        item["candidate_diseases"] = candidate_diseases
        item["coverage"] = _coverage_summary(
            selected, candidate_diseases, available_categories
        )
        item["disease_reason"] = disease_reasons.get(
            item["metadata"].get("disease_protocol"),
            "Stage 2 disease-filtered evidence",
        )
    return selected[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-json", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    results = retrieve(json.loads(args.patient_json.read_text(encoding="utf-8")), args.top_k)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()


# Phase 1: combine semantic hits with local sparse retrieval; do not apply
# disease-specific boosts or infer a diagnosis from similarity alone.
from functools import lru_cache
from collections import Counter
import math

from build_index import retrieval_text


@lru_cache(maxsize=4)
def _resources(persist_dir: str, model_name: str):
    from sentence_transformers import SentenceTransformer
    import chromadb
    model = SentenceTransformer(model_name)
    collection = chromadb.PersistentClient(path=persist_dir).get_collection(
        "clinical_cards_dev" if Path(persist_dir).name == "dev_index" else "clinical_cards"
    )
    return model, collection


def clinical_query(patient: dict[str, Any]) -> str:
    """Clinical values without metadata labels being used as search terms."""
    values = []
    for key in QUERY_FIELDS:
        value = patient.get(key)
        if value not in (None, "", [], {}):
            values.append(_readable(value))
    return ". ".join(values)


class Retriever:
    """Cached embedding model, Chroma collection and in-memory sparse index."""

    def __init__(self, persist_dir: Path = Path("data/index"),
                 cards: Mapping[str, Any] | None = None,
                 model_name: str = "all-MiniLM-L6-v2"):
        self.cards = dict(cards) if cards is not None else _default_cards(persist_dir)
        self.embedder, self.collection = _resources(str(persist_dir), model_name)
        self.documents = {cid: _tokens(retrieval_text(card))
                          for cid, card in self.cards.items()}
        self.doc_count = max(1, len(self.documents))
        frequencies = Counter(token for tokens in self.documents.values() for token in tokens)
        self.idf = {term: math.log(1 + (self.doc_count - freq + .5) / (freq + .5))
                    for term, freq in frequencies.items()}

    def search(self, query: str, *, category: str | None = None,
               disease: str | None = None, roles: set[str] | None = None,
               n: int = 20) -> list[dict[str, Any]]:
        from card_roles import assign_role
        dense = _query(self.collection, self.embedder, query, max(n, 80),
                       category=category, disease=disease)
        qualified = {
            cid for cid, card in self.cards.items()
            if (category is None or card.clinical_category == category)
            and (disease is None or card.disease_protocol == disease)
            and (roles is None or assign_role(card) in roles)
        }
        dense = [hit for hit in dense if hit["card_id"] in qualified]
        query_tokens = _tokens(query)
        sparse = sorted(qualified,
                        key=lambda cid: sum(self.idf.get(t, 0) for t in
                                            query_tokens & self.documents[cid]),
                        reverse=True)
        sparse = [cid for cid in sparse if query_tokens & self.documents[cid]]
        fused = {}
        for rank, hit in enumerate(dense, 1):
            fused[hit["card_id"]] = [1 / (60 + rank), hit]
        for rank, cid in enumerate(sparse, 1):
            if cid not in fused:
                card = self.cards[cid]
                fused[cid] = [0, {
                    "card_id": cid, "similarity_score": 0.0,
                    "metadata": {
                        "card_id": cid, "disease_protocol": card.disease_protocol,
                        "clinical_category": card.clinical_category,
                    },
                    "source": {"title": card.source_pdf_title,
                               "pdf_pages": card.pdf_page_numbers},
                }]
            fused[cid][0] += 1 / (60 + rank)
        ranked = sorted(fused.values(), key=lambda value: value[0], reverse=True)
        return [dict(hit, retrieval_score=score) for score, hit in ranked[:n]]


def retrieve_category(patient: dict[str, Any], category: str, *, diseases: list[str] | None = None,
                      top_k: int = 3, persist_dir: Path = Path("data/index"),
                      cards: Mapping[str, Any] | None = None,
                      model_name: str = "all-MiniLM-L6-v2") -> list[dict[str, Any]]:
    if category not in CATEGORIES or top_k <= 0:
        return []
    engine = Retriever(persist_dir, cards, model_name)
    query = clinical_query(patient) or "clinical protocol"
    names = list(dict.fromkeys(diseases or []))
    if not names:
        return engine.search(query, category=category, n=top_k)
    hits = {}
    for name in names:
        for item in engine.search(query, category=category, disease=name, n=max(top_k, 12)):
            cid = item["card_id"]
            if cid not in hits or item["retrieval_score"] > hits[cid]["retrieval_score"]:
                hits[cid] = item
    return sorted(hits.values(), key=lambda item: item["retrieval_score"], reverse=True)[:top_k]
