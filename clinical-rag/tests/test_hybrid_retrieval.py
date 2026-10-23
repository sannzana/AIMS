"""Offline regression checks for Phase 1 retrieval changes."""
from types import SimpleNamespace

from build_index import retrieval_text
from card_roles import assign_role
from retrieve import clinical_query, Retriever
from schema import Card


def _card(**kw):
    payload = dict(
        card_id="TEST", disease_protocol="Chikungunya",
        clinical_category="Diagnosis", source_pdf_title="Synthetic test",
        source_version_edition=None, pdf_page_numbers=[1],
        exact_supporting_evidence_passages=[{"pdf_page": 1, "passage": "Sample."}],
        verified=True,
    )
    payload.update(kw)
    return Card(**payload)


def test_retrieval_text_shortens_embedding_but_keeps_full_card_document():
    from build_index import card_document
    card = _card(card_title="Acute fever and joint pain",
                 additional_information="extraneous " * 1000)
    assert "Acute fever and joint pain" in retrieval_text(card)
    assert "extraneous" not in retrieval_text(card)
    assert "extraneous" in card_document(card)


def test_clinical_query_has_no_metadata_labels():
    query = clinical_query({"symptoms": "fever and rash", "age": 32,
                            "history": "none", "disease": "made up"})
    assert "fever and rash" in query
    assert "symptoms:" not in query and "history:" not in query
    assert "made up" not in query


def test_differential_card_role_not_primary_definition():
    card = _card(card_title="Differential diagnosis of fever and rash")
    assert assign_role(card) == "differential"


def test_hybrid_search_filters_disease_and_category_without_boosts():
    a = _card(card_id="CHI-A", card_title="Polyarthralgia fever")
    b = _card(card_id="DEN-B", disease_protocol="Dengue",
              card_title="Retroorbital pain fever")
    engine = object.__new__(Retriever)
    engine.cards = {a.card_id: a, b.card_id: b}
    engine.documents = {"CHI-A": {"polyarthralgia", "fever"},
                        "DEN-B": {"retroorbital", "fever"}}
    engine.doc_count = 2
    engine.idf = {"fever": 1.0, "polyarthralgia": 1.0}
    engine.embedder = SimpleNamespace(encode=lambda texts: [[0.0]])
    engine.collection = SimpleNamespace(query=lambda **kw: {
        "metadatas": [[{
            "card_id": "CHI-A", "clinical_category": "Diagnosis",
            "disease_protocol": "Chikungunya", "source_pdf_pages": "[1]",
        }, {
            "card_id": "DEN-B", "clinical_category": "Diagnosis",
            "disease_protocol": "Dengue", "source_pdf_pages": "[1]",
        }]],
        "distances": [[0.1, 0.2]],
    })
    hits = engine.search("polyarthralgia fever", category="Diagnosis",
                         disease="Chikungunya")
    assert [h["card_id"] for h in hits] == ["CHI-A"]
