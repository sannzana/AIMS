import json
from pathlib import Path
import sys

import fitz

from build_index import build_index
from extract_pages import assigned_page_keys, extract_pages, render_flagged_pages
from retrieve import patient_query, retrieve
from schema import Card
from validate_cards import validate_cards


class FakeEmbedder:
    def encode(self, texts):
        return [[float(len(text))] for text in texts]


class FakeCollection:
    def __init__(self):
        self.rows = {}

    def upsert(self, **kwargs):
        for index, card_id in enumerate(kwargs["ids"]):
            self.rows[card_id] = {
                "metadata": kwargs["metadatas"][index],
                "document": kwargs["documents"][index],
            }

    def query(self, query_embeddings, n_results, include):
        rows = list(self.rows.values())[:n_results]
        return {
            "metadatas": [[row["metadata"] for row in rows]],
            "distances": [[0.2 for row in rows]],
        }


class FakeClient:
    def __init__(self):
        self.collection = FakeCollection()

    def get_or_create_collection(self, name):
        return self.collection

    def get_collection(self, name):
        return self.collection


def card(card_id: str, verified: bool) -> dict:
    return {
        "card_id": card_id,
        "disease": "dengue",
        "category": "diagnosis",
        "action": "DEV ONLY fixture; not clinical advice.",
        "source": {"title": "Protocol", "pdf_pages": [1]},
        "evidence": [{"pdf_page": 1, "passage": "Fever is present."}],
        "verified": verified,
    }


def test_extract_pages_preserves_one_based_pages_and_empty_pdf_dir(tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "protocol.pdf"
    document = fitz.open()
    document.new_page().insert_text((72, 72), "Cover")
    document.new_page().insert_text((72, 72), "Fever is present.")
    document.save(pdf_path)
    document.close()

    output = tmp_path / "pages.json"
    extract_pages(pdf_dir, output)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert [page["page"] for page in data["documents"][0]["pages"]] == [1, 2]
    assert data["documents"][0]["pages"][0]["text"].strip() == "Cover"

    empty_output = tmp_path / "empty.json"
    extract_pages(tmp_path / "missing-pdfs", empty_output)
    assert json.loads(empty_output.read_text(encoding="utf-8")) == {"documents": []}


def test_extract_cli_fails_clearly_for_missing_pdf_directory(tmp_path: Path, monkeypatch, capsys):
    from extract_pages import main

    monkeypatch.setattr(
        sys,
        "argv",
        ["extract_pages.py", "--pdf-dir", str(tmp_path / "missing")],
    )
    try:
        main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("missing PDF directory should fail")
    assert "PDF directory not found" in capsys.readouterr().err


def test_extract_page_three_and_render_flagged_page(tmp_path: Path):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_path = pdf_dir / "hypertension.pdf"
    document = fitz.open()
    for page in range(3):
        document.new_page().insert_text((72, 72), f"Page {page + 1}")
    document.save(pdf_path)
    document.close()
    output = tmp_path / "pages.json"
    extract_pages(pdf_dir, output)
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["documents"][0]["pages"][2]["page"] == 3
    data["documents"][0]["pages"][2]["manual_review_flags"] = ["manual"]
    output.write_text(json.dumps(data), encoding="utf-8")
    rendered = render_flagged_pages(pdf_dir, output, tmp_path / "png")
    assert rendered[0].name == "hypertension_page_3.png"
    assert rendered[0].exists()


def test_page_config_has_45_unique_pages():
    config = json.loads(
        (Path(__file__).parents[1] / "data" / "pages_config.json").read_text()
    )
    pages = assigned_page_keys(config)
    assert len(pages) == 45
    assert len(set(pages)) == 45
    assert config["expected_unique_page_count"] == len(set(pages))
    assert ("hypertension.pdf", 64) in pages


def test_validate_cards_checks_normalized_evidence_duplicates_and_pages(tmp_path: Path, capsys):
    pages = {
        "documents": [
            {
                "title": "Protocol",
                "pages": [{"page": 1, "text": "Fever is present."}],
            }
        ]
    }
    pages_path = tmp_path / "pages.json"
    pages_path.write_text(json.dumps(pages), encoding="utf-8")
    cards_path = tmp_path / "cards.jsonl"
    valid = card("same", True)
    valid["evidence"][0]["passage"] = "Fever is-\npresent."
    invalid_page = card("same", True)
    invalid_page["evidence"][0]["pdf_page"] = 2
    cards_path.write_text(
        json.dumps(valid) + "\n" + json.dumps(invalid_page) + "\n", encoding="utf-8"
    )
    count, errors = validate_cards(cards_path, pages_path)
    output = capsys.readouterr().out
    assert count == 2
    assert errors >= 2
    assert "duplicate card_id" in output
    assert "manual verification still required" in output


def test_build_index_excludes_unverified_cards_and_retrieve_returns_metadata(tmp_path: Path):
    cards_path = tmp_path / "cards.jsonl"
    cards_path.write_text(
        json.dumps(card("verified", True))
        + "\n"
        + json.dumps(card("unverified", False))
        + "\n",
        encoding="utf-8",
    )
    client = FakeClient()
    assert build_index(
        cards_path,
        tmp_path / "index",
        embedder=FakeEmbedder(),
        client=client,
    ) == 1
    results = retrieve(
        {"symptoms": ["fever"], "age": 30, "history": "none"},
        embedder=FakeEmbedder(),
        client=client,
    )
    assert results[0]["card_id"] == "verified"
    assert results[0]["similarity_score"] == 0.8
    assert results[0]["source"]["title"] == "Protocol"
    assert results[0]["source"]["pdf_pages"] == [1]
    assert all(isinstance(value, (str, int, float, bool)) for value in client.collection.rows["verified"]["metadata"].values())


def test_patient_query_does_not_require_or_include_disease_name():
    query = patient_query(
        {
            "symptoms": ["cough"],
            "duration": "2 days",
            "age": 30,
            "sex": "female",
            "history": "none",
            "medicines": ["x"],
            "allergies": ["y"],
            "disease": "made-up",
        }
    )
    assert "cough" in query
    assert "2 days" in query
    assert "female" in query
    assert "none" in query
    assert "medicines" not in query
    assert "allergies" not in query
    assert "made-up" not in query
