# clinical-rag

Clinical decision-support prototype for four protocol domains: dengue,
hypertension, cholera, and chikungunya.

## Current implementation

The schema, applicability, PDF processing, and retrieval stages are implemented:

- `src/schema.py` defines the strict Pydantic 1.1 evidence-card schema.
- `src/extract_pages.py` extracts one-based page text (including detected
  tables) from PDFs in `data/pdfs/`.
- `src/validate_cards.py` validates JSONL cards and checks evidence passages
  against normalized source-page text, duplicate IDs, and page references. A
  matching passage is not proof of manual clinical verification.
- `src/build_index.py` builds a local Chroma index using
  `sentence-transformers`; normal mode indexes only `verified: true` cards.
- `src/retrieve.py` retrieves top-k cards from patient information without
  requiring or using a disease-name field.
- `src/applicability.py` evaluates clinical eligibility separately from
  retrieval relevance.

LLM, Streamlit, and evaluation integration are intentionally not wired yet.

## LLM configuration

Recommendation generation is available through `src/llm.py` and
`src/recommend.py`; Streamlit and evaluation are not part of this phase.
Create a local `.env` file (never commit it) using one of these configurations:

```dotenv
LLM_PROVIDER=api
LLM_MODEL=your-model-name
LLM_API_KEY=your-secret
LLM_API_URL=https://api.openai.com/v1/chat/completions
```

or:

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=your-local-model
OLLAMA_URL=http://localhost:11434
```

Temperature, timeout, and generation limits are fixed in `src/llm.py`.
Credentials and patient prompts are not logged. Recommendation tests use a
mock LLM and make no paid API calls.

## Setup and commands

```powershell
python -m pip install -r requirements.txt
python src/extract_pages.py
python src/validate_cards.py --cards data/cards.jsonl --pages data/pages.json
python src/build_index.py --rebuild
python src/build_index.py --dev --rebuild
streamlit run app.py
```

Put source PDFs in `data/pdfs/`. Keep development fixtures in
`data/sample_cards.jsonl`; these are clearly marked DEV ONLY and must not be
used for patient care. Only manually checked cards should be placed in
`data/cards.jsonl` with `verified: true`.

The Streamlit interface loads all cards for display, retrieves from the local
Chroma index, shows evidence and applicability status, and reports validation
warnings. It does not generate unsupported recommendations; the local index
must be built before retrieval.
