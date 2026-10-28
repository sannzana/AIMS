# AIMS — Clinical NLP + RAG Prototype

A Streamlit prototype that retrieves source-backed clinical cards and presents diagnosis candidates, investigations, treatment, and advice. It includes **Baseline** and **Improved** modes using the same card corpus and configured language model.

**Educational prototype only. Outputs are not confirmed diagnoses or medical advice and require qualified clinician review. Use fictional patient information for demonstrations.**

## 1. Requirements

- Git and **Python 3.11 or 3.12** (Python 3.12 was used for the automated checks).
- Internet access to install packages, download the embedding model on first use, and call the language-model API.
- Your own **Groq API key** and access to the configured model. Provider quotas and charges depend on your account.

The local embedding model is `all-MiniLM-L6-v2`. It supports semantic card retrieval through Chroma; it does not generate clinical answers. The language model reads the selected evidence and generates responses through the Groq API. The example configuration uses `openai/gpt-oss-20b`; availability depends on the provider.

## 2. Windows setup — PowerShell

Run each block in order. Clone only once. If you already have the repository, use your existing folder instead.

```powershell
git clone https://github.com/sannzana/AIMS.git
cd AIMS
git switch main
python --version
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
cd clinical-rag
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If `python` is unavailable but the Windows Python launcher is installed, use `py -3.12` instead of `python` to create the environment. After activation, use `python` as shown.

Create your local configuration **once**:

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace `replace_with_your_groq_api_key` with your actual key, save, and close Notepad. Do not overwrite an existing `.env` that already contains your settings. See the configuration section below.

Build the local search index, then start the app:

```powershell
python src/build_index.py --rebuild
python -m streamlit run app.py
```

Open the local URL printed in the terminal, normally **http://localhost:8501**. Keep the terminal running; press **Ctrl+C** to stop the app.

## 3. macOS / Linux setup

```bash
git clone https://github.com/sannzana/AIMS.git
cd AIMS
git switch main
python3 --version
python3 -m venv .venv
source .venv/bin/activate
cd clinical-rag
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `clinical-rag/.env` in your text editor and insert your API key. Do not overwrite an existing configuration. Then run:

```bash
python src/build_index.py --rebuild
python -m streamlit run app.py
```

On Linux, if virtual-environment creation reports that `venv` is unavailable, install the Python venv package for your distribution before retrying.

## 4. Configuration

Save the following in **`clinical-rag/.env`**, not in the repository root:

```dotenv
LLM_PROVIDER=api
LLM_API_URL=https://api.groq.com/openai/v1/chat/completions
LLM_MODEL=openai/gpt-oss-20b
LLM_API_KEY=your_actual_groq_api_key
LLM_CACHE=1
AIMS_TELEMETRY=0
AIMS_LLM_TIMEOUT_SECONDS=30
```

| Setting | Purpose |
| --- | --- |
| `LLM_PROVIDER` | `api` for the supplied Groq configuration; `ollama` is also supported by the adapter. |
| `LLM_API_URL` | Full chat-completions endpoint, including `/openai/v1/chat/completions` for Groq. |
| `LLM_MODEL` | Model identifier accepted by your provider. Set this explicitly; the code's placeholder default is not a usable model configuration. |
| `LLM_API_KEY` | Your private API key. Never commit or share it. |
| `LLM_CACHE` | `1` enables local caching of successful LLM responses; `0` requests fresh responses. Default: `1`. |
| `LLM_CACHE_DIR` | Optional cache directory; default `.cache/llm_responses`, relative to the working directory. |
| `AIMS_TELEMETRY` | `1` records API-request timing, outcomes, and token counts when supplied by the provider. Default: `0`. |
| `AIMS_TELEMETRY_PATH` | Optional metrics file; default `outputs/llm_metrics.jsonl`. |
| `AIMS_LLM_TIMEOUT_SECONDS` | Per-request timeout in seconds, clamped to 5–90. Default: `30`. |

The adapter reads `.env` from the **current working directory**. Existing terminal environment variables take precedence over `.env`. If a setting appears unchanged, check for an existing environment variable and restart Streamlit after editing the configuration. Do not print your API key when troubleshooting.

`.env`, local indexes, output files, and the default response cache are excluded from Git. The response cache contains generated text and may repeat patient details; use fictional data and keep it private. Caching reduces repeated identical API calls but does not eliminate calls for changed prompts, inputs, or models.

### Optional: local Ollama

Install and start Ollama separately, download a suitable model using `ollama pull YOUR_MODEL_NAME`, and set:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_URL=http://localhost:11434
LLM_MODEL=YOUR_MODEL_NAME
```

Replace the model placeholder with the installed model's exact name. An API key is not required for this local adapter. Model quality and JSON compliance vary; this configuration is not evidence of equivalent results to Groq. Other OpenAI-compatible endpoints can be configured with `LLM_PROVIDER=api`, their full endpoint, model, and key; provider compatibility must be checked separately.

## 5. Evidence and the search index

The repository includes `clinical-rag/data/cards.jsonl` (108 cards), extracted page text in `data/pages.json`, and page configuration. The corpus covers Chikungunya, Cholera, Dengue, and Hypertension, with some comparative material.

You **do not need the original PDFs to launch the app using the supplied cards**. Original PDFs are not distributed in this repository; reviewing or regenerating evidence from the original documents requires obtaining those documents separately.

From `clinical-rag`, build or rebuild the index with:

```bash
python src/build_index.py --rebuild
```

This indexes cards marked `verified` into the local Chroma collection `clinical_cards` under `data/index`. The current corpus should report `Indexed 108 card(s)` if all supplied cards remain verified. The first run downloads the embedding model and may take longer. The `verified` flag is a dataset flag, not proof of clinical validity.

Rebuild after changing cards or indexing logic. Keep the default embedding model because retrieval uses the same default; changing only the indexing model can produce incompatible embeddings. Building the index does not call Groq. Do not use `--dev` for the main demonstration: it indexes separate sample cards into a separate development collection.

Optional evidence consistency check:

```bash
python src/validate_cards.py --cards data/cards.jsonl --pages data/pages.json
```

This checks card structure and supplied text/page consistency. It is not independent medical verification of the source documents.

**Known validation result:** running this command against the supplied dataset during README verification reported **108 cards and 142 errors**, including supporting passages not found in the supplied page text, and exited with status 1. Medication entries also produce manual-review notices. These existing evidence-consistency issues require source review; they are separate from the passing software tests. The app reports validation issues as a warning rather than treating them as proof that the evidence is valid.

## 6. Using the app

1. Enter symptoms, clinical findings, measurements, duration, age, sex, history, current medicines, and allergies in their separate fields. You do not need to provide a disease name or paste JSON into the symptoms field.
2. Choose **Baseline** or **Improved** under **Recommendation mode**. Baseline is the initial default selection.
3. Click **Generate recommendation** to run diagnosis.
4. In Improved mode, select a **Working diagnosis for management** from the available candidates. If no supported selection is available, review the diagnostic status and technical diagnostics.
5. Click **Generate Treatment & Management Plan** to generate investigations, treatment, and advice.
6. Review source references, matched conditions, restrictions, and missing information. Conditional or additional source material is not an approved patient-specific treatment.
7. If optional management questions appear, answer what is known and apply the answers using the displayed controls. Leave unknown information unknown. The reassessment updates management source matches without rerunning diagnosis; it does not necessarily regenerate the original LLM prose.

Example fictional input, entered in separate fields:

| Field | Example |
| --- | --- |
| Symptoms | High fever, severe joint pain in both wrists, knees and ankles, headache, rash |
| Clinical findings | Swollen and tender joints, difficulty walking, no reported bleeding |
| Relevant measurements | Temperature 39°C, BP 115/75 mmHg, SpO2 98% |
| Duration | 3 days |
| Age | 32 years |
| Sex | Female |
| History | No pre-existing liver or kidney disease; no other chronic diseases |
| Current medicines | None |
| Allergies | Unknown |

Improved normalizes age and duration notation such as `32yr` / `32 yr` / `32 years` and `12hr` / `12 hr` / `12 hours`. A bare age such as `32` is treated as years; an explicit value such as `32 months` retains its unit. Use explicit units for duration and measurements. Normalization does not make incomplete clinical information complete.

### Baseline and Improved

Both modes use the supplied evidence corpus, shared retrieval infrastructure, and configured language model. Baseline uses a smaller management evidence selection and generated summaries. Improved adds explicit diagnostic evidence checks, clinician selection of a working diagnosis, management-fact collection, instruction-level source conditions, scoped restrictions, age-clause selection, conditional source displays, and optional management clarification questions.

Improved may show more source information, including unresolved requirements. More displayed information does not by itself establish better clinical performance. Diagnosis candidates are not confirmed diagnoses or calibrated disease probabilities. Missing information can prevent a recommendation from being supported.

## 7. Starting again after the first setup

For your existing Windows checkout at `D:\a\AIMS`:

```powershell
cd D:\a\AIMS
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
cd clinical-rag
python -m streamlit run app.py
```

For macOS/Linux, from the repository root:

```bash
source .venv/bin/activate
cd clinical-rag
python -m streamlit run app.py
```

You do not need to reinstall dependencies or rebuild an unchanged index every time.

To update an existing checkout, stop Streamlit with **Ctrl+C**, then run from the repository root:

```bash
git status --short
git switch main
git pull --ff-only origin main
```

Keep or commit your own changes before switching branches if Git reports conflicts; do not discard them blindly. Activate the environment, enter `clinical-rag`, install requirements if they changed, rebuild the index if cards/indexing changed, and restart the app. Refreshing the browser alone does not switch the running code version.

## 8. Tests and evaluation documents

From `clinical-rag` with the virtual environment active:

```bash
python -m pytest -q
```

At the submitted code version, **355 automated tests passed**. These are software checks using fixtures/mocks where appropriate; they do not constitute a live Groq/retrieval evaluation or establish clinical accuracy. Running the app successfully still requires a working API configuration and local index.

The editable six-case expected-results table and patient inputs are available here:

- [Six tested cases — expected results (Word)](docs/evaluation/Six_Tested_Cases_Expected_Results.docx)
- [Six tested cases — inputs (JSON)](docs/evaluation/Six_Tested_Cases_Inputs.json)
- [Case and checklist guide (PDF)](docs/evaluation/AIMS_Cases_and_Checklists.pdf)

Expected results are evaluation materials, not observed model outputs. For a Baseline/Improved comparison, keep the same inputs, model, and corpus; save all six outputs from each mode separately. Report any clarification answers and cache usage, since these affect comparability. Do not add expected answers to retrieval evidence or generation prompts.

## 9. Troubleshooting

| Problem | What to do |
| --- | --- |
| `File does not exist: app.py` | You are probably in the repository root. Run `cd clinical-rag`, then `python -m streamlit run app.py`. |
| `streamlit` not recognized or a module is missing | Activate `.venv`, install `requirements.txt`, and use `python -m streamlit` to select the active interpreter. |
| PowerShell blocks `Activate.ps1` | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in that terminal, then activate again. Alternatively run the virtual environment's Python executable directly. |
| `LLM_API_KEY is required` | Create `clinical-rag/.env`, insert your key, start from `clinical-rag`, and restart Streamlit. |
| HTTP 401 / 403 | Check key validity, endpoint, model access, and provider account permissions. Do not publish the key while seeking help. |
| HTTP 429 / `LLMRateLimitError` | Your provider is limiting requests or quota. Wait for the applicable reset, check your account limits, and avoid repeated submissions. Cached successful identical requests may help; caching cannot fix exhausted quota for new requests. |
| Model unavailable / HTTP 400 or 404 | Confirm `LLM_MODEL` is available to your account and that `LLM_API_URL` is the full supported endpoint. |
| Timeout or transport error | Check connectivity/provider status. If needed set `AIMS_LLM_TIMEOUT_SECONDS=60` and restart. A timeout change does not fix rate limits. |
| Empty retrieval / missing collection / embedding mismatch | From `clinical-rag`, run `python src/build_index.py --rebuild`; use the default embedding model for both indexing and retrieval. |
| Embedding download fails | Check internet access to the model host. The initial model download is separate from Groq API access. |
| Old response or settings appear | Stop and restart Streamlit; verify your checkout with `git log -1 --oneline`. To request uncached responses, set `LLM_CACHE=0` in `.env` and ensure no terminal variable overrides it. This can increase API usage. |
| Port 8501 is occupied | Stop the other app, or run `python -m streamlit run app.py --server.port 8502`. |
| No supported candidate or no validated recommendation | Inspect the technical diagnostics and missing information. This can reflect incomplete evidence, rejected model output, or a provider failure; it is not automatically an installation problem. |

To bypass activation in Windows, from the repository root you can run:

```powershell
cd clinical-rag
..\.venv\Scripts\python.exe -m streamlit run app.py
```

## Streamlit Community Cloud deployment

Deploy repository `sannzana/AIMS`, branch `main`, entrypoint `clinical-rag/app.py`.
Choose Python **3.12** in the deployment's advanced settings when available.
Use the following top-level **Secrets** (TOML), replacing the key privately:

```toml
LLM_PROVIDER = "api"
LLM_API_URL = "https://api.groq.com/openai/v1/chat/completions"
LLM_MODEL = "openai/gpt-oss-20b"
LLM_API_KEY = "your_actual_groq_api_key"
LLM_CACHE = "1"
AIMS_LLM_TIMEOUT_SECONDS = "60"
```

The app prepares a missing or empty local Chroma collection on startup from the
supplied verified cards. Existing populated collections are reused, not rebuilt.
The first startup requires access to download the embedding model and can take
several minutes. Cloud storage can be recreated after redeployment, so the app
can rebuild its index when needed. No source PDFs or local `.env` upload is needed.
For changed cards, explicitly rebuild the index or deploy into fresh storage.

After deploying, test a complete diagnosis and management run, then test the URL
in a private browser window to confirm evaluator access. A page loading alone
does not verify model/API access. If startup fails, inspect **Manage app → Logs**;
if requests return HTTP 429, check the Groq quota. Never submit your API key as
an evaluator credential. Local setup and manual index-building commands above
remain supported.

## 10. Repository layout

```text
AIMS/
├── README.md
├── docs/evaluation/              # Case inputs and expected-result documents
└── clinical-rag/
    ├── app.py                    # Streamlit interface
    ├── requirements.txt
    ├── .env.example              # Safe configuration template
    ├── .env                      # Your local credentials; not committed
    ├── data/
    │   ├── cards.jsonl           # Main evidence cards
    │   ├── pages.json            # Extracted source-page text
    │   └── index/                # Generated local Chroma index; not committed
    ├── src/                      # Retrieval, LLM adapter, rules, validation, UI helpers
    ├── tests/                    # Automated software tests
    ├── .cache/llm_responses/      # Optional local response cache; not committed
    └── outputs/                  # Optional local telemetry; not committed
```

## Limitations

This is a source-bounded demonstration, not a complete clinical system. The corpus is limited, extraction and interpretation can fail, some source tables or prerequisites require manual review, and provider responses can vary. A matched source condition does not establish a verified clinical finding or authorize treatment. Real-world clinical use, privacy controls, and clinical validation require further work.
