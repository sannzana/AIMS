# Improved management source mappings

These mappings affect Improved management only. Diagnosis, Baseline, retrieval,
the shared applicability engine and `data/cards.jsonl` are unchanged.

- `management_exception_scopes.jsonl` contains 14 engineering source-text
  mappings: restrictions scoped to named instructions, interpretation cautions,
  and instructions that are themselves safety guidance. This is not independent
  clinical validation of the source corpus.
- The population/field-alias questions in HTN-021 are intentionally unmapped.
  Unmapped or stale restrictions remain conditional and visible for review.
- `management_prerequisites.jsonl` binds an explicit drinking-safety prerequisite
  to the corresponding source instructions. No clinical threshold is invented.
- Each record is bound to a digest of its card's relevant source fields and an
  exact source quote. Source changes invalidate the mapping until it is reviewed.
- Medication terms are literal terms in source exceptions. A mention is not proof
  of current use. No brand/class equivalence or general interaction lookup is
  inferred. Missing facts never establish overall suitability.
- Age filtering uses explicit numeric source bands. It does not round a
  fractional age into an assumed completed-year cohort. Overlap, unparsed prose,
  or an unmarked dependent clause remains source-review information.

Runtime validation uses ephemeral conservative views so legacy whole-card
exception handling does not suppress unrelated matching source instructions.
Mixed cards remain review-only as a whole; source traceability still uses the
original cards. No source file is rewritten and no additional LLM call is added.

Before changing a mapping, review the source quote, affected rule IDs, exception
meaning and clinical implications; regenerate the digest only after that review.
Do not merely refresh a digest to bypass a stale-source check.
