# Provenance-Aware Memory Core

Store structured knowledge with evidence and source references, then combine vector and lexical retrieval.

This source snapshot contains the SQLite/FTS5 knowledge store and supporting functions from a larger personal-memory experiment. The included demo uses synthetic units and fixed two-dimensional vectors. It runs without an LLM, embedding server, personal conversations, or credentials.

## Run

Python 3.11 or later with SQLite FTS5/trigram support; no Python packages required for this demo.

```sh
python demo.py
python -m unittest discover -s tests -v
```

The demo returns the active checklist policy with its supporting evidence and scope, while excluding a superseded version.

## Engineering Focus

- Structured units keep evidence, scope, lifecycle metadata, and a source reference.
- Hybrid retrieval combines vector ranking and SQLite lexical matches.
- The ingestion ledger distinguishes successful, failed, and resumable work.
- The public snapshot applies one eligibility filter to both retrieval lanes.

Six offline tests cover evidence preservation, superseded/expired exclusion, explicit deadlines, resumable ingestion, and explicit credential configuration.

## Snapshot Corrections And Limits

The source revision allowed a superseded unit to re-enter through the lexical merge. This preparation reproduced that bug and corrected it **in this public snapshot only**. Active units with past, invalid, unresolved, or timezone-free expiry values are also excluded. Empty expiry means no explicit deadline.

Conditions are returned as evidence for the caller; the engine does not infer whether arbitrary natural-language conditions hold. Extraction and similarity-based deduplication remain heuristic. The demo tests retrieval plumbing, not language-model quality or semantic accuracy.

Earlier private-corpus comparisons do not measure this corrected revision. No private corpus, answer key, numerical benchmark result, MCP server, watchdog, or automatic session importer is shipped.

## Provenance And AI Assistance

Most development took place locally before this public snapshot, first published on October 18, 2026. Public commits record the released snapshot and subsequent changes.

sudo-ciallo-world conceived and led the system design, including the knowledge-unit model, retrieval/evaluation choices, architecture direction, integration, debugging, and verification. Implementation was developed with substantial AI-assisted coding. SQLite and model providers are reused infrastructure.

Online functions require explicitly supplied endpoints, model choice, and API key. They no longer discover credentials from another application's home-directory configuration. The documented commands never invoke those functions.

See `SOURCE_MAP.md` and `LICENSE`.

