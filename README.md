# Provenance Memory Core

Retrieve structured knowledge together with the evidence, source, and conditions that support it.

This Python core stores knowledge units in SQLite and combines vector ranking with FTS5 lexical search. Each result carries its lifecycle state and provenance so a caller can inspect why it was stored and where it applies.

The offline demo uses synthetic units and fixed two-dimensional vectors. It needs no LLM, embedding server, API key, or conversation export.

## Quick start

Requires **Python 3.11 or later** with SQLite **FTS5 and the trigram tokenizer**. The demo uses only the Python standard library.

```sh
git clone https://github.com/sudo-ciallo-world/provenance-memory-core.git
cd provenance-memory-core
python demo.py
```

The demo creates a temporary database, inserts an active policy and a superseded policy, and searches for `demo policy`. It returns only the active unit and removes the temporary database on exit.

Selected fields from the result:

```json
{
  "text": "demo policy: use the new review checklist",
  "source_conv": "synthetic-conversation-001",
  "metadata": {
    "status": "active",
    "conditions": "Synthetic demo project only.",
    "evidence": "Synthetic author: use the new review checklist for the demo."
  }
}
```

See [demo.py](demo.py) for the full example and [fixtures/units.json](fixtures/units.json) for both records.

## What a knowledge unit keeps

| Field | Purpose |
| --- | --- |
| `statement`, `topic`, `type` | The knowledge and its classification |
| `evidence`, `source_conv`, `source_note` | Supporting text and source reference |
| `conditions`, `confidence` | Scope and recorded confidence |
| `status`, `expires_at` | Lifecycle and explicit deadline |
| `supersedes`, `superseded_by` | Links between replaced and replacement units |
| `embedding` | Vector used by retrieval |

Retrieval applies the same eligibility rule to both vector and lexical candidates. Only active units are eligible. A nonempty expiry must be a valid, timezone-aware future timestamp; past, invalid, unresolved, and timezone-free values are excluded. An empty expiry means no explicit deadline.

The ingestion ledger records successful and failed sources. Resume mode skips successful ingestion and retries failed work.

## Tests

```sh
python -m unittest discover -s tests -v
```

The six offline tests cover:

- Preserving evidence, conditions, and source references.
- Excluding superseded units even when lexical search matches them.
- Excluding expired units and invalid or unresolved deadlines.
- Resuming ingestion without repeating successful sources.
- Requiring explicitly supplied credentials for online calls.

## Code map

| File | Responsibility |
| --- | --- |
| [tools/arch_c_system.py](tools/arch_c_system.py) | Schema, ingestion ledger, lifecycle filtering, hybrid retrieval, and export |
| [tools/common.py](tools/common.py) | Corpus parsing, vector helpers, and optional provider calls |
| [demo.py](demo.py) | Temporary database and deterministic retrieval example |
| [tests/test_offline.py](tests/test_offline.py) | Offline regressions |

## Online integration

The source modules include extraction and embedding helpers. Online calls require explicit configuration: a chat endpoint through `AI_JOB_BASE_URL` or `SUB2API_BASE`, `LLM_MODEL`, and `LLM_API_KEY` or `AI_JOB_API_KEY`; embeddings require `OLLAMA_BASE`, with `EMBED_MODEL` selecting the model.

The documented demo and test commands never invoke those providers. Credential discovery from another application's configuration has been removed. No corpus, database, answer key, runtime log, or session importer is shipped.

## Corrections and limitations

This public snapshot fixes a source-revision bug that allowed a superseded unit to re-enter results through the lexical merge. The correction and the shared expiry filter are documented in [SOURCE_MAP.md](SOURCE_MAP.md).

Conditions are returned for the caller to assess; the engine does not decide whether arbitrary natural-language conditions hold. Extraction and similarity-based deduplication remain heuristic. The deterministic demo demonstrates retrieval behavior, rather than measuring model quality or semantic accuracy. Earlier private-corpus comparisons do not measure this corrected revision.

If SQLite reports `no such module: fts5` or `no such tokenizer: trigram`, use a Python/SQLite build with those capabilities enabled.

## Contributing

Use synthetic knowledge units and fixed vectors for regression cases. State the expected returned units and lifecycle behavior, then run the tests above.

## Development and license

The project maintainer led the knowledge-unit model, retrieval design, architecture, integration, debugging, and validation. Implementation used substantial AI-assisted coding. SQLite and model providers are reused infrastructure.

Most development took place locally before the first public snapshot on **October 18, 2026**. Public commits record released snapshots and subsequent changes.

Released under the [MIT license](LICENSE). See [SOURCE_MAP.md](SOURCE_MAP.md) and [LICENSE-DECISION.md](LICENSE-DECISION.md) for extraction boundaries and licensing scope.
