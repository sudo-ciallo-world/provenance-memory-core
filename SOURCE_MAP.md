# Source Map

Source modules: `tools/arch_c_system.py` and `tools/common.py` from the larger personal-memory project working tree.

Preparation changes:

1. Removed automatic home-directory credential discovery and implicit service endpoints.
2. Replaced personal example topics in the extraction prompt with neutral examples.
3. Applied a shared active/unexpired eligibility filter to vector and lexical retrieval.
4. Added entirely synthetic units, an offline demo, and focused regression tests.

The public demo fixes embeddings rather than claiming to run a trained model. Original research data and historical metrics are excluded. The source directory had no independent Git history. sudo-ciallo-world confirmed the contribution and release rights for the selected code; see `LICENSE-DECISION.md` and the MIT `LICENSE`.

The original application was not modified, redeployed, or benchmarked by this preparation.
