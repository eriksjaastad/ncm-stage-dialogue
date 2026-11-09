# NCM Stage Dialogue

A local, reproducible research project comparing a small **Neural Cellular Model (NCM)** with a parameter-matched autoregressive language model on human-written, multi-speaker stage dialogue.

## Project intent

- **Purpose:** a portfolio-quality machine-learning research reproduction and benchmark, not a production chatbot.
- **Success:** build a provenance-preserving public-domain play corpus, train both models on Apple Silicon with MLX, and publish honest measurements and samples—even if the NCM loses.
- **Decision rule:** favor reproducibility, legal provenance, controlled comparisons, and negative-result honesty over model size or impressive claims.

## Current milestone

The first end-to-end slice downloads and parses Oscar Wilde's *The Importance of Being Earnest* from Project Gutenberg:

```bash
uv sync
uv run build-corpus
uv run pytest
uv run ruff check .
```

`build-corpus` downloads the pinned source, verifies its byte count and SHA-256, removes the Gutenberg wrapper, parses all three acts into speaker-labeled scene records, preserves stage directions separately, and writes:

```text
data/raw/gutenberg-844-importance-of-being-earnest.txt
data/processed/scenes.jsonl
```

Generated data is intentionally gitignored. A clean checkout rebuilds it from `sources/plays.json`.

## Dataset record

Every source must declare:

- title, author, language, and original publication year;
- source landing page and exact downloadable URL;
- retrieval date, byte count, and SHA-256;
- jurisdiction and the stated public-domain basis;
- source terms, parser version, and an explicit speaker-label inventory;
- explicit whole-scene train/validation/test assignments.

The seed play uses complete acts as scene units: Act I is training, Act II is validation, and Act III is test. Later corpus releases should prefer holding out complete plays when enough material exists; scene-level holdouts remain useful for within-author structure tests.

## Legal and provenance scope

Project Gutenberg ebook 844 is distributed as public-domain material in the United States. The original play was first performed and published in 1895. The downloaded file includes Project Gutenberg's license and trademark language. This repository does not commit or relicense the source text; it stores the source URL, hash, provenance statement, and reproducible transformations. Anyone using the corpus outside the United States must verify the law in their jurisdiction.

The repository's MIT license covers code and original documentation only.

## Planned experiment

1. Expand the manifest with verified comedy, drama, farce, mystery, and domestic plays.
2. Audit parser output and deduplicate editions.
3. Freeze play-level train/validation/test splits and a tokenizer.
4. Train an approximately 27M-parameter MLX NCM.
5. Train an autoregressive Transformer with a matched parameter and token budget.
6. Compare validation loss, generation throughput, repetition, speaker consistency, and multi-speaker structure.
7. Publish configurations, checkpoints where licensing permits, samples, limitations, and negative findings.

## Non-goals

- No paid model APIs or synthetic-dialogue generation.
- No changes to Muffin Pan Recipes production.
- No changes to Cortana's model or adapter.
- No claim that a small research model is a useful production assistant.
