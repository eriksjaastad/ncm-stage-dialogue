# NCM Stage Dialogue

A local, reproducible research project comparing a small **Neural Cellular Model (NCM)** with a parameter-matched autoregressive language model on human-written, multi-speaker stage dialogue.

## Project intent

- **Purpose:** a portfolio-quality machine-learning research reproduction and benchmark, not a production chatbot.
- **Success:** build a provenance-preserving public-domain play corpus, train both models on Apple Silicon with MLX, and publish honest measurements and samples—even if the NCM loses.
- **Decision rule:** favor reproducibility, legal provenance, controlled comparisons, and negative-result honesty over model size or impressive claims.

## Current milestone

The corpus pipeline currently downloads and parses five pinned Project Gutenberg ebooks: Oscar Wilde's *The Importance of Being Earnest*, Susan Glaspell's eight-play collection *Plays*, Julius West's English translation of Anton Chekhov's eight-play *Second Series*, R. Farquharson Sharp's English translation of Henrik Ibsen's *A Doll's House*, and George Bernard Shaw's *Pygmalion*:

```bash
uv sync
uv run build-corpus
uv run pytest
uv run ruff check .
```

`build-corpus` downloads each pinned source, verifies its byte count and SHA-256, removes the Gutenberg wrapper, splits collections into independent works, parses speaker-labeled scene records, preserves stage directions separately, and writes:

```text
data/raw/gutenberg-844-importance-of-being-earnest.txt
data/raw/gutenberg-59432-glaspell-plays.txt
data/raw/gutenberg-7986-chekhov-second-series.txt
data/raw/gutenberg-2542-a-dolls-house.txt
data/raw/gutenberg-3825-pygmalion.txt
data/processed/scenes.jsonl
```

The current manifest produces 19 independent plays, 36 scenes, and 7,416 speaker turns. Generated data is intentionally gitignored. A clean checkout rebuilds it from `sources/plays.json`.

## Dataset record

Every source must declare:

- title, author, language, and original publication year;
- source landing page and exact downloadable URL;
- retrieval date, byte count, and SHA-256;
- jurisdiction and the stated public-domain basis;
- source terms, parser version, and an explicit speaker-label inventory;
- collection boundaries and per-work authorship where an ebook contains multiple plays;
- explicit complete-play train/validation/test assignments for collections.

Every work now stays in exactly one partition. All three acts of Wilde's *The Importance of Being Earnest* are held out for testing. The Glaspell collection uses six training plays, *Woman's Honor* for validation, and *Tickless Time* for testing. The Chekhov collection uses six training plays, *The Bear* for validation, and *The Cherry Orchard* for testing. All acts of *A Doll's House* and *Pygmalion* remain together in training. The current scene counts are 26 train, 2 validation, and 8 test.

## Legal and provenance scope

Project Gutenberg ebooks 844, 59432, 7986, 2542, and 3825 are cataloged as public-domain material in the United States. Ebook 59432 was transcribed from images of the 1920 Small, Maynard & Company edition; *Suppressed Desires* and *Tickless Time* are attributed to Susan Glaspell in collaboration with George Cram Cook. Ebook 7986 is Julius West's English translation of Chekhov's second series, first published in 1915; the Gutenberg text does not identify the exact scanned print copy. Ebook 2542 attributes its English translation to R. Farquharson Sharp but likewise does not identify the exact underlying print edition. Ebook 3825's embedded title page dates *Pygmalion* to 1912 but does not identify a publisher or exact print source, and its transcriber notes that omitted contraction apostrophes were restored electronically. The downloaded files include Project Gutenberg's license and trademark language. This repository does not commit or relicense the source text; it stores source URLs, hashes, provenance statements, and reproducible transformations. Anyone using the corpus outside the United States must verify the law in their jurisdiction.

The repository's MIT license covers code and original documentation only.

## Experiment

The first comparison reproduces the core architecture from *TextNCA: Neural Cellular Automata for Language Modeling via Hierarchical Local Attention* (Findings of EMNLP 2026): three causal local-attention stages with windows 8, 32, and 128; four shared cellular updates per stage; learned update embeddings; and GRU state updates.

The deterministic tokenizer uses UTF-8 bytes plus fixed structural tokens for works, scenes, speakers, dialogue, and stage directions. Complete works cannot cross train, validation, and test streams. Inspect the captured configurations with:

```bash
uv run ncm-experiment inspect --config configs/v1-textnca.json
uv run ncm-experiment inspect --config configs/v1-transformer.json
```

The primary models contain 27.35M and 27.18M trainable parameters respectively, a difference below 1%. They use the same 512-token context, byte vocabulary, corpus streams, batch size, optimizer settings, seed, and token budget. Small parameter-matched configurations are provided for correctness and checkpoint-resume calibration:

```bash
uv run ncm-experiment train --config configs/smoke-textnca.json
uv run ncm-experiment train --config configs/smoke-transformer.json
uv run ncm-experiment evaluate --config configs/smoke-textnca.json
uv run ncm-experiment generate --config configs/smoke-textnca.json --prompt 'ALICE: '
```

The long-term sequence is: complete and review the V1 harness, run both equal-budget models under a supervised detached runner, publish the V1 comparison, expand to a bounded V2 corpus of at least 30 plays based on measured V1 limitations, and repeat the locked comparison. Each phase ends with a verified artifact before the next begins.

## Non-goals

- No paid model APIs or synthetic-dialogue generation.
- No changes to Muffin Pan Recipes production.
- No changes to Cortana's model or adapter.
- No claim that a small research model is a useful production assistant.
