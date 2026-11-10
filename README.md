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

The Wilde parser fixture uses complete acts as scene units: Act I is training, Act II is validation, and Act III is test. Both collections are split only at complete-play boundaries. The Glaspell collection uses six training plays, *Woman's Honor* for validation, and *Tickless Time* for testing. The Chekhov collection uses six training plays, *The Bear* for validation, and *The Cherry Orchard* for testing. All acts of *A Doll's House* and *Pygmalion* remain together in training. Future corpus releases will continue to hold out complete plays and, where scale permits, complete authors or translators.

## Legal and provenance scope

Project Gutenberg ebooks 844, 59432, 7986, 2542, and 3825 are cataloged as public-domain material in the United States. Ebook 59432 was transcribed from images of the 1920 Small, Maynard & Company edition; *Suppressed Desires* and *Tickless Time* are attributed to Susan Glaspell in collaboration with George Cram Cook. Ebook 7986 is Julius West's English translation of Chekhov's second series, first published in 1915; the Gutenberg text does not identify the exact scanned print copy. Ebook 2542 attributes its English translation to R. Farquharson Sharp but likewise does not identify the exact underlying print edition. Ebook 3825's embedded title page dates *Pygmalion* to 1912 but does not identify a publisher or exact print source, and its transcriber notes that omitted contraction apostrophes were restored electronically. The downloaded files include Project Gutenberg's license and trademark language. This repository does not commit or relicense the source text; it stores source URLs, hashes, provenance statements, and reproducible transformations. Anyone using the corpus outside the United States must verify the law in their jurisdiction.

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
