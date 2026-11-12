# Corpus V2

Corpus V2 freezes the bounded expansion selected from the V1 failure analysis. V1 had 19 plays, sparse held-out author/genre coverage, author leakage across partitions, zero held-out-speaker consistency for both models, and severe TextNCA cross-prompt collapse. V2 addresses the data-side weaknesses without turning corpus collection into an open-ended phase.

## Frozen corpus

- **30 independent plays** across **75 scenes** and **14,800 dialogue turns**.
- **2,094,294 serialized byte-level tokens**, up **1,031,970** (**97.14%**) from V1.
- Train: **23 works / 50 scenes / 8,682 turns / 1,189,607 tokens**.
- Validation: **3 works / 11 scenes / 3,065 turns / 426,484 tokens**.
- Test: **4 works / 14 scenes / 3,053 turns / 478,203 tokens**.
- Corpus SHA-256: `1deb3f405f96b81208352de6cb87a5477b6cb2841ac2f3809433e49f7c6d981d`.
- Manifest SHA-256: `e09014769afb9e04d61d668bd60daa883db65f5c9f95d38de48b1580a7dbc1fb`.

## Author-aware partition

- Train: Anton Chekhov; John Galsworthy; Susan Glaspell; Susan Glaspell and George Cram Cook.
- Validation: George Bernard Shaw; Henrik Ibsen.
- Test: Eugene O’Neill; J. M. Synge; Oscar Wilde.
- Overlapping coauthors are normalized before validation. **No author appears in more than one split.**

## Bounded V2 additions

Eleven plays were added: seven Galsworthy works in train, Shaw’s *Arms and the Man* in validation, and Wilde’s *An Ideal Husband*, O’Neill’s *Beyond the Horizon*, and Synge’s *Riders to the Sea* in test. These additions broaden social drama, naturalist drama, tragedy, satire, farce, comedy, and comedy-of-manners coverage. Every new source has a pinned HTTPS Gutenberg URL, byte count, SHA-256, exact source-line ranges, declared speakers, and fail-closed repairs.

## Remaining limitations

- English-language drama only.
- Training remains concentrated in three authorship groups.
- Historical translations confound author and translator style for Chekhov and Ibsen.
- This is one fixed partition, not cross-validation.
- Genre labels are coarse curator labels.
- *Justice* Act III Scene III is silent stage action and is omitted because training records require dialogue turns.

`results.json` contains the complete play, author, genre, split, token, hash, and delta inventory.
