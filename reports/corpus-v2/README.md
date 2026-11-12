# Corpus V2

Corpus V2 freezes the bounded expansion selected from the V1 failure analysis. V1 had 19 plays, sparse held-out author/genre coverage, author leakage across partitions, zero held-out-speaker consistency for both models, and severe TextNCA cross-prompt collapse. V2 addresses the data-side weaknesses without turning corpus collection into an open-ended phase.

## Frozen corpus

- **30 independent plays** across **75 scenes** and **14,797 dialogue turns**.
- **2,094,294 serialized byte-level tokens**, up **1,031,970** (**97.14%**) from V1.
- Train: **23 works / 50 scenes / 8,680 turns / 1,189,603 tokens**.
- Validation: **3 works / 11 scenes / 3,065 turns / 426,484 tokens**.
- Test: **4 works / 14 scenes / 3,052 turns / 478,207 tokens**.
- Corpus SHA-256: `d2126f5d958e3190a708f7a502259196d8d9971df4935e33e53c388e1ee9a7c5`.
- Manifest SHA-256: `52430fe5ead44b4a154e4a628952aa60494e8baad9c5abcc566eba979e2ebc03`.

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
