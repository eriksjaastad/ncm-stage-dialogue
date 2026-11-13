"""Build the reproducible V2 TextNCA-versus-Transformer findings artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

from ncm_stage_dialogue.v1_report import (
    _artifact_hashes,
    _atomic_write,
    _curve_svg,
    _fmt,
    _generate_samples,
    _shared_training_commit,
    _shared_training_seed,
    _training_summary,
)

MODEL_NAMES = ("v2-textnca", "v2-transformer")


def _models(results: dict[str, Any], version: str) -> tuple[dict[str, Any], dict[str, Any]]:
    models = results["training"]["models"]
    return models[f"{version}-textnca"], models[f"{version}-transformer"]


def _cross_prompt(
    results: dict[str, Any], version: str
) -> tuple[dict[str, float], dict[str, float]]:
    summary = results["generation"]["summary"]
    return (
        summary[f"{version}-textnca"]["cross_prompt"],
        summary[f"{version}-transformer"]["cross_prompt"],
    )


def _version_comparison(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    previous_models = _models(previous, "v1")
    current_models = _models(current, "v2")
    previous_cross = _cross_prompt(previous, "v1")
    current_cross = _cross_prompt(current, "v2")
    comparison: dict[str, Any] = {}
    for label, index in (("textnca", 0), ("transformer", 1)):
        old_model = previous_models[index]
        new_model = current_models[index]
        old_cross = previous_cross[index]
        new_cross = current_cross[index]
        comparison[label] = {
            "test_loss_change": new_model["test"]["loss"] - old_model["test"]["loss"],
            "test_loss_change_percent": 100
            * (new_model["test"]["loss"] - old_model["test"]["loss"])
            / old_model["test"]["loss"],
            "elapsed_seconds_change": new_model["elapsed_seconds"]
            - old_model["elapsed_seconds"],
            "unique_continuation_rate_change": new_cross["unique_continuation_rate"]
            - old_cross["unique_continuation_rate"],
            "cross_work_similarity_change": new_cross[
                "mean_cross_work_character_trigram_jaccard"
            ]
            - old_cross["mean_cross_work_character_trigram_jaccard"],
        }
    return comparison


def _investment_decision(results: dict[str, Any]) -> dict[str, Any]:
    textnca, transformer = _models(results, "v2")
    textnca_cross, transformer_cross = _cross_prompt(results, "v2")
    textnca_similarity = textnca_cross["mean_cross_work_character_trigram_jaccard"]
    transformer_similarity = transformer_cross["mean_cross_work_character_trigram_jaccard"]
    criteria = {
        "textnca_test_loss_worse": textnca["test"]["loss"] > transformer["test"]["loss"],
        "textnca_training_slower": textnca["elapsed_seconds"]
        > transformer["elapsed_seconds"],
        "textnca_unique_rate_no_better": textnca_cross["unique_continuation_rate"]
        <= transformer_cross["unique_continuation_rate"],
        "textnca_cross_work_similarity_no_better": (
            textnca_similarity is not None
            and transformer_similarity is not None
            and textnca_similarity >= transformer_similarity
        ),
    }
    stop = all(criteria.values())
    return {
        "decision": "stop" if stop else "continue",
        "criteria": criteria,
        "rule": (
            "Stop further TextNCA investment when it loses on held-out loss and training "
            "speed and shows no advantage on exact continuation diversity or cross-work "
            "collapse."
        ),
    }


def _markdown(results: dict[str, Any]) -> str:
    training = results["training"]
    textnca, transformer = _models(results, "v2")
    comparison = training["comparison"]
    generation = results["generation"]["summary"]
    tn_gen = generation["v2-textnca"]["macro_average_across_works"]
    tr_gen = generation["v2-transformer"]["macro_average_across_works"]
    tn_cross, tr_cross = _cross_prompt(results, "v2")
    version = results["v1_to_v2"]
    decision = results["textnca_investment_decision"]
    test_works = results["dataset"]["split_counts"]["test"]["works"]
    early_samples = [
        sample for sample in results["generation"]["samples"] if sample["boundary"] == "early"
    ]
    sample_sections = []
    for sample in early_samples:
        continuation = sample["continuation"].replace("```", "` ` `").rstrip()
        sample_sections.append(
            f"### {sample['title']} — {sample['model']}\n\n"
            f"Prompt: `{sample['prompt_id']}`; seed {sample['seed']}.\n\n"
            f"```text\n{continuation}\n```"
        )
    decision_word = "Stop" if decision["decision"] == "stop" else "Continue"
    return f"""# V2 TextNCA comparison

## Conclusion

**The Transformer wins V2, and the result supports stopping further TextNCA investment in this project.** At the fixed 10,000-step budget, the Transformer achieved lower held-out loss, trained {comparison["transformer_training_speedup"]:.2f}× faster, and was no worse on either exact continuation diversity or cross-work collapse.

**Decision: {decision_word} TextNCA investment.** The predeclared rule is: {decision["rule"]} All four stop criteria evaluated to `{str(all(decision["criteria"].values())).lower()}`. Preserve the implementation and reports as a negative result; direct future modeling work to the matched Transformer baseline rather than another TextNCA training round.

Corpus V2 improved both models' teacher-forced loss, but the Transformer improved more: TextNCA {abs(version["textnca"]["test_loss_change_percent"]):.1f}% versus Transformer {abs(version["transformer"]["test_loss_change_percent"]):.1f}%.

This remains a single-seed, approximately 27M-parameter comparison, not a universal claim about every neural cellular automaton architecture.

![Validation loss curves](training-curves.svg)

## Controlled comparison

- Training commit: `{results["identity"]["training_commit"]}`
- Run: `{results["identity"]["run_id"]}`
- Corpus SHA-256: `{results["dataset"]["corpus_sha256"]}`
- Corpus: {results["dataset"]["work_count"]} works; {test_works} held-out test works
- Seen tokens per model: {textnca["seen_tokens"]:,}
- Parameters: TextNCA {textnca["parameters"]:,}; Transformer {transformer["parameters"]:,} ({comparison["parameter_difference_percent"]:.3f}% difference)
- Shared seed: {results["protocol"]["training_seed"]}
- Shared context: {results["dataset"]["block_length"]} tokens

## Teacher-forced results

| Metric | TextNCA | Transformer |
|---|---:|---:|
| Final train loss | {textnca["final_train_loss"]:.4f} | {transformer["final_train_loss"]:.4f} |
| Final validation loss | {textnca["final_validation"]["loss"]:.4f} | {transformer["final_validation"]["loss"]:.4f} |
| Best validation loss (step) | {textnca["best_validation"]["loss"]:.4f} ({textnca["best_validation"]["step"]:,}) | {transformer["best_validation"]["loss"]:.4f} ({transformer["best_validation"]["step"]:,}) |
| Test loss | {textnca["test"]["loss"]:.4f} | {transformer["test"]["loss"]:.4f} |
| Test perplexity | {textnca["test"]["perplexity"]:.3f} | {transformer["test"]["perplexity"]:.3f} |
| End-to-end runtime | {textnca["elapsed_seconds"] / 60:.1f} min | {transformer["elapsed_seconds"] / 60:.1f} min |
| Test throughput | {textnca["test"]["tokens_per_second"]:,.0f} tok/s | {transformer["test"]["tokens_per_second"]:,.0f} tok/s |

TextNCA's final test loss was {comparison["textnca_test_loss_relative_percent"]:.1f}% higher. The equal-budget comparison uses the predeclared final checkpoints rather than selecting a post-hoc checkpoint.

## Held-out generation

The prompt manifest was fixed before inspecting output: early, middle, and late completed-turn boundaries from every held-out test work, with identical canonical token prompts, generation seed, temperature, length cap, and EOS policy for both models. {results["protocol"]["truncated_prompt_count"]} of {len(results["prompt_manifest"])} canonical prompts exceeded the 512-token model context; generation used the final 512-token suffix. `results.json` preserves every effective suffix as token IDs plus a SHA-256 hash.

| Macro-average across held-out works | TextNCA | Transformer |
|---|---:|---:|
| Prefix-valid continuation rate | {_fmt(tn_gen["prefix_valid"])} | {_fmt(tr_gen["prefix_valid"])} |
| Complete serialized work rate | {_fmt(tn_gen["structure_valid"])} | {_fmt(tr_gen["structure_valid"])} |
| Length-censored rate | {_fmt(tn_gen["censored_by_length"])} | {_fmt(tr_gen["censored_by_length"])} |
| Turn completion rate | {_fmt(tn_gen["turn_completion_rate"])} | {_fmt(tr_gen["turn_completion_rate"])} |
| Speaker-field format rate | {_fmt(tn_gen["speaker_format_rate"])} | {_fmt(tr_gen["speaker_format_rate"])} |
| Allowed-speaker rate | {_fmt(tn_gen["speaker_consistency"])} | {_fmt(tr_gen["speaker_consistency"])} |
| Multi-speaker sample rate | {_fmt(tn_gen["multi_speaker_generation"])} | {_fmt(tr_gen["multi_speaker_generation"])} |
| Speaker transition rate | {_fmt(tn_gen["speaker_transition_rate"])} | {_fmt(tr_gen["speaker_transition_rate"])} |
| Token repeated-trigram rate | {_fmt(tn_gen["token_repeated_trigram_rate"])} | {_fmt(tr_gen["token_repeated_trigram_rate"])} |
| Character repeated-trigram rate | {_fmt(tn_gen["character_repeated_trigram_rate"])} | {_fmt(tr_gen["character_repeated_trigram_rate"])} |
| Word repeated-trigram rate | {_fmt(tn_gen["word_repeated_trigram_rate"])} | {_fmt(tr_gen["word_repeated_trigram_rate"])} |
| Exact unique-continuation rate | {_fmt(tn_cross["unique_continuation_rate"])} | {_fmt(tr_cross["unique_continuation_rate"])} |
| Cross-work character-trigram similarity | {_fmt(tn_cross["mean_cross_work_character_trigram_jaccard"])} | {_fmt(tr_cross["mean_cross_work_character_trigram_jaccard"])} |

A fixed-length output that remains a valid grammar prefix is censored rather than malformed. Speaker consistency checks only grammar-valid speaker fields against the prompted work. Cross-prompt exact uniqueness and character-trigram similarity distinguish within-sample repetition from prompt-insensitive collapse.

TextNCA retained a limited serialization advantage ({_fmt(tn_gen["prefix_valid"])} prefix-valid and {_fmt(tn_gen["speaker_format_rate"])} speaker-field formatting), while the Transformer completed turns more often ({_fmt(tr_gen["turn_completion_rate"])} versus {_fmt(tn_gen["turn_completion_rate"])}). Neither model generated an allowed held-out speaker or a complete serialized work. TextNCA also had lower within-sample token and character repeated-trigram rates; both models had zero word repeated-trigram rate. Those narrow advantages do not offset TextNCA's {_fmt(tn_cross["unique_continuation_rate"])} unique-continuation rate and {_fmt(tn_cross["mean_cross_work_character_trigram_jaccard"])} cross-work similarity.

## V1 to V2

| Change | TextNCA | Transformer |
|---|---:|---:|
| Test loss | {version["textnca"]["test_loss_change_percent"]:+.1f}% | {version["transformer"]["test_loss_change_percent"]:+.1f}% |
| Exact unique-continuation rate | {version["textnca"]["unique_continuation_rate_change"]:+.3f} | {version["transformer"]["unique_continuation_rate_change"]:+.3f} |
| Cross-work similarity | {version["textnca"]["cross_work_similarity_change"]:+.3f} | {version["transformer"]["cross_work_similarity_change"]:+.3f} |

Negative loss and cross-work-similarity changes are improvements; positive uniqueness changes are improvements.

## Predeclared representative samples

These are the early-boundary samples for every held-out work and both models. They were not selected for quality.

{"\n\n".join(sample_sections)}

## Limitations

- One training seed and one generation seed.
- {results["dataset"]["work_count"]} plays overall and {test_works} held-out test works.
- Near-parameter matching, not exact matching: TextNCA has {comparison["parameter_difference_percent"]:.3f}% more parameters.
- No architecture or optimizer ablations.
- Teacher-forced loss does not measure human preference or long-range dramatic coherence.
- Generation statistics are descriptive paired probes, not population-level significance claims.

## Portable verification and regeneration

The committed report is portable and verifiable from a clean checkout without the ignored training data:

```bash
uv sync --locked --dev
uv run pytest tests/test_v1_report.py tests/test_v2_report.py -q
```

Full regeneration additionally requires restoring the intentionally untracked Corpus V2 build and `data/runs/v2-d8cbe61` checkpoint tree. Before running the generator, verify the corpus SHA-256, training commit, and checkpoint hashes against `results.json`; the command fails if the paired checkpoint commits, seeds, datasets, or prompt manifests disagree.

```bash
uv run ncm-v2-report --run-root data/runs/v2-d8cbe61 --v1-results reports/v1/results.json --output-dir reports/v2
```

The repository does not distribute the roughly 55-million-parameter checkpoint pair. `results.json` is the durable research artifact: it contains the exact effective model-context token IDs and hashes, prompt manifest, checkpoint hashes, raw continuations, metrics, V1 deltas, decision rule, environment, and rendering protocol.
"""


def build_report(
    run_root: Path,
    v1_results_path: Path,
    output_dir: Path,
    *,
    seed: int = 123,
    temperature: float = 0.8,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Run paired generation probes and write the complete V2 report."""
    run_root = run_root.resolve()
    v1_results_path = v1_results_path.resolve()
    output_dir = output_dir.resolve()
    runner = json.loads((run_root / "runner.json").read_text(encoding="utf-8"))
    previous = json.loads(v1_results_path.read_text(encoding="utf-8"))
    training = _training_summary(run_root, MODEL_NAMES)
    samples, generation_summary, prompt_data = _generate_samples(
        run_root,
        seed=seed,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        model_names=MODEL_NAMES,
        include_effective_context=True,
    )
    dataset = prompt_data["dataset"]
    training_commit = _shared_training_commit(run_root, MODEL_NAMES)
    training_seed = _shared_training_seed(run_root, MODEL_NAMES)
    truncated_prompt_count = sum(
        prompt["prompt_tokens"] > dataset["block_length"] for prompt in prompt_data["prompts"]
    )
    results: dict[str, Any] = {
        "schema_version": 2,
        "identity": {
            "run_id": runner["run_id"],
            "training_commit": training_commit,
            "artifact_hashes": _artifact_hashes(run_root, MODEL_NAMES),
            "v1_results_sha256": hashlib.sha256(v1_results_path.read_bytes()).hexdigest(),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mlx": importlib.metadata.version("mlx"),
        },
        "protocol": {
            "training_seed": training_seed,
            "generation_seed": seed,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "prompt_selection": "early, middle, and late completed-turn boundary per test work",
            "model_context_policy": "final block_length token suffix of each canonical full prefix",
            "truncated_prompt_count": truncated_prompt_count,
            "aggregation": "sample metrics, then work means, then macro-average across works",
            "determinism_check": "first prompt replayed per model with exact token equality",
        },
        "dataset": dataset,
        "prompt_manifest": prompt_data["prompts"],
        "training": training,
        "generation": {"summary": generation_summary, "samples": samples},
    }
    results["v1_to_v2"] = _version_comparison(previous, results)
    results["textnca_investment_decision"] = _investment_decision(results)
    _atomic_write(
        output_dir / "results.json",
        (json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _atomic_write(
        output_dir / "training-curves.svg",
        _curve_svg(training, MODEL_NAMES).encode("utf-8"),
    )
    _atomic_write(output_dir / "README.md", _markdown(results).encode("utf-8"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the reproducible V2 comparison report.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--v1-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.temperature <= 0 or args.max_new_tokens < 1:
        parser.error("temperature and max-new-tokens must be positive")
    try:
        results = build_report(
            args.run_root,
            args.v1_results,
            args.output_dir,
            seed=args.seed,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, f"ncm-v2-report: {error}\n")
    print(
        json.dumps(
            {
                "decision": results["textnca_investment_decision"]["decision"],
                "report": str(args.output_dir / "README.md"),
                "samples": len(results["generation"]["samples"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
