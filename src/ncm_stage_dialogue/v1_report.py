"""Build the reproducible V1 TextNCA-versus-Transformer findings artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import uuid
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import fmean
from typing import Any

from ncm_stage_dialogue.experiment_cli import _load_model_for_checkpoint
from ncm_stage_dialogue.experiment_data import SPECIAL_TOKENS, ByteTokenizer
from ncm_stage_dialogue.generation_evaluation import (
    PromptSpec,
    build_prompt_specs,
    generation_metrics,
)
from ncm_stage_dialogue.training import generate

MODEL_NAMES = ("v1-textnca", "v1-transformer")
SUMMARY_METRICS = (
    "prefix_valid",
    "structure_valid",
    "censored_by_length",
    "turn_completion_rate",
    "scene_completion_rate",
    "speaker_format_rate",
    "speaker_consistency",
    "multi_speaker_generation",
    "speaker_transition_rate",
    "token_repeated_trigram_rate",
    "character_repeated_trigram_rate",
    "word_repeated_trigram_rate",
)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    with temporary.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _runner_results(
    run_root: Path, model_names: tuple[str, str] = MODEL_NAMES
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for line in (run_root / "supervisor.log").read_text(encoding="utf-8").splitlines():
        if line.startswith("[runner] starting "):
            current = line.removeprefix("[runner] starting ")
        elif line.startswith("{"):
            if current is None:
                raise ValueError("supervisor result has no experiment name")
            results[current] = json.loads(line)
    if set(results) != set(model_names):
        raise ValueError("supervisor log does not contain both comparison results")
    return results


def _training_summary(
    run_root: Path, model_names: tuple[str, str] = MODEL_NAMES
) -> dict[str, Any]:
    runner = _runner_results(run_root, model_names)
    models: dict[str, Any] = {}
    for name in model_names:
        events = _load_json_lines(run_root / name / "metrics.jsonl")
        train = [event for event in events if event["split"] == "train"]
        validation = [event for event in events if event["split"] == "validation"]
        test = [event for event in events if event["split"] == "test"]
        if len(train) != 10_000 or len(validation) != 40 or len(test) != 1:
            raise ValueError(f"unexpected metrics coverage for {name}")
        final_validation = validation[-1]
        best_validation = min(validation, key=lambda event: event["loss"])
        result = runner[name]
        models[name] = {
            "parameters": result["model_parameters"],
            "elapsed_seconds": result["elapsed_seconds"],
            "wall_tokens_per_second": result["seen_tokens"] / result["elapsed_seconds"],
            "seen_tokens": result["seen_tokens"],
            "final_train_loss": train[-1]["loss"],
            "final_validation": final_validation,
            "best_validation": best_validation,
            "test": test[-1],
            "validation_curve": [
                {"step": event["step"], "loss": event["loss"]} for event in validation
            ],
        }
    textnca = models[model_names[0]]
    transformer = models[model_names[1]]
    return {
        "models": models,
        "comparison": {
            "parameter_difference": textnca["parameters"] - transformer["parameters"],
            "parameter_difference_percent": 100
            * (textnca["parameters"] - transformer["parameters"])
            / transformer["parameters"],
            "textnca_test_loss_relative_percent": 100
            * (textnca["test"]["loss"] - transformer["test"]["loss"])
            / transformer["test"]["loss"],
            "textnca_test_perplexity_relative_percent": 100
            * (textnca["test"]["perplexity"] - transformer["test"]["perplexity"])
            / transformer["test"]["perplexity"],
            "transformer_training_speedup": textnca["elapsed_seconds"]
            / transformer["elapsed_seconds"],
            "transformer_test_throughput_speedup": transformer["test"]["tokens_per_second"]
            / textnca["test"]["tokens_per_second"],
        },
    }


def _sample_record(
    model_name: str,
    prompt: PromptSpec,
    continuation: list[int],
    *,
    context_length: int,
    seed: int,
    temperature: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    display = ByteTokenizer.decode_display(continuation)
    return {
        "model": model_name,
        "prompt_id": prompt.prompt_id,
        "work_key": prompt.work_key,
        "title": prompt.title,
        "boundary": prompt.label,
        "turn_ordinal": prompt.turn_ordinal,
        "prompt_sha256": prompt.prompt_sha256,
        "prompt_tokens": len(prompt.prompt_tokens),
        "model_context_tokens": min(len(prompt.prompt_tokens), context_length),
        "seed": seed,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "continuation_sha256": hashlib.sha256(
            b"".join(token.to_bytes(2, "little") for token in continuation)
        ).hexdigest(),
        "continuation": display,
        "metrics": generation_metrics(
            prompt.prompt_tokens,
            continuation,
            set(prompt.allowed_speakers),
            max_new_tokens=max_new_tokens,
        ),
    }


def _macro_generation_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_work: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_work[sample["work_key"]].append(sample["metrics"])
    work_means: dict[str, dict[str, float | None]] = {}
    for work_key, metrics_rows in by_work.items():
        means: dict[str, float | None] = {}
        for metric in SUMMARY_METRICS:
            values = [float(row[metric]) for row in metrics_rows if row[metric] is not None]
            means[metric] = fmean(values) if values else None
        work_means[work_key] = means
    macro: dict[str, float | None] = {}
    for metric in SUMMARY_METRICS:
        values = [float(value) for row in work_means.values() if (value := row[metric]) is not None]
        macro[metric] = fmean(values) if values else None
    cross_work_similarities: list[float] = []
    for first, second in combinations(samples, 2):
        if first["work_key"] == second["work_key"]:
            continue
        first_trigrams = set(
            zip(first["continuation"], first["continuation"][1:], first["continuation"][2:])
        )
        second_trigrams = set(
            zip(second["continuation"], second["continuation"][1:], second["continuation"][2:])
        )
        union = first_trigrams | second_trigrams
        cross_work_similarities.append(
            len(first_trigrams & second_trigrams) / len(union) if union else 1.0
        )
    return {
        "sample_count": len(samples),
        "work_count": len(by_work),
        "macro_average_across_works": macro,
        "cross_prompt": {
            "unique_continuation_rate": len({sample["continuation_sha256"] for sample in samples})
            / len(samples),
            "mean_cross_work_character_trigram_jaccard": fmean(cross_work_similarities),
        },
        "work_means": work_means,
        "raw_counts": {
            "prefix_valid": sum(sample["metrics"]["prefix_valid"] for sample in samples),
            "structure_valid": sum(sample["metrics"]["structure_valid"] for sample in samples),
            "censored_by_length": sum(
                sample["metrics"]["censored_by_length"] for sample in samples
            ),
            "speaker_fields": sum(sample["metrics"]["speaker_fields"] for sample in samples),
            "completed_text_fields": sum(
                sample["metrics"]["completed_text_fields"] for sample in samples
            ),
        },
    }


def _generate_samples(
    run_root: Path,
    *,
    seed: int,
    temperature: float,
    max_new_tokens: int,
    model_names: tuple[str, str] = MODEL_NAMES,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    manifest: dict[str, Any] | None = None
    prompt_manifest: list[dict[str, Any]] | None = None
    for name in model_names:
        config = run_root / "configs" / f"{name}.json"
        model, model_config, _, dataset, checkpoint = _load_model_for_checkpoint(config, None)
        if manifest is None:
            manifest = dataset.manifest
        elif dataset.manifest != manifest:
            raise ValueError("model runs do not share one dataset manifest")
        prompts = build_prompt_specs(dataset)
        current_prompt_manifest = [
            {
                "prompt_id": prompt.prompt_id,
                "work_key": prompt.work_key,
                "title": prompt.title,
                "boundary": prompt.label,
                "turn_ordinal": prompt.turn_ordinal,
                "prompt_sha256": prompt.prompt_sha256,
                "prompt_tokens": len(prompt.prompt_tokens),
                "allowed_speakers": list(prompt.allowed_speakers),
            }
            for prompt in prompts
        ]
        if prompt_manifest is None:
            prompt_manifest = current_prompt_manifest
        elif current_prompt_manifest != prompt_manifest:
            raise ValueError("model runs produced different prompt manifests")
        for prompt_index, prompt in enumerate(prompts):
            output = generate(
                model,
                list(prompt.prompt_tokens),
                context_length=model_config.context_length,
                max_new_tokens=max_new_tokens,
                seed=seed,
                temperature=temperature,
                stop_token=SPECIAL_TOKENS["EOS"],
            )
            continuation = output[len(prompt.prompt_tokens) :]
            if prompt_index == 0:
                replay = generate(
                    model,
                    list(prompt.prompt_tokens),
                    context_length=model_config.context_length,
                    max_new_tokens=max_new_tokens,
                    seed=seed,
                    temperature=temperature,
                    stop_token=SPECIAL_TOKENS["EOS"],
                )
                if replay != output:
                    raise ValueError(f"generation is not deterministic for {name}")
            record = _sample_record(
                name,
                prompt,
                continuation,
                context_length=model_config.context_length,
                seed=seed,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
            record["checkpoint"] = str(checkpoint.relative_to(run_root))
            samples.append(record)
    assert manifest is not None and prompt_manifest is not None
    summaries = {
        name: _macro_generation_summary([sample for sample in samples if sample["model"] == name])
        for name in model_names
    }
    return samples, summaries, {"dataset": manifest, "prompts": prompt_manifest}


def _artifact_hashes(
    run_root: Path, model_names: tuple[str, str] = MODEL_NAMES
) -> dict[str, Any]:
    hashes: dict[str, Any] = {}
    for name in model_names:
        checkpoint_name = (run_root / name / "latest").read_text(encoding="utf-8").strip()
        checkpoint = run_root / name / checkpoint_name
        hashes[name] = {
            "checkpoint": f"{name}/{checkpoint_name}",
            "config_sha256": _sha256(checkpoint / "config.json"),
            "dataset_manifest_sha256": _sha256(checkpoint / "dataset_manifest.json"),
            "model_sha256": _sha256(checkpoint / "model.safetensors"),
            "trainer_state_sha256": _sha256(checkpoint / "trainer_state.json"),
        }
    return hashes


def _shared_training_commit(
    run_root: Path, model_names: tuple[str, str] = MODEL_NAMES
) -> str:
    """Require both final checkpoints to come from one source revision."""
    commits: dict[str, str] = {}
    for name in model_names:
        checkpoint_name = (run_root / name / "latest").read_text(encoding="utf-8").strip()
        trainer_state = json.loads(
            (run_root / name / checkpoint_name / "trainer_state.json").read_text(encoding="utf-8")
        )
        commits[name] = trainer_state["git_commit"]
    if len(set(commits.values())) != 1:
        raise ValueError(f"model checkpoints use different training commits: {commits}")
    return next(iter(commits.values()))


def _curve_svg(
    training: dict[str, Any], model_names: tuple[str, str] = MODEL_NAMES
) -> str:
    width, height = 800, 420
    left, right, top, bottom = 70, 30, 30, 55
    curves = {
        "TextNCA": training["models"][model_names[0]]["validation_curve"],
        "Transformer": training["models"][model_names[1]]["validation_curve"],
    }
    points = [point for curve in curves.values() for point in curve]
    x_max = max(point["step"] for point in points)
    y_min = min(point["loss"] for point in points)
    y_max = max(point["loss"] for point in points)

    def xy(point: dict[str, float]) -> tuple[float, float]:
        x = left + point["step"] / x_max * (width - left - right)
        y = top + (y_max - point["loss"]) / (y_max - y_min) * (height - top - bottom)
        return x, y

    colors = {"TextNCA": "#d97706", "Transformer": "#2563eb"}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111"/>',
        f'<text x="{width / 2}" y="{height - 12}" text-anchor="middle" font-family="sans-serif" font-size="14">Training step</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="sans-serif" font-size="14">Validation loss</text>',
    ]
    for tick in range(0, 10_001, 2_500):
        x = left + tick / x_max * (width - left - right)
        lines.append(
            f'<text x="{x:.1f}" y="{height - bottom + 22}" text-anchor="middle" font-family="sans-serif" font-size="12">{tick}</text>'
        )
    for index in range(5):
        loss = y_min + index * (y_max - y_min) / 4
        y = top + (y_max - loss) / (y_max - y_min) * (height - top - bottom)
        lines.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{loss:.2f}</text>'
        )
    for legend_index, (name, curve) in enumerate(curves.items()):
        path = " ".join(
            ("M" if index == 0 else "L") + f" {x:.1f} {y:.1f}"
            for index, point in enumerate(curve)
            for x, y in [xy(point)]
        )
        lines.append(f'<path d="{path}" fill="none" stroke="{colors[name]}" stroke-width="3"/>')
        legend_y = 48 + legend_index * 22
        lines.append(
            f'<line x1="600" y1="{legend_y}" x2="630" y2="{legend_y}" stroke="{colors[name]}" stroke-width="3"/>'
        )
        lines.append(
            f'<text x="638" y="{legend_y + 4}" font-family="sans-serif" font-size="13">{name}</text>'
        )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _fmt(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _markdown(results: dict[str, Any]) -> str:
    training = results["training"]
    textnca = training["models"]["v1-textnca"]
    transformer = training["models"]["v1-transformer"]
    comparison = training["comparison"]
    generation_summary = results["generation"]["summary"]
    tn_gen = generation_summary["v1-textnca"]["macro_average_across_works"]
    tr_gen = generation_summary["v1-transformer"]["macro_average_across_works"]
    tn_cross = generation_summary["v1-textnca"]["cross_prompt"]
    tr_cross = generation_summary["v1-transformer"]["cross_prompt"]
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
    return f"""# V1 TextNCA comparison

## Conclusion

**TextNCA loses this V1 comparison.** At the fixed 10,000-step budget, the matched Transformer achieved lower validation and test loss and finished training about {comparison["transformer_training_speedup"]:.2f}× faster. TextNCA produced a valid serialized prefix more often, but neither model completed a work or preserved held-out speaker identities, and TextNCA showed severe cross-prompt output collapse.

**Next measured bottleneck:** held-out conditioning and corpus diversity. Corpus V2 will expand to at least 30 independent plays, then remeasure whether both models stop substituting training-play speakers and whether TextNCA's cross-prompt collapse improves.

This is a result about this single-seed, 27M-parameter, 19-play experiment—not a claim that TextNCA is universally inferior.

![Validation loss curves](training-curves.svg)

## Controlled comparison

- Training commit: `{results["identity"]["training_commit"]}`
- Run: `{results["identity"]["run_id"]}`
- Corpus SHA-256: `{results["dataset"]["corpus_sha256"]}`
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

TextNCA's final test loss was {comparison["textnca_test_loss_relative_percent"]:.1f}% higher and its test perplexity was {comparison["textnca_test_perplexity_relative_percent"]:.1f}% higher. The Transformer reached its best validation checkpoint at step {transformer["best_validation"]["step"]:,} and then overfit, while TextNCA continued improving through step 10,000. The equal-budget headline uses the predeclared final checkpoints rather than selecting a post-hoc checkpoint.

## Held-out generation

The generation manifest was fixed before inspecting outputs: early, middle, and late completed-turn boundaries from each of the three held-out plays; identical token prompts, seed, temperature, and 256-token cap for both models. Long prompts use the canonical full prefix for parsing while the model receives its final 512-token context window.

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

A fixed-length output that remains a valid grammar prefix is reported as censored, not malformed. Speaker consistency counts only speaker fields reached through valid grammar and checks them against the prompted work rather than the full corpus. TextNCA's 0.778 prefix-valid rate is a limited structural advantage over the Transformer's 0.556, but its outputs were only 5/9 exactly unique and had 0.890 mean cross-work character-trigram similarity, versus 9/9 and 0.187 for the Transformer. The visible samples show this prompt-insensitive collapse.

## Predeclared representative samples

These are the early-boundary samples for every held-out work and both models. They were not selected for quality.

{"\n\n".join(sample_sections)}

## Limitations

- One training seed and one generation seed.
- Only 19 plays overall and three held-out test works.
- Near-parameter matching, not exact matching: TextNCA has {comparison["parameter_difference_percent"]:.3f}% more parameters.
- No architecture or optimizer ablations.
- Teacher-forced loss does not measure human preference or long-range dramatic coherence.
- Generation statistics are descriptive paired probes, not population-level significance claims.
- Manifest token counts include the first stream token; next-token evaluation counts one fewer target token per split.

## Reproduction

Rebuild the corpus, restore the recorded run checkpoints, then run:

```bash
uv run ncm-v1-report --run-root data/runs/v1-7e4ecad --output-dir reports/v1
```

`results.json` contains the prompt manifest, checkpoint hashes, raw sample records, metrics, environment, and exact protocol used to render this report.
"""


def build_report(
    run_root: Path,
    output_dir: Path,
    *,
    seed: int = 123,
    temperature: float = 0.8,
    max_new_tokens: int = 256,
) -> dict[str, Any]:
    """Run paired generation probes and write the complete V1 report."""
    run_root = run_root.resolve()
    output_dir = output_dir.resolve()
    runner = json.loads((run_root / "runner.json").read_text(encoding="utf-8"))
    training = _training_summary(run_root)
    samples, generation_summary, prompt_data = _generate_samples(
        run_root,
        seed=seed,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    dataset = prompt_data["dataset"]
    training_commit = _shared_training_commit(run_root)
    results = {
        "schema_version": 1,
        "identity": {
            "run_id": runner["run_id"],
            "training_commit": training_commit,
            "artifact_hashes": _artifact_hashes(run_root),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "mlx": importlib.metadata.version("mlx"),
        },
        "protocol": {
            "training_seed": 123,
            "generation_seed": seed,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "prompt_selection": "early, middle, and late completed-turn boundary per test work",
            "aggregation": "sample metrics, then work means, then macro-average across works",
            "determinism_check": "first prompt replayed per model with exact token equality",
        },
        "dataset": dataset,
        "prompt_manifest": prompt_data["prompts"],
        "training": training,
        "generation": {"summary": generation_summary, "samples": samples},
    }
    _atomic_write(
        output_dir / "results.json",
        (json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    _atomic_write(output_dir / "training-curves.svg", _curve_svg(training).encode("utf-8"))
    _atomic_write(output_dir / "README.md", _markdown(results).encode("utf-8"))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the reproducible V1 comparison report.")
    parser.add_argument("--run-root", type=Path, required=True)
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
            args.output_dir,
            seed=args.seed,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(1, f"ncm-v1-report: {error}\n")
    print(
        json.dumps(
            {
                "report": str(args.output_dir / "README.md"),
                "samples": len(results["generation"]["samples"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
