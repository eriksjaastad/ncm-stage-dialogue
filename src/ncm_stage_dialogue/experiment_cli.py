"""Command-line interface for reproducible TextNCA comparison experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from ncm_stage_dialogue.experiment_data import ByteTokenizer, build_experiment_dataset
from ncm_stage_dialogue.models import ModelConfig, build_model, parameter_count
from ncm_stage_dialogue.training import (
    TrainingConfig,
    TrainingError,
    build_optimizer,
    evaluate,
    generate,
    load_checkpoint,
    repeated_trigram_rate,
    structure_metrics,
    train,
)


def _load_config(path: Path) -> tuple[ModelConfig, TrainingConfig, Path, Path]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"model", "training", "corpus", "run_dir"}:
        raise TrainingError("config requires model, training, corpus, and run_dir")
    model_config = ModelConfig.from_dict(raw["model"])
    training_config = TrainingConfig(**raw["training"])
    training_config.validate()
    corpus = Path(raw["corpus"])
    run_dir = Path(raw["run_dir"])
    return model_config, training_config, corpus, run_dir


def _latest_checkpoint(run_dir: Path) -> Path:
    latest = run_dir / "latest"
    if not latest.is_file():
        raise TrainingError(f"run has no latest checkpoint: {run_dir}")
    checkpoint = run_dir / latest.read_text(encoding="utf-8").strip()
    if not checkpoint.is_dir():
        raise TrainingError(f"latest checkpoint does not exist: {checkpoint}")
    return checkpoint


def _load_model_for_checkpoint(
    config_path: Path, checkpoint: Path | None
) -> tuple[Any, ModelConfig, TrainingConfig, Any, Path]:
    model_config, training_config, corpus, run_dir = _load_config(config_path)
    dataset = build_experiment_dataset(corpus, model_config.context_length)
    model = build_model(model_config)
    optimizer = build_optimizer(model_config, training_config)
    selected = checkpoint or _latest_checkpoint(run_dir)
    load_checkpoint(
        selected,
        model,
        optimizer,
        model_config,
        training_config,
        dataset.manifest,
    )
    return model, model_config, training_config, dataset, selected


def main() -> int:
    """Inspect, train, evaluate, or generate from a captured experiment config."""
    parser = argparse.ArgumentParser(description="Run the MLX TextNCA comparison experiment.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("inspect", "train", "evaluate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", type=Path, required=True)
        if command in {"train", "evaluate"}:
            subparser.add_argument("--checkpoint", type=Path)
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--config", type=Path, required=True)
    generate_parser.add_argument("--checkpoint", type=Path)
    generate_parser.add_argument("--prompt", required=True)
    generate_parser.add_argument("--max-new-tokens", type=int, default=128)
    generate_parser.add_argument("--seed", type=int, default=123)
    generate_parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    try:
        model_config, training_config, corpus, run_dir = _load_config(args.config)
        dataset = build_experiment_dataset(corpus, model_config.context_length)
        if args.command == "inspect":
            mx.random.seed(training_config.seed)
            model = build_model(model_config)
            print(
                json.dumps(
                    {
                        "model": model_config.model_type,
                        "parameters": parameter_count(model),
                        "vocab_size": model_config.vocab_size,
                        "context_length": model_config.context_length,
                        "dataset": dataset.manifest,
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "train":
            result = train(
                dataset,
                model_config,
                training_config,
                run_dir,
                Path.cwd(),
                args.checkpoint,
            )
            print(json.dumps(result, sort_keys=True))
            return 0
        model, _, _, dataset, checkpoint = _load_model_for_checkpoint(args.config, args.checkpoint)
        if args.command == "evaluate":
            result: dict[str, Any] = {
                split: evaluate(model, dataset.blocks[split], training_config.batch_size)
                for split in ("validation", "test")
            }
            result["checkpoint"] = str(checkpoint)
            print(json.dumps(result, sort_keys=True))
            return 0
        tokenizer = ByteTokenizer()
        prompt = tokenizer.encode_text(args.prompt)
        output = generate(
            model,
            prompt,
            context_length=model_config.context_length,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
            temperature=args.temperature,
        )
        continuation = output[len(prompt) :]
        declared_speakers = {
            speaker for work in dataset.manifest["works"] for speaker in work["speakers"]
        }
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "prompt": args.prompt,
                    "continuation_tokens": continuation,
                    "continuation": tokenizer.decode_display(continuation),
                    "repeated_trigram_rate": repeated_trigram_rate(continuation),
                    "structure": structure_metrics(continuation, declared_speakers),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError, TrainingError) as error:
        parser.exit(1, f"ncm-experiment: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
