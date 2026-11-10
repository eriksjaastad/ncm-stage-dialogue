from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlx.core as mx
import pytest

import ncm_stage_dialogue.training as training_module
from ncm_stage_dialogue.corpus import load_sources
from ncm_stage_dialogue.experiment_cli import load_experiment_config
from ncm_stage_dialogue.experiment_data import (
    SPECIAL_TOKENS,
    ByteTokenizer,
    ExperimentDataError,
    build_experiment_dataset,
    pack_stream,
    serialize_work,
)
from ncm_stage_dialogue.models import (
    ModelConfig,
    build_model,
    causal_local_mask,
    parameter_count,
)
from ncm_stage_dialogue.training import (
    TrainingConfig,
    TrainingError,
    repeated_trigram_rate,
    structure_metrics,
    train,
)


def scene(
    *,
    source_id: str = "fixture",
    title: str = "Fixture",
    split: str = "train",
    scene_id: str = "act-1",
    act: int = 1,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "title": title,
        "author": "Test Author",
        "scene_id": scene_id,
        "act": act,
        "split": split,
        "stage_directions": ["A room."],
        "turns": [
            {
                "speaker": "ALICE",
                "text": "Hello.",
                "stage_directions": ["quietly"],
            }
        ],
        "provenance": {"synthetic": True},
    }


def write_fixture_corpus(path: Path) -> None:
    rows = [
        scene(source_id="train", title="Train", split="train"),
        scene(source_id="validation", title="Validation", split="validation"),
        scene(source_id="test", title="Test", split="test"),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_byte_tokenizer_round_trips_all_bytes_and_separates_structure() -> None:
    tokenizer = ByteTokenizer()
    text = bytes(range(256)).decode("latin-1")
    encoded = tokenizer.encode_text(text)

    assert tokenizer.decode_text(encoded) == text
    assert max(encoded) < 256
    assert min(SPECIAL_TOKENS.values()) == 256
    with pytest.raises(ExperimentDataError, match="raw byte"):
        tokenizer.decode_text([SPECIAL_TOKENS["BOS"]])


def test_work_serialization_is_canonical_and_preserves_structure() -> None:
    tokenizer = ByteTokenizer()
    record = scene()

    first = serialize_work([record], tokenizer)
    second = serialize_work([record], tokenizer)

    assert first == second
    assert first[:2] == [SPECIAL_TOKENS["BOS"], SPECIAL_TOKENS["WORK"]]
    assert SPECIAL_TOKENS["SPEAKER"] in first
    assert SPECIAL_TOKENS["DIRECTION"] in first
    assert first[-2:] == [SPECIAL_TOKENS["WORK_END"], SPECIAL_TOKENS["EOS"]]


def test_dataset_keeps_complete_works_in_independent_split_streams(tmp_path: Path) -> None:
    corpus = tmp_path / "scenes.jsonl"
    write_fixture_corpus(corpus)

    dataset = build_experiment_dataset(corpus, block_length=16)

    assert dataset.manifest["work_count"] == 3
    assert dataset.manifest["split_counts"]["train"]["works"] == 1
    assert len(set(dataset.manifest["split_token_sha256"].values())) == 3
    assert all(dataset.blocks[split] for split in ("train", "validation", "test"))


def test_dataset_rejects_a_work_spanning_splits(tmp_path: Path) -> None:
    corpus = tmp_path / "scenes.jsonl"
    rows = [
        scene(scene_id="act-1", act=1, split="train"),
        scene(scene_id="act-2", act=2, split="test"),
    ]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(ExperimentDataError, match="complete work spans splits"):
        build_experiment_dataset(corpus, block_length=16)


def test_stream_packing_shifts_targets_and_masks_padding() -> None:
    block = pack_stream((1, 2, 3), block_length=4)[0]

    assert block.inputs == (1, 2, 3, SPECIAL_TOKENS["PAD"])
    assert block.targets == (2, 3, SPECIAL_TOKENS["PAD"], SPECIAL_TOKENS["PAD"])
    assert block.loss_mask == (1.0, 1.0, 0.0, 0.0)


def test_local_attention_mask_is_causal_and_windowed() -> None:
    mask = causal_local_mask(length=6, window=3)
    mx.eval(mask)

    assert float(mask[4, 4].item()) == 0.0
    assert float(mask[4, 2].item()) == 0.0
    assert float(mask[4, 1].item()) == float("-inf")
    assert float(mask[2, 3].item()) == float("-inf")


@pytest.mark.parametrize("model_type", ["textnca", "transformer"])
def test_models_emit_causal_language_model_logits(model_type: str) -> None:
    config = ModelConfig(
        model_type=model_type,
        context_length=8,
        hidden_size=16,
        num_heads=4,
        ffn_size=32,
        num_layers=2,
    )
    mx.random.seed(7)
    model = build_model(config)
    original = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    changed_future = mx.array([[1, 2, 9, 10]], dtype=mx.int32)

    first = model(original)
    second = model(changed_future)
    mx.eval(first, second)

    assert first.shape == (1, 4, 271)
    assert bool(mx.allclose(first[:, :2], second[:, :2], atol=1e-5, rtol=1e-5).item())


def test_smoke_configs_are_parameter_matched() -> None:
    repository = Path(__file__).parents[1]
    configs = []
    for filename in ("smoke-textnca.json", "smoke-transformer.json"):
        raw = json.loads((repository / "configs" / filename).read_text(encoding="utf-8"))
        config = ModelConfig.from_dict(raw["model"])
        configs.append(parameter_count(build_model(config)))

    difference = abs(configs[0] - configs[1]) / max(configs)
    assert difference < 0.01
    primary = json.loads((repository / "configs" / "v1-textnca.json").read_text(encoding="utf-8"))
    assert primary["model"]["windows"] == [8, 32, 128]
    assert primary["model"]["iterations_per_stage"] == 4


def test_frozen_manifest_has_nineteen_complete_work_splits() -> None:
    repository = Path(__file__).parents[1]
    sources = load_sources(repository / "sources" / "plays.json")
    work_splits: list[set[str]] = []
    for source in sources:
        works = source.metadata.get("works")
        if works is not None:
            work_splits.extend({work["split"]} for work in works)
        else:
            work_splits.append(set(source.scene_splits.values()))

    assert len(work_splits) == 19
    assert all(len(splits) == 1 for splits in work_splits)


def test_repetition_metric_handles_short_and_repeated_sequences() -> None:
    assert repeated_trigram_rate([1, 2]) == 0.0
    assert repeated_trigram_rate([1, 2, 3, 1, 2, 3]) > 0.0


def test_structure_metrics_validate_declared_speakers_and_completion() -> None:
    tokens = [
        SPECIAL_TOKENS["BOS"],
        SPECIAL_TOKENS["WORK"],
        *ByteTokenizer.encode_text("fixture"),
        SPECIAL_TOKENS["FIELD"],
        *ByteTokenizer.encode_text("Fixture"),
        SPECIAL_TOKENS["FIELD"],
        *ByteTokenizer.encode_text("Test Author"),
        SPECIAL_TOKENS["SCENE"],
        *ByteTokenizer.encode_text("act-1"),
        SPECIAL_TOKENS["FIELD"],
        *ByteTokenizer.encode_text("1"),
        SPECIAL_TOKENS["TURN"],
        SPECIAL_TOKENS["SPEAKER"],
        *ByteTokenizer.encode_text("ALICE"),
        SPECIAL_TOKENS["TEXT"],
        *ByteTokenizer.encode_text("Hello."),
        SPECIAL_TOKENS["TURN_END"],
        SPECIAL_TOKENS["SCENE_END"],
        SPECIAL_TOKENS["WORK_END"],
        SPECIAL_TOKENS["EOS"],
    ]

    metrics = structure_metrics(tokens, {"ALICE"})

    assert metrics["structure_valid"] is True
    assert metrics["speaker_consistency"] == 1.0
    assert metrics["turn_completion_rate"] == 1.0

    without_text_marker = list(tokens)
    without_text_marker.remove(SPECIAL_TOKENS["TEXT"])
    out_of_order = list(tokens)
    turn_end = out_of_order.index(SPECIAL_TOKENS["TURN_END"])
    scene_end = out_of_order.index(SPECIAL_TOKENS["SCENE_END"])
    out_of_order[turn_end], out_of_order[scene_end] = (
        out_of_order[scene_end],
        out_of_order[turn_end],
    )
    assert structure_metrics(without_text_marker, {"ALICE"})["structure_valid"] is False
    assert structure_metrics(out_of_order, {"ALICE"})["structure_valid"] is False


def test_config_paths_resolve_from_repository_not_caller(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / ".git").mkdir()
    (repository / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    config = repository / "configs" / "experiment.json"
    config.parent.mkdir()
    payload = {
        "model": {
            "model_type": "transformer",
            "context_length": 8,
            "hidden_size": 16,
            "num_heads": 4,
            "ffn_size": 24,
        },
        "training": {
            "train_steps": 1,
            "warmup_steps": 0,
            "eval_interval": 1,
            "checkpoint_interval": 1,
        },
        "corpus": "data/scenes.jsonl",
        "run_dir": "data/runs/test",
    }
    config.write_text(json.dumps(payload), encoding="utf-8")

    _, _, corpus, run_dir, found_repository = load_experiment_config(config)

    assert found_repository == repository
    assert corpus == repository / "data" / "scenes.jsonl"
    assert run_dir == repository / "data" / "runs" / "test"
    corpus.parent.mkdir()
    write_fixture_corpus(corpus)
    caller = tmp_path / "caller"
    caller.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ncm_stage_dialogue.experiment_cli",
            "inspect",
            "--config",
            str(config),
        ],
        cwd=caller,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert json.loads(result.stdout)["model"] == "transformer"


def train_metric(step: int, *, seen_tokens: int | None = None) -> dict[str, object]:
    return {
        "step": step,
        "split": "train",
        "loss": 1.0,
        "tokens": 7,
        "gradient_norm": 0.5,
        "seen_tokens": seen_tokens if seen_tokens is not None else step * 7,
    }


@pytest.mark.parametrize(
    "events",
    [
        [{**train_metric(1), "step": True}],
        [train_metric(1), train_metric(1, seen_tokens=14)],
        [train_metric(1), train_metric(3, seen_tokens=14)],
        [{**train_metric(1), "split": "unknown"}],
        [{**train_metric(1), "loss": "bad"}],
        [train_metric(1), train_metric(2, seen_tokens=7)],
    ],
)
def test_metrics_reconciliation_rejects_semantic_corruption(
    tmp_path: Path, events: list[dict[str, object]]
) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    with pytest.raises(TrainingError):
        training_module._reconcile_metrics(metrics, 1)


def test_metrics_reconciliation_accepts_valid_mixed_events(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    evaluation = {
        "step": 1,
        "split": "validation",
        "loss": 1.0,
        "perplexity": 2.0,
        "tokens": 7,
        "elapsed_seconds": 0.1,
        "tokens_per_second": 70.0,
    }
    test_event = {**evaluation, "split": "test"}
    events = [train_metric(1), evaluation, test_event]
    metrics.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")

    assert training_module._reconcile_metrics(metrics, 1) == events


def test_metrics_reconciliation_requires_checkpoint_coverage(tmp_path: Path) -> None:
    metrics = tmp_path / "metrics.jsonl"
    metrics.write_text(json.dumps(train_metric(1)) + "\n", encoding="utf-8")

    with pytest.raises(TrainingError, match="durable checkpoint"):
        training_module._reconcile_metrics(metrics, 2)


def test_tiny_training_writes_resumable_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "scenes.jsonl"
    write_fixture_corpus(corpus)
    dataset = build_experiment_dataset(corpus, block_length=8)
    model_config = ModelConfig(
        model_type="transformer",
        context_length=8,
        hidden_size=16,
        num_heads=4,
        ffn_size=24,
        num_layers=1,
    )
    training_config = TrainingConfig(
        seed=3,
        batch_size=1,
        train_steps=2,
        learning_rate=1e-3,
        warmup_steps=0,
        eval_interval=1,
        checkpoint_interval=1,
    )
    run_dir = tmp_path / "run"
    metrics_syncs = 0
    checkpoint_saves = 0
    original_sync = training_module._fsync_existing
    original_save = training_module.save_checkpoint

    def tracking_sync(path: Path) -> None:
        nonlocal metrics_syncs
        if path.name == "metrics.jsonl":
            metrics_syncs += 1
        original_sync(path)

    def checked_save(*args: Any, **kwargs: Any) -> Path:
        nonlocal checkpoint_saves
        assert metrics_syncs > checkpoint_saves
        checkpoint_saves += 1
        return original_save(*args, **kwargs)

    monkeypatch.setattr(training_module, "_fsync_existing", tracking_sync)
    monkeypatch.setattr(training_module, "save_checkpoint", checked_save)

    mismatched_config = ModelConfig(
        model_type="transformer",
        context_length=8,
        hidden_size=16,
        num_heads=4,
        ffn_size=24,
        vocab_size=270,
        num_layers=1,
    )
    with pytest.raises(TrainingError, match="vocabulary sizes"):
        train(
            dataset,
            mismatched_config,
            training_config,
            tmp_path / "mismatch",
            Path(__file__).parents[1],
        )

    result = train(
        dataset,
        model_config,
        training_config,
        run_dir,
        Path(__file__).parents[1],
    )

    checkpoint = Path(result["checkpoint"])
    assert result["global_step"] == 2
    assert metrics_syncs == checkpoint_saves == 2
    assert (checkpoint / "COMPLETE").is_file()
    assert (checkpoint / "model.safetensors").is_file()
    assert (checkpoint / "optimizer.safetensors").is_file()
    assert (run_dir / "latest").read_text(encoding="utf-8").strip() == checkpoint.name
    with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(train_metric(3)) + "\n")
    train(
        dataset,
        model_config,
        training_config,
        run_dir,
        Path(__file__).parents[1],
        checkpoint,
    )
    events = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["step"] for event in events if event["split"] == "train"] == [1, 2]
    interrupted = run_dir / ".checkpoint-00000003.tmp-interrupted"
    interrupted.mkdir()
    assert (run_dir / "latest").read_text(encoding="utf-8").strip() == checkpoint.name
    assert (checkpoint / "COMPLETE").is_file()
    with pytest.raises(TrainingError, match="run directory is not empty"):
        train(dataset, model_config, training_config, run_dir, Path(__file__).parents[1])

    (run_dir / "metrics.jsonl").unlink()
    with pytest.raises(TrainingError, match="durable checkpoint"):
        train(
            dataset,
            model_config,
            training_config,
            run_dir,
            Path(__file__).parents[1],
            checkpoint,
        )
