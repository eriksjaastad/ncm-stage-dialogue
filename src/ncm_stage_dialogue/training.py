"""Training, evaluation, generation, and checkpointing for MLX experiments."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from ncm_stage_dialogue.experiment_data import VOCAB_SIZE, ExperimentDataset, TokenBlock
from ncm_stage_dialogue.models import ModelConfig, build_model, parameter_count


class TrainingError(RuntimeError):
    """Raised when a run or checkpoint violates its captured contract."""


@dataclass(frozen=True)
class TrainingConfig:
    """Optimization settings shared by both comparison models."""

    seed: int = 123
    batch_size: int = 4
    train_steps: int = 100
    learning_rate: float = 6e-4
    warmup_steps: int = 10
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    eval_interval: int = 25
    checkpoint_interval: int = 25

    def validate(self) -> None:
        for name in (
            "batch_size",
            "train_steps",
            "warmup_steps",
            "eval_interval",
            "checkpoint_interval",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise TrainingError(f"{name} must be a non-negative integer")
        if self.batch_size < 1 or self.train_steps < 1:
            raise TrainingError("batch_size and train_steps must be positive")
        if self.eval_interval < 1 or self.checkpoint_interval < 1:
            raise TrainingError("eval_interval and checkpoint_interval must be positive")
        if self.warmup_steps > self.train_steps:
            raise TrainingError("warmup_steps cannot exceed train_steps")
        if not 0 < self.learning_rate or not 0 <= self.weight_decay:
            raise TrainingError("learning_rate must be positive and weight_decay non-negative")
        if self.max_grad_norm <= 0:
            raise TrainingError("max_grad_norm must be positive")


@dataclass
class TrainerState:
    """Minimal deterministic progress required for exact batch resumption."""

    global_step: int = 0
    epoch: int = 0
    next_batch_index: int = 0
    seen_tokens: int = 0


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _sha256_json(value: Any) -> str:
    import hashlib

    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _git_commit(repository: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def _write_file_synced(path: Path, content: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_existing(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def build_optimizer(model_config: ModelConfig, training_config: TrainingConfig) -> optim.AdamW:
    del model_config
    decay_steps = max(1, training_config.train_steps - training_config.warmup_steps)
    if training_config.warmup_steps:
        schedule = optim.join_schedules(
            [
                optim.linear_schedule(
                    0.0, training_config.learning_rate, training_config.warmup_steps
                ),
                optim.cosine_decay(training_config.learning_rate, decay_steps),
            ],
            [training_config.warmup_steps],
        )
    else:
        schedule = optim.cosine_decay(training_config.learning_rate, decay_steps)
    return optim.AdamW(
        learning_rate=schedule,
        betas=[0.9, 0.95],
        weight_decay=training_config.weight_decay,
    )


def _batch_arrays(blocks: tuple[TokenBlock, ...], indices: list[int]) -> tuple[mx.array, ...]:
    selected = [blocks[index] for index in indices]
    return (
        mx.array([block.inputs for block in selected], dtype=mx.int32),
        mx.array([block.targets for block in selected], dtype=mx.int32),
        mx.array([block.loss_mask for block in selected], dtype=mx.float32),
    )


def _next_indices(
    block_count: int, batch_size: int, state: TrainerState
) -> tuple[list[int], TrainerState]:
    indices: list[int] = []
    index = state.next_batch_index
    epoch = state.epoch
    for _ in range(batch_size):
        indices.append(index)
        index += 1
        if index == block_count:
            index = 0
            epoch += 1
    return indices, TrainerState(
        global_step=state.global_step,
        epoch=epoch,
        next_batch_index=index,
        seen_tokens=state.seen_tokens,
    )


def language_model_loss(model: nn.Module, inputs: mx.array, targets: mx.array, mask: mx.array):
    """Return masked next-token cross-entropy and counted target tokens."""
    logits = model(inputs)
    losses = nn.losses.cross_entropy(logits, targets, reduction="none")
    token_count = mask.sum()
    return (losses * mask).sum() / token_count, token_count


def evaluate(model: nn.Module, blocks: tuple[TokenBlock, ...], batch_size: int) -> dict[str, Any]:
    """Aggregate split loss by tokens rather than averaging batch means."""
    started = time.perf_counter()
    total_loss = 0.0
    total_tokens = 0
    for offset in range(0, len(blocks), batch_size):
        indices = list(range(offset, min(offset + batch_size, len(blocks))))
        inputs, targets, mask = _batch_arrays(blocks, indices)
        logits = model(inputs)
        losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        loss_sum = (losses * mask).sum()
        token_count = mask.sum()
        mx.eval(loss_sum, token_count)
        total_loss += float(loss_sum.item())
        total_tokens += int(token_count.item())
    elapsed = time.perf_counter() - started
    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "tokens": total_tokens,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed if elapsed else None,
    }


def repeated_trigram_rate(tokens: Iterable[int]) -> float:
    """Return the fraction of generated trigrams that repeat."""
    values = list(tokens)
    if len(values) < 3:
        return 0.0
    trigrams = list(zip(values, values[1:], values[2:]))
    return 1.0 - len(set(trigrams)) / len(trigrams)


def structure_metrics(tokens: Iterable[int], declared_speakers: set[str]) -> dict[str, Any]:
    """Measure structural completion and declared-speaker consistency in token output."""
    from ncm_stage_dialogue.experiment_data import RAW_BYTE_VOCAB, SPECIAL_TOKENS

    values = list(tokens)
    speaker_names: list[str | None] = []
    index = 0
    while index < len(values):
        if values[index] != SPECIAL_TOKENS["SPEAKER"]:
            index += 1
            continue
        index += 1
        raw = bytearray()
        while index < len(values) and values[index] < RAW_BYTE_VOCAB:
            raw.append(values[index])
            index += 1
        try:
            speaker_names.append(raw.decode("utf-8") if raw else None)
        except UnicodeDecodeError:
            speaker_names.append(None)
    valid_speakers = sum(name in declared_speakers for name in speaker_names)
    turn_count = values.count(SPECIAL_TOKENS["TURN"])
    turn_end_count = values.count(SPECIAL_TOKENS["TURN_END"])
    scene_count = values.count(SPECIAL_TOKENS["SCENE"])
    scene_end_count = values.count(SPECIAL_TOKENS["SCENE_END"])
    work_count = values.count(SPECIAL_TOKENS["WORK"])
    work_end_count = values.count(SPECIAL_TOKENS["WORK_END"])
    unknown_tokens = any(token < 0 or token >= VOCAB_SIZE for token in values)
    fully_valid = bool(
        values
        and values[0] == SPECIAL_TOKENS["BOS"]
        and values[-1] == SPECIAL_TOKENS["EOS"]
        and work_count == work_end_count == 1
        and scene_count == scene_end_count
        and turn_count == turn_end_count
        and all(name is not None for name in speaker_names)
        and not unknown_tokens
    )
    return {
        "structure_valid": fully_valid,
        "speaker_consistency": (valid_speakers / len(speaker_names) if speaker_names else None),
        "speaker_fields": len(speaker_names),
        "turn_completion_rate": min(turn_end_count, turn_count) / turn_count
        if turn_count
        else None,
        "scene_completion_rate": min(scene_end_count, scene_count) / scene_count
        if scene_count
        else None,
        "work_completion_rate": min(work_end_count, work_count) / work_count
        if work_count
        else None,
    }


def generate(
    model: nn.Module,
    prompt: list[int],
    *,
    context_length: int,
    max_new_tokens: int,
    seed: int,
    temperature: float = 0.8,
) -> list[int]:
    """Generate with correct full-prefix recomputation for both architectures."""
    if not prompt or max_new_tokens < 0 or temperature <= 0:
        raise TrainingError(
            "generation requires a prompt, non-negative length, and positive temperature"
        )
    mx.random.seed(seed)
    output = list(prompt)
    for _ in range(max_new_tokens):
        context = output[-context_length:]
        logits = model(mx.array([context], dtype=mx.int32))[0, -1]
        next_token = mx.random.categorical(logits / temperature)
        mx.eval(next_token)
        output.append(int(next_token.item()))
    return output


def save_checkpoint(
    run_dir: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    state: TrainerState,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dataset_manifest: dict[str, Any],
    repository: Path,
) -> Path:
    """Atomically publish a complete model/optimizer/progress checkpoint."""
    checkpoint_name = f"checkpoint-{state.global_step:08d}"
    destination = run_dir / checkpoint_name
    if destination.exists():
        raise TrainingError(f"checkpoint already exists: {destination}")
    temporary = run_dir / f".{checkpoint_name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    model_path = temporary / "model.safetensors"
    optimizer_path = temporary / "optimizer.safetensors"
    model.save_weights(str(model_path))
    flat_optimizer = tree_flatten(optimizer.state, destination={})
    mx.save_safetensors(str(optimizer_path), flat_optimizer)
    mx.eval(model.parameters(), optimizer.state)
    _fsync_existing(model_path)
    _fsync_existing(optimizer_path)
    config_payload = {
        "model": model_config.to_dict(),
        "training": asdict(training_config),
        "model_parameters": parameter_count(model),
    }
    trainer_payload = {
        **asdict(state),
        "checkpoint_version": 1,
        "git_commit": _git_commit(repository),
        "dataset_manifest_hash": _sha256_json(dataset_manifest),
        "config_hash": _sha256_json(config_payload),
    }
    _write_file_synced(temporary / "config.json", _json_bytes(config_payload))
    _write_file_synced(temporary / "dataset_manifest.json", _json_bytes(dataset_manifest))
    _write_file_synced(temporary / "trainer_state.json", _json_bytes(trainer_payload))
    _write_file_synced(temporary / "COMPLETE", b"complete\n")
    directory_fd = os.open(temporary, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    os.replace(temporary, destination)
    latest_temporary = run_dir / f".latest.tmp-{uuid.uuid4().hex}"
    _write_file_synced(latest_temporary, (checkpoint_name + "\n").encode("utf-8"))
    os.replace(latest_temporary, run_dir / "latest")
    return destination


def load_checkpoint(
    checkpoint: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    dataset_manifest: dict[str, Any],
) -> TrainerState:
    """Restore a complete checkpoint after validating dataset and config identity."""
    if not checkpoint.is_dir() or not (checkpoint / "COMPLETE").is_file():
        raise TrainingError(f"checkpoint is incomplete: {checkpoint}")
    config_payload = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    expected_config = {
        "model": model_config.to_dict(),
        "training": asdict(training_config),
        "model_parameters": parameter_count(model),
    }
    if config_payload != expected_config:
        raise TrainingError("checkpoint configuration does not match this run")
    saved_manifest = json.loads((checkpoint / "dataset_manifest.json").read_text(encoding="utf-8"))
    if _sha256_json(saved_manifest) != _sha256_json(dataset_manifest):
        raise TrainingError("checkpoint dataset manifest does not match this run")
    trainer_payload = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
    if trainer_payload["dataset_manifest_hash"] != _sha256_json(dataset_manifest):
        raise TrainingError("checkpoint dataset hash is inconsistent")
    if trainer_payload["config_hash"] != _sha256_json(expected_config):
        raise TrainingError("checkpoint config hash is inconsistent")
    model.load_weights(str(checkpoint / "model.safetensors"))
    optimizer.state = tree_unflatten(mx.load(str(checkpoint / "optimizer.safetensors")))
    mx.eval(model.parameters(), optimizer.state)
    return TrainerState(
        global_step=trainer_payload["global_step"],
        epoch=trainer_payload["epoch"],
        next_batch_index=trainer_payload["next_batch_index"],
        seen_tokens=trainer_payload["seen_tokens"],
    )


def train(
    dataset: ExperimentDataset,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    run_dir: Path,
    repository: Path,
    resume_checkpoint: Path | None = None,
) -> dict[str, Any]:
    """Train one model, writing metrics and atomic resumable checkpoints."""
    training_config.validate()
    model_config.validate()
    if dataset.manifest["block_length"] != model_config.context_length:
        raise TrainingError("dataset block length must match model context length")
    train_blocks = dataset.blocks["train"]
    if not train_blocks:
        raise TrainingError("training split has no blocks")
    run_dir.mkdir(parents=True, exist_ok=True)
    if resume_checkpoint is None and any(run_dir.iterdir()):
        raise TrainingError(f"run directory is not empty: {run_dir}")
    mx.random.seed(training_config.seed)
    model = build_model(model_config)
    optimizer = build_optimizer(model_config, training_config)
    state = TrainerState()
    if resume_checkpoint is not None:
        state = load_checkpoint(
            resume_checkpoint,
            model,
            optimizer,
            model_config,
            training_config,
            dataset.manifest,
        )
    loss_and_grad = nn.value_and_grad(model, language_model_loss)
    metrics_path = run_dir / "metrics.jsonl"
    started = time.perf_counter()
    last_checkpoint: Path | None = resume_checkpoint

    while state.global_step < training_config.train_steps:
        indices, advanced_state = _next_indices(
            len(train_blocks), training_config.batch_size, state
        )
        inputs, targets, mask = _batch_arrays(train_blocks, indices)
        (loss, token_count), gradients = loss_and_grad(model, inputs, targets, mask)
        gradients, gradient_norm = optim.clip_grad_norm(gradients, training_config.max_grad_norm)
        optimizer.update(model, gradients)
        mx.eval(loss, token_count, gradient_norm, model.parameters(), optimizer.state)
        state = TrainerState(
            global_step=state.global_step + 1,
            epoch=advanced_state.epoch,
            next_batch_index=advanced_state.next_batch_index,
            seen_tokens=state.seen_tokens + int(token_count.item()),
        )
        event = {
            "step": state.global_step,
            "split": "train",
            "loss": float(loss.item()),
            "tokens": int(token_count.item()),
            "gradient_norm": float(gradient_norm.item()),
            "seen_tokens": state.seen_tokens,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

        if state.global_step % training_config.eval_interval == 0:
            validation = evaluate(model, dataset.blocks["validation"], training_config.batch_size)
            validation.update({"step": state.global_step, "split": "validation"})
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(validation, sort_keys=True) + "\n")
        if state.global_step % training_config.checkpoint_interval == 0:
            last_checkpoint = save_checkpoint(
                run_dir,
                model,
                optimizer,
                state,
                model_config,
                training_config,
                dataset.manifest,
                repository,
            )

    if last_checkpoint is None or last_checkpoint.name != f"checkpoint-{state.global_step:08d}":
        last_checkpoint = save_checkpoint(
            run_dir,
            model,
            optimizer,
            state,
            model_config,
            training_config,
            dataset.manifest,
            repository,
        )
    test_metrics = evaluate(model, dataset.blocks["test"], training_config.batch_size)
    test_metrics.update({"step": state.global_step, "split": "test"})
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(test_metrics, sort_keys=True) + "\n")
    elapsed = time.perf_counter() - started
    return {
        "checkpoint": str(last_checkpoint),
        "global_step": state.global_step,
        "seen_tokens": state.seen_tokens,
        "elapsed_seconds": elapsed,
        "model_parameters": parameter_count(model),
        "test": test_metrics,
    }
