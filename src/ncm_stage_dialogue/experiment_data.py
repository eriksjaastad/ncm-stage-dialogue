"""Deterministic byte-level serialization for the dialogue experiment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RAW_BYTE_VOCAB = 256
SPECIAL_TOKENS = {
    "PAD": 256,
    "BOS": 257,
    "EOS": 258,
    "WORK": 259,
    "FIELD": 260,
    "SCENE": 261,
    "STAGE": 262,
    "ITEM_END": 263,
    "TURN": 264,
    "SPEAKER": 265,
    "TEXT": 266,
    "DIRECTION": 267,
    "TURN_END": 268,
    "SCENE_END": 269,
    "WORK_END": 270,
}
VOCAB_SIZE = RAW_BYTE_VOCAB + len(SPECIAL_TOKENS)
VALID_SPLITS = ("train", "validation", "test")


class ExperimentDataError(ValueError):
    """Raised when generated corpus data violates the experiment contract."""


@dataclass(frozen=True)
class TokenBlock:
    """One fixed-length next-token prediction block."""

    inputs: tuple[int, ...]
    targets: tuple[int, ...]
    loss_mask: tuple[float, ...]


@dataclass(frozen=True)
class ExperimentDataset:
    """Frozen token streams, packed blocks, and their identity manifest."""

    streams: dict[str, tuple[int, ...]]
    blocks: dict[str, tuple[TokenBlock, ...]]
    manifest: dict[str, Any]


class ByteTokenizer:
    """UTF-8 bytes plus fixed out-of-band structural token IDs."""

    vocab_size = VOCAB_SIZE
    pad_id = SPECIAL_TOKENS["PAD"]

    @staticmethod
    def encode_text(text: str) -> list[int]:
        return list(text.encode("utf-8"))

    @staticmethod
    def decode_text(tokens: Iterable[int]) -> str:
        values = list(tokens)
        if any(token < 0 or token >= RAW_BYTE_VOCAB for token in values):
            raise ExperimentDataError("decode_text accepts raw byte tokens only")
        return bytes(values).decode("utf-8")

    @staticmethod
    def decode_display(tokens: Iterable[int]) -> str:
        names = {token_id: name for name, token_id in SPECIAL_TOKENS.items()}
        output: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                output.append(byte_buffer.decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for token in tokens:
            if 0 <= token < RAW_BYTE_VOCAB:
                byte_buffer.append(token)
            elif token in names:
                flush_bytes()
                output.append(f"<{names[token]}>")
            else:
                raise ExperimentDataError(f"token outside vocabulary: {token}")
        flush_bytes()
        return "".join(output)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExperimentDataError(f"{field} must be a non-empty string")
    return value


def _directions(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ExperimentDataError(f"{field} must be a list of non-empty strings")
    return value


def _append_text(tokens: list[int], tokenizer: ByteTokenizer, text: str) -> None:
    tokens.extend(tokenizer.encode_text(text))


def serialize_work(records: list[dict[str, Any]], tokenizer: ByteTokenizer) -> list[int]:
    """Serialize one work without normalizing its human-written text."""
    if not records:
        raise ExperimentDataError("cannot serialize an empty work")
    first = records[0]
    source_id = _text(first.get("source_id"), "source_id")
    title = _text(first.get("title"), "title")
    author = _text(first.get("author"), "author")
    split = _text(first.get("split"), "split")
    if split not in VALID_SPLITS:
        raise ExperimentDataError(f"unknown split: {split}")

    for record in records:
        identity = (record.get("source_id"), record.get("title"), record.get("author"))
        if identity != (source_id, title, author) or record.get("split") != split:
            raise ExperimentDataError("work identity or split changed between scenes")

    tokens = [SPECIAL_TOKENS["BOS"], SPECIAL_TOKENS["WORK"]]
    for index, value in enumerate((source_id, title, author)):
        _append_text(tokens, tokenizer, value)
        if index < 2:
            tokens.append(SPECIAL_TOKENS["FIELD"])

    ordered = sorted(records, key=lambda record: (record.get("act"), record.get("scene_id")))
    scene_ids: set[str] = set()
    for record in ordered:
        scene_id = _text(record.get("scene_id"), "scene_id")
        if scene_id in scene_ids:
            raise ExperimentDataError(f"duplicate scene id in work: {scene_id}")
        scene_ids.add(scene_id)
        act = record.get("act")
        if type(act) is not int or act < 1:
            raise ExperimentDataError("act must be a positive integer")
        tokens.append(SPECIAL_TOKENS["SCENE"])
        _append_text(tokens, tokenizer, scene_id)
        tokens.append(SPECIAL_TOKENS["FIELD"])
        _append_text(tokens, tokenizer, str(act))

        for direction in _directions(record.get("stage_directions"), "stage_directions"):
            tokens.append(SPECIAL_TOKENS["STAGE"])
            _append_text(tokens, tokenizer, direction)
            tokens.append(SPECIAL_TOKENS["ITEM_END"])

        turns = record.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ExperimentDataError(f"scene has no turns: {scene_id}")
        for turn in turns:
            if not isinstance(turn, dict):
                raise ExperimentDataError("turn must be an object")
            speaker = _text(turn.get("speaker"), "speaker")
            spoken = _text(turn.get("text"), "text")
            tokens.extend((SPECIAL_TOKENS["TURN"], SPECIAL_TOKENS["SPEAKER"]))
            _append_text(tokens, tokenizer, speaker)
            for direction in _directions(turn.get("stage_directions"), "turn directions"):
                tokens.append(SPECIAL_TOKENS["DIRECTION"])
                _append_text(tokens, tokenizer, direction)
                tokens.append(SPECIAL_TOKENS["ITEM_END"])
            tokens.append(SPECIAL_TOKENS["TEXT"])
            _append_text(tokens, tokenizer, spoken)
            tokens.append(SPECIAL_TOKENS["TURN_END"])
        tokens.append(SPECIAL_TOKENS["SCENE_END"])
    tokens.extend((SPECIAL_TOKENS["WORK_END"], SPECIAL_TOKENS["EOS"]))
    return tokens


def _token_bytes(tokens: Iterable[int]) -> bytes:
    return b"".join(int(token).to_bytes(2, "little") for token in tokens)


def token_hash(tokens: Iterable[int]) -> str:
    """Hash token IDs with a portable fixed-width representation."""
    return hashlib.sha256(_token_bytes(tokens)).hexdigest()


def pack_stream(stream: tuple[int, ...], block_length: int) -> tuple[TokenBlock, ...]:
    """Pack one split stream into deterministic shifted prediction blocks."""
    if block_length < 2:
        raise ExperimentDataError("block_length must be at least 2")
    if len(stream) < 2:
        raise ExperimentDataError("split stream must contain at least two tokens")
    blocks: list[TokenBlock] = []
    pad = SPECIAL_TOKENS["PAD"]
    for offset in range(0, len(stream) - 1, block_length):
        inputs = list(stream[offset : offset + block_length])
        targets = list(stream[offset + 1 : offset + block_length + 1])
        real_targets = len(targets)
        inputs.extend([pad] * (block_length - len(inputs)))
        targets.extend([pad] * (block_length - len(targets)))
        mask = [1.0] * real_targets + [0.0] * (block_length - real_targets)
        blocks.append(TokenBlock(tuple(inputs), tuple(targets), tuple(mask)))
    return tuple(blocks)


def build_experiment_dataset(corpus_path: Path, block_length: int) -> ExperimentDataset:
    """Validate and tokenize the frozen JSONL corpus into isolated split streams."""
    raw_bytes = corpus_path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_bytes.decode("utf-8").splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExperimentDataError(f"invalid JSON on line {line_number}") from error
        if not isinstance(row, dict):
            raise ExperimentDataError(f"record {line_number} must be an object")
        rows.append(row)
    if not rows:
        raise ExperimentDataError("corpus is empty")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (_text(row.get("source_id"), "source_id"), _text(row.get("title"), "title"))
        grouped.setdefault(key, []).append(row)

    tokenizer = ByteTokenizer()
    streams_lists = {split: [] for split in VALID_SPLITS}
    works_manifest: list[dict[str, Any]] = []
    split_counts = {split: {"works": 0, "scenes": 0, "turns": 0} for split in VALID_SPLITS}
    for key in sorted(grouped):
        records = grouped[key]
        splits = {record.get("split") for record in records}
        if len(splits) != 1:
            raise ExperimentDataError(f"complete work spans splits: {key}")
        split = next(iter(splits))
        if split not in VALID_SPLITS:
            raise ExperimentDataError(f"unknown split for {key}: {split}")
        serialized = serialize_work(records, tokenizer)
        streams_lists[split].extend(serialized)
        turn_count = sum(len(record["turns"]) for record in records)
        split_counts[split]["works"] += 1
        split_counts[split]["scenes"] += len(records)
        split_counts[split]["turns"] += turn_count
        works_manifest.append(
            {
                "work_key": f"{key[0]}::{key[1]}",
                "source_id": key[0],
                "title": key[1],
                "author": records[0]["author"],
                "split": split,
                "speakers": sorted(
                    {turn["speaker"] for record in records for turn in record["turns"]}
                ),
                "scene_ids": [
                    record["scene_id"]
                    for record in sorted(records, key=lambda item: (item["act"], item["scene_id"]))
                ],
            }
        )

    streams = {split: tuple(tokens) for split, tokens in streams_lists.items()}
    if any(len(streams[split]) < 2 for split in VALID_SPLITS):
        raise ExperimentDataError("every split must contain tokenized data")
    blocks = {split: pack_stream(stream, block_length) for split, stream in streams.items()}
    manifest = {
        "schema_version": 1,
        "corpus_path": str(corpus_path),
        "corpus_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "corpus_bytes": len(raw_bytes),
        "record_count": len(rows),
        "work_count": len(grouped),
        "turn_count": sum(len(row["turns"]) for row in rows),
        "split_counts": split_counts,
        "works": works_manifest,
        "serialization_version": 1,
        "tokenizer_version": 1,
        "block_length": block_length,
        "vocab_size": VOCAB_SIZE,
        "special_token_ids": SPECIAL_TOKENS,
        "split_token_counts": {split: len(tokens) for split, tokens in streams.items()},
        "split_token_sha256": {split: token_hash(tokens) for split, tokens in streams.items()},
    }
    return ExperimentDataset(streams=streams, blocks=blocks, manifest=manifest)
