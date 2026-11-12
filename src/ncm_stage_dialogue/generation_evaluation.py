"""Paired, prefix-aware metrics for structured dialogue generation."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

from ncm_stage_dialogue.experiment_data import (
    RAW_BYTE_VOCAB,
    SPECIAL_TOKENS,
    VOCAB_SIZE,
    ExperimentDataset,
)


@dataclass(frozen=True)
class TurnBoundary:
    """One predeclared early, middle, or late completed-turn boundary."""

    label: str
    turn_ordinal: int
    token_index: int


@dataclass(frozen=True)
class PromptSpec:
    """One held-out serialized prompt shared by both model families."""

    prompt_id: str
    work_key: str
    title: str
    label: str
    turn_ordinal: int
    prompt_tokens: tuple[int, ...]
    prompt_sha256: str
    allowed_speakers: tuple[str, ...]


class _IncompleteStructure(Exception):
    def __init__(self, state: str) -> None:
        self.state = state


class _InvalidStructure(Exception):
    def __init__(self, state: str, index: int) -> None:
        self.state = state
        self.index = index


def select_turn_boundaries(tokens: Iterable[int]) -> tuple[TurnBoundary, ...]:
    """Select the first, middle, and last turn ends without inspecting model output."""
    positions = [index for index, token in enumerate(tokens) if token == SPECIAL_TOKENS["TURN_END"]]
    if not positions:
        raise ValueError("serialized work has no completed turns")
    candidates = (
        ("early", 0),
        ("middle", len(positions) // 2),
        ("late", len(positions) - 1),
    )
    selected: list[TurnBoundary] = []
    used: set[int] = set()
    for label, position_index in candidates:
        token_index = positions[position_index]
        if token_index in used:
            continue
        used.add(token_index)
        selected.append(
            TurnBoundary(
                label=label,
                turn_ordinal=position_index + 1,
                token_index=token_index,
            )
        )
    return tuple(selected)


def split_serialized_works(stream: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    """Split a partition stream at canonical BOS/EOS work boundaries."""
    values = list(stream)
    works: list[tuple[int, ...]] = []
    index = 0
    while index < len(values):
        if values[index] != SPECIAL_TOKENS["BOS"]:
            raise ValueError("split stream has tokens outside a serialized work")
        try:
            end = values.index(SPECIAL_TOKENS["EOS"], index) + 1
        except ValueError as error:
            raise ValueError("serialized work has no EOS marker") from error
        works.append(tuple(values[index:end]))
        index = end
    return tuple(works)


def build_prompt_specs(dataset: ExperimentDataset) -> tuple[PromptSpec, ...]:
    """Build the fixed early/middle/late prompt manifest from the test split."""
    works = split_serialized_works(dataset.streams["test"])
    manifests = [work for work in dataset.manifest["works"] if work["split"] == "test"]
    if len(works) != len(manifests):
        raise ValueError("test stream and manifest work counts differ")
    prompts: list[PromptSpec] = []
    for tokens, work in zip(works, manifests, strict=True):
        for boundary in select_turn_boundaries(tokens):
            prompt_tokens = tokens[: boundary.token_index + 1]
            digest = sha256(
                b"".join(token.to_bytes(2, "little") for token in prompt_tokens)
            ).hexdigest()
            prompts.append(
                PromptSpec(
                    prompt_id=f"{work['work_key']}::{boundary.label}",
                    work_key=work["work_key"],
                    title=work["title"],
                    label=boundary.label,
                    turn_ordinal=boundary.turn_ordinal,
                    prompt_tokens=prompt_tokens,
                    prompt_sha256=digest,
                    allowed_speakers=tuple(work["speakers"]),
                )
            )
    return tuple(prompts)


def _repeated_ngram_rate(values: list[Any], size: int = 3) -> float:
    if len(values) < size:
        return 0.0
    ngrams = list(zip(*(values[offset:] for offset in range(size))))
    return 1.0 - len(set(ngrams)) / len(ngrams)


class _StructureParser:
    def __init__(self, values: list[int], prompt_length: int) -> None:
        self.values = values
        self.prompt_length = prompt_length
        self.index = 0
        self.generated_turns_started = 0
        self.generated_turns_completed = 0
        self.generated_scenes_started = 0
        self.generated_scenes_completed = 0
        self.generated_work_completed = 0
        self.generated_speakers: list[str] = []
        self.completed_text_fields: list[str] = []

    def _current(self, state: str) -> int:
        if self.index >= len(self.values):
            raise _IncompleteStructure(state)
        return self.values[self.index]

    def _consume(self, expected: int, state: str) -> int:
        actual = self._current(state)
        if actual != expected:
            raise _InvalidStructure(state, self.index)
        marker_index = self.index
        self.index += 1
        return marker_index

    def _text(self, state: str) -> tuple[str, int]:
        start = self.index
        while self.index < len(self.values) and self.values[self.index] < RAW_BYTE_VOCAB:
            self.index += 1
        if self.index == start:
            if self.index == len(self.values):
                raise _IncompleteStructure(state)
            raise _InvalidStructure(state, self.index)
        raw = bytes(self.values[start : self.index])
        try:
            value = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise _InvalidStructure(state, start) from error
        if self.index == len(self.values):
            raise _IncompleteStructure(state)
        return value, start

    def parse(self) -> None:
        self._consume(SPECIAL_TOKENS["BOS"], "root")
        self._consume(SPECIAL_TOKENS["WORK"], "after_bos")
        self._text("work_source")
        self._consume(SPECIAL_TOKENS["FIELD"], "after_work_source")
        self._text("work_title")
        self._consume(SPECIAL_TOKENS["FIELD"], "after_work_title")
        self._text("work_author")

        scene_count = 0
        while True:
            if (
                self._current("after_author" if scene_count == 0 else "after_scene")
                != SPECIAL_TOKENS["SCENE"]
            ):
                if scene_count == 0:
                    raise _InvalidStructure("after_author", self.index)
                break
            scene_marker = self._consume(SPECIAL_TOKENS["SCENE"], "after_scene")
            if scene_marker >= self.prompt_length:
                self.generated_scenes_started += 1
            scene_count += 1
            self._text("scene_id")
            self._consume(SPECIAL_TOKENS["FIELD"], "after_scene_id")
            act, act_start = self._text("scene_act")
            if not act.isascii() or not act.isdigit() or int(act) < 1:
                raise _InvalidStructure("scene_act", act_start)

            while self._current("after_act") == SPECIAL_TOKENS["STAGE"]:
                self._consume(SPECIAL_TOKENS["STAGE"], "after_act")
                self._text("stage_direction")
                self._consume(SPECIAL_TOKENS["ITEM_END"], "after_stage_direction")

            turn_count = 0
            while (
                self._current("before_turn" if turn_count == 0 else "after_turn")
                == SPECIAL_TOKENS["TURN"]
            ):
                turn_marker = self._consume(SPECIAL_TOKENS["TURN"], "before_turn")
                if turn_marker >= self.prompt_length:
                    self.generated_turns_started += 1
                turn_count += 1
                speaker_marker = self._consume(SPECIAL_TOKENS["SPEAKER"], "after_turn_marker")
                speaker, _ = self._text("speaker")
                if speaker_marker >= self.prompt_length:
                    self.generated_speakers.append(speaker)
                while self._current("after_speaker") == SPECIAL_TOKENS["DIRECTION"]:
                    self._consume(SPECIAL_TOKENS["DIRECTION"], "after_speaker")
                    self._text("turn_direction")
                    self._consume(SPECIAL_TOKENS["ITEM_END"], "after_turn_direction")
                text_marker = self._consume(SPECIAL_TOKENS["TEXT"], "after_speaker")
                spoken, _ = self._text("turn_text")
                turn_end = self._consume(SPECIAL_TOKENS["TURN_END"], "after_turn_text")
                if turn_end >= self.prompt_length:
                    self.generated_turns_completed += 1
                if text_marker >= self.prompt_length:
                    self.completed_text_fields.append(spoken)
            if turn_count == 0:
                raise _InvalidStructure("before_turn", self.index)
            scene_end = self._consume(SPECIAL_TOKENS["SCENE_END"], "after_turn")
            if scene_end >= self.prompt_length:
                self.generated_scenes_completed += 1

        work_end = self._consume(SPECIAL_TOKENS["WORK_END"], "after_scene")
        if work_end >= self.prompt_length:
            self.generated_work_completed += 1
        self._consume(SPECIAL_TOKENS["EOS"], "after_work")
        if self.index != len(self.values):
            raise _InvalidStructure("trailing_tokens", self.index)


def generation_metrics(
    prompt_tokens: Iterable[int],
    continuation_tokens: Iterable[int],
    allowed_speakers: set[str],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    """Measure one generated continuation against the serialized prefix state."""
    prompt = list(prompt_tokens)
    continuation = list(continuation_tokens)
    values = prompt + continuation
    if not prompt or max_new_tokens < 1:
        raise ValueError("generation metrics require a prompt and positive token cap")
    if any(type(token) is not int or token < 0 or token >= VOCAB_SIZE for token in values):
        raise ValueError("generation contains a token outside the vocabulary")

    parser = _StructureParser(values, len(prompt))
    prefix_valid = True
    structure_valid = False
    incomplete_state: str | None = None
    syntax_error_state: str | None = None
    syntax_error_index: int | None = None
    try:
        parser.parse()
        structure_valid = True
    except _IncompleteStructure as error:
        incomplete_state = error.state
    except _InvalidStructure as error:
        prefix_valid = False
        syntax_error_state = error.state
        syntax_error_index = error.index - len(prompt)

    normalized_text = unicodedata.normalize("NFKC", "\n".join(parser.completed_text_fields))
    words = re.findall(r"\w+", normalized_text.casefold())
    unknown_speakers = sorted(
        {speaker for speaker in parser.generated_speakers if speaker not in allowed_speakers}
    )
    formatted_speakers = sum(
        speaker == speaker.strip()
        and any(character.isalpha() for character in speaker)
        and not any(character.islower() for character in speaker)
        and not any(character.isspace() and character != " " for character in speaker)
        for speaker in parser.generated_speakers
    )
    speaker_transitions = list(zip(parser.generated_speakers, parser.generated_speakers[1:]))
    hit_length_cap = len(continuation) >= max_new_tokens
    return {
        "prefix_valid": prefix_valid,
        "structure_valid": structure_valid,
        "completed_before_cap": structure_valid and not hit_length_cap,
        "hit_length_cap": hit_length_cap,
        "censored_by_length": prefix_valid and not structure_valid and hit_length_cap,
        "incomplete_state": incomplete_state,
        "syntax_error_state": syntax_error_state,
        "syntax_error_index": syntax_error_index,
        "generated_turns_started": parser.generated_turns_started,
        "generated_turns_completed": parser.generated_turns_completed,
        "turn_completion_rate": (
            parser.generated_turns_completed / parser.generated_turns_started
            if parser.generated_turns_started
            else None
        ),
        "generated_scenes_started": parser.generated_scenes_started,
        "generated_scenes_completed": parser.generated_scenes_completed,
        "scene_completion_rate": (
            parser.generated_scenes_completed / parser.generated_scenes_started
            if parser.generated_scenes_started
            else None
        ),
        "generated_work_completed": parser.generated_work_completed,
        "speaker_fields": len(parser.generated_speakers),
        "speaker_format_rate": (
            formatted_speakers / len(parser.generated_speakers)
            if parser.generated_speakers
            else None
        ),
        "speaker_consistency": (
            sum(speaker in allowed_speakers for speaker in parser.generated_speakers)
            / len(parser.generated_speakers)
            if parser.generated_speakers
            else None
        ),
        "distinct_speakers": len(set(parser.generated_speakers)),
        "multi_speaker_generation": len(set(parser.generated_speakers)) >= 2,
        "speaker_transition_rate": (
            sum(first != second for first, second in speaker_transitions) / len(speaker_transitions)
            if speaker_transitions
            else None
        ),
        "unknown_speakers": unknown_speakers,
        "completed_text_fields": len(parser.completed_text_fields),
        "generated_text_characters": len(normalized_text),
        "token_repeated_trigram_rate": _repeated_ngram_rate(continuation),
        "character_repeated_trigram_rate": _repeated_ngram_rate(list(normalized_text)),
        "word_repeated_trigram_rate": _repeated_ngram_rate(words),
    }
