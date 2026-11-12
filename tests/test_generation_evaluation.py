from __future__ import annotations

import json

import mlx.core as mx

import ncm_stage_dialogue.training as training_module
from ncm_stage_dialogue.experiment_data import (
    SPECIAL_TOKENS,
    ByteTokenizer,
    build_experiment_dataset,
)
from ncm_stage_dialogue.generation_evaluation import (
    build_prompt_specs,
    generation_metrics,
    select_turn_boundaries,
)
from ncm_stage_dialogue.training import generate


def text(value: str) -> list[int]:
    return ByteTokenizer.encode_text(value)


def scene(source_id: str, title: str, split: str) -> dict[str, object]:
    return {
        "source_id": source_id,
        "title": title,
        "author": "Test Author",
        "scene_id": "act-1",
        "act": 1,
        "split": split,
        "stage_directions": ["A room."],
        "turns": [{"speaker": "ALICE", "text": "Hello.", "stage_directions": []}],
    }


def completed_first_turn() -> list[int]:
    return [
        SPECIAL_TOKENS["BOS"],
        SPECIAL_TOKENS["WORK"],
        *text("fixture"),
        SPECIAL_TOKENS["FIELD"],
        *text("Fixture"),
        SPECIAL_TOKENS["FIELD"],
        *text("Test Author"),
        SPECIAL_TOKENS["SCENE"],
        *text("act-1"),
        SPECIAL_TOKENS["FIELD"],
        *text("1"),
        SPECIAL_TOKENS["TURN"],
        SPECIAL_TOKENS["SPEAKER"],
        *text("ALICE"),
        SPECIAL_TOKENS["TEXT"],
        *text("Hello."),
        SPECIAL_TOKENS["TURN_END"],
    ]


def second_turn(speaker: str = "BOB") -> list[int]:
    return [
        SPECIAL_TOKENS["TURN"],
        SPECIAL_TOKENS["SPEAKER"],
        *text(speaker),
        SPECIAL_TOKENS["TEXT"],
        *text("Yes, yes, yes."),
        SPECIAL_TOKENS["TURN_END"],
    ]


def test_generation_metrics_accept_complete_structured_continuation() -> None:
    continuation = [
        *second_turn(),
        SPECIAL_TOKENS["SCENE_END"],
        SPECIAL_TOKENS["WORK_END"],
        SPECIAL_TOKENS["EOS"],
    ]

    metrics = generation_metrics(
        completed_first_turn(),
        continuation,
        {"ALICE", "BOB"},
        max_new_tokens=len(continuation) + 1,
    )

    assert metrics["prefix_valid"] is True
    assert metrics["structure_valid"] is True
    assert metrics["completed_before_cap"] is True
    assert metrics["censored_by_length"] is False
    assert metrics["generated_turns_started"] == 1
    assert metrics["generated_turns_completed"] == 1
    assert metrics["speaker_fields"] == 1
    assert metrics["speaker_format_rate"] == 1.0
    assert metrics["speaker_consistency"] == 1.0
    assert metrics["multi_speaker_generation"] is False
    assert metrics["completed_text_fields"] == 1


def test_generation_metrics_distinguishes_censoring_from_syntax_failure() -> None:
    continuation = [
        SPECIAL_TOKENS["TURN"],
        SPECIAL_TOKENS["SPEAKER"],
        *text("BOB"),
        SPECIAL_TOKENS["TEXT"],
        *text("Still speaking"),
    ]

    censored = generation_metrics(
        completed_first_turn(), continuation, {"ALICE", "BOB"}, max_new_tokens=len(continuation)
    )
    malformed = generation_metrics(
        completed_first_turn(),
        [SPECIAL_TOKENS["TEXT"], *text("out of order")],
        {"ALICE", "BOB"},
        max_new_tokens=20,
    )

    assert censored["prefix_valid"] is True
    assert censored["structure_valid"] is False
    assert censored["censored_by_length"] is True
    assert censored["generated_turns_started"] == 1
    assert censored["generated_turns_completed"] == 0
    assert malformed["prefix_valid"] is False
    assert malformed["censored_by_length"] is False
    assert malformed["syntax_error_state"] == "after_turn"


def test_generation_metrics_scopes_speakers_to_prompted_work() -> None:
    continuation = [
        *second_turn("CAROL"),
        SPECIAL_TOKENS["SCENE_END"],
        SPECIAL_TOKENS["WORK_END"],
        SPECIAL_TOKENS["EOS"],
    ]

    metrics = generation_metrics(
        completed_first_turn(), continuation, {"ALICE", "BOB"}, max_new_tokens=100
    )

    assert metrics["structure_valid"] is True
    assert metrics["speaker_fields"] == 1
    assert metrics["speaker_format_rate"] == 1.0
    assert metrics["speaker_consistency"] == 0.0
    assert metrics["unknown_speakers"] == ["CAROL"]


def test_generation_metrics_compare_multi_speaker_format_and_transitions() -> None:
    continuation = [
        *second_turn("BOB"),
        *second_turn("Carol"),
        SPECIAL_TOKENS["SCENE_END"],
        SPECIAL_TOKENS["WORK_END"],
        SPECIAL_TOKENS["EOS"],
    ]

    metrics = generation_metrics(
        completed_first_turn(), continuation, {"ALICE", "BOB"}, max_new_tokens=100
    )

    assert metrics["structure_valid"] is True
    assert metrics["distinct_speakers"] == 2
    assert metrics["multi_speaker_generation"] is True
    assert metrics["speaker_transition_rate"] == 1.0
    assert metrics["speaker_format_rate"] == 0.5


def test_turn_boundary_selection_is_predeclared_early_middle_late() -> None:
    turn_end = SPECIAL_TOKENS["TURN_END"]
    tokens = [1, turn_end, 2, turn_end, 3, turn_end, 4, turn_end, 5, turn_end]

    selected = select_turn_boundaries(tokens)

    assert [(item.label, item.turn_ordinal, item.token_index) for item in selected] == [
        ("early", 1, 1),
        ("middle", 3, 5),
        ("late", 5, 9),
    ]


def test_prompt_manifest_uses_only_test_works_and_canonical_boundaries(tmp_path) -> None:
    corpus = tmp_path / "scenes.jsonl"
    rows = [
        scene(source_id="train", title="Train", split="train"),
        scene(source_id="validation", title="Validation", split="validation"),
        scene(source_id="test", title="Test", split="test"),
    ]
    corpus.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dataset = build_experiment_dataset(corpus, block_length=16)

    prompts = build_prompt_specs(dataset)

    assert len(prompts) == 1
    assert prompts[0].title == "Test"
    assert prompts[0].label == "early"
    assert prompts[0].prompt_tokens[-1] == SPECIAL_TOKENS["TURN_END"]
    assert len(prompts[0].prompt_sha256) == 64


def test_generation_can_stop_at_the_serialized_eos(monkeypatch) -> None:
    class FixtureModel:
        def __call__(self, inputs):
            return mx.zeros((1, inputs.shape[1], 271))

    monkeypatch.setattr(
        training_module.mx.random,
        "categorical",
        lambda logits: mx.array(SPECIAL_TOKENS["EOS"]),
    )

    output = generate(
        FixtureModel(),
        [SPECIAL_TOKENS["BOS"]],
        context_length=8,
        max_new_tokens=4,
        seed=1,
        stop_token=SPECIAL_TOKENS["EOS"],
    )

    assert output == [SPECIAL_TOKENS["BOS"], SPECIAL_TOKENS["EOS"]]
