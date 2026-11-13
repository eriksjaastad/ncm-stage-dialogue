from __future__ import annotations

import json
from pathlib import Path

import pytest

from ncm_stage_dialogue.experiment_data import SPECIAL_TOKENS, ByteTokenizer
from ncm_stage_dialogue.generation_evaluation import PromptSpec
from ncm_stage_dialogue.v1_report import (
    _curve_svg,
    _macro_generation_summary,
    _markdown,
    _prompt_manifest_record,
    _sample_record,
    _shared_training_commit,
    _shared_training_seed,
)


def sample(work: str, continuation: str, digest: str) -> dict[str, object]:
    return {
        "work_key": work,
        "continuation": continuation,
        "continuation_sha256": digest,
        "metrics": {
            "prefix_valid": True,
            "structure_valid": False,
            "censored_by_length": True,
            "turn_completion_rate": 0.5,
            "scene_completion_rate": None,
            "speaker_format_rate": 1.0,
            "speaker_consistency": 0.0,
            "multi_speaker_generation": True,
            "speaker_transition_rate": 1.0,
            "token_repeated_trigram_rate": 0.1,
            "character_repeated_trigram_rate": 0.2,
            "word_repeated_trigram_rate": 0.0,
            "speaker_fields": 1,
            "completed_text_fields": 1,
        },
    }


def prompt_fixture() -> PromptSpec:
    text = ByteTokenizer.encode_text
    tokens = (
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
    )
    return PromptSpec(
        prompt_id="fixture::early",
        work_key="fixture",
        title="Fixture",
        label="early",
        turn_ordinal=1,
        prompt_tokens=tokens,
        prompt_sha256="0" * 64,
        allowed_speakers=("ALICE",),
    )


def test_v1_schema_defaults_preserve_legacy_context_fields() -> None:
    prompt = prompt_fixture()

    manifest = _prompt_manifest_record(prompt, 16)
    sample_record = _sample_record(
        "v1-textnca",
        prompt,
        [],
        context_length=16,
        seed=123,
        temperature=0.8,
        max_new_tokens=1,
    )
    effective_manifest = _prompt_manifest_record(prompt, 16, include_effective_context=True)
    effective_sample = _sample_record(
        "v2-textnca",
        prompt,
        [],
        context_length=16,
        seed=123,
        temperature=0.8,
        max_new_tokens=1,
        include_effective_context=True,
    )

    assert "model_context_token_ids" not in manifest
    assert "model_context_sha256" not in manifest
    assert "model_context_sha256" not in sample_record
    assert len(effective_manifest["model_context_token_ids"]) == 16
    assert len(effective_manifest["model_context_sha256"]) == 64
    assert effective_sample["model_context_sha256"] == effective_manifest["model_context_sha256"]


def test_generation_summary_macro_averages_works_and_detects_cross_prompt_collapse() -> None:
    samples = [
        sample("work-a", "the same continuation", "one"),
        sample("work-a", "the same continuation", "one"),
        sample("work-b", "the same continuation", "one"),
    ]

    summary = _macro_generation_summary(samples)

    assert summary["sample_count"] == 3
    assert summary["work_count"] == 2
    assert summary["macro_average_across_works"]["prefix_valid"] == 1.0
    assert summary["cross_prompt"]["unique_continuation_rate"] == 1 / 3
    assert summary["cross_prompt"]["mean_cross_work_character_trigram_jaccard"] == 1.0


def test_cross_work_similarity_macro_averages_work_pairs_with_uneven_prompt_counts() -> None:
    samples = [
        sample("work-a", "abc", "a"),
        sample("work-b", "abc", "b1"),
        sample("work-b", "xyz", "b2"),
        sample("work-c", "xyz", "c"),
    ]

    summary = _macro_generation_summary(samples)

    assert summary["cross_prompt"]["mean_cross_work_character_trigram_jaccard"] == pytest.approx(
        1 / 3
    )


def test_cross_work_similarity_is_not_applicable_for_one_work() -> None:
    summary = _macro_generation_summary([sample("work-a", "abc", "a")])

    assert summary["cross_prompt"]["mean_cross_work_character_trigram_jaccard"] is None


def test_shared_training_commit_rejects_mixed_checkpoint_sources(tmp_path: Path) -> None:
    for model, commit in (("v1-textnca", "a" * 40), ("v1-transformer", "b" * 40)):
        run_dir = tmp_path / model
        checkpoint = run_dir / "checkpoint-00010000"
        checkpoint.mkdir(parents=True)
        (run_dir / "latest").write_text(checkpoint.name + "\n", encoding="utf-8")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"git_commit": commit}), encoding="utf-8"
        )

    with pytest.raises(ValueError, match="different training commits"):
        _shared_training_commit(tmp_path)


def test_shared_training_seed_reads_checkpoints_and_rejects_mismatch(tmp_path: Path) -> None:
    for model, seed in (("v1-textnca", 123), ("v1-transformer", 456)):
        run_dir = tmp_path / model
        checkpoint = run_dir / "checkpoint-00010000"
        checkpoint.mkdir(parents=True)
        (run_dir / "latest").write_text(checkpoint.name + "\n", encoding="utf-8")
        (checkpoint / "config.json").write_text(
            json.dumps({"training": {"seed": seed}}), encoding="utf-8"
        )

    with pytest.raises(ValueError, match="different training seeds"):
        _shared_training_seed(tmp_path)


def test_committed_v1_report_is_self_consistent_and_portable() -> None:
    repository = Path(__file__).parents[1]
    report_dir = repository / "reports" / "v1"
    results_text = (report_dir / "results.json").read_text(encoding="utf-8")
    results = json.loads(results_text)

    assert results["schema_version"] == 1
    assert results["identity"]["training_commit"] == ("7e4ecad7071f2ac4ea2bb99c520aac7291333713")
    assert len(results["prompt_manifest"]) == 9
    assert len(results["generation"]["samples"]) == 18
    assert all("model_context_token_ids" not in prompt for prompt in results["prompt_manifest"])
    assert all(
        "model_context_sha256" not in sample for sample in results["generation"]["samples"]
    )
    assert "/Users/" not in results_text
    for model in ("v1-textnca", "v1-transformer"):
        samples = [
            sample for sample in results["generation"]["samples"] if sample["model"] == model
        ]
        assert _macro_generation_summary(samples) == results["generation"]["summary"][model]
        hashes = results["identity"]["artifact_hashes"][model]
        assert all(
            len(hashes[field]) == 64
            for field in (
                "config_sha256",
                "dataset_manifest_sha256",
                "model_sha256",
                "trainer_state_sha256",
            )
        )
    assert (report_dir / "README.md").read_text(encoding="utf-8") == _markdown(results)
    assert (report_dir / "training-curves.svg").read_text(encoding="utf-8") == _curve_svg(
        results["training"]
    )
