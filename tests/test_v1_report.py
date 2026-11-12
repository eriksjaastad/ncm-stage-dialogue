from __future__ import annotations

import json
from pathlib import Path

import pytest

from ncm_stage_dialogue.v1_report import (
    _curve_svg,
    _macro_generation_summary,
    _markdown,
    _shared_training_commit,
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


def test_committed_v1_report_is_self_consistent_and_portable() -> None:
    repository = Path(__file__).parents[1]
    report_dir = repository / "reports" / "v1"
    results_text = (report_dir / "results.json").read_text(encoding="utf-8")
    results = json.loads(results_text)

    assert results["schema_version"] == 1
    assert results["identity"]["training_commit"] == ("7e4ecad7071f2ac4ea2bb99c520aac7291333713")
    assert len(results["prompt_manifest"]) == 9
    assert len(results["generation"]["samples"]) == 18
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
