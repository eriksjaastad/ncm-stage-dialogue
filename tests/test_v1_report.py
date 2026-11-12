from __future__ import annotations

from ncm_stage_dialogue.v1_report import _macro_generation_summary


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
            "speaker_consistency": 0.0,
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
