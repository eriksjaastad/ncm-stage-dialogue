import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from ncm_stage_dialogue.v1_report import _macro_generation_summary
from ncm_stage_dialogue.v2_report import (
    MODEL_NAMES,
    _curve_svg,
    _investment_decision,
    _markdown,
    _version_comparison,
)


def _results(
    *,
    prefix: str,
    textnca_loss: float,
    transformer_loss: float,
    textnca_seconds: float,
    transformer_seconds: float,
    textnca_unique: float,
    transformer_unique: float,
    textnca_similarity: float,
    transformer_similarity: float,
) -> dict[str, Any]:
    return {
        "training": {
            "models": {
                f"{prefix}-textnca": {
                    "test": {"loss": textnca_loss},
                    "elapsed_seconds": textnca_seconds,
                },
                f"{prefix}-transformer": {
                    "test": {"loss": transformer_loss},
                    "elapsed_seconds": transformer_seconds,
                },
            }
        },
        "generation": {
            "summary": {
                f"{prefix}-textnca": {
                    "cross_prompt": {
                        "unique_continuation_rate": textnca_unique,
                        "mean_cross_work_character_trigram_jaccard": textnca_similarity,
                    }
                },
                f"{prefix}-transformer": {
                    "cross_prompt": {
                        "unique_continuation_rate": transformer_unique,
                        "mean_cross_work_character_trigram_jaccard": transformer_similarity,
                    }
                },
            }
        },
    }


def test_version_comparison_reports_model_specific_loss_and_collapse_changes() -> None:
    previous = _results(
        prefix="v1",
        textnca_loss=2.5,
        transformer_loss=2.3,
        textnca_seconds=100.0,
        transformer_seconds=25.0,
        textnca_unique=0.5,
        transformer_unique=1.0,
        textnca_similarity=0.9,
        transformer_similarity=0.2,
    )
    current = _results(
        prefix="v2",
        textnca_loss=2.25,
        transformer_loss=1.8,
        textnca_seconds=120.0,
        transformer_seconds=30.0,
        textnca_unique=0.75,
        transformer_unique=1.0,
        textnca_similarity=0.4,
        transformer_similarity=0.1,
    )

    comparison = _version_comparison(previous, current)

    assert comparison["textnca"]["test_loss_change_percent"] == -10.0
    assert comparison["textnca"]["unique_continuation_rate_change"] == 0.25
    assert comparison["textnca"]["cross_work_similarity_change"] == -0.5
    assert comparison["transformer"]["test_loss_change_percent"] == pytest.approx(
        -21.7391304348
    )


def test_investment_decision_stops_when_textnca_loses_loss_speed_and_collapse() -> None:
    current = _results(
        prefix="v2",
        textnca_loss=2.25,
        transformer_loss=1.8,
        textnca_seconds=120.0,
        transformer_seconds=30.0,
        textnca_unique=0.75,
        transformer_unique=1.0,
        textnca_similarity=0.4,
        transformer_similarity=0.1,
    )

    decision = _investment_decision(current)

    assert decision["decision"] == "stop"
    assert decision["criteria"] == {
        "textnca_test_loss_worse": True,
        "textnca_training_slower": True,
        "textnca_unique_rate_no_better": True,
        "textnca_cross_work_similarity_no_better": True,
    }


def test_investment_decision_fails_closed_when_cross_work_similarity_is_unavailable() -> None:
    current = _results(
        prefix="v2",
        textnca_loss=2.25,
        transformer_loss=1.8,
        textnca_seconds=120.0,
        transformer_seconds=30.0,
        textnca_unique=0.75,
        transformer_unique=1.0,
        textnca_similarity=0.4,
        transformer_similarity=0.1,
    )
    current["generation"]["summary"]["v2-textnca"]["cross_prompt"][
        "mean_cross_work_character_trigram_jaccard"
    ] = None

    decision = _investment_decision(current)

    assert decision["decision"] == "continue"
    assert decision["criteria"]["textnca_cross_work_similarity_no_better"] is False


def test_committed_v2_report_is_self_consistent_and_portable() -> None:
    repository = Path(__file__).parents[1]
    report_dir = repository / "reports" / "v2"
    results_text = (report_dir / "results.json").read_text(encoding="utf-8")
    results = json.loads(results_text)

    assert results["schema_version"] == 2
    assert results["identity"]["training_commit"] == (
        "d8cbe61523f2627cb786ab1d7d19d0fbc04e5a19"
    )
    assert results["identity"]["run_id"] == "v2-d8cbe61"
    assert len(results["prompt_manifest"]) == 12
    assert len(results["generation"]["samples"]) == 24
    assert results["protocol"]["training_seed"] == 123
    assert results["protocol"]["truncated_prompt_count"] == 11
    assert results["textnca_investment_decision"]["decision"] == "stop"
    assert "/Users/" not in results_text
    contexts_by_prompt: dict[str, str] = {}
    for prompt in results["prompt_manifest"]:
        context = prompt["model_context_token_ids"]
        assert len(context) == prompt["model_context_tokens"] <= 512
        assert prompt["model_context_sha256"] == hashlib.sha256(
            b"".join(token.to_bytes(2, "little") for token in context)
        ).hexdigest()
        contexts_by_prompt[prompt["prompt_id"]] = prompt["model_context_sha256"]
    for sample_record in results["generation"]["samples"]:
        assert sample_record["model_context_sha256"] == contexts_by_prompt[
            sample_record["prompt_id"]
        ]
    for model in MODEL_NAMES:
        samples = [
            sample for sample in results["generation"]["samples"] if sample["model"] == model
        ]
        assert _macro_generation_summary(samples) == results["generation"]["summary"][model]
    assert (report_dir / "README.md").read_text(encoding="utf-8") == _markdown(results)
    assert (report_dir / "training-curves.svg").read_text(encoding="utf-8") == _curve_svg(
        results["training"], MODEL_NAMES
    )
