from __future__ import annotations

import json
import shutil
from pathlib import Path

from ncm_stage_dialogue.publication import build_publication

REPOSITORY_URL = "https://github.com/example/research"


def _build(repository: Path, output: Path) -> None:
    build_publication(repository, output, "https://example.test/research", REPOSITORY_URL)


def test_publication_builds_from_committed_reports(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    output = tmp_path / "docs"

    _build(repository, output)

    index = (output / "index.html").read_text(encoding="utf-8")
    privacy = (output / "privacy.html").read_text(encoding="utf-8")
    sitemap = (output / "sitemap.xml").read_text(encoding="utf-8")
    robots = (output / "robots.txt").read_text(encoding="utf-8")
    public_data = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted((output / "data").glob("*.json"))
    )

    assert "Review draft" in index
    assert 'content="noindex,nofollow,noarchive"' in index
    assert "We tested whether a TextNCA could learn stage dialogue. It did not." in index
    assert "1.813 vs 2.258" in index
    assert "3.30× faster" in index
    assert "100% vs 50%" in index
    assert "Stop TextNCA" in index
    assert "Every stage" in index
    assert "Corpus V1" in index and "Corpus V2" in index
    assert "Token repeated trigrams" in index
    assert "Character repeated trigrams" in index
    assert "Provenance and locked protocol" in index
    assert "Allowed held-out speaker" in index
    assert "Neither V2 model generated a correct held-out speaker" in index
    assert REPOSITORY_URL in index
    assert "/Users/" not in index + public_data
    assert "data/processed" not in public_data
    assert "data/runs" not in public_data
    assert "checkpoint-" not in public_data
    assert "brain.db" not in index + public_data
    assert "does not use analytics, cookies" in privacy
    assert "GitHub Pages hosts" in privacy
    assert "https://example.test/research/privacy.html" in sitemap
    assert "Disallow: /" in robots
    assert (output / ".nojekyll").exists()


def test_headline_cards_follow_mutated_report_data(tmp_path: Path) -> None:
    source = Path(__file__).parents[1]
    repository = tmp_path / "repository"
    for directory in ("configs", "reports", "site", "sources"):
        shutil.copytree(source / directory, repository / directory)
    v2_path = repository / "reports/v2/results.json"
    v2 = json.loads(v2_path.read_text(encoding="utf-8"))
    v2["training"]["models"]["v2-textnca"]["test"]["loss"] = 9.876
    v2["training"]["models"]["v2-transformer"]["test"]["loss"] = 8.765
    v2["training"]["comparison"]["transformer_training_speedup"] = 9.99
    v2["generation"]["summary"]["v2-textnca"]["cross_prompt"][
        "unique_continuation_rate"
    ] = 0.25
    v2["generation"]["summary"]["v2-transformer"]["cross_prompt"][
        "unique_continuation_rate"
    ] = 0.75
    v2["textnca_investment_decision"]["decision"] = "continue"
    v2_path.write_text(json.dumps(v2), encoding="utf-8")
    config_path = repository / "configs/v2-textnca.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["training"]["train_steps"] = 4321
    config_path.write_text(json.dumps(config), encoding="utf-8")

    output = tmp_path / "mutated-docs"
    _build(repository, output)
    index = (output / "index.html").read_text(encoding="utf-8")

    assert "8.765 vs 9.876" in index
    assert "9.99× faster" in index
    assert "75% vs 25%" in index
    assert "Continue TextNCA" in index
    assert "four 4,321-step runs" in index


def test_publication_has_only_declared_public_files(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    output = tmp_path / "docs"

    _build(repository, output)

    relative_files = {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    }
    assert relative_files == {
        ".nojekyll",
        "assets/style.css",
        "assets/v1-training-curves.svg",
        "assets/v2-training-curves.svg",
        "data/corpus-v2-evidence.json",
        "data/experiment-configs.json",
        "data/v1-evidence.json",
        "data/v2-evidence.json",
        "index.html",
        "privacy.html",
        "robots.txt",
        "sitemap.xml",
    }
