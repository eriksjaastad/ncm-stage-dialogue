from __future__ import annotations

import json
from pathlib import Path

from ncm_stage_dialogue.publication import build_publication


def test_publication_builds_from_committed_reports(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    output = tmp_path / "public"

    build_publication(repository, output, "https://example.test")

    index = (output / "index.html").read_text(encoding="utf-8")
    privacy = (output / "privacy.html").read_text(encoding="utf-8")
    sitemap = (output / "sitemap.xml").read_text(encoding="utf-8")
    robots = (output / "robots.txt").read_text(encoding="utf-8")

    assert "We tested whether a TextNCA could learn stage dialogue. It did not." in index
    assert "1.813 vs 2.258" in index
    assert "3.30× faster" in index
    assert "100% vs 50%" in index
    assert "Stop TextNCA" in index
    assert "Every stage" in index
    assert "Corpus V1" in index and "Corpus V2" in index
    assert "Allowed held-out speaker" in index
    assert "Neither V2 model generated a correct held-out speaker" in index
    assert "/Users/" not in index
    assert "brain.db" not in index
    assert "does not use analytics, cookies" in privacy
    assert "https://example.test/privacy.html" in sitemap
    assert "Sitemap: https://example.test/sitemap.xml" in robots

    v1_source = json.loads((repository / "reports/v1/results.json").read_text(encoding="utf-8"))
    v2_source = json.loads((repository / "reports/v2/results.json").read_text(encoding="utf-8"))
    assert json.loads((output / "data/v1-results.json").read_text(encoding="utf-8")) == v1_source
    assert json.loads((output / "data/v2-results.json").read_text(encoding="utf-8")) == v2_source


def test_publication_has_only_declared_public_pages(tmp_path: Path) -> None:
    repository = Path(__file__).parents[1]
    output = tmp_path / "public"

    build_publication(repository, output, "https://example.test")

    relative_files = {
        path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()
    }
    assert relative_files == {
        "assets/style.css",
        "assets/v1-training-curves.svg",
        "assets/v2-training-curves.svg",
        "data/corpus-v2-results.json",
        "data/v1-results.json",
        "data/v2-results.json",
        "data/v2-textnca-config.json",
        "data/v2-transformer-config.json",
        "index.html",
        "privacy.html",
        "robots.txt",
        "sitemap.xml",
    }
