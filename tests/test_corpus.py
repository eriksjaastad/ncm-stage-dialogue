from __future__ import annotations

import json
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest

from ncm_stage_dialogue.cli import main
from ncm_stage_dialogue.corpus import (
    CorpusError,
    HttpsOnlyRedirectHandler,
    Source,
    build_corpus,
    download_source,
    load_sources,
    parse_gutenberg_stage_play,
    sha256_bytes,
    strip_gutenberg_wrapper,
)


def source_for(content: bytes, scene_splits: dict[str, str] | None = None) -> Source:
    return Source(
        id="fixture-play",
        title="Fixture Play",
        author="Test Author",
        source_url="https://example.invalid/play.txt",
        source_sha256=sha256_bytes(content),
        source_bytes=len(content),
        parser="gutenberg_stage_play_v1",
        scene_splits=scene_splits or {"act-1": "train"},
        metadata={"jurisdiction": "Test", "public_domain_basis": "Synthetic fixture"},
    )


def wrapped(body: str) -> bytes:
    return (
        "Project Gutenberg header\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK FIXTURE ***\n"
        f"{body}\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK FIXTURE ***\n"
        "Project Gutenberg license footer\n"
    ).encode()


def complete_manifest_source(content: bytes) -> dict[str, object]:
    return {
        "id": "fixture-play",
        "title": "Fixture Play",
        "author": "Test Author",
        "language": "en",
        "original_publication_year": 1900,
        "source_landing_page": "https://www.gutenberg.org/ebooks/999999",
        "source_url": "https://www.gutenberg.org/cache/epub/999999/pg999999.txt",
        "source_sha256": sha256_bytes(content),
        "source_bytes": len(content),
        "retrieved_on": "2026-11-08",
        "jurisdiction": "Test",
        "public_domain_basis": "Synthetic fixture",
        "source_terms": "Synthetic fixture terms",
        "parser": "gutenberg_stage_play_v1",
        "scene_splits": {"act-1": "train"},
    }


def test_strip_gutenberg_wrapper_removes_header_and_footer() -> None:
    text = wrapped("FIRST ACT\nALICE.\nHello.").decode()

    assert strip_gutenberg_wrapper(text) == "FIRST ACT\nALICE.\nHello."


def test_download_fails_closed_when_source_hash_changes(tmp_path: Path) -> None:
    expected = wrapped("FIRST ACT\nALICE.\nHello.")
    source = source_for(expected)

    with pytest.raises(CorpusError, match="source byte count changed"):
        download_source(source, tmp_path / "play.txt", fetch=lambda _: expected + b"changed")

    assert not (tmp_path / "play.txt").exists()


def test_parser_emits_speaker_turns_and_separate_directions() -> None:
    content = wrapped(
        "FIRST ACT\n"
        "SCENE\n"
        "Morning in a drawing room.\n"
        "[A drawing room.]\n"
        "ALICE.\n"
        "[Quietly.] Hello there.\n"
        "BOB.\n"
        "Good morning."
    )

    scenes = parse_gutenberg_stage_play(source_for(content), content.decode())

    assert scenes == [
        {
            "source_id": "fixture-play",
            "title": "Fixture Play",
            "author": "Test Author",
            "scene_id": "act-1",
            "act": 1,
            "split": "train",
            "stage_directions": ["Morning in a drawing room.", "A drawing room."],
            "turns": [
                {
                    "speaker": "ALICE",
                    "text": "Hello there.",
                    "stage_directions": ["Quietly."],
                },
                {
                    "speaker": "BOB",
                    "text": "Good morning.",
                    "stage_directions": [],
                },
            ],
        }
    ]


def test_scene_splits_keep_every_turn_in_one_complete_scene() -> None:
    content = wrapped(
        "FIRST ACT\nALICE.\nTrain line.\n"
        "SECOND ACT\nALICE.\nValidation line.\n"
        "THIRD ACT\nALICE.\nTest line."
    )
    source = source_for(
        content,
        {"act-1": "train", "act-2": "validation", "act-3": "test"},
    )

    scenes = parse_gutenberg_stage_play(source, content.decode())

    assert [(scene["scene_id"], scene["split"]) for scene in scenes] == [
        ("act-1", "train"),
        ("act-2", "validation"),
        ("act-3", "test"),
    ]
    assert all(len(scene["turns"]) == 1 for scene in scenes)


def test_manifest_requires_provenance_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "plays.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "sources": [{"id": "incomplete"}]}),
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match="missing required fields"):
        load_sources(manifest)


def test_manifest_rejects_source_id_path_traversal(tmp_path: Path) -> None:
    content = wrapped("FIRST ACT\nALICE.\nHello.")
    raw_source = complete_manifest_source(content)
    raw_source["id"] = "../escaped"
    manifest = tmp_path / "plays.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "sources": [raw_source]}),
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match="unsafe source id"):
        load_sources(manifest)


def test_parser_rejects_duplicate_scene_ids() -> None:
    content = wrapped(
        "FIRST ACT\nALICE.\nFirst version.\n"
        "FIRST ACT\nBOB.\nDuplicate version."
    )

    with pytest.raises(CorpusError, match="duplicate scene id: act-1"):
        parse_gutenberg_stage_play(source_for(content), content.decode())


def test_processed_records_retain_complete_source_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = wrapped("FIRST ACT\nALICE.\nHello.")
    raw_source = complete_manifest_source(content)
    manifest = tmp_path / "plays.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "sources": [raw_source]}),
        encoding="utf-8",
    )

    def fake_download(source: Source, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        return destination

    monkeypatch.setattr("ncm_stage_dialogue.corpus.download_source", fake_download)
    build_corpus(manifest, tmp_path / "data")

    record = json.loads(
        (tmp_path / "data" / "processed" / "scenes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    assert record["provenance"]["source"] == raw_source
    assert record["provenance"]["transformation"] == {
        "parser": "gutenberg_stage_play_v1",
        "project_gutenberg_wrapper_removed": True,
        "stage_directions_separated": True,
    }


@pytest.mark.parametrize(
    "missing_field",
    [
        "language",
        "original_publication_year",
        "source_landing_page",
        "retrieved_on",
        "source_terms",
    ],
)
def test_manifest_requires_complete_provenance(
    tmp_path: Path, missing_field: str
) -> None:
    content = wrapped("FIRST ACT\nALICE.\nHello.")
    raw_source = complete_manifest_source(content)
    del raw_source[missing_field]
    manifest = tmp_path / "plays.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "sources": [raw_source]}),
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match=missing_field):
        load_sources(manifest)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update(sources={}), "sources must be a list"),
        (lambda payload: payload.update(sources=[None]), "source 0 must be an object"),
        (
            lambda payload: payload["sources"][0].update(scene_splits=[]),
            "scene_splits must be an object",
        ),
        (
            lambda payload: payload["sources"][0].update(source_bytes="100"),
            "source_bytes must be an integer",
        ),
    ],
)
def test_malformed_manifest_raises_corpus_error(
    tmp_path: Path,
    mutate: Callable[[dict[str, object]], None],
    message: str,
) -> None:
    content = wrapped("FIRST ACT\nALICE.\nHello.")
    payload: dict[str, object] = {
        "schema_version": 1,
        "sources": [complete_manifest_source(content)],
    }
    mutate(payload)
    manifest = tmp_path / "plays.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CorpusError, match=message):
        load_sources(manifest)


def test_cli_reports_malformed_manifest_without_traceback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    manifest = tmp_path / "plays.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "sources": [None]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["build-corpus", "--manifest", str(manifest), "--data-root", str(tmp_path / "data")],
    )

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == 1
    assert capsys.readouterr().err == "build-corpus: source 0 must be an object\n"


@pytest.mark.parametrize(
    "source_url",
    [
        "file:///etc/hosts",
        "http://www.gutenberg.org/cache/epub/1/pg1.txt",
        "https://example.invalid/play.txt",
        "https://user:password@www.gutenberg.org/play.txt",
    ],
)
def test_manifest_rejects_unsafe_source_urls(tmp_path: Path, source_url: str) -> None:
    content = wrapped("FIRST ACT\nALICE.\nHello.")
    raw_source = complete_manifest_source(content)
    raw_source["source_url"] = source_url
    manifest = tmp_path / "plays.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "sources": [raw_source]}),
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match="unsafe source_url"):
        load_sources(manifest)


def test_https_redirect_handler_rejects_scheme_or_host_changes() -> None:
    handler = HttpsOnlyRedirectHandler({"www.gutenberg.org"})
    request = urllib.request.Request("https://www.gutenberg.org/ebooks/1")

    for target in (
        "http://www.gutenberg.org/ebooks/1",
        "file:///etc/hosts",
        "https://example.invalid/ebooks/1",
    ):
        with pytest.raises(CorpusError, match="unsafe redirect URL"):
            handler.redirect_request(request, None, 302, "Found", {}, target)

    redirected = handler.redirect_request(
        request,
        None,
        302,
        "Found",
        {},
        "https://www.gutenberg.org/cache/epub/1/pg1.txt",
    )
    assert redirected is not None
    assert redirected.full_url == "https://www.gutenberg.org/cache/epub/1/pg1.txt"


def test_unsupported_parser_causes_no_fetch_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = wrapped("FIRST ACT\nALICE.\nHello.")
    raw_source = complete_manifest_source(content)
    raw_source["parser"] = "unknown_parser"
    manifest = tmp_path / "plays.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "sources": [raw_source]}),
        encoding="utf-8",
    )
    fetched = False

    def fake_download(source: Source, destination: Path) -> Path:
        nonlocal fetched
        fetched = True
        return destination

    monkeypatch.setattr("ncm_stage_dialogue.corpus.download_source", fake_download)

    with pytest.raises(CorpusError, match="unsupported parser"):
        build_corpus(manifest, tmp_path / "data")

    assert fetched is False
    assert not (tmp_path / "data").exists()
