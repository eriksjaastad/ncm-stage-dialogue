from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from ncm_stage_dialogue.corpus import (
    CorpusError,
    Source,
    build_corpus,
    parse_gutenberg_stage_collection,
    sha256_bytes,
)


def wrapped(body: str) -> str:
    return (
        "*** START OF THE PROJECT GUTENBERG EBOOK FIXTURE ***\n"
        f"{body}\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK FIXTURE ***"
    )


def collection_source(text: str) -> Source:
    content = text.encode()
    works = [
        {
            "id": "work-one",
            "title": "WORK ONE",
            "author": "Author One",
            "heading": "WORK ONE",
            "heading_occurrence": 2,
            "split": "train",
            "scene_markers": [{"prefix": "SCENE:", "scene_id": "scene-1"}],
            "speaker_labels": ["ALICE"],
        },
        {
            "id": "work-two",
            "title": "WORK TWO",
            "author": "Author Two",
            "heading": "WORK TWO",
            "heading_occurrence": 2,
            "split": "validation",
            "scene_markers": [
                {"prefix": "SCENE I:", "scene_id": "scene-1"},
                {"prefix": "SCENE II:", "scene_id": "scene-2"},
            ],
            "speaker_labels": ["BOB"],
        },
    ]
    return Source(
        id="fixture-collection",
        title="Fixture Collection",
        author="Various",
        source_url="https://www.gutenberg.org/cache/epub/999999/pg999999.txt",
        source_sha256=sha256_bytes(content),
        source_bytes=len(content),
        parser="gutenberg_stage_collection_v1",
        scene_splits={},
        speaker_labels=(),
        metadata={"works": works},
    )


def collection_manifest_record(text: str) -> dict[str, object]:
    source = collection_source(text)
    return {
        "id": source.id,
        "title": source.title,
        "author": source.author,
        "language": "en",
        "original_publication_year": 1920,
        "source_landing_page": "https://www.gutenberg.org/ebooks/999999",
        "source_url": source.source_url,
        "source_sha256": source.source_sha256,
        "source_bytes": source.source_bytes,
        "retrieved_on": "2026-11-09",
        "jurisdiction": "United States",
        "public_domain_basis": "Fixture public-domain statement.",
        "source_terms": "Fixture source terms.",
        "parser": source.parser,
        "works": source.metadata["works"],
    }


def fixture_text() -> str:
    return wrapped(
        "CONTENTS\n"
        "WORK ONE\n"
        "WORK TWO\n"
        "WORK ONE\n"
        "SCENE: _A kitchen._\n"
        "ALICE\n"
        "Hello from the first work.\n"
        "(CURTAIN)\n"
        "WORK TWO\n"
        "SCENE I: _Morning._\n"
        "BOB\n"
        "First scene.\n"
        "(CURTAIN)\n"
        "SCENE II: _Evening._\n"
        "BOB\n"
        "Second scene.\n"
        "(CURTAIN)"
    )


def test_collection_parser_emits_independent_works_and_complete_work_splits() -> None:
    text = fixture_text()

    scenes = parse_gutenberg_stage_collection(collection_source(text), text)

    assert [(scene["work_id"], scene["scene_id"], scene["split"]) for scene in scenes] == [
        ("work-one", "work-one-scene-1", "train"),
        ("work-two", "work-two-scene-1", "validation"),
        ("work-two", "work-two-scene-2", "validation"),
    ]
    assert [scene["turns"][0]["speaker"] for scene in scenes] == ["ALICE", "BOB", "BOB"]
    assert scenes[0]["stage_directions"] == ["A kitchen."]
    assert scenes[1]["stage_directions"] == ["Morning."]
    assert scenes[2]["stage_directions"] == ["Evening."]
    assert all("CURTAIN" not in scene["turns"][-1]["text"] for scene in scenes)


def test_build_corpus_dispatches_collection_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = fixture_text()
    manifest = tmp_path / "plays.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "sources": [collection_manifest_record(text)]}),
        encoding="utf-8",
    )

    def fake_download(source: Source, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text, encoding="utf-8")
        return destination

    monkeypatch.setattr("ncm_stage_dialogue.corpus.download_source", fake_download)

    counts = build_corpus(manifest, tmp_path / "data")
    records = [
        json.loads(line)
        for line in (tmp_path / "data" / "processed" / "scenes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert counts == {
        "scenes": 3,
        "test": 0,
        "train": 1,
        "turns": 3,
        "validation": 2,
    }
    assert {record["work_id"] for record in records} == {"work-one", "work-two"}
    assert all(record["source_id"] == "fixture-collection" for record in records)


def test_collection_parser_rejects_missing_work_boundary() -> None:
    text = fixture_text().replace("\nWORK TWO\nSCENE I:", "\nMISSING WORK\nSCENE I:")

    with pytest.raises(CorpusError, match="work heading occurrence changed"):
        parse_gutenberg_stage_collection(collection_source(text), text)


def test_collection_parser_rejects_missing_scene_boundary() -> None:
    text = fixture_text().replace("SCENE II: _Evening._", "MISSING SCENE")

    with pytest.raises(CorpusError, match="scene marker occurrence changed"):
        parse_gutenberg_stage_collection(collection_source(text), text)


def test_collection_parser_rejects_suspicious_undeclared_speaker() -> None:
    text = fixture_text().replace(
        "SCENE: _A kitchen._\nALICE", "SCENE: _A kitchen._\nINTRUDER\nALICE"
    )

    with pytest.raises(CorpusError, match="suspicious undeclared speaker label"):
        parse_gutenberg_stage_collection(collection_source(text), text)


def test_collection_parser_requires_every_declared_speaker() -> None:
    text = fixture_text()
    source = collection_source(text)
    metadata = deepcopy(source.metadata)
    metadata["works"][0]["speaker_labels"].append("CAROL")

    with pytest.raises(CorpusError, match="declared speakers missing.*CAROL"):
        parse_gutenberg_stage_collection(replace(source, metadata=metadata), text)


@pytest.mark.parametrize("invalid_split", [["train"], {"split": "train"}])
def test_collection_parser_rejects_non_string_split(invalid_split: object) -> None:
    text = fixture_text()
    source = collection_source(text)
    metadata = deepcopy(source.metadata)
    metadata["works"][0]["split"] = invalid_split

    with pytest.raises(CorpusError, match="invalid work split"):
        parse_gutenberg_stage_collection(replace(source, metadata=metadata), text)


def test_collection_parser_rejects_duplicate_work_ids() -> None:
    text = fixture_text()
    source = collection_source(text)
    metadata = deepcopy(source.metadata)
    metadata["works"][1]["id"] = "work-one"
    duplicate_source = Source(
        id=source.id,
        title=source.title,
        author=source.author,
        source_url=source.source_url,
        source_sha256=source.source_sha256,
        source_bytes=source.source_bytes,
        parser=source.parser,
        scene_splits=source.scene_splits,
        speaker_labels=source.speaker_labels,
        metadata=metadata,
    )

    with pytest.raises(CorpusError, match="duplicate work id"):
        parse_gutenberg_stage_collection(duplicate_source, text)
