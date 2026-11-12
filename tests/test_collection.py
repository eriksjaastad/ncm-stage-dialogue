from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from ncm_stage_dialogue.corpus import (
    VALID_SPLITS,
    CorpusError,
    Source,
    build_corpus,
    load_sources,
    parse_gutenberg_stage_collection,
    parse_gutenberg_stage_explicit,
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


def test_collection_parser_supports_included_start_and_exact_end_markers() -> None:
    text = wrapped(
        "CONTENTS\nWORK ONE\nWORK ONE\nThe action starts here.\n[A room.]\n"
        "ALICE. Hello.\nHIGH ROAD\nA VOICE FROM THE CORNER. Reply.\nCurtain."
    )
    source = collection_source(text)
    work = {
        "id": "work-one",
        "title": "WORK ONE",
        "author": "Author One",
        "heading": "WORK ONE",
        "heading_occurrence": 2,
        "split": "train",
        "scene_markers": [
            {
                "prefix": "The action starts here.",
                "scene_id": "scene-1",
                "include_marker": True,
                "end_marker": "Curtain.",
            }
        ],
        "speaker_labels": ["ALICE", "A VOICE FROM THE CORNER"],
        "ignored_lines": ["HIGH ROAD"],
    }
    source = replace(source, metadata={"works": [work]})

    scenes = parse_gutenberg_stage_collection(source, text)

    assert scenes[0]["stage_directions"] == ["The action starts here.", "A room."]
    assert [turn["speaker"] for turn in scenes[0]["turns"]] == [
        "ALICE",
        "A VOICE FROM THE CORNER",
    ]
    assert "HIGH ROAD" not in scenes[0]["turns"][0]["text"]

    with pytest.raises(CorpusError, match="scene end marker changed"):
        parse_gutenberg_stage_collection(source, text.replace("Curtain.", "Curtain"))
    with pytest.raises(CorpusError, match="ignored line occurrence changed"):
        parse_gutenberg_stage_collection(source, text.replace("HIGH ROAD\n", ""))


def test_collection_parser_validates_optional_marker_fields() -> None:
    text = fixture_text()
    source = collection_source(text)
    metadata = deepcopy(source.metadata)
    metadata["works"][0]["scene_markers"][0]["include_marker"] = "yes"

    with pytest.raises(CorpusError, match="include_marker must be a boolean"):
        parse_gutenberg_stage_collection(replace(source, metadata=metadata), text)

    metadata = deepcopy(source.metadata)
    metadata["works"][0]["scene_markers"][0]["match_mode"] = []
    with pytest.raises(CorpusError, match="match_mode must be exact or prefix"):
        parse_gutenberg_stage_collection(replace(source, metadata=metadata), text)


def test_chekhov_manifest_declares_complete_play_splits() -> None:
    manifest_path = Path(__file__).parents[1] / "sources" / "plays.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = next(
        item for item in manifest["sources"] if item["id"] == "gutenberg-7986-chekhov-second-series"
    )

    assert source["translator"] == "Julius West (1891–1918)"
    assert len(source["works"]) == 8
    assert {work["split"] for work in source["works"]} == {"train"}
    assert sum(len(work["scene_markers"]) for work in source["works"]) == 14
    assert all(
        marker["match_mode"] == "exact"
        for work in source["works"]
        for marker in work["scene_markers"]
    )


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


def test_explicit_parser_uses_pinned_source_line_ranges() -> None:
    text = wrapped(
        "FRONT MATTER\n"
        "ACT I\n"
        "A room.\n"
        "ALICE. Hello.\n"
        "BOB. Goodbye.\n"
        "ACT II\n"
        "A garden.\n"
        "ALICE. [Quietly.] Again.\n"
        "CURTAIN\n"
        "POSTMATTER"
    )
    lines = text.splitlines()
    content = text.encode()
    source = Source(
        id="fixture-explicit",
        title="Fixture Explicit Collection",
        author="Fixture Author",
        source_url="https://www.gutenberg.org/cache/epub/999998/pg999998.txt",
        source_sha256=sha256_bytes(content),
        source_bytes=len(content),
        parser="gutenberg_stage_explicit_v2",
        scene_splits={},
        speaker_labels=(),
        metadata={
            "works": [
                {
                    "id": "work-one",
                    "title": "Work One",
                    "author": "Author One",
                    "split": "train",
                    "speaker_labels": ["ALICE", "BOB"],
                    "ignored_line_numbers": [3, 7, 10],
                    "scenes": [
                        {
                            "scene_id": "act-1",
                            "act": 1,
                            "start_line": 3,
                            "end_line": 7,
                            "start_text": "ACT I",
                            "end_text": "BOB. Goodbye.",
                        },
                        {
                            "scene_id": "act-2",
                            "act": 2,
                            "start_line": 7,
                            "end_line": 11,
                            "start_text": "ACT II",
                            "end_text": "CURTAIN",
                        },
                    ],
                }
            ]
        },
    )

    scenes = parse_gutenberg_stage_explicit(source, text)

    assert [(scene["scene_id"], scene["act"], scene["split"]) for scene in scenes] == [
        ("work-one-act-1", 1, "train"),
        ("work-one-act-2", 2, "train"),
    ]
    assert scenes[0]["stage_directions"] == ["A room."]
    assert [turn["speaker"] for turn in scenes[0]["turns"]] == ["ALICE", "BOB"]
    assert scenes[1]["turns"][0] == {
        "speaker": "ALICE",
        "text": "Again.",
        "stage_directions": ["Quietly."],
    }
    assert scenes[0]["source_line_start"] == 3
    assert scenes[0]["source_line_end"] == 6
    assert lines[scenes[1]["source_line_end"] - 1].strip() == "CURTAIN"


def test_explicit_parser_rejects_boundary_drift() -> None:
    text = wrapped("ACT I\nALICE. Hello.\nCURTAIN")
    content = text.encode()
    source = Source(
        id="fixture-explicit",
        title="Fixture Explicit Collection",
        author="Fixture Author",
        source_url="https://www.gutenberg.org/cache/epub/999998/pg999998.txt",
        source_sha256=sha256_bytes(content),
        source_bytes=len(content),
        parser="gutenberg_stage_explicit_v2",
        scene_splits={},
        speaker_labels=(),
        metadata={
            "works": [
                {
                    "id": "work-one",
                    "title": "Work One",
                    "author": "Author One",
                    "split": "train",
                    "speaker_labels": ["ALICE"],
                    "ignored_line_numbers": [2, 4],
                    "scenes": [
                        {
                            "scene_id": "act-1",
                            "act": 1,
                            "start_line": 2,
                            "end_line": 5,
                            "start_text": "WRONG ACT",
                            "end_text": "CURTAIN",
                        }
                    ],
                }
            ]
        },
    )

    with pytest.raises(CorpusError, match="explicit scene start changed"):
        parse_gutenberg_stage_explicit(source, text)


@pytest.mark.parametrize(
    ("metadata_update", "message"),
    [
        ({"ignored_line_numbers": [2]}, "ignored line outside scene ranges"),
        (
            {
                "line_replacements": [
                    {"line_number": 2, "source": "FRONT MATTER", "replacement": "Front matter"}
                ]
            },
            "replacement line outside scene ranges",
        ),
    ],
)
def test_explicit_parser_rejects_line_operations_outside_scene_ranges(
    metadata_update: dict[str, object], message: str
) -> None:
    text = wrapped("FRONT MATTER\nACT I\nALICE. Hello.\nCURTAIN")
    content = text.encode()
    work = {
        "id": "work-one",
        "title": "Work One",
        "author": "Author One",
        "split": "train",
        "speaker_labels": ["ALICE"],
        "scenes": [
            {
                "scene_id": "act-1",
                "act": 1,
                "start_line": 3,
                "end_line": 6,
                "start_text": "ACT I",
                "end_text": "CURTAIN",
            }
        ],
        **metadata_update,
    }
    source = Source(
        id="fixture-explicit",
        title="Fixture Explicit Collection",
        author="Fixture Author",
        source_url="https://www.gutenberg.org/cache/epub/999998/pg999998.txt",
        source_sha256=sha256_bytes(content),
        source_bytes=len(content),
        parser="gutenberg_stage_explicit_v2",
        scene_splits={},
        speaker_labels=(),
        metadata={"works": [work]},
    )

    with pytest.raises(CorpusError, match=message):
        parse_gutenberg_stage_explicit(source, text)


def test_explicit_parser_rejects_ignored_line_between_scene_ranges() -> None:
    text = wrapped("ACT I\nALICE. Hello.\nUNCLAIMED\nACT II\nALICE. Goodbye.")
    content = text.encode()
    source = Source(
        id="fixture-explicit",
        title="Fixture Explicit Collection",
        author="Fixture Author",
        source_url="https://www.gutenberg.org/cache/epub/999998/pg999998.txt",
        source_sha256=sha256_bytes(content),
        source_bytes=len(content),
        parser="gutenberg_stage_explicit_v2",
        scene_splits={},
        speaker_labels=(),
        metadata={
            "works": [
                {
                    "id": "work-one",
                    "title": "Work One",
                    "author": "Author One",
                    "split": "train",
                    "speaker_labels": ["ALICE"],
                    "ignored_line_numbers": [4],
                    "scenes": [
                        {
                            "scene_id": "act-1",
                            "act": 1,
                            "start_line": 2,
                            "end_line": 4,
                            "start_text": "ACT I",
                            "end_text": "ALICE. Hello.",
                        },
                        {
                            "scene_id": "act-2",
                            "act": 2,
                            "start_line": 5,
                            "end_line": 7,
                            "start_text": "ACT II",
                            "end_text": "ALICE. Goodbye.",
                        },
                    ],
                }
            ]
        },
    )

    with pytest.raises(CorpusError, match="ignored line outside scene ranges"):
        parse_gutenberg_stage_explicit(source, text)


def test_author_split_isolation_rejects_coauthor_overlap(tmp_path: Path) -> None:
    text = fixture_text()
    source = collection_manifest_record(text)
    works = source["works"]
    assert isinstance(works, list)
    works[0]["author"] = "Author One"
    works[0]["split"] = "train"
    works[1]["author"] = "Author One and Author Two"
    works[1]["split"] = "test"
    manifest = tmp_path / "plays.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "author_split_isolation": True,
                "sources": [source],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match="author split leakage.*Author One"):
        load_sources(manifest)


@pytest.mark.parametrize("invalid_value", [False, 0])
def test_author_split_isolation_rejects_falsey_values(
    tmp_path: Path, invalid_value: object
) -> None:
    text = fixture_text()
    manifest = tmp_path / "plays.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "author_split_isolation": invalid_value,
                "sources": [collection_manifest_record(text)],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match="author_split_isolation must be true"):
        load_sources(manifest)


def test_v2_manifest_freezes_30_author_isolated_works() -> None:
    manifest_path = Path(__file__).parents[1] / "sources" / "plays.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert payload["author_split_isolation"] is True
    sources = load_sources(manifest_path)
    works: list[tuple[str, str]] = []
    for source in sources:
        if source.parser == "gutenberg_stage_play_v1":
            split = next(iter(set(source.scene_splits.values())))
            works.append((source.author, split))
        else:
            works.extend((work["author"], work["split"]) for work in source.metadata["works"])

    assert len(works) == 30
    assert {
        split: sum(work_split == split for _, work_split in works) for split in VALID_SPLITS
    } == {
        "train": 23,
        "validation": 3,
        "test": 4,
    }
    author_splits: dict[str, set[str]] = {}
    for author, split in works:
        for author_part in author.split(" and "):
            author_splits.setdefault(author_part, set()).add(split)
    assert all(len(splits) == 1 for splits in author_splits.values())


def test_corpus_v2_report_records_frozen_inventory() -> None:
    report = json.loads(
        (Path(__file__).parents[1] / "reports" / "corpus-v2" / "results.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["v2"]["work_count"] == 30
    assert report["v2"]["scenes"] == 75
    assert report["v2"]["turns"] == 14797
    assert report["v2"]["tokens"] == 2094294
    assert report["v2"]["author_leakage"] == {}
    assert len(report["v2"]["works"]) == 30
