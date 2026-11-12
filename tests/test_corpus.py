from __future__ import annotations

import json
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import replace
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


def source_for(
    content: bytes,
    scene_splits: dict[str, str] | None = None,
    speaker_labels: tuple[str, ...] = ("ALICE", "BOB"),
) -> Source:
    return Source(
        id="fixture-play",
        title="Fixture Play",
        author="Test Author",
        source_url="https://example.invalid/play.txt",
        source_sha256=sha256_bytes(content),
        source_bytes=len(content),
        parser="gutenberg_stage_play_v1",
        scene_splits=scene_splits or {"act-1": "train"},
        speaker_labels=speaker_labels,
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
        "speaker_labels": ["ALICE", "BOB"],
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
    content = wrapped("FIRST ACT\nALICE.\nFirst version.\nFIRST ACT\nBOB.\nDuplicate version.")

    with pytest.raises(CorpusError, match="duplicate scene id: act-1"):
        parse_gutenberg_stage_play(source_for(content), content.decode())


def test_uppercase_prose_is_not_misclassified_as_a_speaker() -> None:
    content = wrapped(
        "FIRST ACT\n"
        "ALICE.\n"
        "Before the emphasis.\n"
        "THIS IS A SENTENCE.\n"
        "After the emphasis.\n"
        "BOB.\n"
        "Reply."
    )

    scenes = parse_gutenberg_stage_play(source_for(content), content.decode())

    assert scenes[0]["turns"][0] == {
        "speaker": "ALICE",
        "text": "Before the emphasis. THIS IS A SENTENCE. After the emphasis.",
        "stage_directions": [],
    }


def test_title_case_names_and_direction_continuations_are_not_speakers() -> None:
    content = wrapped(
        "FIRST ACT\n"
        "BOB.\n"
        "Alice. Your name.\n"
        "ALICE._] remains part of the damaged direction.\n"
        "ALICE.\n"
        "A real turn."
    )

    scenes = parse_gutenberg_stage_play(source_for(content), content.decode())

    assert scenes[0]["turns"][0]["speaker"] == "BOB"
    assert scenes[0]["turns"][0]["text"] == (
        "Alice. Your name. ALICE.] remains part of the damaged direction."
    )
    assert scenes[0]["turns"][1]["speaker"] == "ALICE"


def test_parser_recognizes_joint_speakers_with_inline_direction_and_text() -> None:
    content = wrapped(
        "FIRST ACT\n"
        "ALICE.\n"
        "A solo line.\n"
        "ALICE and BOB [Speaking together.] Our joint line.\n"
        "BOB.\n"
        "A final line."
    )
    source = source_for(
        content,
        speaker_labels=("ALICE", "BOB", "ALICE AND BOB"),
    )

    scenes = parse_gutenberg_stage_play(source, content.decode())

    assert scenes[0]["turns"][1] == {
        "speaker": "ALICE AND BOB",
        "text": "Our joint line.",
        "stage_directions": ["Speaking together."],
    }


def test_parser_separates_parenthesized_italic_stage_directions() -> None:
    content = wrapped(
        "FIRST ACT\nALICE. (_quietly_) Hello.\nBOB. (_crossing to the door_) Goodbye."
    )

    scenes = parse_gutenberg_stage_play(source_for(content), content.decode())

    assert scenes[0]["turns"] == [
        {"speaker": "ALICE", "text": "Hello.", "stage_directions": ["quietly"]},
        {
            "speaker": "BOB",
            "text": "Goodbye.",
            "stage_directions": ["crossing to the door"],
        },
    ]


def test_strict_parser_accepts_uppercase_names_inside_multiline_directions() -> None:
    content = wrapped(
        "FIRST ACT\nALICE. Hello.\n[BOB crosses the room,\nCHARLIE. watches him.]\nBOB. Goodbye."
    )
    source = replace(
        source_for(content),
        metadata={"strict_speaker_labels": True},
    )

    scenes = parse_gutenberg_stage_play(source, content.decode())

    assert scenes[0]["turns"][0]["stage_directions"] == [
        "BOB crosses the room, CHARLIE. watches him."
    ]


def test_declared_act_markers_skip_toc_and_enforce_complete_play_split() -> None:
    content = wrapped("CONTENTS\nACT I.\nACT II.\nACT I\nALICE.\nHello.\nACT II\nBOB.\nGoodbye.")
    source = source_for(
        content,
        scene_splits={"act-1": "train", "act-2": "train"},
    )
    source = replace(
        source,
        metadata={
            **source.metadata,
            "act_markers": [
                {"act_number": 1, "scene_id": "act-1", "line": "ACT I", "occurrence": 1},
                {"act_number": 2, "scene_id": "act-2", "line": "ACT II", "occurrence": 1},
            ],
            "complete_play_split": True,
            "strict_speaker_labels": True,
        },
    )

    scenes = parse_gutenberg_stage_play(source, content.decode())

    assert [(scene["scene_id"], scene["split"]) for scene in scenes] == [
        ("act-1", "train"),
        ("act-2", "train"),
    ]
    assert [scene["turns"][0]["speaker"] for scene in scenes] == ["ALICE", "BOB"]

    bad_metadata = dict(source.metadata)
    bad_metadata["act_markers"] = [
        {"act_number": 1, "scene_id": "act-1", "line": "ACT I", "occurrence": 2},
        {"act_number": 2, "scene_id": "act-2", "line": "ACT II", "occurrence": 1},
    ]
    with pytest.raises(CorpusError, match="act marker occurrence changed"):
        parse_gutenberg_stage_play(replace(source, metadata=bad_metadata), content.decode())

    bogus_metadata = dict(source.metadata)
    bogus_metadata["act_markers"] = [
        {"act_number": 1, "scene_id": "act-1", "line": "BOGUS MARKER", "occurrence": 1},
        {"act_number": 2, "scene_id": "act-2", "line": "ACT II", "occurrence": 1},
    ]
    with pytest.raises(CorpusError, match="act marker identity mismatch"):
        parse_gutenberg_stage_play(replace(source, metadata=bogus_metadata), content.decode())

    reordered_metadata = dict(source.metadata)
    reordered_metadata["act_markers"] = list(reversed(source.metadata["act_markers"]))
    with pytest.raises(CorpusError, match="act markers must be sequential"):
        parse_gutenberg_stage_play(replace(source, metadata=reordered_metadata), content.decode())

    with pytest.raises(CorpusError, match="suspicious undeclared speaker label"):
        parse_gutenberg_stage_play(
            source,
            content.decode().replace("BOB.\nGoodbye.", "INTRUDER. Wrong speaker."),
        )


def test_declared_line_replacement_preserves_nonstandard_speaker_layout() -> None:
    content = wrapped("FIRST ACT\nALICE   {exclaiming  Hello!\n***\nALICE. Leaked prose.")
    source = source_for(content, speaker_labels=("ALICE",))
    source = replace(
        source,
        metadata={
            **source.metadata,
            "line_replacements": [
                {
                    "source": "ALICE   {exclaiming  Hello!",
                    "replacement": "ALICE. [exclaiming] Hello!",
                    "occurrence": 1,
                }
            ],
            "play_end_marker": {"line": "***", "occurrence": 1},
            "strict_speaker_labels": True,
        },
    )

    scenes = parse_gutenberg_stage_play(source, content.decode())

    assert scenes[0]["turns"] == [
        {"speaker": "ALICE", "text": "Hello!", "stage_directions": ["exclaiming"]}
    ]
    with pytest.raises(CorpusError, match="line replacement occurrence changed"):
        parse_gutenberg_stage_play(source, content.decode().replace("exclaiming", "calling"))
    with pytest.raises(CorpusError, match="play end marker occurrence changed"):
        parse_gutenberg_stage_play(source, content.decode().replace("***", "---"))


def test_strict_parser_separates_unlabelled_action_paragraphs() -> None:
    content = wrapped(
        "FIRST ACT\nALICE. Hello.\n\n[She smiles].\nStill speaking.\n\nShe exits.\n\nBOB. Goodbye."
    )
    source = source_for(content)
    source = replace(
        source,
        metadata={**source.metadata, "strict_speaker_labels": True},
    )

    scenes = parse_gutenberg_stage_play(source, content.decode())

    assert scenes[0]["turns"] == [
        {
            "speaker": "ALICE",
            "text": "Hello. Still speaking.",
            "stage_directions": ["She smiles"],
        },
        {"speaker": "BOB", "text": "Goodbye.", "stage_directions": []},
    ]
    assert scenes[0]["stage_directions"] == ["She exits."]


def test_pygmalion_manifest_pins_five_act_complete_play() -> None:
    manifest = Path(__file__).parents[1] / "sources" / "plays.json"
    source = next(item for item in load_sources(manifest) if item.id == "gutenberg-3825-pygmalion")

    assert source.source_sha256 == (
        "bf2518b5c0f1e46e3190ab2e77a0ecb3e35a1e120b233e6cb27134395dff048a"
    )
    assert source.source_bytes == 215094
    assert source.metadata["act_markers"] == [
        {
            "act_number": act_number,
            "scene_id": f"act-{act_number}",
            "line": f"ACT {roman}",
            "occurrence": 1,
        }
        for act_number, roman in enumerate(("I", "II", "III", "IV", "V"), start=1)
    ]
    assert source.scene_splits == {f"act-{number}": "validation" for number in range(1, 6)}
    assert source.metadata["complete_play_split"] is True
    assert source.metadata["strict_speaker_labels"] is True
    assert source.metadata["play_end_marker"] == {
        "line": "***********************",
        "occurrence": 1,
    }
    assert "THE GENTLEMAN" in source.speaker_labels
    assert "THE GENTLEMEN" in source.speaker_labels


def test_manifest_rejects_mixed_complete_play_splits(tmp_path: Path) -> None:
    content = wrapped("ACT I\nALICE.\nHello.\nACT II\nBOB.\nGoodbye.")
    raw_source = complete_manifest_source(content)
    raw_source["scene_splits"] = {"act-1": "train", "act-2": "validation"}
    raw_source["act_markers"] = [
        {"act_number": 1, "scene_id": "act-1", "line": "ACT I", "occurrence": 1},
        {"act_number": 2, "scene_id": "act-2", "line": "ACT II", "occurrence": 1},
    ]
    raw_source["complete_play_split"] = True
    manifest = tmp_path / "plays.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "sources": [raw_source]}),
        encoding="utf-8",
    )

    with pytest.raises(CorpusError, match="complete play must use one split"):
        load_sources(manifest)


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
def test_manifest_requires_complete_provenance(tmp_path: Path, missing_field: str) -> None:
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
