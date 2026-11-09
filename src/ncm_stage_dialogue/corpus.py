"""Download and parse provenance-pinned public-domain stage plays."""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SPEAKER_RE = re.compile(r"^[A-Z][A-Z .’'\-]{1,60}\.$")
ACT_RE = re.compile(r"^(FIRST|SECOND|THIRD|FOURTH|FIFTH) ACT$")
DIRECTION_RE = re.compile(r"\[([^]]+)\]", re.DOTALL)
VALID_SPLITS = {"train", "validation", "test"}
ORDINALS = {"FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5}


class CorpusError(RuntimeError):
    """Raised when source provenance or parsing invariants fail."""


@dataclass(frozen=True)
class Source:
    """One immutable source-text record from the corpus manifest."""

    id: str
    title: str
    author: str
    source_url: str
    source_sha256: str
    source_bytes: int
    parser: str
    scene_splits: dict[str, str]
    metadata: dict[str, Any]


def load_sources(manifest_path: Path) -> list[Source]:
    """Load and validate corpus sources from a JSON manifest."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise CorpusError("manifest schema_version must be 1")

    required = {
        "id",
        "title",
        "author",
        "source_url",
        "source_sha256",
        "source_bytes",
        "parser",
        "scene_splits",
        "jurisdiction",
        "public_domain_basis",
    }
    sources: list[Source] = []
    seen: set[str] = set()
    for raw in payload.get("sources", []):
        missing = sorted(required - raw.keys())
        if missing:
            raise CorpusError(f"source is missing required fields: {', '.join(missing)}")
        if raw["id"] in seen:
            raise CorpusError(f"duplicate source id: {raw['id']}")
        seen.add(raw["id"])
        if not re.fullmatch(r"[0-9a-f]{64}", raw["source_sha256"]):
            raise CorpusError(f"invalid SHA-256 for {raw['id']}")
        if set(raw["scene_splits"].values()) - VALID_SPLITS:
            raise CorpusError(f"invalid scene split for {raw['id']}")
        sources.append(
            Source(
                id=raw["id"],
                title=raw["title"],
                author=raw["author"],
                source_url=raw["source_url"],
                source_sha256=raw["source_sha256"],
                source_bytes=int(raw["source_bytes"]),
                parser=raw["parser"],
                scene_splits=dict(raw["scene_splits"]),
                metadata=dict(raw),
            )
        )
    if not sources:
        raise CorpusError("manifest contains no sources")
    return sources


def sha256_bytes(content: bytes) -> str:
    """Return the lowercase SHA-256 digest for content."""
    return hashlib.sha256(content).hexdigest()


def download_source(
    source: Source,
    destination: Path,
    fetch: Callable[[str], bytes] | None = None,
) -> Path:
    """Download one source, fail closed on byte-count or hash drift, and save atomically."""

    def fetch_url(url: str) -> bytes:
        request = urllib.request.Request(url, headers={"User-Agent": "ncm-stage-dialogue/0.1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()

    content = (fetch or fetch_url)(source.source_url)
    actual_hash = sha256_bytes(content)
    if len(content) != source.source_bytes:
        raise CorpusError(
            f"source byte count changed for {source.id}: "
            f"expected {source.source_bytes}, got {len(content)}"
        )
    if actual_hash != source.source_sha256:
        raise CorpusError(
            f"source hash changed for {source.id}: "
            f"expected {source.source_sha256}, got {actual_hash}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(destination)
    return destination


def strip_gutenberg_wrapper(text: str) -> str:
    """Remove the Project Gutenberg header and license footer from a text."""
    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if "START OF THE PROJECT GUTENBERG EBOOK" in line),
        None,
    )
    end = next(
        (index for index, line in enumerate(lines) if "END OF THE PROJECT GUTENBERG EBOOK" in line),
        None,
    )
    if start is None or end is None or end <= start:
        raise CorpusError("Project Gutenberg start/end markers were not found")
    return "\n".join(lines[start + 1 : end]).strip()


def _normalize_passage(lines: list[str]) -> tuple[str, list[str]]:
    raw = " ".join(line.strip() for line in lines if line.strip())
    raw = re.sub(r"\s+", " ", raw).strip()
    directions = [re.sub(r"\s+", " ", item).strip() for item in DIRECTION_RE.findall(raw)]
    spoken = DIRECTION_RE.sub(" ", raw)
    spoken = re.sub(r"\s+", " ", spoken).strip()
    return spoken, directions


def parse_gutenberg_stage_play(source: Source, text: str) -> list[dict[str, Any]]:
    """Parse an act-structured Gutenberg play into scene-level records."""
    body = strip_gutenberg_wrapper(text)
    scenes: list[dict[str, Any]] = []
    current_scene: dict[str, Any] | None = None
    current_speaker: str | None = None
    turn_lines: list[str] = []
    prelude_lines: list[str] = []

    def flush_turn() -> None:
        nonlocal current_speaker, turn_lines
        if current_scene is None or current_speaker is None:
            return
        spoken, directions = _normalize_passage(turn_lines)
        if spoken:
            current_scene["turns"].append(
                {
                    "speaker": current_speaker,
                    "text": spoken,
                    "stage_directions": directions,
                }
            )
        elif directions:
            current_scene["stage_directions"].extend(directions)
        current_speaker = None
        turn_lines = []

    def flush_scene() -> None:
        nonlocal current_scene, prelude_lines
        flush_turn()
        if current_scene is None:
            return
        prelude_text, prelude_directions = _normalize_passage(prelude_lines)
        if prelude_text:
            current_scene["stage_directions"].append(prelude_text)
        current_scene["stage_directions"].extend(prelude_directions)
        if not current_scene["turns"]:
            raise CorpusError(f"scene has no dialogue: {current_scene['scene_id']}")
        scenes.append(current_scene)
        current_scene = None
        prelude_lines = []

    for raw_line in body.splitlines():
        line = raw_line.strip()
        act_match = ACT_RE.fullmatch(line)
        if act_match:
            flush_scene()
            act_number = ORDINALS[act_match.group(1)]
            scene_id = f"act-{act_number}"
            split = source.scene_splits.get(scene_id)
            if split not in VALID_SPLITS:
                raise CorpusError(f"missing valid split for {source.id}/{scene_id}")
            current_scene = {
                "source_id": source.id,
                "title": source.title,
                "author": source.author,
                "scene_id": scene_id,
                "act": act_number,
                "split": split,
                "stage_directions": [],
                "turns": [],
            }
            continue
        if current_scene is None:
            continue
        if SPEAKER_RE.fullmatch(line):
            flush_turn()
            current_speaker = line[:-1]
            continue
        if current_speaker is None:
            if line != "SCENE":
                prelude_lines.append(line)
        else:
            turn_lines.append(line)

    flush_scene()
    expected_scenes = set(source.scene_splits)
    actual_scenes = {scene["scene_id"] for scene in scenes}
    if actual_scenes != expected_scenes:
        raise CorpusError(
            f"scene set changed for {source.id}: expected {sorted(expected_scenes)}, "
            f"got {sorted(actual_scenes)}"
        )
    return scenes


def build_corpus(manifest_path: Path, data_root: Path) -> dict[str, int]:
    """Download all manifest sources and emit validated scene-level JSONL."""
    sources = load_sources(manifest_path)
    output_records: list[dict[str, Any]] = []
    for source in sources:
        raw_path = data_root / "raw" / f"{source.id}.txt"
        download_source(source, raw_path)
        text = raw_path.read_text(encoding="utf-8-sig")
        if source.parser != "gutenberg_stage_play_v1":
            raise CorpusError(f"unsupported parser: {source.parser}")
        scenes = parse_gutenberg_stage_play(source, text)
        for scene in scenes:
            scene["provenance"] = {
                "source_url": source.source_url,
                "source_sha256": source.source_sha256,
                "parser": source.parser,
                "jurisdiction": source.metadata["jurisdiction"],
                "public_domain_basis": source.metadata["public_domain_basis"],
            }
        output_records.extend(scenes)

    processed = data_root / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    output_path = processed / "scenes.jsonl"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=processed, delete=False
    ) as handle:
        for record in output_records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        temporary = Path(handle.name)
    temporary.replace(output_path)

    counts = {split: 0 for split in sorted(VALID_SPLITS)}
    for record in output_records:
        counts[record["split"]] += 1
    counts["scenes"] = len(output_records)
    counts["turns"] = sum(len(record["turns"]) for record in output_records)
    return counts
