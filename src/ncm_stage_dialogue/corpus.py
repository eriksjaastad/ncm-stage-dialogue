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
from urllib.parse import urlsplit

ACT_RE = re.compile(r"^(FIRST|SECOND|THIRD|FOURTH|FIFTH) ACT$")
DIRECTION_RE = re.compile(r"\[([^]]+)\]", re.DOTALL)
VALID_SPLITS = {"train", "validation", "test"}
ORDINALS = {"FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5}
SOURCE_ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
PARSER_ALLOWED_HOSTS = {
    "gutenberg_stage_play_v1": {"gutenberg.org", "www.gutenberg.org"},
}


class CorpusError(RuntimeError):
    """Raised when source provenance or parsing invariants fail."""


def _validate_https_url(url: str, field: str, allowed_hosts: set[str]) -> None:
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as error:
        raise CorpusError(f"unsafe {field}: {url}") from error
    if (
        parsed.scheme != "https"
        or parsed.hostname not in allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        raise CorpusError(f"unsafe {field}: {url}")


class HttpsOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Allow redirects only to HTTPS URLs on the parser's trusted hosts."""

    def __init__(self, allowed_hosts: set[str]) -> None:
        super().__init__()
        self.allowed_hosts = allowed_hosts

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        _validate_https_url(newurl, "redirect URL", self.allowed_hosts)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


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
    speaker_labels: tuple[str, ...]
    metadata: dict[str, Any]


def load_sources(manifest_path: Path) -> list[Source]:
    """Load and validate corpus sources from a JSON manifest."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CorpusError("manifest must be an object")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise CorpusError("manifest schema_version must be 1")

    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list):
        raise CorpusError("manifest sources must be a list")
    if not raw_sources:
        raise CorpusError("manifest contains no sources")

    required = {
        "id",
        "title",
        "author",
        "language",
        "original_publication_year",
        "source_landing_page",
        "source_url",
        "source_sha256",
        "source_bytes",
        "retrieved_on",
        "jurisdiction",
        "public_domain_basis",
        "source_terms",
        "parser",
        "scene_splits",
        "speaker_labels",
    }
    string_fields = {
        "id",
        "title",
        "author",
        "language",
        "source_landing_page",
        "source_url",
        "source_sha256",
        "retrieved_on",
        "jurisdiction",
        "public_domain_basis",
        "source_terms",
        "parser",
    }
    sources: list[Source] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_sources):
        if not isinstance(raw, dict):
            raise CorpusError(f"source {index} must be an object")
        missing = sorted(required - raw.keys())
        if missing:
            raise CorpusError(f"source is missing required fields: {', '.join(missing)}")
        for field in sorted(string_fields):
            if not isinstance(raw[field], str) or not raw[field].strip():
                raise CorpusError(f"source {field} must be a non-empty string")
        if type(raw["original_publication_year"]) is not int:
            raise CorpusError("source original_publication_year must be an integer")
        if type(raw["source_bytes"]) is not int:
            raise CorpusError("source source_bytes must be an integer")
        if raw["source_bytes"] < 0:
            raise CorpusError("source source_bytes must not be negative")
        if not isinstance(raw["scene_splits"], dict):
            raise CorpusError("source scene_splits must be an object")
        if not raw["scene_splits"]:
            raise CorpusError("source scene_splits must not be empty")
        if not all(
            isinstance(scene_id, str) and isinstance(split, str)
            for scene_id, split in raw["scene_splits"].items()
        ):
            raise CorpusError("source scene_splits keys and values must be strings")
        if not isinstance(raw["speaker_labels"], list):
            raise CorpusError("source speaker_labels must be a list")
        if not raw["speaker_labels"]:
            raise CorpusError("source speaker_labels must not be empty")
        if not all(
            isinstance(label, str) and label and label == label.upper()
            for label in raw["speaker_labels"]
        ):
            raise CorpusError("source speaker_labels must be non-empty uppercase strings")
        if len(set(raw["speaker_labels"])) != len(raw["speaker_labels"]):
            raise CorpusError("source speaker_labels must be unique")

        source_id = raw["id"]
        if source_id in seen:
            raise CorpusError(f"duplicate source id: {source_id}")
        if not SOURCE_ID_RE.fullmatch(source_id):
            raise CorpusError(f"unsafe source id: {source_id}")
        seen.add(source_id)
        if not re.fullmatch(r"[0-9a-f]{64}", raw["source_sha256"]):
            raise CorpusError(f"invalid SHA-256 for {source_id}")
        allowed_hosts = PARSER_ALLOWED_HOSTS.get(raw["parser"])
        if allowed_hosts is None:
            raise CorpusError(f"unsupported parser: {raw['parser']}")
        _validate_https_url(raw["source_url"], "source_url", allowed_hosts)
        _validate_https_url(
            raw["source_landing_page"], "source_landing_page", allowed_hosts
        )
        if set(raw["scene_splits"].values()) - VALID_SPLITS:
            raise CorpusError(f"invalid scene split for {source_id}")
        sources.append(
            Source(
                id=source_id,
                title=raw["title"],
                author=raw["author"],
                source_url=raw["source_url"],
                source_sha256=raw["source_sha256"],
                source_bytes=raw["source_bytes"],
                parser=raw["parser"],
                scene_splits=dict(raw["scene_splits"]),
                speaker_labels=tuple(raw["speaker_labels"]),
                metadata=dict(raw),
            )
        )
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
        allowed_hosts = PARSER_ALLOWED_HOSTS[source.parser]
        opener = urllib.request.build_opener(HttpsOnlyRedirectHandler(allowed_hosts))
        with opener.open(request, timeout=60) as response:
            _validate_https_url(response.geturl(), "final source URL", allowed_hosts)
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


def _match_speaker_line(line: str, speaker_labels: tuple[str, ...]) -> tuple[str, str] | None:
    """Return a declared speaker and any same-line direction/dialogue."""
    for label in sorted(speaker_labels, key=len, reverse=True):
        if line[: len(label)].casefold() != label.casefold():
            continue
        remainder = line[len(label) :]
        if remainder.startswith("."):
            return label, remainder[1:].strip()
        stripped = remainder.lstrip()
        if remainder != stripped and stripped.startswith("["):
            return label, stripped
    return None


def parse_gutenberg_stage_play(source: Source, text: str) -> list[dict[str, Any]]:
    """Parse an act-structured Gutenberg play into scene-level records."""
    body = strip_gutenberg_wrapper(text)
    scenes: list[dict[str, Any]] = []
    current_scene: dict[str, Any] | None = None
    current_speaker: str | None = None
    turn_lines: list[str] = []
    prelude_lines: list[str] = []
    seen_scene_ids: set[str] = set()

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
            if scene_id in seen_scene_ids:
                raise CorpusError(f"duplicate scene id: {scene_id}")
            seen_scene_ids.add(scene_id)
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
        speaker_match = _match_speaker_line(line, source.speaker_labels)
        if speaker_match is not None:
            flush_turn()
            current_speaker, same_line_content = speaker_match
            if same_line_content:
                turn_lines.append(same_line_content)
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
        raw_root = (data_root / "raw").resolve()
        raw_path = (raw_root / f"{source.id}.txt").resolve()
        if not raw_path.is_relative_to(raw_root):
            raise CorpusError(f"source path escapes data root: {source.id}")
        download_source(source, raw_path)
        text = raw_path.read_text(encoding="utf-8-sig")
        if source.parser != "gutenberg_stage_play_v1":
            raise CorpusError(f"unsupported parser: {source.parser}")
        scenes = parse_gutenberg_stage_play(source, text)
        for scene in scenes:
            scene["provenance"] = {
                "source": source.metadata,
                "transformation": {
                    "parser": source.parser,
                    "project_gutenberg_wrapper_removed": True,
                    "stage_directions_separated": True,
                },
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
