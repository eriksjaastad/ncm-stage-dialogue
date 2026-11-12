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
DIRECTION_RE = re.compile(r"\[([^]]+)\]|\(_(.*?)_\)", re.DOTALL)
VALID_SPLITS = {"train", "validation", "test"}
ORDINALS = {"FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5}
ACT_ROMANS = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V"}
ACT_CARDINALS = {1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 5: "FIVE"}
SOURCE_ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
SUSPICIOUS_SPEAKER_RE = re.compile(r"^[A-Z][A-Z .’'&-]+$")
INLINE_SPEAKER_RE = re.compile(r"^([A-Z](?:[A-Z .’'&-]*[A-Z])?)(?:\.(?:\s|$)|\s+\[)")
PARSER_ALLOWED_HOSTS = {
    "gutenberg_stage_play_v1": {"gutenberg.org", "www.gutenberg.org"},
    "gutenberg_stage_collection_v1": {"gutenberg.org", "www.gutenberg.org"},
    "gutenberg_stage_explicit_v2": {"gutenberg.org", "www.gutenberg.org"},
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


def _validate_scene_splits(value: Any, context: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise CorpusError("source scene_splits must be an object")
    if not value:
        raise CorpusError("source scene_splits must not be empty")
    if not all(
        isinstance(scene_id, str) and isinstance(split, str) for scene_id, split in value.items()
    ):
        raise CorpusError("source scene_splits keys and values must be strings")
    if set(value.values()) - VALID_SPLITS:
        raise CorpusError(f"invalid scene split for {context}")
    return dict(value)


def _validate_speaker_labels(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise CorpusError("source speaker_labels must be a list")
    if not value:
        raise CorpusError("source speaker_labels must not be empty")
    if not all(isinstance(label, str) and label and label == label.upper() for label in value):
        raise CorpusError("source speaker_labels must be non-empty uppercase strings")
    if len(set(value)) != len(value):
        raise CorpusError(f"duplicate speaker label in {context}")
    return tuple(value)


def _validate_act_markers(
    value: Any, scene_splits: dict[str, str], source_id: str
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise CorpusError(f"act_markers must be a non-empty list: {source_id}")
    seen_scene_ids: set[str] = set()
    seen_boundaries: set[tuple[str, int]] = set()
    act_numbers: list[int] = []
    for marker in value:
        if not isinstance(marker, dict) or set(marker) != {
            "act_number",
            "scene_id",
            "line",
            "occurrence",
        }:
            raise CorpusError("act marker requires act_number, scene_id, line, and occurrence")
        if (
            type(marker["act_number"]) is not int
            or marker["act_number"] not in ACT_ROMANS
            or not isinstance(marker["scene_id"], str)
            or SOURCE_ID_RE.fullmatch(marker["scene_id"]) is None
            or not isinstance(marker["line"], str)
            or not marker["line"]
            or type(marker["occurrence"]) is not int
            or marker["occurrence"] < 1
        ):
            raise CorpusError(f"invalid act marker for {source_id}")
        act_number = marker["act_number"]
        ordinal = next(name for name, number in ORDINALS.items() if number == act_number)
        valid_headings = {
            f"{ordinal} ACT",
            f"ACT {ACT_ROMANS[act_number]}",
            f"ACT {ACT_CARDINALS[act_number]}",
        }
        if marker["scene_id"] != f"act-{act_number}" or marker["line"] not in valid_headings:
            raise CorpusError(f"act marker identity mismatch for {source_id}")
        boundary = (marker["line"], marker["occurrence"])
        if marker["scene_id"] in seen_scene_ids or boundary in seen_boundaries:
            raise CorpusError(f"duplicate act marker for {source_id}")
        seen_scene_ids.add(marker["scene_id"])
        seen_boundaries.add(boundary)
        act_numbers.append(act_number)
    if act_numbers != list(range(1, len(value) + 1)):
        raise CorpusError(f"act markers must be sequential for {source_id}")
    if seen_scene_ids != set(scene_splits):
        raise CorpusError(
            f"act marker scene set changed for {source_id}: "
            f"expected {sorted(scene_splits)}, got {sorted(seen_scene_ids)}"
        )
    return value


def _validate_line_replacements(value: Any, source_id: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise CorpusError(f"line_replacements must be a non-empty list: {source_id}")
    seen: set[tuple[str, int]] = set()
    for replacement in value:
        if not isinstance(replacement, dict) or set(replacement) != {
            "source",
            "replacement",
            "occurrence",
        }:
            raise CorpusError("line replacement requires source, replacement, and occurrence")
        if (
            not isinstance(replacement["source"], str)
            or not replacement["source"]
            or not isinstance(replacement["replacement"], str)
            or not replacement["replacement"]
            or type(replacement["occurrence"]) is not int
            or replacement["occurrence"] < 1
        ):
            raise CorpusError(f"invalid line replacement for {source_id}")
        identity = (replacement["source"], replacement["occurrence"])
        if identity in seen:
            raise CorpusError(f"duplicate line replacement for {source_id}")
        seen.add(identity)
    return value


def _validate_play_end_marker(value: Any, source_id: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"line", "occurrence"}:
        raise CorpusError(f"play_end_marker requires line and occurrence: {source_id}")
    if (
        not isinstance(value["line"], str)
        or not value["line"]
        or type(value["occurrence"]) is not int
        or value["occurrence"] < 1
    ):
        raise CorpusError(f"invalid play_end_marker for {source_id}")
    return value


def _validate_collection_works(value: Any, source_id: str) -> None:
    if not isinstance(value, list) or not value:
        raise CorpusError("collection works must be a non-empty list")
    seen_work_ids: set[str] = set()

    required = {
        "id",
        "title",
        "author",
        "heading",
        "heading_occurrence",
        "split",
        "scene_markers",
        "speaker_labels",
    }
    for index, work in enumerate(value):
        if not isinstance(work, dict):
            raise CorpusError(f"collection work {index} must be an object")
        missing = sorted(required - work.keys())
        if missing:
            raise CorpusError(f"collection work is missing fields: {', '.join(missing)}")
        for field in ("id", "title", "author", "heading"):
            if not isinstance(work[field], str) or not work[field].strip():
                raise CorpusError(f"collection work {field} must be a non-empty string")
        if not SOURCE_ID_RE.fullmatch(work["id"]):
            raise CorpusError(f"unsafe work id: {work['id']}")
        if work["id"] in seen_work_ids:
            raise CorpusError(f"duplicate work id: {work['id']}")
        seen_work_ids.add(work["id"])
        if type(work["heading_occurrence"]) is not int or work["heading_occurrence"] < 1:
            raise CorpusError("collection heading_occurrence must be a positive integer")
        if not isinstance(work["split"], str) or work["split"] not in VALID_SPLITS:
            raise CorpusError(f"invalid work split for {source_id}/{work['id']}")
        ignored_lines = work.get("ignored_lines", [])
        if (
            not isinstance(ignored_lines, list)
            or not all(isinstance(line, str) and line for line in ignored_lines)
            or len(set(ignored_lines)) != len(ignored_lines)
        ):
            raise CorpusError(f"ignored_lines must be unique non-empty strings: {work['id']}")

        _validate_speaker_labels(work["speaker_labels"], f"work {work['id']}")
        markers = work["scene_markers"]
        if not isinstance(markers, list) or not markers:
            raise CorpusError("collection scene_markers must be a non-empty list")
        seen_scene_ids: set[str] = set()
        seen_prefixes: set[str] = set()
        for marker in markers:
            if not isinstance(marker, dict):
                raise CorpusError("collection scene marker must be an object")
            required_marker_fields = {"prefix", "scene_id"}
            allowed_marker_fields = required_marker_fields | {
                "end_marker",
                "include_marker",
                "match_mode",
            }
            if not required_marker_fields <= marker.keys() or marker.keys() - allowed_marker_fields:
                raise CorpusError(
                    "collection scene marker requires prefix and scene_id with only "
                    "optional end_marker/include_marker/match_mode"
                )
            if not all(
                isinstance(marker[field], str) and marker[field] for field in required_marker_fields
            ):
                raise CorpusError("collection scene marker prefix and scene_id must be strings")
            if "end_marker" in marker and (
                not isinstance(marker["end_marker"], str) or not marker["end_marker"]
            ):
                raise CorpusError("collection scene end_marker must be a non-empty string")
            if "include_marker" in marker and not isinstance(marker["include_marker"], bool):
                raise CorpusError("collection scene include_marker must be a boolean")
            match_mode = marker.get("match_mode", "prefix")
            if not isinstance(match_mode, str) or match_mode not in {"exact", "prefix"}:
                raise CorpusError("collection scene match_mode must be exact or prefix")
            if not SOURCE_ID_RE.fullmatch(marker["scene_id"]):
                raise CorpusError(f"unsafe scene id: {marker['scene_id']}")
            if marker["scene_id"] in seen_scene_ids:
                raise CorpusError(f"duplicate scene id in work {work['id']}")
            if marker["prefix"] in seen_prefixes:
                raise CorpusError(f"duplicate scene prefix in work {work['id']}")
            seen_scene_ids.add(marker["scene_id"])
            seen_prefixes.add(marker["prefix"])


def _validate_explicit_works(value: Any, source_id: str) -> None:
    """Validate exact source-line ranges for a hash-pinned Gutenberg text."""
    if not isinstance(value, list) or not value:
        raise CorpusError("explicit works must be a non-empty list")
    seen_work_ids: set[str] = set()
    previous_end = 0
    required_work_fields = {"id", "title", "author", "split", "speaker_labels", "scenes"}
    required_scene_fields = {
        "scene_id",
        "act",
        "start_line",
        "end_line",
        "start_text",
        "end_text",
    }
    for work in value:
        if not isinstance(work, dict):
            raise CorpusError("explicit work must be an object")
        missing = sorted(required_work_fields - work.keys())
        if missing:
            raise CorpusError(f"explicit work is missing fields: {', '.join(missing)}")
        for field in ("id", "title", "author"):
            if not isinstance(work[field], str) or not work[field].strip():
                raise CorpusError(f"explicit work {field} must be a non-empty string")
        if SOURCE_ID_RE.fullmatch(work["id"]) is None:
            raise CorpusError(f"unsafe work id: {work['id']}")
        if work["id"] in seen_work_ids:
            raise CorpusError(f"duplicate work id: {work['id']}")
        seen_work_ids.add(work["id"])
        if work["split"] not in VALID_SPLITS:
            raise CorpusError(f"invalid work split for {source_id}/{work['id']}")
        _validate_speaker_labels(work["speaker_labels"], f"work {work['id']}")

        scenes = work["scenes"]
        if not isinstance(scenes, list) or not scenes:
            raise CorpusError("explicit work scenes must be a non-empty list")
        seen_scene_ids: set[str] = set()
        for scene in scenes:
            if not isinstance(scene, dict) or set(scene) != required_scene_fields:
                raise CorpusError(
                    "explicit scene requires scene_id, act, start_line, end_line, "
                    "start_text, and end_text"
                )
            if (
                not isinstance(scene["scene_id"], str)
                or SOURCE_ID_RE.fullmatch(scene["scene_id"]) is None
                or type(scene["act"]) is not int
                or scene["act"] < 1
                or type(scene["start_line"]) is not int
                or type(scene["end_line"]) is not int
                or scene["start_line"] < 1
                or scene["end_line"] <= scene["start_line"]
                or not isinstance(scene["start_text"], str)
                or not scene["start_text"]
                or not isinstance(scene["end_text"], str)
            ):
                raise CorpusError(f"invalid explicit scene for {source_id}/{work['id']}")
            if scene["scene_id"] in seen_scene_ids:
                raise CorpusError(f"duplicate scene id in work {work['id']}")
            if scene["start_line"] < previous_end:
                raise CorpusError(
                    f"explicit scene ranges overlap or are out of order: {work['id']}"
                )
            seen_scene_ids.add(scene["scene_id"])
            previous_end = scene["end_line"]

        ignored = work.get("ignored_line_numbers", [])
        if (
            not isinstance(ignored, list)
            or any(type(line_number) is not int or line_number < 1 for line_number in ignored)
            or len(set(ignored)) != len(ignored)
        ):
            raise CorpusError(f"invalid ignored_line_numbers for {work['id']}")
        scene_ranges = [range(scene["start_line"], scene["end_line"]) for scene in scenes]
        if any(
            not any(line_number in scene_range for scene_range in scene_ranges)
            for line_number in ignored
        ):
            raise CorpusError(f"ignored line outside scene ranges for {work['id']}")
        replacements = work.get("line_replacements", [])
        if not isinstance(replacements, list):
            raise CorpusError(f"line_replacements must be a list: {work['id']}")
        replacement_lines: set[int] = set()
        for replacement in replacements:
            if (
                not isinstance(replacement, dict)
                or set(replacement) != {"line_number", "source", "replacement"}
                or type(replacement["line_number"]) is not int
                or replacement["line_number"] < 1
                or not isinstance(replacement["source"], str)
                or not replacement["source"]
                or not isinstance(replacement["replacement"], str)
                or not replacement["replacement"]
                or replacement["line_number"] in replacement_lines
            ):
                raise CorpusError(f"invalid explicit line replacement for {work['id']}")
            if not any(replacement["line_number"] in scene_range for scene_range in scene_ranges):
                raise CorpusError(f"replacement line outside scene ranges for {work['id']}")
            replacement_lines.add(replacement["line_number"])


def _author_parts(author: str) -> set[str]:
    return {part.strip() for part in re.split(r"\s+and\s+", author) if part.strip()}


def _validate_author_split_isolation(sources: list[Source]) -> None:
    author_splits: dict[str, set[str]] = {}
    for source in sources:
        works = source.metadata.get("works")
        if source.parser in {"gutenberg_stage_collection_v1", "gutenberg_stage_explicit_v2"}:
            if not isinstance(works, list):
                raise CorpusError(f"works missing for author split validation: {source.id}")
            authored_splits = [(work["author"], work["split"]) for work in works]
        else:
            authored_splits = [
                (source.author, split) for split in sorted(set(source.scene_splits.values()))
            ]
        for author, split in authored_splits:
            for part in _author_parts(author):
                author_splits.setdefault(part, set()).add(split)
    leaking = sorted(
        (author, splits) for author, splits in author_splits.items() if len(splits) > 1
    )
    if leaking:
        author, splits = leaking[0]
        raise CorpusError(f"author split leakage for {author}: {', '.join(sorted(splits))}")


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

    common_required = {
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
        missing = sorted(common_required - raw.keys())
        if missing:
            raise CorpusError(f"source is missing required fields: {', '.join(missing)}")
        for field in sorted(string_fields):
            if not isinstance(raw[field], str) or not raw[field].strip():
                raise CorpusError(f"source {field} must be a non-empty string")
        if (
            raw["original_publication_year"] is not None
            and type(raw["original_publication_year"]) is not int
        ):
            raise CorpusError("source original_publication_year must be an integer or null")
        if type(raw["source_bytes"]) is not int:
            raise CorpusError("source source_bytes must be an integer")
        if raw["source_bytes"] < 0:
            raise CorpusError("source source_bytes must not be negative")
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
        _validate_https_url(raw["source_landing_page"], "source_landing_page", allowed_hosts)
        if raw["parser"] == "gutenberg_stage_play_v1":
            parser_required = {"scene_splits", "speaker_labels"}
            parser_missing = sorted(parser_required - raw.keys())
            if parser_missing:
                raise CorpusError(f"source is missing required fields: {', '.join(parser_missing)}")
            scene_splits = _validate_scene_splits(raw["scene_splits"], source_id)
            speaker_labels = _validate_speaker_labels(raw["speaker_labels"], f"source {source_id}")
            if "act_markers" in raw:
                _validate_act_markers(raw["act_markers"], scene_splits, source_id)
            if "line_replacements" in raw:
                _validate_line_replacements(raw["line_replacements"], source_id)
            if "play_end_marker" in raw:
                _validate_play_end_marker(raw["play_end_marker"], source_id)
            for flag in ("complete_play_split", "strict_speaker_labels"):
                if flag in raw and not isinstance(raw[flag], bool):
                    raise CorpusError(f"source {flag} must be a boolean")
            if raw.get("complete_play_split", False) and len(set(scene_splits.values())) != 1:
                raise CorpusError(f"complete play must use one split: {source_id}")
        elif raw["parser"] == "gutenberg_stage_collection_v1":
            _validate_collection_works(raw.get("works"), source_id)
            scene_splits = {}
            speaker_labels = ()
        else:
            _validate_explicit_works(raw.get("works"), source_id)
            scene_splits = {}
            speaker_labels = ()
        sources.append(
            Source(
                id=source_id,
                title=raw["title"],
                author=raw["author"],
                source_url=raw["source_url"],
                source_sha256=raw["source_sha256"],
                source_bytes=raw["source_bytes"],
                parser=raw["parser"],
                scene_splits=scene_splits,
                speaker_labels=speaker_labels,
                metadata=dict(raw),
            )
        )
    if "author_split_isolation" in payload:
        if payload["author_split_isolation"] is not True:
            raise CorpusError("author_split_isolation must be true when present")
        _validate_author_split_isolation(sources)
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
        (
            index
            for index, line in enumerate(lines)
            if "START OF THE PROJECT GUTENBERG EBOOK" in line
        ),
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
    directions = [
        re.sub(r"\s+", " ", next(group for group in match.groups() if group is not None)).strip()
        for match in DIRECTION_RE.finditer(raw)
    ]
    spoken = DIRECTION_RE.sub(" ", raw)
    spoken = spoken.replace("_", "")
    spoken = re.sub(r"(?<!\S)[.;:](?!\S)", " ", spoken)
    spoken = re.sub(r"\s+", " ", spoken).strip()
    return spoken, directions


def _match_speaker_line(line: str, speaker_labels: tuple[str, ...]) -> tuple[str, str] | None:
    """Return a declared speaker and any same-line direction/dialogue."""
    for label in sorted(speaker_labels, key=len, reverse=True):
        variants = {label, label.replace(" AND ", " and ")}
        matched = next((variant for variant in variants if line.startswith(variant)), None)
        if matched is None:
            continue
        remainder = line[len(matched) :]
        if not remainder:
            return label, ""
        if remainder.startswith("."):
            after_period = remainder[1:]
            if not after_period or after_period.startswith((" ", "[")):
                return label, after_period.strip()
            continue
        if remainder.startswith(" "):
            stripped = remainder.strip()
            if stripped.startswith("["):
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
    paragraph_break = True
    direction_depth = 0
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

    def start_scene(scene_id: str, act_number: int) -> None:
        nonlocal current_scene
        flush_scene()
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

    body_lines = body.splitlines()
    line_replacements = source.metadata.get("line_replacements")
    if line_replacements is not None:
        replacements = _validate_line_replacements(line_replacements, source.id)
        for replacement in replacements:
            occurrences = [
                index
                for index, line in enumerate(body_lines)
                if line.strip() == replacement["source"]
            ]
            expected_occurrences = replacement["occurrence"]
            if len(occurrences) != expected_occurrences:
                raise CorpusError(
                    f"line replacement occurrence changed for {source.id}: "
                    f"expected {expected_occurrences} {replacement['source']!r}, "
                    f"got {len(occurrences)}"
                )
            body_lines[occurrences[expected_occurrences - 1]] = replacement["replacement"]
    play_end_marker = source.metadata.get("play_end_marker")
    if play_end_marker is not None:
        end_marker = _validate_play_end_marker(play_end_marker, source.id)
        occurrences = [
            index for index, line in enumerate(body_lines) if line.strip() == end_marker["line"]
        ]
        expected_occurrences = end_marker["occurrence"]
        if len(occurrences) != expected_occurrences:
            raise CorpusError(
                f"play end marker occurrence changed for {source.id}: "
                f"expected {expected_occurrences}, got {len(occurrences)}"
            )
        body_lines = body_lines[: occurrences[expected_occurrences - 1]]
    act_markers = source.metadata.get("act_markers")
    marker_by_position: dict[int, tuple[str, int]] = {}
    if act_markers is not None:
        markers = _validate_act_markers(act_markers, source.scene_splits, source.id)
        positions: list[int] = []
        for marker in markers:
            act_number = marker["act_number"]
            occurrences = [
                index for index, line in enumerate(body_lines) if line.strip() == marker["line"]
            ]
            expected_occurrences = marker["occurrence"]
            if len(occurrences) != expected_occurrences:
                raise CorpusError(
                    f"act marker occurrence changed for {source.id}/{marker['scene_id']}: "
                    f"expected {expected_occurrences}, got {len(occurrences)}"
                )
            position = occurrences[expected_occurrences - 1]
            positions.append(position)
            marker_by_position[position] = (marker["scene_id"], act_number)
        if positions != sorted(positions) or len(set(positions)) != len(positions):
            raise CorpusError(f"act markers are missing, duplicated, or out of order: {source.id}")

    for line_index, raw_line in enumerate(body_lines):
        line = raw_line.strip()
        declared_marker = marker_by_position.get(line_index)
        if declared_marker is not None:
            start_scene(*declared_marker)
            paragraph_break = True
            continue
        act_match = ACT_RE.fullmatch(line) if act_markers is None else None
        if act_match is not None:
            act_number = ORDINALS[act_match.group(1)]
            start_scene(f"act-{act_number}", act_number)
            paragraph_break = True
            continue
        if current_scene is None:
            continue
        if not line:
            paragraph_break = True
            direction_depth = 0
            continue
        inside_direction = direction_depth > 0
        speaker_match = (
            None if inside_direction else _match_speaker_line(line, source.speaker_labels)
        )
        if (
            source.metadata.get("strict_speaker_labels", False)
            and current_speaker is not None
            and paragraph_break
            and speaker_match is None
            and not line.startswith("[")
        ):
            flush_turn()
        inline_speaker_match = INLINE_SPEAKER_RE.match(line)
        if (
            source.metadata.get("strict_speaker_labels", False)
            and not inside_direction
            and paragraph_break
            and speaker_match is None
            and (
                SUSPICIOUS_SPEAKER_RE.fullmatch(line)
                or (
                    inline_speaker_match is not None
                    and inline_speaker_match.group(1) not in source.speaker_labels
                )
            )
        ):
            raise CorpusError(f"suspicious undeclared speaker label in {source.id}: {line}")
        if speaker_match is not None:
            flush_turn()
            current_speaker, same_line_content = speaker_match
            if same_line_content:
                turn_lines.append(same_line_content)
                direction_depth = max(
                    0,
                    direction_depth
                    + same_line_content.count("[")
                    - same_line_content.count("]")
                    + same_line_content.count("(_")
                    - same_line_content.count("_)"),
                )
            paragraph_break = False
            continue
        if current_speaker is None:
            if line != "SCENE":
                prelude_lines.append(line)
        else:
            turn_lines.append(line)
        direction_depth = max(
            0,
            direction_depth
            + line.count("[")
            - line.count("]")
            + line.count("(_")
            - line.count("_)"),
        )
        paragraph_break = False

    flush_scene()
    expected_scenes = set(source.scene_splits)
    actual_scenes = {scene["scene_id"] for scene in scenes}
    if actual_scenes != expected_scenes:
        raise CorpusError(
            f"scene set changed for {source.id}: expected {sorted(expected_scenes)}, "
            f"got {sorted(actual_scenes)}"
        )
    if source.metadata.get("strict_speaker_labels", False) and source.metadata.get(
        "require_all_speakers", True
    ):
        observed_speakers = {turn["speaker"] for scene in scenes for turn in scene["turns"]}
        missing_speakers = sorted(set(source.speaker_labels) - observed_speakers)
        if missing_speakers:
            raise CorpusError(
                f"declared speakers missing from {source.id}: {', '.join(missing_speakers)}"
            )
    return scenes


def parse_gutenberg_stage_collection(source: Source, text: str) -> list[dict[str, Any]]:
    """Split a manifest-declared Gutenberg collection into independent works."""
    body_lines = strip_gutenberg_wrapper(text).splitlines()
    works = source.metadata.get("works")
    _validate_collection_works(works, source.id)
    if not isinstance(works, list):
        raise CorpusError("collection works must be a non-empty list")
    work_starts: list[int] = []
    seen_work_ids: set[str] = set()

    for work in works:
        if work["id"] in seen_work_ids:
            raise CorpusError(f"duplicate work id: {work['id']}")
        seen_work_ids.add(work["id"])
        occurrences = [
            index for index, line in enumerate(body_lines) if line.strip() == work["heading"]
        ]
        expected_occurrences = work["heading_occurrence"]
        if len(occurrences) != expected_occurrences:
            raise CorpusError(
                f"work heading occurrence changed for {work['id']}: "
                f"expected {expected_occurrences}, got {len(occurrences)}"
            )
        work_starts.append(occurrences[expected_occurrences - 1])

    if work_starts != sorted(work_starts) or len(set(work_starts)) != len(work_starts):
        raise CorpusError("collection work headings are missing, duplicated, or out of order")

    output: list[dict[str, Any]] = []
    ordinal_names = ("FIRST", "SECOND", "THIRD", "FOURTH", "FIFTH")
    for work_index, work in enumerate(works):
        start = work_starts[work_index]
        end = work_starts[work_index + 1] if work_index + 1 < len(works) else len(body_lines)
        section = body_lines[start:end]
        for ignored_line in work.get("ignored_lines", []):
            occurrences = [line for line in section if line.strip() == ignored_line]
            if len(occurrences) != 1:
                raise CorpusError(
                    f"ignored line occurrence changed for {work['id']}: "
                    f"expected 1 {ignored_line!r}, got {len(occurrences)}"
                )

        marker_positions: list[int] = []
        for marker in work["scene_markers"]:
            match_mode = marker.get("match_mode", "prefix")
            occurrences = [
                index
                for index, line in enumerate(section)
                if (
                    line.strip() == marker["prefix"]
                    if match_mode == "exact"
                    else line.strip().startswith(marker["prefix"])
                )
            ]
            if len(occurrences) != 1:
                raise CorpusError(
                    f"scene marker occurrence changed for {work['id']}/{marker['scene_id']}: "
                    f"expected 1, got {len(occurrences)}"
                )
            marker_positions.append(occurrences[0])
        if marker_positions != sorted(marker_positions):
            raise CorpusError(f"scene markers are out of order for {work['id']}")
        if len(marker_positions) > len(ordinal_names):
            raise CorpusError(f"too many scenes in work {work['id']}")

        normalized_lines: list[str] = []
        for scene_index, (marker, marker_position) in enumerate(
            zip(work["scene_markers"], marker_positions, strict=True)
        ):
            scene_end = (
                marker_positions[scene_index + 1]
                if scene_index + 1 < len(marker_positions)
                else len(section)
            )
            end_marker = marker.get("end_marker", "(CURTAIN)")
            end_positions = [
                index
                for index in range(marker_position + 1, scene_end)
                if section[index].strip() == end_marker
            ]
            if len(end_positions) != 1:
                raise CorpusError(
                    f"scene end marker changed for {work['id']}/{marker['scene_id']}: "
                    f"expected 1 {end_marker!r}, got {len(end_positions)}"
                )
            scene_end = end_positions[0]
            marker_line = section[marker_position].strip()
            remainder = marker_line[len(marker["prefix"]) :].strip()
            include_marker = marker.get("include_marker", False)
            segment_start = marker_position if include_marker else marker_position + 1
            segment = section[segment_start:scene_end]
            ignored_lines = set(work.get("ignored_lines", []))
            segment = [line for line in segment if line.strip() not in ignored_lines]
            if segment and segment[0].strip().startswith("SCENE:"):
                scene_line = segment[0].strip()
                segment[0] = scene_line[len("SCENE:") :].strip()
            for line in segment:
                candidate = line.strip()
                if SUSPICIOUS_SPEAKER_RE.fullmatch(candidate) and not _match_speaker_line(
                    candidate, tuple(work["speaker_labels"])
                ):
                    raise CorpusError(
                        f"suspicious undeclared speaker label in {work['id']}: {candidate}"
                    )
            normalized_lines.append(f"{ordinal_names[scene_index]} ACT")
            if remainder and not include_marker:
                normalized_lines.append(remainder)
            normalized_lines.extend(segment)

        scene_splits = {
            f"act-{index + 1}": work["split"] for index in range(len(work["scene_markers"]))
        }
        work_source = Source(
            id=f"{source.id}-{work['id']}",
            title=work["title"],
            author=work["author"],
            source_url=source.source_url,
            source_sha256=source.source_sha256,
            source_bytes=source.source_bytes,
            parser="gutenberg_stage_play_v1",
            scene_splits=scene_splits,
            speaker_labels=tuple(work["speaker_labels"]),
            metadata=source.metadata,
        )
        normalized = (
            "*** START OF THE PROJECT GUTENBERG EBOOK NORMALIZED WORK ***\n"
            + "\n".join(normalized_lines)
            + "\n*** END OF THE PROJECT GUTENBERG EBOOK NORMALIZED WORK ***"
        )
        parsed_scenes = parse_gutenberg_stage_play(work_source, normalized)
        observed_speakers = {turn["speaker"] for scene in parsed_scenes for turn in scene["turns"]}
        missing_speakers = sorted(set(work["speaker_labels"]) - observed_speakers)
        if missing_speakers:
            raise CorpusError(
                f"declared speakers missing from {work['id']}: {', '.join(missing_speakers)}"
            )
        for scene_index, scene in enumerate(parsed_scenes):
            declared_scene_id = work["scene_markers"][scene_index]["scene_id"]
            scene["source_id"] = source.id
            scene["work_id"] = work["id"]
            scene["scene_id"] = f"{work['id']}-{declared_scene_id}"
        output.extend(parsed_scenes)
    return output


def parse_gutenberg_stage_explicit(source: Source, text: str) -> list[dict[str, Any]]:
    """Parse hash-pinned works from explicit 1-indexed source-line ranges."""
    lines = text.splitlines()
    works = source.metadata.get("works")
    _validate_explicit_works(works, source.id)
    if not isinstance(works, list):
        raise CorpusError("explicit works must be a non-empty list")

    output: list[dict[str, Any]] = []
    for work in works:
        work_output: list[dict[str, Any]] = []
        work_lines = list(lines)
        for replacement in work.get("line_replacements", []):
            line_number = replacement["line_number"]
            if (
                line_number > len(work_lines)
                or work_lines[line_number - 1] != replacement["source"]
            ):
                raise CorpusError(
                    f"explicit line replacement changed for {source.id}/{work['id']}: "
                    f"line {line_number}"
                )
            work_lines[line_number - 1] = replacement["replacement"]
        ignored = set(work.get("ignored_line_numbers", []))

        for scene in work["scenes"]:
            start_line = scene["start_line"]
            end_line = scene["end_line"]
            if end_line - 1 > len(lines):
                raise CorpusError(
                    f"explicit scene range exceeds source for {source.id}/{work['id']}/"
                    f"{scene['scene_id']}"
                )
            if lines[start_line - 1].strip() != scene["start_text"]:
                raise CorpusError(
                    f"explicit scene start changed for {source.id}/{work['id']}/{scene['scene_id']}"
                )
            if lines[end_line - 2].strip() != scene["end_text"]:
                raise CorpusError(
                    f"explicit scene end changed for {source.id}/{work['id']}/{scene['scene_id']}"
                )
            segment = [
                work_lines[index - 1]
                for index in range(start_line, end_line)
                if index not in ignored
            ]
            normalized = (
                "*** START OF THE PROJECT GUTENBERG EBOOK NORMALIZED WORK ***\n"
                "FIRST ACT\n"
                + "\n".join(segment)
                + "\n*** END OF THE PROJECT GUTENBERG EBOOK NORMALIZED WORK ***"
            )
            scene_source = Source(
                id=f"{source.id}-{work['id']}",
                title=work["title"],
                author=work["author"],
                source_url=source.source_url,
                source_sha256=source.source_sha256,
                source_bytes=source.source_bytes,
                parser="gutenberg_stage_play_v1",
                scene_splits={"act-1": work["split"]},
                speaker_labels=tuple(work["speaker_labels"]),
                metadata={
                    "act_markers": [
                        {
                            "act_number": 1,
                            "scene_id": "act-1",
                            "line": "FIRST ACT",
                            "occurrence": 1,
                        }
                    ],
                    "strict_speaker_labels": True,
                    "require_all_speakers": False,
                },
            )
            parsed = parse_gutenberg_stage_play(scene_source, normalized)
            if len(parsed) != 1:
                raise CorpusError(
                    f"explicit scene did not parse exactly once: {source.id}/{work['id']}/"
                    f"{scene['scene_id']}"
                )
            record = parsed[0]
            record["source_id"] = source.id
            record["work_id"] = work["id"]
            record["scene_id"] = f"{work['id']}-{scene['scene_id']}"
            record["act"] = scene["act"]
            record["source_line_start"] = start_line
            record["source_line_end"] = end_line - 1
            work_output.append(record)
        observed_speakers = {turn["speaker"] for record in work_output for turn in record["turns"]}
        missing_speakers = sorted(set(work["speaker_labels"]) - observed_speakers)
        if missing_speakers:
            raise CorpusError(
                f"declared speakers missing from {source.id}/{work['id']}: "
                f"{', '.join(missing_speakers)}"
            )
        output.extend(work_output)
    return output


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
        if source.parser == "gutenberg_stage_play_v1":
            scenes = parse_gutenberg_stage_play(source, text)
        elif source.parser == "gutenberg_stage_collection_v1":
            scenes = parse_gutenberg_stage_collection(source, text)
        elif source.parser == "gutenberg_stage_explicit_v2":
            scenes = parse_gutenberg_stage_explicit(source, text)
        else:
            raise CorpusError(f"unsupported parser: {source.parser}")
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
