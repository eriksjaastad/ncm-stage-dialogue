"""Manual tmux+caffeinate supervision for checkpointed experiment runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ncm_stage_dialogue.experiment_cli import load_experiment_config
from ncm_stage_dialogue.training import TrainingError

RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=True, text=True)


def _tmux_exists(session: str) -> bool:
    return _run(["tmux", "has-session", "-t", session], check=False).returncode == 0


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _validate_run_id(run_id: str) -> None:
    if RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise TrainingError("run ID must match [a-z0-9][a-z0-9._-]{0,63}")


def _metadata_path(repository: Path, run_id: str) -> Path:
    return repository / "data" / "runs" / run_id / "runner.json"


def _load_metadata(repository: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    path = _metadata_path(repository, run_id)
    if not path.is_file():
        raise TrainingError(f"runner metadata does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return path, payload


def _repository_for_configs(configs: list[Path]) -> Path:
    repositories = {load_experiment_config(path)[4] for path in configs}
    if len(repositories) != 1:
        raise TrainingError("all configs must belong to one repository")
    return repositories.pop()


def _training_command(metadata: dict[str, Any], *, resume: bool) -> str:
    commands: list[str] = ["set -euo pipefail"]
    for item in metadata["experiments"]:
        run_dir = Path(item["run_dir"])
        train_steps = item["train_steps"]
        latest_path = run_dir / "latest"
        if resume and latest_path.is_file():
            checkpoint_name = latest_path.read_text(encoding="utf-8").strip()
            state_path = run_dir / checkpoint_name / "trainer_state.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state["global_step"] >= train_steps:
                continue
            checkpoint = run_dir / checkpoint_name
            command = [
                sys.executable,
                "-m",
                "ncm_stage_dialogue.experiment_cli",
                "train",
                "--config",
                item["config"],
                "--checkpoint",
                str(checkpoint),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "ncm_stage_dialogue.experiment_cli",
                "train",
                "--config",
                item["config"],
            ]
        commands.append(f"echo {shlex.quote('[runner] starting ' + item['name'])}")
        commands.append(shlex.join(command))
        commands.append(f"echo {shlex.quote('[runner] completed ' + item['name'])}")
    if len(commands) == 1:
        commands.append("echo '[runner] all experiments already complete'")
    return "; ".join(commands)


def _launch(metadata_path: Path, metadata: dict[str, Any], *, resume: bool) -> None:
    session = metadata["session"]
    if _tmux_exists(session):
        raise TrainingError(f"tmux session already exists: {session}")
    log_path = Path(metadata["log"])
    command = _training_command(metadata, resume=resume)
    wrapped = f"{{ {command}; }} >> {shlex.quote(str(log_path))} 2>&1"
    shell = (
        f"cd {shlex.quote(metadata['repository'])} && exec caffeinate -dimsu "
        f"bash -lc {shlex.quote(wrapped)}"
    )
    metadata["last_launch_at"] = datetime.now(UTC).isoformat()
    metadata["last_launch_mode"] = "resume" if resume else "start"
    metadata["last_log_offset"] = log_path.stat().st_size if log_path.is_file() else 0
    metadata["command"] = shell
    _atomic_json(metadata_path, metadata)
    result = _run(["tmux", "new-session", "-d", "-s", session, shell], check=False)
    if result.returncode != 0:
        raise TrainingError(result.stderr.strip() or "failed to start tmux session")


def start(run_id: str, configs: list[Path]) -> dict[str, Any]:
    _validate_run_id(run_id)
    if not configs:
        raise TrainingError("at least one config is required")
    repository = _repository_for_configs(configs)
    root = repository / "data" / "runs" / run_id
    if root.exists():
        raise TrainingError(f"run ID already exists: {run_id}")
    root.mkdir(parents=True)
    copied = root / "configs"
    copied.mkdir()
    experiments: list[dict[str, Any]] = []
    names: set[str] = set()
    for source in configs:
        raw = json.loads(source.resolve().read_text(encoding="utf-8"))
        name = source.stem
        if name in names:
            raise TrainingError(f"duplicate config name: {name}")
        names.add(name)
        run_dir = root / name
        raw["run_dir"] = str(run_dir.relative_to(repository))
        destination = copied / source.name
        _atomic_json(destination, raw)
        experiments.append(
            {
                "name": name,
                "config": str(destination),
                "run_dir": str(run_dir),
                "train_steps": raw["training"]["train_steps"],
            }
        )
    commit = _run(["git", "-C", str(repository), "rev-parse", "HEAD"]).stdout.strip()
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "session": f"ncm-{run_id}",
        "repository": str(repository),
        "commit": commit,
        "created_at": datetime.now(UTC).isoformat(),
        "log": str(root / "supervisor.log"),
        "experiments": experiments,
    }
    metadata_path = root / "runner.json"
    _launch(metadata_path, metadata, resume=False)
    return status(repository, run_id)


def resume(repository: Path, run_id: str) -> dict[str, Any]:
    metadata_path, metadata = _load_metadata(repository.resolve(), run_id)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    for item in metadata["experiments"]:
        run_dir = Path(item["run_dir"])
        if run_dir.is_dir() and any(run_dir.iterdir()) and not (run_dir / "latest").is_file():
            archive = run_dir.with_name(f"{run_dir.name}-precheckpoint-{stamp}")
            os.replace(run_dir, archive)
            metadata.setdefault("precheckpoint_archives", []).append(str(archive))
            parent_fd = os.open(run_dir.parent, os.O_RDONLY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    _launch(metadata_path, metadata, resume=True)
    return status(repository, run_id)


def _last_error(log_path: Path, offset: int) -> str | None:
    if not log_path.is_file():
        return None
    with log_path.open("rb") as handle:
        handle.seek(offset)
        content = handle.read().decode("utf-8", errors="replace")
    for line in reversed(content.splitlines()):
        lowered = line.lower()
        if "error" in lowered or "traceback" in lowered:
            return line
    return None


def status(repository: Path, run_id: str) -> dict[str, Any]:
    _, metadata = _load_metadata(repository.resolve(), run_id)
    session = metadata["session"]
    running = _tmux_exists(session)
    pane = None
    if running:
        pane_result = _run(
            [
                "tmux",
                "list-panes",
                "-t",
                session,
                "-F",
                "pid=#{pane_pid} command=#{pane_current_command}",
            ],
            check=False,
        )
        if pane_result.returncode == 0:
            pane = pane_result.stdout.strip()
        else:
            running = False
    experiments: list[dict[str, Any]] = []
    complete = True
    for item in metadata["experiments"]:
        run_dir = Path(item["run_dir"])
        latest = (
            (run_dir / "latest").read_text(encoding="utf-8").strip()
            if (run_dir / "latest").is_file()
            else None
        )
        step = 0
        checkpoint_complete = False
        if latest:
            checkpoint = run_dir / latest
            checkpoint_complete = (checkpoint / "COMPLETE").is_file()
            if (checkpoint / "trainer_state.json").is_file():
                step = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))[
                    "global_step"
                ]
        item_complete = checkpoint_complete and step >= item["train_steps"]
        complete = complete and item_complete
        experiments.append(
            {
                "name": item["name"],
                "run_dir": str(run_dir),
                "step": step,
                "train_steps": item["train_steps"],
                "latest": latest,
                "checkpoint_complete": checkpoint_complete,
                "complete": item_complete,
            }
        )
    state = "running" if running else ("complete" if complete else "stopped")
    return {
        "run_id": run_id,
        "state": state,
        "session": session,
        "pane": pane,
        "commit": metadata["commit"],
        "log": metadata["log"],
        "last_error": _last_error(Path(metadata["log"]), metadata.get("last_log_offset", 0)),
        "experiments": experiments,
    }


def stop(repository: Path, run_id: str, timeout: float = 10.0) -> dict[str, Any]:
    _, metadata = _load_metadata(repository.resolve(), run_id)
    session = metadata["session"]
    if not _tmux_exists(session):
        raise TrainingError(f"tmux session is not running: {session}")
    _run(["tmux", "send-keys", "-t", session, "C-c"])
    deadline = time.monotonic() + timeout
    while _tmux_exists(session) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _tmux_exists(session):
        _run(["tmux", "kill-session", "-t", session])
    return status(repository, run_id)


def log_text(repository: Path, run_id: str, lines: int) -> str:
    _, metadata = _load_metadata(repository.resolve(), run_id)
    log_path = Path(metadata["log"])
    if not log_path.is_file():
        return ""
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ncm-runner")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    start_parser = subparsers.add_parser("start")
    start_parser.add_argument("--run-id", required=True)
    start_parser.add_argument("--config", action="append", type=Path, required=True)
    for name in ("status", "resume", "stop", "log"):
        child = subparsers.add_parser(name)
        child.add_argument("--run-id", required=True)
        if name == "stop":
            child.add_argument("--timeout", type=float, default=10.0)
        if name == "log":
            child.add_argument("--lines", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "start":
            result: Any = start(args.run_id, args.config)
        elif args.command == "resume":
            result = resume(args.repository, args.run_id)
        elif args.command == "status":
            result = status(args.repository, args.run_id)
        elif args.command == "stop":
            result = stop(args.repository, args.run_id, args.timeout)
        else:
            print(log_text(args.repository, args.run_id, args.lines))
            return 0
    except (OSError, subprocess.SubprocessError, TrainingError, ValueError, KeyError) as error:
        print(f"ncm-runner: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
