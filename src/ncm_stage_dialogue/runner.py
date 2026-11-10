"""Manual tmux+caffeinate supervision for checkpointed experiment runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ncm_stage_dialogue.experiment_cli import load_experiment_config
from ncm_stage_dialogue.training import TrainingError

RUN_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")


def _run(
    command: list[str], *, check: bool = True, timeout: float = 5.0
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=check, capture_output=True, text=True, timeout=timeout)


def _tmux_exists(session: str) -> bool:
    return _run(["tmux", "has-session", "-t", session], check=False, timeout=2).returncode == 0


@contextmanager
def _operation_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "runner.lock"
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise TrainingError(f"another runner operation is active: {root.name}") from error
        yield


def _atomic_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


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
    _validate_run_id(run_id)
    return repository / "data" / "runs" / run_id / "runner.json"


def _load_metadata(repository: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    path = _metadata_path(repository, run_id)
    root = path.parent
    runs_root = repository / "data" / "runs"
    if root.is_symlink() or root.resolve().parent != runs_root.resolve():
        raise TrainingError(f"invalid runner directory: {root}")
    if path.is_symlink() or not path.is_file():
        raise TrainingError(f"runner metadata does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return path, payload


def _repository_for_configs(configs: list[Path]) -> Path:
    repositories = {load_experiment_config(path)[4] for path in configs}
    if len(repositories) != 1:
        raise TrainingError("all configs must belong to one repository")
    return repositories.pop()


def _latest_checkpoint(run_dir: Path) -> tuple[Path, dict[str, Any]] | None:
    latest_path = run_dir / "latest"
    if not latest_path.exists():
        if latest_path.is_symlink():
            raise TrainingError(f"invalid latest checkpoint pointer: {latest_path}")
        return None
    if latest_path.is_symlink() or not latest_path.is_file():
        raise TrainingError(f"invalid latest checkpoint pointer: {latest_path}")
    checkpoint_name = latest_path.read_text(encoding="utf-8").strip()
    if re.fullmatch(r"checkpoint-\d{8}", checkpoint_name) is None:
        raise TrainingError(f"invalid checkpoint name: {checkpoint_name!r}")
    checkpoint = run_dir / checkpoint_name
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise TrainingError(f"invalid checkpoint directory: {checkpoint}")
    if checkpoint.resolve().parent != run_dir.resolve():
        raise TrainingError(f"checkpoint escapes run directory: {checkpoint}")
    state_path = checkpoint / "trainer_state.json"
    complete_path = checkpoint / "COMPLETE"
    for path in (state_path, complete_path):
        if path.is_symlink() or not path.is_file() or path.resolve().parent != checkpoint.resolve():
            raise TrainingError(f"invalid checkpoint file: {path}")
    return checkpoint, json.loads(state_path.read_text(encoding="utf-8"))


def _training_command(metadata: dict[str, Any], *, resume: bool) -> str:
    commands: list[str] = ["set -euo pipefail"]
    for item in metadata["experiments"]:
        run_dir = Path(item["run_dir"])
        train_steps = item["train_steps"]
        latest = _latest_checkpoint(run_dir) if resume else None
        if latest is not None:
            checkpoint, state = latest
            if state["global_step"] >= train_steps:
                continue
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
    for state_name in ("runner.pgid", "runner.exit"):
        state_path = metadata_path.parent / state_name
        if state_path.is_symlink():
            raise TrainingError(f"invalid runner state file: {state_path}")
        if state_path.exists():
            state_path.unlink()
    log_path = Path(metadata["log"])
    worker = [
        sys.executable,
        "-m",
        "ncm_stage_dialogue.runner",
        "_worker",
        "--metadata",
        str(metadata_path),
    ]
    if resume:
        worker.append("--resume")
    shell = shlex.join(worker)
    metadata["last_launch_at"] = datetime.now(UTC).isoformat()
    metadata["last_launch_mode"] = "resume" if resume else "start"
    metadata["last_log_offset"] = log_path.stat().st_size if log_path.is_file() else 0
    metadata["command"] = shell
    _atomic_json(metadata_path, metadata)
    result = _run(["tmux", "new-session", "-d", "-s", session, shell], check=False)
    if result.returncode != 0:
        raise TrainingError(result.stderr.strip() or "failed to start tmux session")


def _worker(metadata_path: Path, *, resume: bool) -> int:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    root = metadata_path.parent
    log_path = Path(metadata["log"])
    command = _training_command(metadata, resume=resume)
    with log_path.open("ab", buffering=0) as log_handle:
        child = subprocess.Popen(
            ["caffeinate", "-dimsu", "bash", "-lc", command],
            cwd=metadata["repository"],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        _atomic_text(root / "runner.pgid", f"{child.pid}\n")

        def forward(signum: int, _frame: Any) -> None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

        signal.signal(signal.SIGINT, forward)
        signal.signal(signal.SIGTERM, forward)
        return_code = child.wait()
    _atomic_text(root / "runner.exit", f"{return_code}\n")
    return return_code


def _process_group_alive(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    return True


def _runner_process_group(root: Path) -> int | None:
    pgid_path = root / "runner.pgid"
    exit_path = root / "runner.exit"
    if exit_path.exists():
        if exit_path.is_symlink() or not exit_path.is_file():
            raise TrainingError(f"invalid runner exit file: {exit_path}")
        return None
    if not pgid_path.exists():
        return None
    if pgid_path.is_symlink() or not pgid_path.is_file():
        raise TrainingError(f"invalid runner process group file: {pgid_path}")
    raw_pgid = pgid_path.read_text(encoding="utf-8").strip()
    if not raw_pgid.isascii() or not raw_pgid.isdigit() or int(raw_pgid) < 2:
        raise TrainingError(f"invalid runner process group: {raw_pgid!r}")
    return int(raw_pgid)


def _wait_for_process_group(process_group: int, deadline: float) -> bool:
    while time.monotonic() < deadline:
        if not _process_group_alive(process_group):
            return True
        time.sleep(0.05)
    return not _process_group_alive(process_group)


def _terminate_process_group(process_group: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    for signum, fraction in ((signal.SIGINT, 0.5), (signal.SIGTERM, 0.85)):
        try:
            os.killpg(process_group, signum)
        except ProcessLookupError:
            return
        stage_deadline = min(deadline, time.monotonic() + timeout * fraction)
        if _wait_for_process_group(process_group, stage_deadline):
            return
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return
    if not _wait_for_process_group(process_group, deadline):
        raise TrainingError(f"process group did not stop: {process_group}")


def start(run_id: str, configs: list[Path]) -> dict[str, Any]:
    _validate_run_id(run_id)
    if not configs:
        raise TrainingError("at least one config is required")
    repository = _repository_for_configs(configs)
    root = repository / "data" / "runs" / run_id
    try:
        root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as error:
        raise TrainingError(f"run ID already exists: {run_id}") from error
    with _operation_lock(root):
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
    root = metadata_path.parent
    with _operation_lock(root):
        if _tmux_exists(metadata["session"]):
            raise TrainingError(f"tmux session already exists: {metadata['session']}")
        process_group = _runner_process_group(root)
        if process_group is not None and _process_group_alive(process_group):
            raise TrainingError(f"training process group is still active: {process_group}")
        stamp = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{time.time_ns()}"
        for item in metadata["experiments"]:
            run_dir = Path(item["run_dir"])
            if run_dir.is_dir() and any(run_dir.iterdir()) and not (run_dir / "latest").exists():
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
    metadata_path, metadata = _load_metadata(repository.resolve(), run_id)
    process_group = _runner_process_group(metadata_path.parent)
    processes_running = process_group is not None and _process_group_alive(process_group)
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
        latest_info = _latest_checkpoint(run_dir)
        latest = None
        step = 0
        checkpoint_complete = False
        if latest_info is not None:
            checkpoint, trainer_state = latest_info
            latest = checkpoint.name
            checkpoint_complete = True
            step = trainer_state["global_step"]
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
    state = "running" if running or processes_running else ("complete" if complete else "stopped")
    return {
        "run_id": run_id,
        "state": state,
        "session": session,
        "pane": pane,
        "process_group": process_group if processes_running else None,
        "commit": metadata["commit"],
        "log": metadata["log"],
        "last_error": _last_error(Path(metadata["log"]), metadata.get("last_log_offset", 0)),
        "experiments": experiments,
    }


def stop(repository: Path, run_id: str, timeout: float = 10.0) -> dict[str, Any]:
    if timeout <= 0:
        raise TrainingError("stop timeout must be positive")
    deadline = time.monotonic() + timeout
    metadata_path, metadata = _load_metadata(repository.resolve(), run_id)
    root = metadata_path.parent
    session = metadata["session"]
    with _operation_lock(root):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TrainingError("stop timed out before checking the session")
        session_result = _run(
            ["tmux", "has-session", "-t", session],
            check=False,
            timeout=remaining,
        )
        process_group = _runner_process_group(root)
        while (
            process_group is None and session_result.returncode == 0 and time.monotonic() < deadline
        ):
            time.sleep(0.05)
            process_group = _runner_process_group(root)
        if process_group is None:
            raise TrainingError(f"runner has no active process group: {run_id}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TrainingError("stop timed out before signaling training")
        _terminate_process_group(process_group, remaining)
        if _process_group_alive(process_group):
            raise TrainingError(f"training processes are still running: {process_group}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TrainingError("training stopped but tmux cleanup exceeded the timeout")
        session_result = _run(
            ["tmux", "has-session", "-t", session], check=False, timeout=remaining
        )
        if session_result.returncode == 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TrainingError("training stopped but tmux cleanup exceeded the timeout")
            _run(["tmux", "kill-session", "-t", session], timeout=remaining)
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
    worker_parser = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    worker_parser.add_argument("--metadata", type=Path, required=True)
    worker_parser.add_argument("--resume", action="store_true")
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
        if args.command == "_worker":
            return _worker(args.metadata, resume=args.resume)
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
