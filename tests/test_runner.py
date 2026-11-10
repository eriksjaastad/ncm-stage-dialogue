from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest

from ncm_stage_dialogue.runner import (
    _process_group_alive,
    _process_identity,
    _runner_process_group,
    _terminate_process_group,
    resume,
    start,
    status,
    stop,
)
from ncm_stage_dialogue.training import TrainingError


def _scene(split: str, scene_id: str) -> dict[str, object]:
    return {
        "source_id": f"fixture-{scene_id}",
        "title": f"Fixture {scene_id}",
        "author": "Test Author",
        "scene_id": scene_id,
        "act": 1,
        "split": split,
        "stage_directions": ["A room."],
        "turns": [
            {"speaker": "ALICE", "text": "Hello.", "stage_directions": ["quietly"]},
            {"speaker": "BOB", "text": "Goodbye.", "stage_directions": ["firmly"]},
        ],
        "provenance": {"synthetic": True},
    }


def _repository(tmp_path: Path, train_steps: int) -> tuple[Path, Path]:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / "pyproject.toml").write_text(
        "[project]\nname='runner-fixture'\nversion='0.0.0'\n", encoding="utf-8"
    )
    data = repository / "data"
    data.mkdir()
    corpus = data / "scenes.jsonl"
    corpus.write_text(
        "".join(
            json.dumps(_scene(split, scene_id)) + "\n"
            for split, scene_id in (
                ("train", "train-1"),
                ("validation", "validation-1"),
                ("test", "test-1"),
            )
        ),
        encoding="utf-8",
    )
    configs = repository / "configs"
    configs.mkdir()
    config = configs / "tiny.json"
    config.write_text(
        json.dumps(
            {
                "model": {
                    "model_type": "transformer",
                    "context_length": 8,
                    "hidden_size": 16,
                    "num_heads": 4,
                    "ffn_size": 24,
                    "num_layers": 1,
                },
                "training": {
                    "seed": 3,
                    "batch_size": 1,
                    "train_steps": train_steps,
                    "learning_rate": 0.001,
                    "warmup_steps": 0,
                    "weight_decay": 0.0,
                    "max_grad_norm": 1.0,
                    "eval_interval": max(1, train_steps // 2),
                    "checkpoint_interval": 1,
                },
                "corpus": "data/scenes.jsonl",
                "run_dir": "unused",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "-c",
            "user.name=Runner Test",
            "-c",
            "user.email=runner@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return repository, config


def _wait(repository: Path, run_id: str, expected: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    current = status(repository, run_id)
    while current["state"] != expected and time.monotonic() < deadline:
        time.sleep(0.1)
        current = status(repository, run_id)
    assert current["state"] == expected, current
    return current


def _kill_if_running(session: str) -> None:
    subprocess.run(["tmux", "kill-session", "-t", session], check=False, capture_output=True)


def test_runner_refuses_duplicates_and_reports_clean_completion(tmp_path: Path) -> None:
    repository, config = _repository(tmp_path, train_steps=2)
    run_id = f"clean-{uuid.uuid4().hex[:8]}"
    session = f"ncm-{run_id}"
    try:
        launched = start(run_id, [config])
        assert launched["session"] == session
        assert launched["commit"]
        with pytest.raises(TrainingError, match="already exists"):
            start(run_id, [config])

        completed = _wait(repository, run_id, "complete")
        experiment = completed["experiments"][0]
        assert experiment["step"] == 2
        assert experiment["checkpoint_complete"] is True
        assert completed["last_error"] is None
        assert Path(completed["log"]).is_file()
        assert "[runner] completed tiny" in Path(completed["log"]).read_text(encoding="utf-8")
    finally:
        _kill_if_running(session)


def test_runner_stops_at_checkpoint_and_resumes_exactly(tmp_path: Path) -> None:
    repository, config = _repository(tmp_path, train_steps=200)
    run_id = f"resume-{uuid.uuid4().hex[:8]}"
    session = f"ncm-{run_id}"
    try:
        start(run_id, [config])
        run_dir = repository / "data" / "runs" / run_id / "tiny"
        with pytest.raises(TrainingError, match="session already exists"):
            resume(repository, run_id)
        assert not list(run_dir.parent.glob("tiny-precheckpoint-*"))
        deadline = time.monotonic() + 30
        current = status(repository, run_id)
        while current["experiments"][0]["step"] == 0 and time.monotonic() < deadline:
            time.sleep(0.05)
            current = status(repository, run_id)
        assert current["experiments"][0]["checkpoint_complete"] is True
        process_group = current["process_group"]
        assert process_group is not None
        subprocess.run(["tmux", "kill-session", "-t", session], check=True)
        assert _process_group_alive(process_group)
        assert status(repository, run_id)["state"] == "running"

        stopped = stop(repository, run_id, timeout=5)
        assert stopped["state"] in {"stopped", "complete"}
        assert stopped["process_group"] is None
        stopped_step = stopped["experiments"][0]["step"]
        assert stopped_step >= 1
        if stopped["state"] != "complete":
            resumed = resume(repository, run_id)
            assert resumed["state"] == "running"
            completed = _wait(repository, run_id, "complete")
        else:
            completed = stopped

        run_dir = Path(completed["experiments"][0]["run_dir"])
        events = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        train_steps = [event["step"] for event in events if event["split"] == "train"]
        assert train_steps == list(range(1, 201))
        assert len([event for event in events if event["split"] == "test"]) == 1
    finally:
        _kill_if_running(session)


def test_runner_rejects_checkpoint_pointer_escape(tmp_path: Path) -> None:
    repository, config = _repository(tmp_path, train_steps=2)
    run_id = f"escape-{uuid.uuid4().hex[:8]}"
    session = f"ncm-{run_id}"
    try:
        start(run_id, [config])
        _wait(repository, run_id, "complete")
        run_dir = repository / "data" / "runs" / run_id / "tiny"
        (run_dir / "latest").write_text("../outside\n", encoding="utf-8")
        with pytest.raises(TrainingError, match="invalid checkpoint name"):
            status(repository, run_id)
    finally:
        _kill_if_running(session)


def test_runner_rejects_symlinked_run_directory(tmp_path: Path) -> None:
    repository, config = _repository(tmp_path, train_steps=2)
    run_id = f"run-link-{uuid.uuid4().hex[:8]}"
    session = f"ncm-{run_id}"
    try:
        start(run_id, [config])
        _wait(repository, run_id, "complete")
        run_dir = repository / "data" / "runs" / run_id / "tiny"
        external = repository / "external-run"
        os.replace(run_dir, external)
        run_dir.symlink_to(external, target_is_directory=True)
        with pytest.raises(TrainingError, match="run directory"):
            status(repository, run_id)
    finally:
        _kill_if_running(session)


def test_runner_rejects_symlinked_checkpoint_payload(tmp_path: Path) -> None:
    repository, config = _repository(tmp_path, train_steps=2)
    run_id = f"payload-link-{uuid.uuid4().hex[:8]}"
    session = f"ncm-{run_id}"
    try:
        start(run_id, [config])
        completed = _wait(repository, run_id, "complete")
        run_dir = Path(completed["experiments"][0]["run_dir"])
        checkpoint = run_dir / completed["experiments"][0]["latest"]
        model = checkpoint / "model.safetensors"
        outside = repository / "outside.safetensors"
        outside.write_bytes(model.read_bytes())
        model.unlink()
        model.symlink_to(outside)
        with pytest.raises(TrainingError, match="checkpoint file"):
            status(repository, run_id)
    finally:
        _kill_if_running(session)


def test_runner_rejects_stale_reused_process_identity(tmp_path: Path) -> None:
    repository, config = _repository(tmp_path, train_steps=2)
    run_id = f"stale-{uuid.uuid4().hex[:8]}"
    session = f"ncm-{run_id}"
    try:
        start(run_id, [config])
        _wait(repository, run_id, "complete")
        root = repository / "data" / "runs" / run_id
        identity = _process_identity(os.getpid())
        assert identity is not None
        (root / "runner.process.json").write_text(
            json.dumps(
                {
                    "process_group": os.getpid(),
                    "started": identity["started"],
                    "metadata": str((root / "runner.json").resolve()),
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(TrainingError, match="identity mismatch"):
            _runner_process_group(root)
    finally:
        _kill_if_running(session)


def test_bounded_stop_kills_stubborn_process_group() -> None:
    code = """
import os, signal, time
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
if os.fork() == 0:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(60)
time.sleep(60)
"""
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    waiter = threading.Thread(target=process.wait, daemon=True)
    waiter.start()
    try:
        _terminate_process_group(process.pid, timeout=2)
        waiter.join(timeout=1)
        assert not _process_group_alive(process.pid)
    finally:
        if _process_group_alive(process.pid):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=1)


def test_group_sentinel_waits_after_child_leader_exits() -> None:
    code = """
import os, subprocess, time
from ncm_stage_dialogue.runner import _wait_for_group_descendants
started = time.monotonic()
subprocess.run(["sh", "-c", "sleep 1 &"], check=True)
_wait_for_group_descendants(os.getpid(), os.getpid())
print(time.monotonic() - started)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
        start_new_session=True,
    )
    assert float(result.stdout.strip()) >= 0.8
