"""Collect CLI agent runs as raw training-data artifacts."""
from __future__ import annotations

import datetime
import json
import os
import pty
import subprocess
import sys
import threading
from pathlib import Path


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _stamp() -> str:
    return _now().strftime("%Y%m%dT%H%M%SZ")


def collection_dir(base: str | Path, agent: str) -> Path:
    return Path(base) / f"{_stamp()}-{agent}"


def build_command(agent: str, prompt: str, out_dir: str | Path,
                  extra: list[str] | None = None) -> list[str]:
    extra = _clean_extra(extra)
    if agent == "codex":
        return ["codex", "exec", "--json", *extra, prompt]
    if agent == "claude":
        return [
            "claude", "-p", prompt,
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
            "--include-hook-events",
            "--debug", "api,mcp,hooks",
            "--debug-file", str(Path(out_dir) / "debug.log"),
            *extra,
        ]
    raise ValueError(f"unsupported agent: {agent}")


def build_session_command(agent: str, extra: list[str] | None = None) -> list[str]:
    if agent not in {"codex", "claude"}:
        raise ValueError(f"unsupported agent: {agent}")
    return [agent, *_clean_extra(extra)]


def _clean_extra(extra: list[str] | None = None) -> list[str]:
    extra = list(extra or [])
    if extra and extra[0] == "--":
        extra = extra[1:]
    return extra


def _base_dir(base_dir: str | Path, cwd: str | Path) -> Path:
    work_dir = Path(cwd).expanduser().resolve()
    if not work_dir.is_dir():
        raise FileNotFoundError(f"collect cwd does not exist or is not a directory: {work_dir}")
    base = Path(base_dir).expanduser()
    if not base.is_absolute():
        base = work_dir / base
    return base


def collect_cli_run(agent: str, prompt: str, base_dir: str | Path = ".marrow/collections",
                    extra: list[str] | None = None, dry_run: bool = False,
                    cwd: str | Path = ".", task_id: str | None = None) -> tuple[Path, int]:
    """Run one CLI agent and capture the raw artifacts Marrow can later normalize."""
    work_dir = Path(cwd).expanduser().resolve()
    out = collection_dir(_base_dir(base_dir, work_dir), agent)
    cmd = build_command(agent, prompt, out, extra)
    if dry_run:
        return out, 0

    out.mkdir(parents=True, exist_ok=False)
    start = _now()
    prompt_path = out / "prompt.txt"
    events_path = out / "events.jsonl"
    stderr_path = out / "stderr.log"
    prompt_path.write_text(prompt, encoding="utf-8")

    with open(events_path, "w", encoding="utf-8") as stdout:
        with open(stderr_path, "w", encoding="utf-8") as stderr:
            proc = subprocess.run(cmd, cwd=str(work_dir), stdout=stdout,
                                  stderr=stderr, text=True)

    end = _now()
    manifest = {
        "collector": "cli-run",
        "agent": agent,
        "task_id": task_id,
        "command": cmd,
        "cwd": str(work_dir),
        "started_at": start.isoformat(),
        "finished_at": end.isoformat(),
        "duration_seconds": round((end - start).total_seconds(), 3),
        "returncode": proc.returncode,
        "prompt": prompt_path.name,
        "events": events_path.name,
        "stderr": stderr_path.name,
    }
    if agent == "claude":
        manifest["debug"] = "debug.log"
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                        encoding="utf-8")
    return out, proc.returncode


def collect_cli_session(agent: str, base_dir: str | Path = ".marrow/collections",
                        extra: list[str] | None = None, dry_run: bool = False,
                        cwd: str | Path = ".", task_id: str | None = None) -> tuple[Path, int]:
    """Run the real Codex/Claude CLI and record the visible terminal session."""
    work_dir = Path(cwd).expanduser().resolve()
    out = collection_dir(_base_dir(base_dir, work_dir), agent)
    cmd = build_session_command(agent, extra)
    if dry_run:
        return out, 0

    out.mkdir(parents=True, exist_ok=False)
    start = _now()
    if sys.stdin.isatty() and sys.stdout.isatty():
        mode = "pty"
        returncode = _run_pty(cmd, work_dir, out / "terminal.typescript")
        artifacts = {"terminal": "terminal.typescript"}
    else:
        mode = "pipes"
        returncode = _run_pipes(cmd, work_dir, out / "stdout.log", out / "stderr.log")
        artifacts = {"stdout": "stdout.log", "stderr": "stderr.log"}
    end = _now()

    manifest = {
        "collector": "cli-session",
        "agent": agent,
        "task_id": task_id,
        "command": cmd,
        "cwd": str(work_dir),
        "mode": mode,
        "started_at": start.isoformat(),
        "finished_at": end.isoformat(),
        "duration_seconds": round((end - start).total_seconds(), 3),
        "returncode": returncode,
        **artifacts,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                        encoding="utf-8")
    return out, returncode


def _run_pty(cmd: list[str], cwd: Path, transcript_path: Path) -> int:
    old_cwd = Path.cwd()
    with open(transcript_path, "ab", buffering=0) as transcript:
        def read(fd: int) -> bytes:
            data = os.read(fd, 4096)
            transcript.write(data)
            return data

        os.chdir(cwd)
        try:
            status = pty.spawn(cmd, read)
        finally:
            os.chdir(old_cwd)
    try:
        return os.waitstatus_to_exitcode(status)
    except ValueError:
        return int(status)


def _run_pipes(cmd: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    proc = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None and proc.stderr is not None
    with open(stdout_path, "wb") as stdout, open(stderr_path, "wb") as stderr:
        threads = [
            threading.Thread(target=_tee, args=(proc.stdout, sys.stdout, stdout)),
            threading.Thread(target=_tee, args=(proc.stderr, sys.stderr, stderr)),
        ]
        for thread in threads:
            thread.start()
        returncode = proc.wait()
        for thread in threads:
            thread.join()
    return returncode


def _tee(src, dst, log) -> None:
    while True:
        chunk = src.read(8192)
        if not chunk:
            break
        log.write(chunk)
        log.flush()
        stream = getattr(dst, "buffer", dst)
        stream.write(chunk)
        stream.flush()


def shell_hook(shell: str = "bash") -> str:
    if shell not in {"bash", "zsh"}:
        raise ValueError(f"unsupported shell: {shell}")
    return '''# Marrow auto-collection for Codex and Claude CLI sessions.
# Set MARROW_AUTO_COLLECT=0 to bypass recording for one command or shell.
__marrow_collect_cli() {
  local agent="$1"
  shift
  if [ "${MARROW_AUTO_COLLECT:-1}" = "0" ] || [ -n "${MARROW_RECORDING_ACTIVE:-}" ]; then
    command "$agent" "$@"
    return $?
  fi
  MARROW_RECORDING_ACTIVE=1 command marrow collect cli-session "$agent" --cwd "$PWD" -- "$@"
}

codex() { __marrow_collect_cli codex "$@"; }
claude() { __marrow_collect_cli claude "$@"; }
'''
