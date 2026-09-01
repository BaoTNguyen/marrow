"""Self-check for raw CLI-run collection. Run: python3 tests/test_collect.py"""
from __future__ import annotations

import contextlib
import fcntl
import io
import os
import struct
import sys
import termios
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from marrow.cli import main as cli_main  # noqa: E402
from marrow.collect import (  # noqa: E402
    build_command,
    build_session_command,
    collect_cli_run,
    collect_cli_session,
    shell_hook,
)
from marrow.collect import _run_pty  # noqa: E402


class PtyWindowSizeTest(unittest.TestCase):
    def test_child_inherits_real_window_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            transcript = work / "terminal.typescript"
            master, slave = os.openpty()
            fcntl.ioctl(master, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
            pid = os.fork()
            if pid == 0:  # child: pretend the pty is our terminal
                try:
                    os.setsid()
                    fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
                    for fd in (0, 1, 2):
                        os.dup2(slave, fd)
                    sys.stdin = os.fdopen(0, "r")
                    sys.stdout = os.fdopen(1, "w")
                    sys.stderr = os.fdopen(2, "w")
                    _run_pty(["stty", "size"], work, transcript)
                finally:
                    os._exit(0)
            os.close(slave)
            os.waitpid(pid, 0)
            os.close(master)
            self.assertIn(b"40 120", transcript.read_bytes())


class TestCollect(unittest.TestCase):
    def test_build_codex_command(self):
        cmd = build_command("codex", "do the thing", "/tmp/out", ["--", "--model", "gpt-5.5"])
        self.assertEqual(cmd, ["codex", "exec", "--json", "--model", "gpt-5.5", "do the thing"])

    def test_build_claude_command(self):
        cmd = build_command("claude", "do the thing", "/tmp/out", ["--model", "sonnet"])
        self.assertEqual(cmd[:3], ["claude", "-p", "do the thing"])
        self.assertIn("--output-format", cmd)
        self.assertIn("stream-json", cmd)
        self.assertIn("--verbose", cmd)
        self.assertIn("--include-partial-messages", cmd)
        self.assertIn("--include-hook-events", cmd)
        self.assertIn("--debug-file", cmd)
        self.assertIn("/tmp/out/debug.log", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("sonnet", cmd)

    def test_build_session_command_preserves_cli_args(self):
        self.assertEqual(build_session_command("codex", ["--", "--model", "gpt-5.5"]),
                         ["codex", "--model", "gpt-5.5"])
        self.assertEqual(build_session_command("claude", ["--help"]), ["claude", "--help"])

    def test_relative_out_resolves_under_cwd(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "repo"
            repo.mkdir()
            out, code = collect_cli_run("codex", "do it", dry_run=True, cwd=repo)
            self.assertEqual(code, 0)
            self.assertEqual(out.parent, repo / ".marrow" / "collections")
            self.assertTrue(out.name.endswith("-codex"))
            out, code = collect_cli_session("claude", dry_run=True, cwd=repo)
            self.assertEqual(code, 0)
            self.assertEqual(out.parent, repo / ".marrow" / "collections")
            self.assertTrue(out.name.endswith("-claude"))

    def test_cli_run_dry_run_after_prompt_is_not_provider_arg(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "repo"
            repo.mkdir()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = cli_main(["collect", "cli-run", "claude", "do it",
                                 "--cwd", str(repo), "--dry-run", "--",
                                 "--model", "sonnet"])
            self.assertEqual(code, 0)
            out = buf.getvalue()
            self.assertNotIn("--dry-run", out)
            self.assertIn("--model sonnet", out)
            self.assertIn(str(repo / ".marrow" / "collections"), out)

    def test_cli_session_dry_run_preserves_codex_and_claude(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d) / "repo"
            repo.mkdir()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = cli_main(["collect", "cli-session", "codex",
                                 "--cwd", str(repo), "--dry-run", "--", "--help"])
            self.assertEqual(code, 0)
            self.assertEqual(buf.getvalue().strip(), "codex --help")

            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = cli_main(["collect", "cli-session", "claude",
                                 "--cwd", str(repo), "--dry-run", "--", "--help"])
            self.assertEqual(code, 0)
            self.assertEqual(buf.getvalue().strip(), "claude --help")

    def test_shell_hook_wraps_both_clis(self):
        hook = shell_hook("bash")
        self.assertIn("codex()", hook)
        self.assertIn("claude()", hook)
        self.assertIn("marrow collect cli-session", hook)
        self.assertIn("MARROW_RECORDING_ACTIVE", hook)
        self.assertIn("MARROW_AUTO_COLLECT", hook)

    def test_shell_hook_command_prints_hook(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cli_main(["shell-hook", "zsh"]), 0)
        out = buf.getvalue()
        self.assertIn("codex()", out)
        self.assertIn("claude()", out)


if __name__ == "__main__":
    unittest.main()
