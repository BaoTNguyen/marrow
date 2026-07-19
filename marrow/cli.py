"""marrow CLI: data collection and training entry points."""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="marrow")
    sub = p.add_subparsers(dest="cmd", required=True)

    collect = sub.add_parser("collect", help="collect raw artifacts for training data")
    collect_sub = collect.add_subparsers(dest="collect_cmd", required=True)

    s = collect_sub.add_parser("cli-run", help="record a structured Codex or Claude one-shot run")
    s.add_argument("agent", choices=("codex", "claude"))
    s.add_argument("prompt")
    s.add_argument("--cwd", default=".", help="repo directory where the agent CLI should run")
    s.add_argument("--out", default=".marrow/collections",
                   help="base collection directory; relative paths resolve under --cwd")
    s.add_argument("--task-id", default=None, help="optional stable task id for dataset joins")
    s.add_argument("--dry-run", action="store_true",
                   help="print the provider command without executing it")

    s = collect_sub.add_parser("cli-session", help="record a visible Codex or Claude CLI session")
    s.add_argument("agent", choices=("codex", "claude"))
    s.add_argument("--cwd", default=".", help="repo directory where the agent CLI should run")
    s.add_argument("--out", default=".marrow/collections",
                   help="base collection directory; relative paths resolve under --cwd")
    s.add_argument("--task-id", default=None, help="optional stable task id for dataset joins")
    s.add_argument("--dry-run", action="store_true",
                   help="print the provider command without executing it")

    s = sub.add_parser("shell-hook", help="print shell functions that auto-record codex and claude")
    s.add_argument("shell", nargs="?", default="bash", choices=("bash", "zsh"))

    args, extra = p.parse_known_args(argv)
    collect_cli = args.cmd == "collect" and args.collect_cmd in {"cli-run", "cli-session"}
    if extra and not collect_cli:
        p.error("unrecognized arguments: " + " ".join(extra))

    if args.cmd == "shell-hook":
        from .collect import shell_hook
        print(shell_hook(args.shell), end="")
        return 0

    if args.cmd == "collect" and args.collect_cmd == "cli-run":
        from .collect import build_command, collect_cli_run
        out, code = collect_cli_run(args.agent, args.prompt, args.out, extra,
                                    dry_run=args.dry_run, cwd=args.cwd,
                                    task_id=args.task_id)
        if args.dry_run:
            print(" ".join(build_command(args.agent, args.prompt, out, extra)))
            return 0
        print(f"collected {args.agent} run in {out}")
        return code

    if args.cmd == "collect" and args.collect_cmd == "cli-session":
        from .collect import build_session_command, collect_cli_session
        out, code = collect_cli_session(args.agent, args.out, extra,
                                        dry_run=args.dry_run, cwd=args.cwd,
                                        task_id=args.task_id)
        if args.dry_run:
            print(" ".join(build_session_command(args.agent, extra)))
            return 0
        print(f"collected {args.agent} session in {out}")
        return code

    return 2


if __name__ == "__main__":
    sys.exit(main())
