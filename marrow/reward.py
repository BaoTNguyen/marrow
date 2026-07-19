"""Patch scoring for GRPO: apply a generated diff to a fresh worktree at the
task's base commit and run the task's verifiers. Pure heart + stdlib — no torch,
so it's testable without GPUs and runs CPU-side during training.
"""
from __future__ import annotations

import re
import subprocess

from heart.env import Workspace
from heart.reward import diff_changed_lines
from heart.taskspec import Verifier
from heart.verify import run_verifiers

FENCE_RE = re.compile(r"```(?:diff|patch)?\n(.*?)```", re.DOTALL)
# aider-style edit blocks: filename line, then <<<<<<< SEARCH / ======= / >>>>>>> REPLACE
BLOCK_RE = re.compile(
    r"^(\S+)\n<{7} SEARCH\n(.*?)\n={7}\n(.*?)\n>{7} REPLACE",
    re.DOTALL | re.MULTILINE,
)


def extract_diff(text: str) -> str:
    for block in FENCE_RE.findall(text):
        if "--- " in block or block.lstrip().startswith("diff --git"):
            return block
    for marker in ("diff --git", "--- "):
        idx = text.find(marker)
        if idx != -1:
            return text[idx:]
    return ""


def apply_blocks(root, text: str) -> bool:
    """Apply SEARCH/REPLACE blocks directly to files. All blocks must land."""
    blocks = BLOCK_RE.findall(text)
    if not blocks:
        return False
    for fname, search, replace in blocks:
        path = root / fname.strip("`")
        if not path.is_file():
            return False
        content = path.read_text()
        if search not in content:
            return False
        path.write_text(content.replace(search, replace, 1))
    return True


def score_patch(task: dict, completion: str, timeout: int = 120) -> float:
    """0.8 * verifier pass fraction + 0.2 * diff-size quality. Accepts unified
    diffs or SEARCH/REPLACE blocks; completions that apply neither way score 0
    (that shows up in training stats as the apply-failure rate)."""
    verifiers = [Verifier(**v) for v in task.get("public_verifiers", [])]
    ws = Workspace(task["repo_path"], task["base_commit"])
    try:
        applied = False
        diff = extract_diff(completion)
        if diff.strip():
            try:
                ws.apply(diff)
                applied = True
            except RuntimeError:
                pass
        if not applied:
            applied = apply_blocks(ws.path, completion)
        if not applied:
            return 0.0
        final_diff = ws.diff()  # actual change size, whichever format applied
        results = run_verifiers(verifiers, str(ws.path), timeout)
        pass_frac = (
            sum(r["passed"] for r in results.values()) / len(results) if results else 0.0
        )
    finally:
        ws.destroy()
    changed = diff_changed_lines(final_diff)
    quality = 1.0 if changed <= 50 else max(0.0, 1.0 - (changed - 50) / 450)
    return round(0.8 * pass_frac + 0.2 * quality, 4)


DIFF_INSTRUCTION = (
    "Output a unified diff (git apply compatible, paths relative to the repo root) "
    "inside a ```diff fence. No other changes."
)
BLOCKS_INSTRUCTION = (
    "Output your edits as one or more blocks in exactly this format "
    "(filename line, then the markers):\n"
    "path/to/file.py\n<<<<<<< SEARCH\nexact existing lines\n=======\n"
    "replacement lines\n>>>>>>> REPLACE"
)


def build_prompt(task: dict, fmt: str = "diff", max_chars_per_file: int = 6000) -> str:
    """Single-shot patch prompt: task text + base-commit contents of the files
    the fix touched (known from fix_commit on mined tasks)."""
    repo, base = task["repo_path"], task["base_commit"]
    sections = [task["prompt"]]
    if task.get("fix_commit"):
        names = subprocess.run(
            ["git", "-C", repo, "diff", "--name-only", base, task["fix_commit"]],
            capture_output=True, text=True,
        ).stdout.split()
        for name in names[:8]:
            show = subprocess.run(
                ["git", "-C", repo, "show", f"{base}:{name}"],
                capture_output=True, text=True,
            )
            if show.returncode == 0:
                sections.append(f"### {name}\n```\n{show.stdout[:max_chars_per_file]}\n```")
    sections.append(BLOCKS_INSTRUCTION if fmt == "blocks" else DIFF_INSTRUCTION)
    return "\n\n".join(sections)
