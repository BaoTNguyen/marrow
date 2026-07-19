"""Self-check for the GRPO reward path — runs without torch or GPUs.
Run: python3 tests/test_reward.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "heart" / "src"))

from marrow.reward import build_prompt, extract_diff, score_patch  # noqa: E402

GOOD_BLOCKS = """\
calc.py
<<<<<<< SEARCH
    return a - b
=======
    return a + b
>>>>>>> REPLACE
"""

GOOD_DIFF = """\
--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,2 @@
 def add(a, b):
-    return a - b
+    return a + b
"""


class TestReward(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        repo = Path(self.tmp.name) / "repo"
        repo.mkdir()
        (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        (repo / "test_calc.py").write_text(
            "from calc import add\n\nassert add(2, 3) == 5\n"
        )
        git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t"]
        subprocess.run([*git[:3], "init", "-q"], check=True)
        subprocess.run([*git, "add", "-A"], check=True)
        subprocess.run([*git, "commit", "-qm", "bug"], check=True)
        base = subprocess.run(
            [*git[:3], "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        self.task = {
            "task_id": "toy",
            "repo_path": str(repo),
            "base_commit": base,
            "prompt": "Fix add.",
            "public_verifiers": [{"name": "unit", "command": "python3 test_calc.py"}],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract_diff(self):
        fenced = f"Here you go:\n```diff\n{GOOD_DIFF}```\ndone"
        self.assertIn("+    return a + b", extract_diff(fenced))
        self.assertIn("+    return a + b", extract_diff(GOOD_DIFF))
        self.assertEqual(extract_diff("no patch here"), "")

    def test_score_patch(self):
        self.assertEqual(score_patch(self.task, f"```diff\n{GOOD_DIFF}```"), 1.0)
        bad = GOOD_DIFF.replace("a + b", "a * b")
        self.assertEqual(score_patch(self.task, f"```diff\n{bad}```"), 0.2)  # quality only
        self.assertEqual(score_patch(self.task, "I could not produce a diff."), 0.0)
        self.assertEqual(score_patch(self.task, "```diff\ngarbage patch\n```"), 0.0)

    def test_search_replace_blocks(self):
        self.assertEqual(score_patch(self.task, GOOD_BLOCKS), 1.0)
        missing = GOOD_BLOCKS.replace("a - b", "not in the file")
        self.assertEqual(score_patch(self.task, missing), 0.0)

    def test_build_prompt(self):
        prompt = build_prompt(self.task)
        self.assertIn("Fix add.", prompt)
        self.assertIn("```diff", prompt)
        self.assertIn("SEARCH", build_prompt(self.task, fmt="blocks"))


if __name__ == "__main__":
    unittest.main()
