"""Self-check for the training-event tee — runs without torch or transformers.
Run: python3 tests/test_spine.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "heart" / "src"))

from marrow.spine import SpineCallback  # noqa: E402


class SpineTests(unittest.TestCase):
    def test_callback_emits_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("HEART_SPOOL_DIR")
            os.environ["HEART_SPOOL_DIR"] = tmp
            try:
                cb = SpineCallback("sft")
                args = types.SimpleNamespace(output_dir="checkpoints/sft")
                state = types.SimpleNamespace(
                    is_world_process_zero=True, max_steps=100, global_step=0)
                cb.on_train_begin(args, state, None)
                state.global_step = 5
                cb.on_log(args, state, None, logs={"loss": 1.25, "note": "skip-me"})
                cb.on_train_end(args, state, None)

                events = [json.loads(line)
                          for p in sorted(Path(tmp).glob("*.ndjson"))
                          for line in p.read_text().splitlines()]
                kinds = [e["kind"] for e in events]
                self.assertEqual(kinds, ["training.started", "training.progress",
                                         "training.finished"])
                self.assertEqual(events[1]["payload"]["loss"], 1.25)
                self.assertNotIn("note", events[1]["payload"])  # numeric only
                self.assertEqual(events[1]["source"], "marrow")
            finally:
                if old is None:
                    os.environ.pop("HEART_SPOOL_DIR", None)
                else:
                    os.environ["HEART_SPOOL_DIR"] = old


if __name__ == "__main__":
    unittest.main()
