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
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "heart" / "src"))

from marrow.spine import SpineCallback  # noqa: E402


class SpineTests(unittest.TestCase):
    def test_callback_emits_lifecycle(self):
        # VASCULAR_HOME too: heart's emitter keeps per-context markers there
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"EVENT_JOURNAL_DIR": tmp,
                                        "VASCULAR_HOME": str(Path(tmp) / "home")}):
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
            # a fresh emitter context opens with context.started (pulse's envelope)
            events = [e for e in events if e["kind"] != "context.started"]
            kinds = [e["kind"] for e in events]
            self.assertEqual(kinds, ["training.started", "training.progress",
                                     "training.finished"])
            self.assertEqual(events[1]["payload"]["loss"], 1.25)
            self.assertNotIn("note", events[1]["payload"])  # numeric only
            self.assertEqual(events[1]["source"], "marrow")

if __name__ == "__main__":
    unittest.main()
