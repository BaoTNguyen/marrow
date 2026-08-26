"""Tee training progress into heart's event spine (see heart's SPINE.md).

Training runs were the last dark corner of the stack: `heart pulse tail`
shows episodes and memory decisions but nothing about the GPUs. This
callback emits training.started / training.progress / training.finished
so a run is watchable live and its loss curve lands in the same NDJSON
journal as everything else.
"""
from __future__ import annotations

from heart.events import emit

try:
    from transformers import TrainerCallback
except ImportError:  # reward-only installs (no torch) must stay importable
    TrainerCallback = object


class SpineCallback(TrainerCallback):
    def __init__(self, stage: str):
        self.stage = stage

    def on_train_begin(self, args, state, control, **kw):
        if state.is_world_process_zero:
            emit("marrow", "training.started", stage=self.stage,
                 max_steps=state.max_steps, output_dir=args.output_dir)

    def on_log(self, args, state, control, logs=None, **kw):
        if logs and state.is_world_process_zero:
            emit("marrow", "training.progress", stage=self.stage,
                 step=state.global_step,
                 **{k: v for k, v in logs.items() if isinstance(v, (int, float))})

    def on_train_end(self, args, state, control, **kw):
        if state.is_world_process_zero:
            emit("marrow", "training.finished", stage=self.stage,
                 step=state.global_step)
