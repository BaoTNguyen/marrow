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


def train_with_spine(trainer, stage: str, out: str) -> None:
    """Attach the spine callback, train, save.

    All three stages ended with these same four lines. Forgetting the first
    one is the quiet failure: the run trains fine and emits nothing, so the
    GPUs go dark again in exactly the way this module exists to prevent.
    """
    trainer.add_callback(SpineCallback(stage))
    trainer.train()
    trainer.save_model(out)
