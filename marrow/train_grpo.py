"""Stage 3: GRPO on single-shot patch generation. Reward = heart verifiers on
the generated diff (marrow.reward.score_patch).

Topology on 2x3090:
  GPU 0:  CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model checkpoints/dpo
  GPU 1:  CUDA_VISIBLE_DEVICES=1 python3 marrow/train_grpo.py \
              --tasks ../heart/tasks --base checkpoints/dpo

Only run against tasks that passed `heart check-task` (deterministic verifiers);
flaky rewards are worse than no training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from .reward import build_prompt, score_patch
from .train_sft import DEFAULT_MODEL, LORA, load_model


def build_dataset(tasks_dir: str, fmt: str) -> Dataset:
    rows = []
    for path in sorted(Path(tasks_dir).glob("*.json")):
        task = json.loads(path.read_text())
        rows.append({"prompt": build_prompt(task, fmt=fmt), "task_json": json.dumps(task)})
    if not rows:
        raise SystemExit(f"no task JSONs in {tasks_dir}")
    return Dataset.from_list(rows)


def verifier_reward(prompts, completions, task_json, **kwargs) -> list[float]:
    return [score_patch(json.loads(t), c) for c, t in zip(completions, task_json)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="heart TaskSpec directory (not holdout/)")
    ap.add_argument("--base", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="checkpoints/grpo")
    ap.add_argument("--format", default="diff", choices=["diff", "blocks"],
                    help="completion format; switch to blocks if apply-failure rate stays high")
    args = ap.parse_args()

    trainer = GRPOTrainer(
        model=load_model(args.base),
        processing_class=AutoTokenizer.from_pretrained(args.base),
        reward_funcs=verifier_reward,
        train_dataset=build_dataset(args.tasks, args.format),
        peft_config=LORA,
        args=GRPOConfig(
            output_dir=args.out,
            use_vllm=True,  # generation happens on the GPU-0 vllm-serve process
            num_generations=8,
            max_prompt_length=6144,
            max_completion_length=2048,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            learning_rate=1e-6,
            bf16=True,
            logging_steps=1,
            save_steps=25,
        ),
    )
    from marrow.spine import SpineCallback
    trainer.add_callback(SpineCallback("grpo"))
    trainer.train()
    trainer.save_model(args.out)


if __name__ == "__main__":
    main()
