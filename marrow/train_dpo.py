"""Stage 2: DPO on heart's same-task pass/fail diff pairs (dpo.jsonl).

Launch:  accelerate launch --num_processes 2 marrow/train_dpo.py \
             --data dpo.jsonl --base checkpoints/sft
"""
from __future__ import annotations

import argparse

from datasets import load_dataset
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

from .train_sft import DEFAULT_MODEL, LORA, load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="heart dpo.jsonl")
    ap.add_argument("--base", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="checkpoints/dpo")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda r: {
        "prompt": r["prompt"],
        "chosen": f"```diff\n{r['chosen']}```",
        "rejected": f"```diff\n{r['rejected']}```",
    })

    trainer = DPOTrainer(
        model=load_model(args.base),
        processing_class=tok,
        train_dataset=ds,
        peft_config=LORA,  # ref model implicit via disabled adapters — fits one GPU
        args=DPOConfig(
            output_dir=args.out,
            beta=0.1,
            max_length=4096,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            learning_rate=5e-6,
            num_train_epochs=1,
            bf16=True,
            logging_steps=5,
            save_strategy="epoch",
        ),
    )
    trainer.train()
    trainer.save_model(args.out)


if __name__ == "__main__":
    main()
