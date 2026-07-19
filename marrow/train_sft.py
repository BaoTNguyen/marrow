"""Stage 1: SFT (QLoRA) on passing episode diffs from heart's sft.jsonl.

Launch:  accelerate launch --num_processes 2 marrow/train_sft.py --data sft.jsonl
Fits one 3090 at 4k context; two GPUs just double throughput via DDP.
"""
from __future__ import annotations

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"


def load_model(name: str):
    return AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )


LORA = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, target_modules="all-linear")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="heart sft.jsonl")
    ap.add_argument("--base", default=DEFAULT_MODEL)
    ap.add_argument("--out", default="checkpoints/sft")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base)
    ds = load_dataset("json", data_files=args.data, split="train")
    ds = ds.map(lambda r: {"messages": [
        {"role": "user", "content": r["prompt"]},
        {"role": "assistant", "content": f"```diff\n{r['completion']}```"},
    ]})

    trainer = SFTTrainer(
        model=load_model(args.base),
        processing_class=tok,
        train_dataset=ds,
        peft_config=LORA,
        args=SFTConfig(
            output_dir=args.out,
            max_length=4096,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            learning_rate=1e-4,
            num_train_epochs=2,
            bf16=True,
            logging_steps=5,
            save_strategy="epoch",
        ),
    )
    from marrow.spine import SpineCallback
    trainer.add_callback(SpineCallback("sft"))
    trainer.train()
    trainer.save_model(args.out)


if __name__ == "__main__":
    main()
