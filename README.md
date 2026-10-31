# marrow

RL training for coding agents. Marrow trains models on episodes an agent
harness already ran and checked against real test suites, and hands the
checkpoint back as an OpenAI-compatible endpoint the harness can drive. Its
input is an exported `episodes.jsonl`; it scores patches itself
(`marrow/reward.py`), never talks to Postgres directly, and touches the agent
CLIs only to record your own sessions (`marrow collect`).

Why a separate repo: training pulls in torch, TRL, PEFT, bitsandbytes, and
vLLM — a multi-GB CUDA stack that has no business in a tool you install into
every project.

## Where it sits in the stack

Marrow is part of [vascular](https://github.com/BaoTNguyen/vascular), the
monorepo that holds the whole stack in one clone; vascular's README shows how
episodes are produced and exported for it. Marrow reads the exported
`episodes.jsonl`, scores patches with the tasks' own verifiers, trains a
checkpoint, and serves it back as the harness's API agent. The loop closes
there: a better checkpoint lands more scope per budget, which produces more
training data.

## What sets it apart

Most open RL-for-code setups train on synthetic tasks with a proxy reward and
assume a datacenter. Marrow does neither:

- **The reward is the actual test suite.** Scoring applies the generated diff to
  a fresh worktree at the base commit and runs the task's real verifiers (the episode
  runner's own code, CPU-side). No learned reward model to game, no BLEU-against-a-patch.
- **It trains on episodes you actually ran.** Every stage feeds on exported
  episodes and the memory layer's decision ledger. Best-of-N traffic from your daily
  work is the data engine; nothing is fabricated to fill a training set.
- **Staged with data gates, cheapest first.** Stages 0 through 4 each unlock the
  next — best-of-N generates the pairs SFT needs, SFT stabilizes the checkpoint
  DPO needs, and so on. You can't skip ahead past a gate, which is the point.
- **48 GB is the whole budget.** Everything targets 2× RTX 3090 with no NVLink.
  GRPO was chosen over PPO specifically because a value network would eat the VRAM
  the policy needs; 4-bit QLoRA keeps a 7B model training on one card while the
  other serves rollouts.
- **It captures real CLI sessions as traces.** A shell hook wraps ordinary
  `codex` and `claude` runs and records them under `.marrow/`, so agentic
  behavior can be learned from how you already work — without touching hidden
  chain-of-thought.
- **The only score that counts is holdout.** `tasks/holdout/` is never trained
  on and runs through the same episode machinery as everything else. A checkpoint
  is judged the way it will actually be used.
- **Training is visible on the same board as everything else.** The SFT/DPO/GRPO
  loops emit progress into the stack's event journal (`marrow/spine.py`), so a run's loss
  and step cadence show up next to the episodes it trains on —
  no separate dashboard.

## Hardware assumptions

2× RTX 3090 (24 GB each, Ampere SM86, PCIe). bf16 works. FlashAttention-2
works. No NVLink assumed. This budget comfortably trains a 7B model with QLoRA
and serves the same model on the other GPU for rollouts.

## Setup

Install from vascular, which puts the episode runner marrow imports beside it:

```bash
git clone --recursive https://github.com/BaoTNguyen/vascular && cd vascular/marrow
uv sync                          # plus torch/trl/peft/bitsandbytes
uv pip install vllm              # rollout serving (GRPO, and serving a checkpoint as an agent);
                                 # `uv run` keeps it, a plain `uv sync` removes it
source .venv/bin/activate        # or prefix the commands below with `uv run`
```

The tests need no CUDA stack: `uv run --only-group dev pytest -q`.

## Base model

`Qwen/Qwen2.5-Coder-7B-Instruct` — the best coding model per GB of VRAM in this
class. 4-bit QLoRA training uses ~9–12 GB at 4k context, leaving headroom on one
card. 14B fits with QLoRA at reduced context if 7B plateaus; 1.5B/3B variants
are the right size for the later memory extractor/gate models.

## Training sequence (do them in this order)

Each stage has a data gate. Do not skip ahead — earlier stages are cheaper and
produce the data the later ones need.

| Stage | What | Needs | GPUs |
|---|---|---|---|
| 0 | Best-of-N: each task run N times, verifier selection | tasks only, no training | 0 (or 1 serving vLLM as the API agent) |
| 1 | SFT (QLoRA) on passing diffs | 100+ passing episodes | 1–2 (DDP) |
| 2 | DPO on pass/fail pairs | 20+ same-task pairs | 1–2 |
| 3 | GRPO, single-shot patch generation, task verifiers as reward | stable stage-2 checkpoint + deterministic tasks | 2 (see topology) |
| 4 | Reranker fine-tune (mxbai-rerank-base-v2) & gate classifier | memory decision ledger + retrieval-labeled episodes | 1 |

Stage 0 is the data engine: run each task eight times against the served
model, export, and stages 1–2 fall out of it. Stage 4 waits on the memory
decision ledger — that's where retrieval/gate labels come from.

### Why GRPO and not PPO

GRPO needs no value model — on 48 GB total, a PPO value network would eat the
VRAM the policy needs. Group-relative advantages from N samples per task fit
the best-of-N data engine the episode runner already provides.

### GRPO topology on 2×3090

```text
GPU 0: trl vllm-serve --model <stage-2 checkpoint>      (rollout generation)
GPU 1: accelerate launch marrow/train_grpo.py           (policy + QLoRA update)
```

Weights sync from trainer to the vLLM server between steps over local HTTP.
Reward = apply the generated diff to a fresh worktree at base_commit, run the
task's verifiers (CPU-side), plus a diff-size term. Verifier runs
are CPU-bound, so keep tasks' test suites fast (<30 s) or GPU utilization dies.

### Scope honesty

Stage 3 trains single-shot patch generation, not the full multi-turn tool-use
loop. Multi-turn agentic RL on 48 GB is bleeding-edge and data-hungry; the
agentic behavior comes from SFT on pipeline traces (the API agent's
logs), while RL sharpens patch quality. Revisit agentic GRPO only after stage 3
beats stage 2 on the holdout tasks.

Known brittleness: models emit malformed unified diffs and `git apply` rejects
them (reward 0). `reward.py` already accepts SEARCH/REPLACE blocks as a
fallback; if apply-failure rate stays high after SFT, train with
`--format blocks` instead of diffs.

## Collect raw CLI runs

Use `marrow collect cli-run` to gather raw Codex or Claude Code runs for later
normalization, labeling, and training exports. This records the trace surfaces
the CLIs expose; it does not expose hidden chain-of-thought.

```bash
marrow collect cli-run codex "summarize this repo" --cwd ../some-repo -- --model gpt-5.5
marrow collect cli-run claude "summarize this repo" --cwd ../some-repo -- --model sonnet
```

Each run writes under `<repo>/.marrow/collections/<timestamp>-<agent>/` by
default. If `--out` is relative, it is resolved under `--cwd`; absolute `--out`
paths are used as-is. The bundle contains:

- `manifest.json` — collector, agent, task id, command, cwd, timestamps, duration, and return code
- `prompt.txt` — the submitted prompt
- `events.jsonl` — Codex `exec --json` or Claude `--output-format stream-json` output
- `stderr.log` — provider CLI stderr
- `debug.log` — Claude Code debug log (`claude` only)

Use `--task-id` when the run should join back to a task or later label row:

```bash
marrow collect cli-run claude "fix auth bug" --cwd ../app --task-id auth-001 -- --model sonnet
```

Use `--dry-run` to inspect the exact provider command without launching it.

### Auto-collect normal Codex and Claude CLI sessions

For normal CLI usage, load Marrow's shell hook in your current shell:

```bash
eval "$(marrow shell-hook bash)"   # zsh users: marrow shell-hook zsh
```

After that, ordinary `codex ...` and `claude ...` commands are wrapped by
`marrow collect cli-session` and recorded under the current repo's
`.marrow/collections/` directory. Interactive sessions are captured as
`terminal.typescript`; non-interactive sessions fall back to `stdout.log` and
`stderr.log`. The wrapper delegates to the real CLI and preserves the arguments
you typed.

```bash
codex
claude
codex --model gpt-5.5 "explain this repo"
claude --help
```

Bypass recording for a command or shell with:

```bash
MARROW_AUTO_COLLECT=0 codex
MARROW_AUTO_COLLECT=0 claude
```

To load it automatically for new shells, add the `eval` line to your shell rc
file after `marrow` is on `PATH`.

## Commands

`sft.jsonl`, `dpo.jsonl` and `tasks/` come from the episode export; vascular's
README has the commands that produce them and run the holdout.

```bash
# stage 1
accelerate launch --num_processes 2 marrow/train_sft.py --data sft.jsonl

# stage 2
accelerate launch --num_processes 2 marrow/train_dpo.py --data dpo.jsonl \
    --base checkpoints/sft

# stage 3
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model checkpoints/dpo &
CUDA_VISIBLE_DEVICES=1 python3 marrow/train_grpo.py --tasks tasks/ \
    --base checkpoints/dpo

# evaluate any checkpoint the same way it will be used: serve it, then run the
# holdout tasks against it as the API agent
CUDA_VISIBLE_DEVICES=0 vllm serve checkpoints/grpo --port 8000 &
```

The holdout run is the only score that matters: tasks never trained on, run
through the same episode machinery as everything else.
