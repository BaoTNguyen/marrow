# marrow

RL training for the heart/arteries/capillaries stack. Marrow trains the models;
heart runs the episodes and computes rewards. The contract between them is
heart's `episodes.jsonl` export plus heart imported as a library for patch
scoring — marrow never talks to Postgres or the agent CLIs directly.

Why a separate repo: heart is a daily-workflow orchestrator that stays
stdlib-only so it installs anywhere in seconds. Training pulls in torch, TRL,
PEFT, bitsandbytes, and vLLM — a multi-GB CUDA stack that has no business in a
tool you install into every project.

## Where it sits in the stack

Marrow is the training layer at the bottom of a five-repo agent stack:

```text
capillaries  prompt/skill retrieval
arteries     memory + trace substrate
heart        orchestration + environment + reward
plexus       goal decomposition + acceptance loop
marrow       RL training on heart's episodes           <- this repo
```

Heart runs the episodes and scores them; marrow reads the exported
`episodes.jsonl`, trains a checkpoint, and hands it back as heart's `api` agent.
The loop closes there: a better checkpoint lands more scope per budget, which
produces more training data.

## What sets it apart

Most open RL-for-code setups train on synthetic tasks with a proxy reward and
assume a datacenter. Marrow does neither:

- **The reward is the actual test suite.** Scoring applies the generated diff to
  a fresh worktree at the base commit and runs the task's real verifiers (heart's
  own code, CPU-side). No learned reward model to game, no BLEU-against-a-patch.
- **It trains on episodes you actually ran.** Every stage feeds on heart's
  exported runs and arteries' decision ledger. Best-of-N traffic from your daily
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
  loops tee progress into heart's event spine (`marrow/spine.py`), so a run's loss
  and step cadence show up in `heart pulse` next to the episodes it trains on —
  no separate dashboard.

## Hardware assumptions

2× RTX 3090 (24 GB each, Ampere SM86, PCIe). bf16 works. FlashAttention-2
works. No NVLink assumed. This budget comfortably trains a 7B model with QLoRA
and serves the same model on the other GPU for rollouts.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ../heart          # reward computation imports heart
pip install -e .                 # torch/trl/peft/bitsandbytes
pip install vllm                 # rollout serving (GRPO + the heart "local" agent)
```

## Base model

`Qwen/Qwen2.5-Coder-7B-Instruct` — the best coding model per GB of VRAM in this
class. 4-bit QLoRA training uses ~9–12 GB at 4k context, leaving headroom on one
card. 14B fits with QLoRA at reduced context if 7B plateaus; 1.5B/3B variants
are the right size for the later arteries extractor/gate models.

## Training sequence (do them in this order)

Each stage has a data gate. Do not skip ahead — earlier stages are cheaper and
produce the data the later ones need.

| Stage | What | Needs | GPUs |
|---|---|---|---|
| 0 | Best-of-N via `heart batch --repeat N` + verifier selection | tasks only, no training | 0 (or 1 serving vLLM for heart's `api` agent) |
| 1 | SFT (QLoRA) on passing diffs | 100+ passing episodes | 1–2 (DDP) |
| 2 | DPO on pass/fail pairs | 20+ same-task pairs | 1–2 |
| 3 | GRPO, single-shot patch generation, heart verifiers as reward | stable stage-2 checkpoint + deterministic tasks | 2 (see topology) |
| 4 | Reranker fine-tune (mxbai-rerank-base-v2) & gate classifier | arteries decision ledger + retrieval-labeled episodes | 1 |

Stage 0 is the data engine: run `heart batch tasks/ --agent api --repeat 8`,
export, and stages 1–2 fall out of it. Stage 4 waits on the arteries decision
ledger — that's where retrieval/gate labels come from.

### Why GRPO and not PPO

GRPO needs no value model — on 48 GB total, a PPO value network would eat the
VRAM the policy needs. Group-relative advantages from N samples per task fit
the best-of-N data engine heart already provides. This is the same reasoning as
the handoff doc's "do not start with PPO," with hardware to back it.

### GRPO topology on 2×3090

```text
GPU 0: trl vllm-serve --model <stage-2 checkpoint>      (rollout generation)
GPU 1: accelerate launch marrow/train_grpo.py           (policy + QLoRA update)
```

Weights sync from trainer to the vLLM server between steps over local HTTP.
Reward = apply the generated diff to a fresh worktree at base_commit, run the
task's verifiers (heart code, CPU-side), plus a diff-size term. Verifier runs
are CPU-bound, so keep tasks' test suites fast (<30 s) or GPU utilization dies.

### Scope honesty

Stage 3 trains single-shot patch generation, not the full multi-turn tool-use
loop. Multi-turn agentic RL on 48 GB is bleeding-edge and data-hungry; the
agentic behavior comes from SFT on heart pipeline traces (the `api` agent's
logs), while RL sharpens patch quality. Revisit agentic GRPO only after stage 3
beats stage 2 on heart's holdout tasks.

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

```bash
# stage 1
accelerate launch --num_processes 2 marrow/train_sft.py --data ../heart/sft.jsonl

# stage 2
accelerate launch --num_processes 2 marrow/train_dpo.py --data ../heart/dpo.jsonl \
    --base checkpoints/sft

# stage 3
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model checkpoints/dpo &
CUDA_VISIBLE_DEVICES=1 python3 marrow/train_grpo.py --tasks ../heart/tasks \
    --base checkpoints/dpo

# evaluate any checkpoint the same way it will be used
CUDA_VISIBLE_DEVICES=0 vllm serve checkpoints/grpo --port 8000 &
cd ../heart && heart batch tasks/holdout --agent api
```

The last command is the only score that matters: heart's holdout tasks, never
trained on, run through the same episode machinery as everything else.
