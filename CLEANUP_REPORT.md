# marrow cleanup: import placement and duplication

Branch `dev`, which was already 6 commits ahead of `main`.

## Counts

| | |
|---|---|
| function-local imports audited | 4 |
| hoisted to module top | 3 |
| kept local | 1 |

marrow is the smallest repo in the stack and the cleanest going in. The three
hoists were all the same line — `from marrow.spine import SpineCallback`,
sitting one line above its only use in each of the three trainers, which is
habit rather than design. `spine.py` guards its own `transformers` import, so
nothing was being deferred by putting it there.

The one kept local is that guard: `from transformers import TrainerCallback`
inside a `try`/`except ImportError` that falls back to `object`, so a
reward-only install with no torch stays importable. It already said so.

## Within-repo duplication

### Extracted

**`train_with_spine()` in `spine.py`** — all three stages ended with the same
four lines: import the callback, attach it, train, save. Forgetting the
attach is the quiet one. The run trains fine, saves fine, and emits nothing,
so the GPUs go dark in exactly the way this module exists to prevent. Now
it is attach-and-train or neither.

### Already deduplicated before this pass

`train_dpo` and `train_grpo` both import `DEFAULT_MODEL`, `LORA` and
`load_model` from `train_sft` rather than copying them. That is the pattern;
the trainer tail was the one place it had not been applied.

### Left alone

| what | why |
|---|---|
| the shared `TrainingArguments` kwargs (`max_length`, `gradient_accumulation_steps`, `bf16`, `logging_steps`…) across `SFTConfig`, `DPOConfig` and `GRPOConfig` | they look identical and are not: GRPO sets `save_steps=25` where the others set `save_strategy="epoch"`, and the batch/accumulation pair differs because GRPO generates 8 completions per prompt. A shared dict would need an override for every stage that has one, which is every stage |
| the three `argparse` blocks | `--base` and `--out` are common; `--data` vs `--tasks` are not, and their help strings carry the stage-specific meaning |

## Cross-repo: fixed, not just reported

**The diff-size reward curve.** `score_patch` had
`1.0 if changed <= 50 else max(0.0, 1.0 - (changed - 50) / 450)` written out
character for character, matching `heart.reward.compute`'s `diff_quality`
component exactly. Two lines, and the most dangerous duplication in the stack:
it is the number GRPO optimises against and the number the runtime scores
with. Drift teaches the trainer to satisfy a reward heart no longer pays.

heart now exports `reward.diff_quality(diff_text)` and marrow calls it.
marrow already imported `heart.reward.diff_changed_lines` from the module next
door, so this cost one changed import line.

## Cross-repo: still reported only

| # | logic | files | ~lines | worth it? |
|---|---|---|---|---|
| 1 | Building `claude` / `codex` command lines. `collect.build_command` produces `["codex", "exec", "--json", ...]` and `["claude", "-p", ...]`; heart's `AGENT_COMMANDS` and `_agent_command` build the same two with ~30 lines of comments about why each flag is there. | `marrow/collect.py:32` vs `heart/runner.py:32` | ~16 | **No, not as one builder.** The flag sets genuinely differ: collect wants `stream-json` plus debug tracing to capture a trajectory, heart wants one JSON result to score. What is actually shared is the agent-name allowlist and the `--model` pinning rule. Export `heart.runner.resolve_model` if anything; leave the command tables apart. |
| 2 | Probing a model server for its slot count. | `marrow` is not involved; see arteries' report | — | — |

## Verification

| check | result |
|---|---|
| `PYTHONPATH=.:../heart/src pytest -q` | 25 passed |
| `import marrow.reward, marrow.spine` with no torch | passes — the `TrainerCallback` fallback still holds |

## Note on CI

marrow already has the branch policy the rest of the stack is adopting, and
has had it longer: `.github/workflows/ci.yml` runs on PRs into `main` and
pushes to `dev`, and checks out the sibling heart at a matching ref — a PR
into main is judged against heart's main, work on dev against heart's dev.
It also deliberately skips installing torch, because nothing in the suite
reaches it and a CUDA install would make the gate slow enough to route
around.

That file is the model for the other four repos, which currently have no
equivalent gate.
