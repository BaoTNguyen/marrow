# Extra context — marrow

Session handoff, written 2026-08-26. `README.md` explains what marrow is;
`marrow/capillaries_opt/SPEC.md` is the label format law. This file holds
current state and what a fresh session can't derive from either.

## State

Branch `dev`, in sync with origin, at `c45eaf5` (merge of
`event-journal-and-sandbox`). Tests: `python3 -m pytest -q` → 25 passed, under
two seconds.

The merge pulled in dev-side work that has **not been reviewed**, notably
`marrow/capillaries_opt/labels.py`. It passes; that isn't the same as read.

Recent line, newest first:

- `d9aa0ad` follow the spool → journal rename (arteries renamed the event spool;
  marrow tracks it)
- `b77906d` tier A gate labeling
- `9ed4dc1` label format spec and holdout oracle
- `fbcaa83` tee training progress into heart's event spine

## `capillaries_opt` is the live thread

The retrieval label set: which prompt *should* have come back for a query, and
whether anything should have. Capillaries' `docs/rework_actions.md` calls it the
keystone and names what it blocks — the chunk cutover, the reranker choice, the
embedding bake-off. Chunking is built, backfilled, and sitting unused because
nobody can say whether it helped.

Two decisions here are load-bearing:

**Labels key on `title`, never `prompt_id`.** `prompts.prompt_id` is
`gen_random_uuid()`; every corpus rebuild reassigns every ID, and one on
2026-08-17 did exactly that to all 1,026 rows. `title` has a `UNIQUE`
constraint and is what ingest upserts on. This isn't a preference — capillaries'
golden set keys on title substrings and survived that rebuild untouched, while a
`relevant_prompt_ids` field would have been silently invalidated.

**`content_hash` is a staleness marker, not an identity.** When a prompt's text
changes, its hash changes, and any label mentioning it is flagged for re-review
rather than quietly coming to mean something else.

Labeling interface keys: `a` answerable, `n` nothing_relevant, `x`
not_a_retrieval_query, `s` skip, `u` undo, `q` quit. Notes attach to any
decision. When torn between "the corpus has a gap" and "the router erred",
`n` is the safer default — `x` asserts the query was never retrievable.

## Direction of imports

Marrow imports capillaries to pool candidates: top of the stack reading from the
bottom, which is the direction that works. **Capillaries never imports marrow.**
The loop is a subcommand and is not invoked automatically.

Marrow reads heart's exported `episodes.jsonl` and imports heart as a library
for patch scoring. It never talks to Postgres or the agent CLIs directly. CI has
to check out heart for this reason (`d6bde67`).

## What marrow owns

Off-policy calibration of retrieval acceptance. Arteries gates retrieval and
applies an injection floor; capillaries owns candidate retrieval and Qwen
reranking; marrow owns learning what the floor should be.

The standing finding: **eliminating arbitrary numeric injection floors requires
labeled inject/useful-vs-abstain examples. Rank alone is insufficient.** That is
the argument for `capillaries_opt` existing at all.

A related result from the arteries side, so it isn't rediscovered: gating on the
**margin** between top-1 and runner-up rerank scores does not work. It inverts.
The worst observed retrieval had a margin of 0.526; an apt one had 0.035.

## Training strategy under discussion

Let initially suboptimal runs finish their subtasks — two or three iterations —
so the agent learns from failure and converges, rather than cutting them off.
Not implemented.

The hard negative marrow can't get anywhere else is the "heart passed /
acceptance failed" cell: heart's verifiers judge the task, plexus's acceptance
judges the goal against ground truth, and the disagreement is the signal. That's
why those two checks are deliberately not merged (plexus `LEDGER.md` law 5).

## Traps

- `.marrow/collections/` is where sessions land, and it accumulates. As of today
  arteries alone holds eight collection directories, all untracked.
- `.arteries/hooks/` is generated and gitignored. Capillaries lost its entire
  hook install that way and ran blind for a week. Verify the directory exists,
  not just that `settings.local.json` references it.
- Marrow has essentially no arteries memory of its own — one uncompiled
  ephemeral row from 2026-08-18. It reads the `harness` scope, so it sees
  arteries' memory, which is arteries-flavored.
