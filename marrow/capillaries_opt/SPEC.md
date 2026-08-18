# Capillaries retrieval labels — format spec

Ground truth for capillaries' retrieval layer: which prompt *should* have come
back for a given query, and whether anything should have come back at all.

`docs/rework_actions.md` in capillaries calls this the keystone, and lists what
it blocks: the chunk cutover (2.2), the reranker choice (6.1), and the embedding
bake-off (6.3). Chunking is built, backfilled, and sitting unused because
nobody can say whether it helped.

## Why it lives here

Marrow already collects the raw material. Every query in the set comes from a
real session, and `.marrow/collections/` is where sessions land. Labels are
annotations on data marrow already owns.

Marrow imports capillaries to pool candidates — top of the stack reading from
the bottom, which is the direction that works. Capillaries never imports this
package.

## Identity: title, never prompt_id

`prompts.prompt_id` is `gen_random_uuid()`. Every database rebuild assigns new
IDs to the same content — a full rebuild of the 1,026-prompt corpus takes about
four minutes, and one on 2026-08-17 replaced every UUID in the table.

So labels key on `title`, which has a `UNIQUE` constraint and is what ingest
upserts on (`ON CONFLICT (title)`). This is not a theoretical preference:
capillaries' existing golden set in `tests/test_search.py` keys on title
substrings and survived that rebuild untouched, while a `relevant_prompt_ids`
field in `eval/plexus_queries.jsonl` would have been silently invalidated.

`content_hash` rides along as a **staleness marker**, not an identity. When a
prompt's text changes its hash changes, and any label mentioning it is flagged
for re-review rather than quietly coming to mean something else.

## Record format

One JSON object per line. `labels.jsonl`:

```json
{
  "query": "build a 13-week cash flow model for a SaaS startup",
  "query_id": "sha256:8f14e45f...",
  "source": "plexus-session",
  "harvested_at": "2026-08-17T19:31:00Z",

  "label": "answerable",
  "relevant": ["13-Week Cash Flow Model", "3-Year Business Plan Financials"],
  "judged_pool": ["13-Week Cash Flow Model", "Cohort Retention Model", "..."],
  "content_hashes": {"13-Week Cash Flow Model": "1c2a1ce8e9f1fd2d"},

  "pool_config": ["vector@10", "keyword@10", "union@20"],
  "corpus_size": 1026,
  "judged_by": "bao",
  "judged_at": "2026-08-18T04:12:00Z",
  "notes": "wants the 13-week variant specifically, not the 3-year plan"
}
```

### `label` — the gate axis

Three values, and they are not decoration. Capillaries is allowed to return
nothing; that is the behaviour its README leads with. Ranking quality and gate
correctness are separate measurements and need separate ground truth.

| value | meaning |
|---|---|
| `answerable` | a good prompt exists; the gate should open |
| `nothing_relevant` | the corpus does not cover this; silence is correct |
| `not_a_retrieval_query` | chatter, a follow-up, a "thanks" — should not reach retrieval |

Without the second and third you can measure ranking forever and never measure
the gate. Roughly a third of a real query log is the third category.

### `judged_pool` — the field people skip

Every title shown to the judge, whether marked relevant or not.

Without it, "this title is absent from `relevant`" is ambiguous between *judged
and rejected* and *never seen*. With it, a future retriever that surfaces
something unjudged scores as **unknown**, not wrong, and you re-pool only those
rather than redoing the whole set. This is what makes labels outlive the
retriever that produced the pool.

## Pooling, not scanning

Judging a query against all 1,026 prompts is 1,026 decisions; at 50 queries
that is 51,300. Nobody finishes that.

Pool instead: run several retrieval configurations, take the top 10 from each,
union and dedupe. Three configs give roughly 15–30 unique candidates. Judge
those. Standard IR practice since TREC-1.

Capillaries exposes the channels separately for exactly this — `channels.py`
has `vector_search` and `keyword_search` as standalone functions, and
`union.py` has `union_candidates_broad`.

**Known bias:** a relevant prompt that no configuration retrieved can never be
labeled. You buy this down by pooling from diverse configs, not by pretending
it is absent. Record `pool_config` so a later re-pool is comparable.

## Judge at prompt level

Never at chunk level. Chunk IDs move whenever `TARGET` changes in
`capillaries/chunk.py`, so chunk-level labels expire the next time chunking is
touched — which is the very decision these labels exist to settle.

## Split, and the holdout

Split on `query_id` hash at creation, not at read time, so the assignment
cannot drift:

```
train/  labels.jsonl        full records, readable
holdout/queries.txt         query_ids only — plaintext, so training can EXCLUDE them
holdout/oracle.json         HMAC membership set — stored here, unreadable here
```

Training must know *which* queries are held out (to exclude them) and must not
know *their answers*. Those split cleanly, which is what makes this workable.

`oracle.json` holds `HMAC(key, query_id || title)` for every relevant pair, plus
`HMAC(key, query_id || label)` for the gate verdict. No plaintext titles, no
answer list. Every metric you need — recall@k, MRR, precision@k, gate accuracy —
reduces to *"is this returned title relevant to this query?"*, which is a
membership test. The scorer hashes what the system under test already produced
and checks the set. **It never materializes the answers**, so even an authorized
scoring run cannot dump them.

### Where the key lives

`CAPILLARIES_HOLDOUT_KEY`, absent from the training environment and present
only in the scoring one. Two job definitions, one variable. This is the whole
security boundary, and it is stronger than file permissions because the
*default* path is the one without access.

Keep the oracle a **file**. Marrow has database credentials now, so anything in
Postgres is readable by the training process.

### The leak nobody plans for

A perfect oracle still leaks under repeated use. Score fifty checkpoints against
one holdout and you have fit it by selection, without reading a single label.

- log every scoring call: run id, checkpoint, timestamp, score
- cap submissions per holdout generation
- rotate a fresh slice periodically and retire the burned one

Seed a few canary queries with deliberately atypical relevant sets. A checkpoint
that nails those while performing normally elsewhere is evidence of leakage
rather than skill.

## What flows back to capillaries

Nothing automatic. Capillaries keeps a small frozen smoke set — a dozen
title-keyed cases exported from `train/` — so its own `pytest` still catches a
retrieval regression without the CUDA stack installed. That export is a
derivative, not a second source of truth; regenerate it, never edit it in place.

## Staleness

Before a scoring run, compare each `content_hashes` entry against the live
corpus. A changed hash means the prompt was edited after judgment. Report those
counts; do not silently drop them, and do not silently trust them.
