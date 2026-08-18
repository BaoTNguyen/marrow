"""Pool candidates from capillaries, judge them, write labels.

    python -m marrow.capillaries_opt.labels pool   --queries q.jsonl --out pool.jsonl
    python -m marrow.capillaries_opt.labels judge  --pool pool.jsonl --out labels.jsonl
    python -m marrow.capillaries_opt.labels split  --labels labels.jsonl --out-dir .

Pooling is the whole trick. Judging a query against 1,026 prompts is 1,026
decisions; at 50 queries nobody finishes. Running three retrieval configs and
judging their union is 15-30 per query instead, which is an hour of work rather
than a fortnight.

STUB: `pool` and `split` are complete. `judge` has the loop and the record
shape but prints candidates rather than rendering a real review UI — swap the
`_ask` function for whatever you want to look at.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Marrow imports capillaries; capillaries never imports marrow. Deferred to call
# time so `--help` and the split subcommand work without a database.
POOL_CONFIGS = ("vector@10", "keyword@10", "union@20")

# Queries whose sha256 starts below this go to holdout. ~20% at 0x33.
HOLDOUT_PREFIX_MAX = 0x33


def query_id(query: str) -> str:
    """Stable id from the text. Same query, same id, across machines and runs."""
    return "sha256:" + hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def is_holdout(qid: str) -> bool:
    """Split on the id, decided once, so it cannot drift between runs."""
    return int(qid.split(":")[1][:2], 16) <= HOLDOUT_PREFIX_MAX


def _read_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict]) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


# --- pool ------------------------------------------------------------------

def pool_one(query: str, per_channel: int = 10) -> tuple[list[str], dict[str, str]]:
    """Union of each channel's top-k, as titles. Returns (titles, content_hashes).

    Titles, not prompt_ids: capillaries assigns prompt_id with gen_random_uuid(),
    so every corpus rebuild invalidates id-keyed labels. Title has a UNIQUE
    constraint and is what ingest upserts on. See SPEC.md.
    """
    import asyncio
    import psycopg2
    from capillaries.config.paths import DB_CONFIG
    from capillaries.search.channels import keyword_search, vector_search
    from capillaries.search.union import union_candidates_broad

    seen: dict[str, None] = {}
    for hit in vector_search(query, top_k=per_channel):
        seen.setdefault(hit["title"], None)
    for hit in keyword_search(query, top_k=per_channel):
        seen.setdefault(hit["title"], None)
    for cand in asyncio.run(union_candidates_broad(query, per_channel=per_channel * 2)):
        seen.setdefault(getattr(cand, "title", None) or cand.prompt_id, None)

    titles = [t for t in seen if t]

    # content_hash is a staleness marker, not identity: an edited prompt keeps
    # its title and changes its hash, which flags the label instead of silently
    # changing what it means.
    hashes: dict[str, str] = {}
    if titles:
        with psycopg2.connect(**DB_CONFIG) as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT title, content_hash FROM prompts WHERE title = ANY(%s)", (titles,)
            )
            hashes = dict(cur.fetchall())
    return titles, hashes


def cmd_pool(args: argparse.Namespace) -> int:
    rows = _read_jsonl(args.queries)
    out = []
    for i, row in enumerate(rows, 1):
        q = row["query"]
        titles, hashes = pool_one(q, args.per_channel)
        out.append({
            "query": q,
            "query_id": query_id(q),
            "source": row.get("source", "unknown"),
            "candidates": titles,
            "content_hashes": hashes,
            "pool_config": list(POOL_CONFIGS),
        })
        print(f"  [{i}/{len(rows)}] {len(titles):3d} candidates  {q[:60]}", file=sys.stderr)
    _write_jsonl(args.out, out)
    print(f"pooled {len(out)} queries -> {args.out}")
    return 0


# --- judge -----------------------------------------------------------------

LABELS = ("answerable", "nothing_relevant", "not_a_retrieval_query")


def _ask(query: str, candidates: list[str]) -> tuple[str, list[str]]:
    """STUB. Replace with a real review surface.

    Must return (label, relevant_titles). Whatever renders it, keep two
    properties: write after every judgment (a thousand decisions will be
    interrupted), and show the whole pool so `judged_pool` means what it says.
    """
    print(f"\n{query}")
    for i, t in enumerate(candidates, 1):
        print(f"  {i:2d}. {t}")
    print(f"  label one of: {', '.join(LABELS)}")
    raise NotImplementedError("wire up a review surface, then delete this line")


def cmd_judge(args: argparse.Namespace) -> int:
    pooled = _read_jsonl(args.pool)
    done = {r["query_id"] for r in _read_jsonl(args.out)} if Path(args.out).exists() else set()
    out_path = Path(args.out)

    # Append-per-judgment, and skip what is already judged: a thousand decisions
    # is several sittings, and a run that loses work on interrupt gets abandoned.
    with out_path.open("a", encoding="utf-8") as fh:
        for row in pooled:
            if row["query_id"] in done:
                continue
            label, relevant = _ask(row["query"], row["candidates"])
            assert label in LABELS, f"bad label: {label}"
            assert all(t in row["candidates"] for t in relevant), "relevant must come from the pool"
            fh.write(json.dumps({
                "query": row["query"],
                "query_id": row["query_id"],
                "source": row["source"],
                "label": label,
                "relevant": relevant,
                # Every title shown, not just the chosen ones. Without this,
                # "absent from relevant" cannot be told apart from "never seen",
                # and the set stops being reusable the moment retrieval changes.
                "judged_pool": row["candidates"],
                "content_hashes": row["content_hashes"],
                "pool_config": row["pool_config"],
                "judged_by": args.judge,
            }) + "\n")
            fh.flush()
    return 0


# --- split -----------------------------------------------------------------

def cmd_split(args: argparse.Namespace) -> int:
    from marrow.capillaries_opt import holdout

    rows = _read_jsonl(args.labels)
    train = [r for r in rows if not is_holdout(r["query_id"])]
    held = [r for r in rows if is_holdout(r["query_id"])]

    out = Path(args.out_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train" / "labels.jsonl", train)

    info = holdout.build(held, out / "holdout")
    print(f"train:   {len(train)} records -> {out / 'train' / 'labels.jsonl'}")
    print(f"holdout: {info['queries']} queries, {info['tags']} tags -> {info['dir']}")
    print("\nThe holdout answers are not written anywhere in readable form.")
    print("Training reads holdout/queries.txt to EXCLUDE those ids.")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="marrow.capillaries_opt.labels", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("pool", help="run retrieval configs, collect candidates to judge")
    s.add_argument("--queries", required=True, help="jsonl with a `query` field per line")
    s.add_argument("--out", required=True)
    s.add_argument("--per-channel", type=int, default=10)
    s.set_defaults(fn=cmd_pool)

    s = sub.add_parser("judge", help="label a pooled file (resumable)")
    s.add_argument("--pool", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--judge", default="unknown")
    s.set_defaults(fn=cmd_judge)

    s = sub.add_parser("split", help="write train/ and an unreadable holdout/")
    s.add_argument("--labels", required=True)
    s.add_argument("--out-dir", default=".")
    s.set_defaults(fn=cmd_split)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
