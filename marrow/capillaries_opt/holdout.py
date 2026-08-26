"""Holdout oracle: answerable without being readable.

The training process must know which queries are held out, so it can exclude
them, and must not know their answers. Those are separable, which is what makes
this work: query ids go in a plaintext file, answers become an HMAC membership
set.

Scoring asks "is this title relevant to this query?" — a membership test the
oracle answers without ever holding the answer list. Retrieval metrics all
reduce to that test, so nothing is lost by not being able to enumerate.

The key is the boundary. Keep CAPILLARIES_HOLDOUT_KEY out of the training
environment; its absence there is the guarantee, not file permissions.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path

KEY_ENV = "CAPILLARIES_HOLDOUT_KEY"


def _key() -> bytes:
    k = os.environ.get(KEY_ENV)
    if not k:
        raise RuntimeError(
            f"{KEY_ENV} is not set. Scoring needs it; training must run without "
            f"it. If you are seeing this inside a training job, that is the "
            f"boundary working."
        )
    return k.encode()


def tag(query_id: str, value: str, key: bytes | None = None) -> str:
    """One membership tag. Same inputs, same tag; nothing reversible."""
    return hmac.new(key or _key(), f"{query_id}\x00{value}".encode(), hashlib.sha256).hexdigest()


def build(records: list[dict], out_dir: str | Path) -> dict:
    """Write queries.txt (plaintext ids) and oracle.json (opaque tags)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    key = _key()

    tags: set[str] = set()
    for r in records:
        qid = r["query_id"]
        for title in r.get("relevant", []):
            tags.add(tag(qid, title, key))
        # The gate verdict is hidden too. Leaving it plaintext would reveal
        # which queries are supposed to return nothing, which is half the answer.
        tags.add(tag(qid, f"__label__{r['label']}", key))

    (out / "queries.txt").write_text(
        "\n".join(r["query_id"] for r in records) + "\n", encoding="utf-8"
    )
    (out / "oracle.json").write_text(
        json.dumps({"version": 1, "count": len(records), "tags": sorted(tags)}, indent=2),
        encoding="utf-8",
    )
    return {"queries": len(records), "tags": len(tags), "dir": str(out)}


class Oracle:
    """Read-side. Answers yes/no; cannot list."""

    def __init__(self, oracle_path: str | Path):
        self._tags = set(json.loads(Path(oracle_path).read_text())["tags"])
        self._key = _key()

    def is_relevant(self, query_id: str, title: str) -> bool:
        return tag(query_id, title, self._key) in self._tags

    def label_is(self, query_id: str, candidate: str) -> bool:
        return tag(query_id, f"__label__{candidate}", self._key) in self._tags

    def recall_at_k(self, query_id: str, ranked_titles: list[str], k: int) -> float:
        """Fraction of the top-k that is relevant.

        Note this is precision-flavoured on purpose: true recall needs the
        relevant *count*, which the oracle deliberately cannot reveal. Store
        per-query relevant counts alongside the oracle if you need true recall —
        a count leaks far less than the titles themselves.
        """
        if k <= 0:
            return 0.0
        hits = sum(1 for t in ranked_titles[:k] if self.is_relevant(query_id, t))
        return hits / min(k, len(ranked_titles)) if ranked_titles else 0.0
