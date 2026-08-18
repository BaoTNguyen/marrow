"""Guards for the capillaries label format and holdout oracle.

The oracle is the piece worth testing: it is what lets a holdout live in the
same repo as the training that must not see it. If membership tags were
reversible, or the split drifted between runs, the holdout would be quietly
worthless and every number measured against it would be too.

    python -m pytest tests/test_capillaries_opt.py
"""
from __future__ import annotations

import json

import pytest

from marrow.capillaries_opt import holdout
from marrow.capillaries_opt.labels import is_holdout, query_id

KEY = "test-key-not-a-real-one"


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv(holdout.KEY_ENV, KEY)


def test_query_id_is_stable_and_normalized():
    """Same question, same id — across machines, runs, and casing."""
    assert query_id("Build a cash flow model") == query_id("  build a CASH FLOW model  ")
    assert query_id("a") != query_id("b")


def test_split_is_deterministic():
    """Assignment is decided by the id, so it cannot drift between runs."""
    ids = [query_id(f"query number {i}") for i in range(200)]
    first = [is_holdout(i) for i in ids]
    assert first == [is_holdout(i) for i in ids]
    held = sum(first)
    assert 0 < held < len(ids), "split put everything on one side"


def test_oracle_answers_without_revealing(tmp_path):
    records = [
        {"query_id": query_id("cash flow"), "label": "answerable",
         "relevant": ["13-Week Cash Flow Model"]},
        {"query_id": query_id("nonsense zzz"), "label": "nothing_relevant", "relevant": []},
    ]
    info = holdout.build(records, tmp_path / "holdout")
    assert info["queries"] == 2

    oracle = holdout.Oracle(tmp_path / "holdout" / "oracle.json")
    qid = query_id("cash flow")
    assert oracle.is_relevant(qid, "13-Week Cash Flow Model")
    assert not oracle.is_relevant(qid, "Marketing Budget Allocation")
    assert oracle.label_is(qid, "answerable")
    assert not oracle.label_is(qid, "nothing_relevant")

    # The stored form must not contain the answer in readable shape — that is
    # the entire point of storing a holdout next to the training that uses it.
    raw = (tmp_path / "holdout" / "oracle.json").read_text()
    assert "13-Week Cash Flow Model" not in raw
    assert "answerable" not in raw
    assert "relevant" not in json.loads(raw)

    # Query ids ARE plaintext, deliberately: training has to know what to skip.
    assert qid in (tmp_path / "holdout" / "queries.txt").read_text()


def test_oracle_is_useless_without_the_key(tmp_path, monkeypatch):
    """Absence of the key in the training environment is the whole boundary."""
    records = [{"query_id": query_id("q"), "label": "answerable", "relevant": ["T"]}]
    holdout.build(records, tmp_path / "h")

    monkeypatch.delenv(holdout.KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=holdout.KEY_ENV):
        holdout.Oracle(tmp_path / "h" / "oracle.json")


def test_wrong_key_does_not_silently_pass(tmp_path, monkeypatch):
    """A mismatched key must answer 'no', never accidentally 'yes'."""
    qid = query_id("q")
    holdout.build([{"query_id": qid, "label": "answerable", "relevant": ["T"]}], tmp_path / "h")

    monkeypatch.setenv(holdout.KEY_ENV, "a-different-key")
    assert not holdout.Oracle(tmp_path / "h" / "oracle.json").is_relevant(qid, "T")
