"""Ground truth and optimization for capillaries' retrieval layer.

Marrow owns the queries (they come from sessions it already collects), the
labels, the holdout, and the training. Capillaries owns the corpus and the
acceptance gate. This package is the only place marrow reaches into capillaries,
and it reaches in one direction: top of the stack importing the bottom.

See SPEC.md for the record format and the holdout design.
"""
from __future__ import annotations

__all__ = ["labels", "holdout"]
