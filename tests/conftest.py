"""Suite-wide setup.

THE LAYOUT CORPUS. Every ratified layout morphism has exactly ONE producer --
tensorfs' ``spec/v2/layouts`` records (tcg#87) -- and tensorfs' wheel
deliberately carries no corpus, so a consumer supplies one. In production
torchcg is vendored beside a vendored tensorfs and finds the records
automatically; this suite has no vendored tensorfs tree, so it names the
byte-identical copy under ``tests/testdata/spec/v2``.

That copy is a FIXTURE, not a second author: nothing in ``src/`` reads it, and
``python-gen-worker``'s cross-vendor fence (pgw#1645) holds it byte-identical to
the records the vendored tensorfs actually ships. A fixture that drifted from
the corpus would make this suite prove agreement with itself.
"""

from __future__ import annotations

import os
from pathlib import Path

CORPUS = Path(__file__).resolve().parent / "testdata" / "spec" / "v2"

# `setdefault`, not assignment: a developer pointing the suite at a real
# tensorfs checkout is doing something deliberate and should win.
os.environ.setdefault("TENSORFS_SPEC_V2", str(CORPUS))


def pytest_runtest_teardown() -> None:
    """The layout pairing is `functools.cache`d on a corpus ROOT, and arms that
    move the corpus move the root with it. Clearing after every test keeps one
    arm's private corpus from being the next arm's answer -- a cache that
    outlives its input is a guard that cannot go red."""

    from torchcg import layout

    layout._paired.cache_clear()
