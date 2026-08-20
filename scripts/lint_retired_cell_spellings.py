#!/usr/bin/env python3
"""pgw#1547: the word `cell` is retired for compiled graphs, and stays retired.

Paul, 2026-08-20: *"we stopped using the term 'cell' a while ago. They are now
'graphs' and 'graph specializations'."* A target has ONE graph and MANY graph
SPECIALIZATIONS (tcg#56); the durable artifact is a COMPILED GRAPH keyed by a
`cg-key-v1` value. Nothing in this vocabulary is a "cell".

WHY A FENCE AND NOT A SWEEP. This package is the UPSTREAM of the vocabulary —
python-gen-worker vendors `contracts/` and `identity.py` byte-for-byte — so a
`cell` that lands here is copied into a consumer rather than merely written. The
word had already survived one retirement: pgw#1363 renamed it across the worker
on 2026-08-18, and two days later `docs/RECIPE.md` still described a compiled
graph as "one cell". `lint_retired_graph_class_spellings.py` in that repo makes
the identical argument for `graph class`, which was renamed here in tcg#56.

WHAT IT LOOKS FOR is the substring `cell` NOT PRECEDED BY A LETTER, swept over
`.py`/`.md`/`.toml`/`.json` sources, case-insensitively. A TEXT sweep, not an
AST one, because prose is the failure mode: a docstring calling a compiled graph
"the cell" sends the next reader to a vocabulary the fleet does not speak.

WHY THE LOOKBEHIND, and it is load-bearing. `cancelled`, `CancelledError` and
`cancellation` all contain the letters `cell`. They are excluded STRUCTURALLY
(the `c` of `can-cell-ed` is a letter) rather than by an allowlist that would
have to grow every time something is cancelled. `cell_contents` — the *Python*
closure-cell API — is a different word and is allowlisted by exact token.

EXCLUSIONS, structural rather than a list of findings:

* this file, which DEFINES the retired spelling and so must spell it;
* `src/torchcg/contracts/` and `tests/testdata/` — the shared conformance
  corpora. These are vendored BYTE-IDENTICALLY into python-gen-worker (all four
  copies hash alike) and pinned by `KEY_GRAMMAR_DIGEST` in both repos, so
  editing their prose reddens two repos at once and is a coordinated corpus
  bump rather than a rename. They are CLEAN as of tcg#74 / pgw#1554, which
  corrected `"gen_worker.cell_key.is_key"` — a module the worker does not have —
  to `gen_worker._vendor.torchcg.identity.is_compiled_graph_key`, and respelled
  the one remaining prose note. The exclusion stays because the COUPLING is
  what makes these files special, not their current contents.

Everything else is recognised by a PROOF AT THE LINE: a `cell-spelling:` marker
with a reason after the colon. A bare marker is not a proof.

Usage:

    python scripts/lint_retired_cell_spellings.py [PATH ...]
    python scripts/lint_retired_cell_spellings.py --selftest
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parents[1]

PATTERN = re.compile(r"(?<![A-Za-z])cell", re.IGNORECASE)
ALLOWED_TOKENS = ("cell_contents",)
MARKER = "cell-spelling:"
SUCCESSOR = "graph / compiled graph / graph specialization"

EXCLUDED_PARTS = ("__pycache__", ".git", ".venv", "target", "node_modules")
EXCLUDED_SUFFIXES = ("lint_retired_cell_spellings.py",)
EXCLUDED_RELATIVE = ("src/torchcg/contracts", "tests/testdata")


def _excluded(path: Path) -> bool:
    if any(part in EXCLUDED_PARTS for part in path.parts):
        return True
    if str(path).endswith(EXCLUDED_SUFFIXES):
        return True
    try:
        rel = path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        return False
    return any(rel == e or rel.startswith(e + "/") for e in EXCLUDED_RELATIVE)


def scan_text(text: str) -> list[tuple[int, str]]:
    """Every retired spelling in `text`, minus lines carrying a proof."""
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if MARKER in line and line.split(MARKER, 1)[1].strip():
            continue
        probe = line
        for token in ALLOWED_TOKENS:
            probe = probe.replace(token, "")
        m = PATTERN.search(probe)
        if m:
            start = max(0, m.start() - 12)
            hits.append((lineno, probe[start:m.end() + 20].strip()))
    return hits


def _walk(roots: list[Path]) -> Iterator[Path]:
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            if not _excluded(root):
                yield root
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".md", ".toml", ".json"}:
                if not _excluded(path):
                    yield path


def selftest() -> int:
    """A sweep that can only ever print 'clean' guards nothing."""
    ok = True

    def check(label: str, text: str, want: bool) -> None:
        nonlocal ok
        hits = scan_text(text)
        if bool(hits) != want:
            print(f"SELFTEST FAIL [{label}]: want hits={want}, got {hits}")
            ok = False

    check("retired prose", '"""One cell serves sixteen fine-tunes."""\n', True)
    check("retired plural", "# two cells per family\n", True)
    check("retired identifier", "CELL_DIR = 'aot-cells'\n", True)
    check("retired CamelCase", "class CellRef: ...\n", True)
    check("current vocabulary",
          '"""One graph, many graph specializations, keyed by cg-key-v1."""\n', False)
    # If this arm goes red the pattern over-reached; fix the PATTERN, not the source.
    check("cancel family",
          "from asyncio import CancelledError\n"
          "task.cancel()\n"
          "# cancellation is cooperative\n"
          "was_cancelled = True\n", False)
    check("python closure api",
          "for slot in fn.__closure__ or ():\n"
          "    candidate = slot.cell_contents\n", False)
    check("proof at the line",
          "note = 'a cell REF'  # cell-spelling: vendored corpus, digest-pinned\n", False)
    check("bare marker is not a proof",
          "note = 'a cell REF'  # cell-spelling:\n", True)

    print("selftest: OK" if ok else "selftest: FAILED")
    return 0 if ok else 1


def main(argv: list[str]) -> int:
    if "--selftest" in argv:
        return selftest()
    roots = [Path(a) for a in argv] or [
        REPO / "src", REPO / "tests", REPO / "docs", REPO / "benchmarks",
        REPO / "README.md",
    ]
    problems: list[str] = []
    for path in _walk(roots):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, excerpt in scan_text(text):
            rel = path.resolve()
            try:
                rel = rel.relative_to(REPO)
            except ValueError:
                pass
            problems.append(f"{rel}:{lineno}: retired spelling in {excerpt!r}")
    if problems:
        for p in problems:
            print(p)
        print(
            f"\nlint_retired_cell_spellings: {len(problems)} finding(s). The word `cell` is "
            f"retired for compiled graphs — write {SUCCESSOR}.\n"
            f"If a line must keep it, prove it at the line with `# {MARKER} <reason>`."
        )
        return 1
    print("pgw#1547: clean — no retired `cell` spelling survives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
