"""The compile stack a TEST states, in one place.

Production reads it off the endpoint's `uv.lock` (`compile_stack`). A test
has no endpoint, so it states the torch it is actually running under --
which is what makes an adopt audit in-process pass for the right reason.
"""

from __future__ import annotations


def local_stack() -> tuple[tuple[str, str], ...]:
    import importlib.metadata

    return (("torch", importlib.metadata.version("torch")),)


STACK: tuple[tuple[str, str], ...] = (
    ("nvidia-cublas-cu12", "12.8.4"),
    ("torch", "2.13.0"),
)
