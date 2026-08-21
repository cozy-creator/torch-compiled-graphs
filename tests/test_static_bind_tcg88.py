"""tcg#88 (pgw#1603): the symbolic export is TRACE-ONLY under ``static_bind``.

One symbolic export per structural group, and what is DECLARED is one static
record per covered call, bound from the parent — byte-identical (graph hash
AND ingress) to the per-bucket static fan it replaces. The falsifier is the
byte-compare: any delta kills the mechanism, not the reader of the numbers.

The subject is the tcg#41/#77 tiny SD1.5-class pipeline: real diffusers
author code, generated weights, nothing downloaded.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("diffusers")
pytest.importorskip("transformers")

import sd15_tiny  # noqa: E402

from torchcg.discovery import discover_lane  # noqa: E402

SHAPES = ((64, 64), (48, 80), (80, 48))


def _drive(pipe: object) -> None:
    for height, width in SHAPES:
        for guidance in (7.5, 1.0):
            pipe(  # type: ignore[operator]
                prompt="a cat",
                num_inference_steps=1,
                guidance_scale=guidance,
                height=height,
                width=width,
                output_type="pt",
            )


def _spatial_only(_target: str, _name: str, axis: int) -> bool:
    """pgw's own derive policy shape: batch axes stay structural forks."""

    return axis >= 2


def _lane(**kwargs: Any) -> object:
    pipe = sd15_tiny.build_pipe()
    return discover_lane(
        sd15_tiny.LANE_CONTRACT,
        ("unet",),
        pipe.components,
        lambda: _drive(pipe),
        **kwargs,
    )


def _rows(lane: object) -> set[tuple[str, str, str]]:
    return {
        (record.target, record.graph,
         json.dumps(record.ingress.as_dict(), sort_keys=True))
        for record in lane.graphs  # type: ignore[attr-defined]
    }


@pytest.fixture(scope="module")
def static_fan() -> object:
    return _lane(dynamic_dims=None)


def test_static_bind_is_byte_identical_to_the_static_fan(
    static_fan: object,
) -> None:
    bound = _lane(dynamic_dims=_spatial_only, static_bind=True)
    assert _rows(bound) == _rows(static_fan)
    # And it really is per-bucket: 3 aspects x 2 CFG batches, nothing merged
    # into a range record — no ingress states a symbol.
    for record in bound.graphs:  # type: ignore[attr-defined]
        assert not record.ingress.symbols


def test_the_store_banks_one_symbolic_parent_and_the_compile_seam_rebinds(
    static_fan: object,
) -> None:
    from torchcg.bind import bind_static_spec, respecialize
    from torchcg.declaration import GraphSpecialization
    from torchcg.graph_identity import graph_hash

    banked: dict[str, object] = {}
    pipe = sd15_tiny.build_pipe()
    lane = discover_lane(
        sd15_tiny.LANE_CONTRACT,
        ("unet",),
        pipe.components,
        lambda: _drive(pipe),
        dynamic_dims=_spatial_only,
        static_bind=True,
        program_sink=lambda graph, program: banked.__setitem__(graph, program),
    )
    records = list(lane.graphs)
    assert len(banked) == len(records)
    # Every record banked a SYMBOLIC parent, and per structural group the
    # parent is ONE object — the bytes dedup in a content-addressed store.
    assert all(getattr(p, "range_constraints", None) for p in banked.values())
    assert len({id(p) for p in banked.values()}) < len(records)
    # The compile seam re-derives the exact requested identity from it.
    record = records[0]
    parent = banked[record.graph]
    rebound = respecialize(parent, record.ingress)
    assert graph_hash(rebound, record.ingress) == record.graph
    spec = GraphSpecialization(
        name=record.graph, target=record.target, program=parent,
        ingress=record.ingress, strict=False,
    )
    bound_spec = bind_static_spec(spec)
    assert not getattr(bound_spec.program, "range_constraints", None)
    assert bound_spec.name == record.graph


def test_a_demoted_axis_is_told_to_the_caller() -> None:
    # dynamic over EVERYTHING marks the batch axis too; sd15's own guards
    # narrow batch [1, 2] to [2, 2] (tcg#78), so the group re-plans with the
    # batch held static — and under static_bind the caller must HEAR that,
    # not just a logger.
    notes: list[str] = []
    bound = _lane(dynamic_dims=True, static_bind=True, notes=notes)
    assert notes, "a demotion happened and the caller was not told"
    assert any("REFUSED" in note for note in notes)
    # The records still restate the full fan, byte-identically.
    assert _rows(bound) == _rows(_lane(dynamic_dims=None))
