from __future__ import annotations

import pytest

from torchcg import spans


def _closed_table() -> dict[str, float]:
    return {
        "compile_s": 10.0,
        "child_boot_s": 1.0,
        "child_wall_s": 8.5,
        "reap_lag_s": 0.5,
        "parent_other_s": 0.0,
        "child_seal_s": 0.1,
        "child_torch_import_s": 1.5,
        "child_devlock_s": 0.0,
        "child_setup_s": 0.4,
        "child_trace_s": 0.0,
        "child_pack_s": 0.0,
        "compile_wall_s": 6.0,
        "child_other_s": 0.5,
        "lowering_s": 0.5,
        "codegen_s": 1.0,
        "graph_passes_s": 0.3,
        "host_compile_s": 3.7,
        "compile_other_s": 0.5,
    }


def test_partition_is_closed_and_names_dark_time() -> None:
    table = _closed_table()
    assert spans.check(table) == []
    assert spans.dark_fraction(table) == pytest.approx(0.10)


def test_partition_breaks_loudly_for_missing_or_double_counted_members() -> None:
    broken = dict(_closed_table(), host_compile_s=1.0)
    assert any("compile_wall_s" in problem for problem in spans.check(broken))

    missing = _closed_table()
    del missing["child_torch_import_s"]
    assert any("child_torch_import_s" in problem for problem in spans.check(missing))

    doubled = dict(_closed_table(), compile_other_s=-1.0)
    assert any("double-counted" in problem for problem in spans.check(doubled))


def test_every_partition_has_one_dedicated_residual() -> None:
    measured = {
        member
        for members in spans.PARTITIONS.values()
        for member in members
        if member not in spans.RESIDUALS
    }
    assert measured.isdisjoint(spans.RESIDUALS)
    for members in spans.PARTITIONS.values():
        assert len([member for member in members if member in spans.RESIDUALS]) == 1
    assert spans.SPANS_V >= 2


def test_phase_delta_keeps_overlays_out_of_the_partition() -> None:
    partition, overlays, raw = spans.phase_delta(
        {},
        {
            "GraphLowering.run": 1.0,
            "GraphLowering.codegen": 2.0,
            "AotCodeCompiler.compile": 3.0,
            "CachingAutotuner.benchmark_all_configs": 0.25,
            "triton_kernel": 0.5,
        },
    )
    assert partition == {
        "lowering_s": 1.0,
        "codegen_s": 2.0,
        "host_compile_s": 3.0,
        "graph_passes_s": 0,
    }
    assert overlays == {"autotune_s": 0.25, "triton_s": 0.5}
    assert sum(partition.values()) == 6.0
    assert sum(raw.values()) == 6.75


def test_span_ledger_always_records_the_residual(monkeypatch: pytest.MonkeyPatch) -> None:
    ticks = iter((10.0, 12.0, 15.0, 15.0))
    monkeypatch.setattr("torchcg.spans.time.monotonic", lambda: next(ticks))
    ledger = spans.SpanLedger()
    with ledger.span("work_s"):
        pass
    table = ledger.close("total_s", "other_s")
    assert table == {"work_s": 3.0, "total_s": 5.0, "other_s": 2.0}
