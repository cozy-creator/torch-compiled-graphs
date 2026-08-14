from __future__ import annotations

import platform
import threading
from typing import Any

import pytest

import torch_compiled_graphs.host_isa as host_isa

torch: Any = pytest.importorskip("torch")


@pytest.mark.skipif(platform.machine() != "x86_64", reason="x86-64 ISA-level semantics")
def test_real_host_policy_is_capped_and_process_wide() -> None:
    facts = host_isa._impose_host_policy()
    assert host_isa._RANK[facts["host_isa_level"]] <= host_isa._RANK["x86-64-v3"]
    assert host_isa._decoded_features(facts["host_isa_features"]) <= host_isa._cpu_features()

    from torch._inductor import cpp_builder

    result: dict[str, object] = {}

    def build_flags() -> None:
        try:
            compiler = cpp_builder.get_cpp_compiler()
            result["flags"] = cpp_builder._get_cpu_arch_cflags(compiler)
        except Exception as exc:  # pragma: no cover - guards Torch internal drift
            result["error"] = exc

    thread = threading.Thread(target=build_flags)
    thread.start()
    thread.join()
    assert "error" not in result, result.get("error")
    flags = result.get("flags")
    assert flags == [f"march={facts['cpp_march']}"]


def test_concurrent_policy_calls_converge_on_one_process_default() -> None:
    results: list[dict[str, str]] = []
    errors: list[BaseException] = []

    def impose() -> None:
        try:
            results.append(host_isa._impose_host_policy())
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    threads = [threading.Thread(target=impose) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(results) == len(threads)
    assert all(result == results[0] for result in results)


def test_non_x86_policy_is_native_but_feature_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    with monkeypatch.context() as patch:
        patch.setattr(platform, "machine", lambda: "aarch64")
        patch.setattr(host_isa, "_cpu_features", lambda: frozenset({"asimd", "fp"}))
        facts = host_isa._impose_host_policy()
        assert facts == {
            "machine": "aarch64",
            "host_isa_level": "native",
            "host_isa_features": "asimd,fp",
            "cpp_march": "native",
            "cpp_simdlen": "native",
        }
        host_isa._admit_host(facts)
    host_isa._impose_host_policy()


def test_unknown_native_host_without_features_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "riscv64")
    monkeypatch.setattr(host_isa, "_cpu_features", frozenset)
    with pytest.raises(host_isa.HostISAError, match="cannot state a native ISA"):
        host_isa._impose_host_policy()


@pytest.mark.parametrize(
    "flag",
    [
        "-march=native",
        "-march=x86-64-v4",
        "-march=skylake-avx512",
        "-mavx512f",
    ],
)
def test_host_compile_command_refuses_an_x86_isa_escape(
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    with pytest.raises(host_isa.HostISAError, match="ISA clamp"):
        host_isa._assert_command_is_clamped(f"g++ -march=x86-64-v3 {flag} wrapper.cpp -c")


def test_x86_host_compile_requires_the_exact_policy_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    host_isa._assert_command_is_clamped("g++ -march=x86-64-v3 -mavx2 -mfma -mf16c wrapper.cpp -c")
    with pytest.raises(host_isa.HostISAError, match="ISA clamp"):
        host_isa._assert_command_is_clamped("g++ wrapper.cpp -c")


def test_non_x86_host_compile_keeps_its_native_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    host_isa._assert_command_is_clamped("g++ -march=native wrapper.cpp -c")
