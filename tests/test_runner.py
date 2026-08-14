from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from hashrepo import CASRef

import torch_compiled_graphs.runner as runner_module
from torch_compiled_graphs import CompiledGraphRunner, ConstantBindingError
from torch_compiled_graphs.storage import StoredCompiledGraph


class FakePackage:
    def __init__(self, names: tuple[str, ...]) -> None:
        self.names = names
        self.loaded: dict[str, Any] | None = None
        self.calls = 0

    def get_constant_fqns(self) -> tuple[str, ...]:
        return self.names

    def load_constants(
        self,
        values: dict[str, Any],
        *,
        check_full_update: bool,
        user_managed: bool,
    ) -> None:
        assert check_full_update and user_managed
        self.loaded = values

    def __call__(self, *feeds: object) -> tuple[object, ...]:
        self.calls += 1
        return feeds


def _graph(tmp_path: Path, constants: list[dict[str, object]]) -> StoredCompiledGraph:
    directory = tmp_path / "graph"
    directory.mkdir()
    (directory / "model.pt2").write_bytes(b"package")
    return StoredCompiledGraph(
        "cg-key-v1-" + "0" * 56,
        directory,
        {"graph_class": {"name": "denoiser", "constants": constants}},
        CASRef.parse("sha256:" + "1" * 64),
    )


def test_runner_refuses_call_until_exact_complete_bind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = FakePackage(("weight",))
    monkeypatch.setattr(runner_module, "_load_package", lambda path, name: package)
    graph = _graph(
        tmp_path,
        [{"fqn": "weight", "source": "state_dict", "dtype": "float32", "shape": [1]}],
    )
    runner = CompiledGraphRunner(graph)
    with pytest.raises(ConstantBindingError) as caught:
        runner("input")
    assert caught.value.reason == "constants_unbound"

    value = object()
    runner.bind({"weight": value}, device="cpu")
    assert runner.bound
    assert runner.bound_fqns == ("weight",)
    assert package.loaded == {"weight": value}
    assert runner("input") == ("input",)
    assert runner.calls == 1


def test_runner_refuses_manifest_package_constant_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = FakePackage(("other",))
    monkeypatch.setattr(runner_module, "_load_package", lambda path, name: package)
    runner = CompiledGraphRunner(
        _graph(
            tmp_path,
            [{"fqn": "weight", "source": "state_dict", "dtype": "float32", "shape": [1]}],
        )
    )
    with pytest.raises(ConstantBindingError) as caught:
        runner.bind({"weight": object()}, device="cpu")
    assert caught.value.reason == "constant_set_mismatch"
    assert package.loaded is None


def test_failed_partial_bind_is_not_retried_on_the_same_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DeviceOOM(RuntimeError):
        pass

    DeviceOOM.__name__ = "OutOfMemoryError"
    package = FakePackage(("weight",))

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("wrapper") from DeviceOOM("CUDA out of memory")

    package.load_constants = fail  # type: ignore[method-assign]
    monkeypatch.setattr(runner_module, "_load_package", lambda path, name: package)
    runner = CompiledGraphRunner(
        _graph(
            tmp_path,
            [{"fqn": "weight", "source": "state_dict", "dtype": "float32", "shape": [1]}],
        )
    )
    with pytest.raises(ConstantBindingError) as caught:
        runner.bind({"weight": object()}, device="cuda")
    assert caught.value.reason == "out_of_memory"
    assert not runner.bound
    with pytest.raises(ConstantBindingError) as retry:
        runner.bind({"weight": object()}, device="cuda")
    assert retry.value.reason == "binding_failed"


def test_unresolved_constants_fail_before_package_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = FakePackage(("weight",))
    monkeypatch.setattr(runner_module, "_load_package", lambda path, name: package)
    runner = CompiledGraphRunner(
        _graph(
            tmp_path,
            [{"fqn": "weight", "source": "state_dict", "dtype": "float32", "shape": [1]}],
        )
    )
    with pytest.raises(ConstantBindingError) as caught:
        runner.bind({}, device="cpu")
    assert caught.value.reason == "constant_unresolved"
    assert package.loaded is None
