from __future__ import annotations

import gc
import weakref
from pathlib import Path
from typing import Any

import pytest
from tensorfs import CASRef

import torchcg.runner as runner_module
from torchcg import CompiledGraphRunner, ConstantBindingError
from torchcg.storage import StoredCompiledGraph


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
        {
            "graph_specialization": {"name": "denoiser", "constants": constants},
            # tcg#83: the binder reads the artifact's declared layout, so the
            # fixture states one exactly as a real artifact must.
            "declared_input_layout": "torch.contiguous@1",
        },
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
    runner = CompiledGraphRunner._from_verified_graph(graph)
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
    runner = CompiledGraphRunner._from_verified_graph(
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
    runner = CompiledGraphRunner._from_verified_graph(
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
    runner = CompiledGraphRunner._from_verified_graph(
        _graph(
            tmp_path,
            [{"fqn": "weight", "source": "state_dict", "dtype": "float32", "shape": [1]}],
        )
    )
    with pytest.raises(ConstantBindingError) as caught:
        runner.bind({}, device="cpu")
    assert caught.value.reason == "constant_unresolved"
    assert package.loaded is None


def test_runner_cannot_be_constructed_around_unverified_bytes() -> None:
    with pytest.raises(TypeError, match="only by Engine.runner"):
        CompiledGraphRunner()


def test_empty_update_still_runs_the_package_full_update_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = FakePackage(())
    monkeypatch.setattr(runner_module, "_load_package", lambda path, name: package)
    runner = CompiledGraphRunner._from_verified_graph(_graph(tmp_path, []))

    runner.bind({}, device="cpu")

    assert package.loaded == {}
    assert runner.bound


def test_literal_load_oom_is_typed_and_poisons_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class OutOfMemoryError(RuntimeError):
        pass

    package = FakePackage(("table",))
    monkeypatch.setattr(runner_module, "_load_package", lambda path, name: package)

    def fail_literals(path: Path | None, device: str) -> dict[str, Any]:
        del path, device
        try:
            raise OutOfMemoryError("CUDA out of memory")
        except OutOfMemoryError as exc:
            raise ConstantBindingError("literal_load_failed", "wrapped") from exc

    monkeypatch.setattr(runner_module, "_load_literals", fail_literals)
    runner = CompiledGraphRunner._from_verified_graph(
        _graph(tmp_path, [{"fqn": "table", "source": "literal"}])
    )

    with pytest.raises(ConstantBindingError) as caught:
        runner.bind({}, device="cuda")
    assert caught.value.reason == "out_of_memory"
    with pytest.raises(ConstantBindingError) as retry:
        runner.bind({}, device="cuda")
    assert retry.value.reason == "binding_failed"


def test_an_unreadable_layout_is_typed_and_poisons_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """tcg#83 replaced the bind-time contiguity COPY with a bind-time QUESTION.

    There is no repack here any more -- bytes in the declared layout bind by
    reference and bytes in any other layout refuse -- so the failure this arm
    guards moved with it: asking a value about its layout can still exhaust the
    device, and that stays typed rather than generic.
    """

    class OutOfMemoryError(RuntimeError):
        pass

    class Value:
        def dim(self) -> int:
            return 1

        def is_contiguous(self, memory_format: object = None) -> bool:
            raise OutOfMemoryError("allocation failed")

    package = FakePackage(("weight",))
    monkeypatch.setattr(runner_module, "_load_package", lambda path, name: package)
    runner = CompiledGraphRunner._from_verified_graph(
        _graph(tmp_path, [{"fqn": "weight", "source": "state_dict"}])
    )

    with pytest.raises(ConstantBindingError) as caught:
        runner.bind({"weight": Value()}, device="cuda")
    assert caught.value.reason == "out_of_memory"
    with pytest.raises(ConstantBindingError) as retry:
        runner.bind({"weight": Value()}, device="cuda")
    assert retry.value.reason == "binding_failed"


def test_failed_partial_user_managed_update_keeps_attempted_values_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Value:
        pass

    package = FakePackage(("weight",))
    observed: weakref.ReferenceType[Value] | None = None

    def fail(values: dict[str, Any], *, check_full_update: bool, user_managed: bool) -> None:
        nonlocal observed
        assert check_full_update and user_managed
        observed = weakref.ref(values["weight"])
        raise RuntimeError("partial update")

    package.load_constants = fail  # type: ignore[method-assign]
    monkeypatch.setattr(runner_module, "_load_package", lambda path, name: package)
    runner = CompiledGraphRunner._from_verified_graph(
        _graph(tmp_path, [{"fqn": "weight", "source": "state_dict"}])
    )
    value = Value()

    with pytest.raises(ConstantBindingError) as caught:
        runner.bind({"weight": value}, device="cpu")
    assert caught.value.reason == "injection_failed"
    del value
    gc.collect()
    assert observed is not None and observed() is not None
    assert runner.bound_fqns == ()


def test_package_load_nested_oom_is_typed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class OutOfMemoryError(RuntimeError):
        pass

    def fail(path: Path, graph_specialization: str) -> Any:
        del path, graph_specialization
        try:
            raise OutOfMemoryError("CUDA out of memory")
        except OutOfMemoryError as exc:
            raise ConstantBindingError("package_load_failed", "wrapped") from exc

    monkeypatch.setattr(runner_module, "_load_package", fail)
    with pytest.raises(ConstantBindingError) as caught:
        CompiledGraphRunner._from_verified_graph(_graph(tmp_path, []))
    assert caught.value.reason == "out_of_memory"
