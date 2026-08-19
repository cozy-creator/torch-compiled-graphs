"""pgw#1458: a graph's device is established at TRACE time, and it is stated.

The defect this closes cost four escalating attempts, each defeated by a
different layer, and terminated in AOTInductor's own internal ``Expected: cpu,
Got: cuda:0`` -- an assertion that fires minutes into a compile and names no
graph specialization. hollow.py had claimed to be "device-neutral" while emitting
device-CPU artifacts; the trace stamps a concrete device into every node's
meta, so there was nothing neutral about it.

Everything here runs with NO visible CUDA device on purpose. That is the
finding: a fake-``cuda`` trace needs no silicon, so the publish derive stays
on a CPU-only box. The permanent asymmetry is that LOADING a cuda-stamped
program does need one -- which is why the last link (load, then compile) is
developer-box-and-pod-only and is not tested here.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

from torch._subclasses.fake_tensor import FakeTensorMode  # noqa: E402

from torchcg import (  # noqa: E402
    DeclarationError,
    GraphSpecialization,
    RuntimeCompatibility,
    build_call_ingress,
    graph_hash,
)
from torchcg.discovery import _demote_example_inputs  # noqa: E402
from torchcg.hollow import (  # noqa: E402
    HollowSession,
    TraceDeviceUnavailable,
    exported_program_devices,
    load_exported_program,
    virtualize_parameters,
)

pytestmark = pytest.mark.skipif(
    torch.cuda.is_available(),
    reason=(
        "these assert the GPU-LESS behaviour -- a fake-cuda trace needing no "
        "silicon, and a cuda-stamped blob refusing to load. On a box with a "
        "visible device both are trivially true and prove nothing."
    ),
)


class Tiny(torch.nn.Module):  # type: ignore[misc]
    """A parameter, a buffer, and a LIFTED CONSTANT -- all three must move."""

    def __init__(self, table: list[float]) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(4, 4)
        self.register_buffer("scale", torch.arange(4, dtype=torch.float32))
        self.table = torch.tensor(table)

    def forward(self, value: Any) -> Any:
        return (self.lin(value) * self.scale).relu() + self.table.sum()


def _derive(device: str, table: list[float]) -> tuple[HollowSession, Any, Any]:
    session = HollowSession(
        fake_mode=FakeTensorMode(allow_non_fake_inputs=True), device=device
    )
    module = Tiny(table)
    virtualize_parameters(module, session)
    with session.fake_mode:
        example = torch.zeros((2, 4), dtype=torch.float32, device=device)
        program = torch.export.export(module, (example,))
    session.restore_constants(program)
    return session, program, example


def _placement(program: Any) -> list[str]:
    devices = set()
    for node in program.graph_module.graph.nodes:
        value = node.meta.get("val")
        for tensor in value if isinstance(value, (list, tuple)) else [value]:
            device = getattr(tensor, "device", None)
            if device is not None:
                devices.add(str(device))
    return sorted(devices)


def _declare(program: Any, example: Any) -> Any:
    ingress = build_call_ingress(program, ("value",), (example,), {})
    return GraphSpecialization("tiny", "denoiser", program, ingress).declare()


def test_a_cuda_trace_needs_no_visible_gpu_and_is_uniformly_placed() -> None:
    """The whole design rests on this: no silicon, uniform cuda placement.

    Uniformity is the load-bearing half. Moving only the parameters leaves the
    buffers and lifted constants real-on-cpu, and the mixed result EXPORTS
    CLEANLY -- it is rejected four layers down by AOTI, not here.
    """

    assert not torch.cuda.is_available()
    _, program, _ = _derive("cuda", [1.0, 2.0])
    assert _placement(program) == ["cuda:0"]


def test_faking_a_constant_would_collide_two_models_and_does_not() -> None:
    """The identity hazard the constant swap-back exists to prevent.

    ``graph_hash`` folds ``_literal_digest`` in, so a graph whose lifted
    constants were left FAKE keys off a table of zeros -- and every model
    differing only in its constants becomes one graph specialization. This is the red
    arm: without ``restore_constants`` these two hashes are equal.
    """

    session, program, _ = _derive("cuda", [1.0, 2.0])
    assert session.real_constants, "a GPU-less cuda trace must fake the constants"

    def identity(table: list[float]) -> str:
        _, exported, example = _derive("cuda", table)
        return graph_hash(exported, build_call_ingress(exported, ("value",), (example,), {}))

    first, second = identity([1.0, 2.0]), identity([7.0, 9.0])
    assert first != second, "faked constants collapsed two models into one graph specialization"
    assert identity([1.0, 2.0]) == first, "graph identity is not deterministic"
    assert program.constants["table"].tolist() == [1.0, 2.0]
    assert not isinstance(program.constants["table"], torch._subclasses.fake_tensor.FakeTensor)


def test_the_declaration_records_the_trace_device_it_derived() -> None:
    """Always recorded, including single-device -- the invisibility is the bug.

    v3 dropped placement whenever it was single-device ("single-device
    placement must be omitted"), so an artifact's own device was absent from
    its declaration and a mismatch had nowhere to surface but inside inductor.
    """

    _, program, example = _derive("cuda", [1.0, 2.0])
    declaration = _declare(program, example)
    assert declaration.placement == ("cuda:0",)
    assert declaration.device_types == ("cuda",)


def test_a_cpu_mint_of_a_cuda_trace_refuses_by_name_before_compiling() -> None:
    """One 30-second named refusal in place of four burned attempts."""

    _, program, example = _derive("cuda", [1.0, 2.0])
    declaration = _declare(program, example)
    runtime = RuntimeCompatibility("cpu", toolchain={"torch": torch.__version__})
    with pytest.raises(DeclarationError, match="was TRACED on"):
        runtime.key(declaration)


def test_device_is_not_arch_so_a_cpu_trace_is_a_different_graph_specialization() -> None:
    """cpu and cuda traces are different GRAPHS; sm never enters graph identity."""

    _, cuda_program, cuda_example = _derive("cuda", [1.0, 2.0])
    _, cpu_program, cpu_example = _derive("cpu", [1.0, 2.0])
    cuda_declaration = _declare(cuda_program, cuda_example)
    cpu_declaration = _declare(cpu_program, cpu_example)

    assert cuda_declaration.specialization_hash != cpu_declaration.specialization_hash
    assert cpu_declaration.device_types == ("cpu",)
    # The cpu runtime keys the cpu graph without complaint: the refusal is
    # about the DEVICE class, never about the arch.
    RuntimeCompatibility("cpu", toolchain={"torch": torch.__version__}).key(cpu_declaration)


def test_a_cpu_trace_fakes_nothing_and_keeps_its_constants_real() -> None:
    """The cpu path is bit-for-bit what it was: real values, nothing captured."""

    session, program, _ = _derive("cpu", [1.0, 2.0])
    assert not session.real_constants
    assert _placement(program) == ["cpu"]
    assert program.constants["table"].tolist() == [1.0, 2.0]


def test_saving_a_cuda_trace_needs_no_gpu_but_loading_one_refuses_by_name(
    tmp_path: Any,
) -> None:
    """The permanent asymmetry, stated so no lane rediscovers it as a gap.

    ``torch.export.load`` reports the real cause -- ``No CUDA GPUs are
    available`` -- to the WARNING log and raises a generic "error when
    deserializing" with nothing chained, so the check is POSITIVE: read the
    archive's own recorded devices first.
    """

    _, program, _ = _derive("cuda", [1.0, 2.0])
    blob = tmp_path / "program.pt2"
    torch.export.save(program, blob)
    assert os.path.getsize(blob) > 0

    assert "cuda" in exported_program_devices(blob)
    with pytest.raises(TraceDeviceUnavailable, match="no visible CUDA device"):
        load_exported_program(blob)


def test_a_cpu_trace_round_trips_through_the_named_loader(tmp_path: Any) -> None:
    """The loader only refuses what this host genuinely cannot rebuild."""

    _, program, _ = _derive("cpu", [1.0, 2.0])
    _demote_example_inputs(program)
    blob = tmp_path / "program.pt2"
    torch.export.save(program, blob)
    assert exported_program_devices(blob) == ("cpu",)
    assert load_exported_program(blob).constants["table"].tolist() == [1.0, 2.0]


def test_a_hollow_program_keeping_its_fake_example_inputs_cannot_be_RELOADED(
    tmp_path: Any,
) -> None:
    """Why the derive demotes them -- the red arm for _demote_example_inputs.

    The fake example inputs pickle ``_reconstruct_fake_tensor``, which torch's
    OWN loader then refuses under ``weights_only=True``. It fails on CPU, so
    it has nothing to do with the device work; it is the second thing that
    made a derived blob unusable, and it is fixed at the derive, not worked
    around at the mint.
    """

    _, program, _ = _derive("cpu", [1.0, 2.0])
    assert program.example_inputs is not None
    kept = tmp_path / "kept.pt2"
    torch.export.save(program, kept)
    with pytest.raises(Exception) as refusal:
        load_exported_program(kept)
    assert not isinstance(refusal.value, TraceDeviceUnavailable)

    _demote_example_inputs(program)
    demoted = tmp_path / "demoted.pt2"
    torch.export.save(program, demoted)
    assert load_exported_program(demoted).constants["table"].tolist() == [1.0, 2.0]
