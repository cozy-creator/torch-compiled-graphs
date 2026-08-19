"""pgw#1458 + tcg#64: the device is STATED at trace time, and stating it is free.

pgw#1458 established the fact: a graph's device is stamped into every node's
meta, so it is part of the graph's identity and cannot be re-homed by a mint.
Four escalating attempts to move it downstream were each defeated by a
different layer and terminated in AOTInductor's own ``Expected: cpu, Got:
cuda:0`` -- an assertion that fires minutes into a compile and names no graph.

tcg#64 removes the cost of that fact. The session now has TWO devices: the one
it STATES (identity) and the one the drive actually RUNS on (a device that
exists). A GPU-less box drives on cpu -- real sigmas, real token ids, real
``.item()`` -- and each exported program is restated onto the stated device
before it is hashed. So ``cuda`` needs no silicon, and there is no cpu-class
fallback for a user to be told about.

Everything here runs with NO visible CUDA device on purpose. That is the
finding. The permanent asymmetry that remains is that LOADING a cuda-stamped
program does need one, which is why the last link (load, then compile) is
developer-box-and-pod-only and is not tested here.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

torch: Any = pytest.importorskip("torch")

from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode  # noqa: E402

from torchcg import (  # noqa: E402
    DeclarationError,
    GraphSpecialization,
    RuntimeCompatibility,
    build_call_ingress,
    graph_hash,
)
from torchcg.declaration import _canonical_graph  # noqa: E402
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
        # `torch.zeros(..., device=)` is here so a device burnt into a node's
        # kwargs is on the restate path too, not only the node metas.
        pad = torch.zeros(value.shape, device=value.device, dtype=value.dtype)
        return ((self.lin(value) + pad) * self.scale).relu() + self.table.sum()


def _derive(device: str, table: list[float]) -> tuple[HollowSession, Any, Any]:
    session = HollowSession(
        fake_mode=FakeTensorMode(allow_non_fake_inputs=True), device=device
    )
    module = Tiny(table)
    virtualize_parameters(module, session)
    with session.fake_mode:
        example = torch.zeros(
            (2, 4), dtype=torch.float32, device=session.drive_device
        )
        program = torch.export.export(module, (example,))
    _derive.stranded = session.restate(program)  # type: ignore[attr-defined]
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


def test_a_cuda_trace_needs_no_visible_gpu_and_the_GRAPH_is_uniform() -> None:
    """The whole design rests on this: no silicon, uniform cuda GRAPH.

    Uniformity of the graph is the load-bearing half -- it is what enters
    identity and what AOTI compiles for. A weights-free program is fake all
    the way through, so it moves whole and its state dict is uniform too.
    """

    assert not torch.cuda.is_available()
    session, program, _ = _derive("cuda", [1.0, 2.0])
    assert session.drive_device == "cpu", "a GPU-less box drives where it can"
    assert _placement(program) == ["cuda:0"]
    fake = {
        str(value.device)
        for value in program.state_dict.values()
        if isinstance(value, FakeTensor)
    }
    assert fake == {"cuda:0"}, "every weightless entry states the trace device"


def test_the_drive_device_is_the_stated_one_wherever_it_exists() -> None:
    """No split on a host that can hold the tensor -- the fallback is the case."""

    session = HollowSession(fake_mode=FakeTensorMode(), device="cpu")
    assert session.drive_device == "cpu"
    assert session.restate(object()) == (), "a same-device restate touches nothing"


def test_a_real_value_is_never_destroyed_to_reach_uniformity() -> None:
    """The one thing a GPU-less restate cannot do, done the recoverable way.

    A computed buffer holds a VALUE, and a real tensor cannot live on a device
    this host does not have. Faking it would reach uniformity by deleting the
    value -- unrecoverable, and for a lifted constant it would corrupt the
    literal digest as well. So it stays real on the drive device, it is NAMED
    in the restate's return, and ``load_exported_program`` re-homes it where
    the device exists. Placement is recoverable; a value is not.
    """

    session, program, _ = _derive("cuda", [1.0, 2.0])
    stranded = _derive.stranded  # type: ignore[attr-defined]
    assert stranded == ("scale", "table"), stranded
    assert program.state_dict["scale"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert program.constants["table"].tolist() == [1.0, 2.0]
    assert session.restate(program) == stranded, "the restate is idempotent"


def test_a_gpu_less_cuda_trace_keeps_every_value_it_read() -> None:
    """The identity hazard the old GPU-less path had, gone by construction.

    ``graph_hash`` folds ``_literal_digest`` in, so a graph whose lifted
    constants were faked keys off a table of ZEROS and every model differing
    only in its constants collapses into one graph specialization. The session
    used to fake them (a real tensor could not live on the stated device) and
    swap the values back. It no longer fakes anything value-bearing.
    """

    _, program, _ = _derive("cuda", [1.0, 2.0])
    table = program.constants["table"]
    assert not isinstance(table, FakeTensor)
    assert table.tolist() == [1.0, 2.0]

    def identity(values: list[float]) -> str:
        _, exported, example = _derive("cuda", values)
        return graph_hash(exported, build_call_ingress(exported, ("value",), (example,), {}))

    first, second = identity([1.0, 2.0]), identity([7.0, 9.0])
    assert first != second, "faked constants collapsed two models into one graph"
    assert identity([1.0, 2.0]) == first, "graph identity is not deterministic"


def _native_cuda_export(table: list[float]) -> Any:
    """A fake-cuda export done the OLD way: every tensor born on the device.

    This is the thing a GPU box produces, reproduced here without one -- the
    old GPU-less path could build it (a fake tensor needs no silicon), it just
    could not DRIVE a pipeline under it. It is the reference the restate has
    to equal.
    """

    mode = FakeTensorMode(allow_non_fake_inputs=True)
    module = Tiny(table)
    for _prefix, submodule in module.named_modules():
        for name, parameter in list(submodule._parameters.items()):
            if parameter is None:
                continue
            with mode:
                submodule._parameters[name] = torch.nn.Parameter(
                    torch.empty(
                        tuple(parameter.shape), dtype=parameter.dtype, device="cuda"
                    ),
                    requires_grad=False,
                )
        for name, buffer in list(submodule._buffers.items()):
            if buffer is None:
                continue
            with mode:
                submodule._buffers[name] = torch.empty(
                    tuple(buffer.shape), dtype=buffer.dtype, device="cuda"
                )
    with mode:
        module.table = torch.empty(
            tuple(module.table.shape), dtype=module.table.dtype, device="cuda"
        )
        return torch.export.export(
            module, (torch.zeros((2, 4), dtype=torch.float32, device="cuda"),)
        )


def test_a_restated_cpu_trace_is_the_cuda_trace_line_for_line() -> None:
    """The claim the whole fix rests on, asserted as an equality.

    If a restated cpu export differed from a native fake-cuda export by so
    much as one canonical line, a GPU-less derive would key differently from a
    GPU one and every document it published would be unadoptable. Node metas,
    the ``device=`` burnt into a factory node, symbol ranges and the graph
    signature are all in this comparison.
    """

    _, restated, _ = _derive("cuda", [1.0, 2.0])
    native = _native_cuda_export([1.0, 2.0])
    assert list(_canonical_graph(restated)) == list(_canonical_graph(native))


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


def test_a_cpu_trace_keeps_its_constants_real() -> None:
    """The cpu path is bit-for-bit what it was: real values, nothing moved."""

    _, program, _ = _derive("cpu", [1.0, 2.0])
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
    _demote_example_inputs(program)
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


def test_the_restate_survives_the_SESSION_it_runs_inside() -> None:
    """The bug this catches shipped a cuda-labelled document full of cpu graphs.

    ``restate_program`` runs inside the session, where ``OneTraceDevice`` is
    live -- and ``torch.device`` construction goes through
    ``__torch_function__``, so ``torch.device("cuda", 0)`` evaluated in there
    comes back ``cpu:0``. Every move became a no-op, the derive reported a cuda
    trace, and the graphs it emitted were cpu. Nothing above caught it: they
    all call the restate from OUTSIDE any session, which is not how the derive
    calls it. So this one goes through the real front door --
    ``hollow_session`` + ``discover_modules`` -- and asserts the placement of
    what the sink would have stored.
    """

    from torchcg import discover_modules
    from torchcg.hollow import hollow_session

    stored: dict[str, Any] = {}

    with hollow_session("cuda") as session:
        assert session.drive_device == "cpu"
        module = Tiny([1.0, 2.0])
        virtualize_parameters(module, session)

        def drive() -> None:
            with session.fake_mode:
                module(torch.zeros((2, 4), dtype=torch.float32))

        discover_modules(
            "tiny.lane@1",
            {"denoiser": module},
            drive,
            program_sink=lambda graph, program: stored.setdefault(graph, program) and "",
            session=session,
        )

    assert stored, "discovery observed nothing"
    for program in stored.values():
        assert _placement(program) == ["cuda:0"]


def test_the_restate_leaves_the_LIVE_MODULE_where_the_drive_put_it() -> None:
    """A non-strict export shares its tensors with the module it exported.

    The same FakeTensor object is the module's parameter, the program's
    state-dict entry and a placeholder's ``meta['val']``. Re-homing it by
    assignment re-homed the live module, and the NEXT observed call of the
    same target then exported cuda parameters against cpu synthesized inputs
    -- ``FakeTensorDeviceMismatchError: cpu and cuda:0``, raised from a module
    nobody moved. It survived sd1.5 and sdxl (one observed call per target per
    pass) and was caught by a fixture with two.
    """

    session = HollowSession(
        fake_mode=FakeTensorMode(allow_non_fake_inputs=True), device="cuda"
    )
    module = Tiny([1.0, 2.0])
    virtualize_parameters(module, session)

    for _ in range(2):
        assert next(module.parameters()).device.type == "cpu"
        with session.fake_mode:
            program = torch.export.export(
                module, (torch.zeros((2, 4), dtype=torch.float32, device="cpu"),)
            )
        session.restate(program)
        assert _placement(program) == ["cuda:0"]

    assert next(module.parameters()).device.type == "cpu", (
        "the restate re-homed the module it was handed a program of"
    )
